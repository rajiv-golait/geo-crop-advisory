"""
FastAPI backend for the refactored crop recommendation pipeline.

New flow:
1. Client sends only latitude + longitude.
2. Backend finds the nearest district in CropDataset-Enhanced.csv using
   Haversine distance.
3. Backend derives approximate N/P/K values from the dominant soil category
   percentages for that district.
4. Backend fetches temperature, rainfall, and humidity from Open-Meteo
   (recent past + forecast, ``best_match`` model, blended current + daily stats).
5. Backend predicts the crop with random_forest_model.pkl and decodes it with
   label_encoder.pkl.

Run:
    uvicorn backend_api:app --reload
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "CropDataset-Enhanced.csv"
MODEL_PATH = APP_DIR / "models" / "random_forest_model.pkl"
ENCODER_PATH = APP_DIR / "models" / "label_encoder.pkl"

FEATURE_ORDER = ["N", "P", "K", "pH", "rainfall", "temperature", "humidity"]
PH_DEFAULT = 6.5
# Only used if Open-Meteo omits humidity (should be rare).
FALLBACK_HUMIDITY = 60.0

N_VALS = {"Low": 25.0, "Medium": 40.0, "High": 70.0}
P_VALS = {"Low": 27.0, "Medium": 43.0, "High": 60.0}
K_VALS = {"Low": 23.0, "Medium": 38.0, "High": 52.0}

N_LIMITS = (0.0, 140.0)
P_LIMITS = (5.0, 145.0)
K_LIMITS = (5.0, 205.0)

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Recent observed/reanalysis (past_days) + short forecast improves realism vs
# forecast-only means. Rainfall = sum of daily mm over the full window.
PAST_DAYS = 14
FORECAST_DAYS = 16
# Blend: weight on "now" (current) vs multi-day mean — stabilises noise but
# keeps conditions representative.
CURRENT_BLEND_WEIGHT = 0.35

app = FastAPI(
    title="Crop Recommendation Backend",
    version="2.0.0",
    description="Lat/lon driven crop recommendation API using nearest-district soil lookup.",
)


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="User latitude")
    longitude: float = Field(..., ge=-180, le=180, description="User longitude")


class PredictResponse(BaseModel):
    recommended_crop: str
    context: dict[str, Any]


@lru_cache(maxsize=1)
def load_soil_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Enhanced soil dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {
        "Address",
        "Formatted address",
        "Latitude",
        "Longitude",
        "Nitrogen - Low",
        "Nitrogen - Medium",
        "Nitrogen - High",
        "Phosphorous - Low",
        "Phosphorous - Medium",
        "Phosphorous - High",
        "Potassium - Low",
        "Potassium - Medium",
        "Potassium - High",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Enhanced soil dataset is missing columns: {missing}")

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Enhanced soil dataset has no usable latitude/longitude rows.")
    return df


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_label_encoder():
    if not ENCODER_PATH.is_file():
        raise FileNotFoundError(f"Label encoder not found: {ENCODER_PATH}")
    return joblib.load(ENCODER_PATH)


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Compute great-circle distance in kilometers.

    Haversine works on latitude/longitude over a sphere, which is much better
    than plain Euclidean distance for geographic nearest-neighbor lookup.
    """
    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = lat2.apply(math.radians)
    lon2_rad = lon2.apply(math.radians)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        (dlat / 2).apply(math.sin) ** 2
        + math.cos(lat1_rad) * lat2_rad.apply(math.cos) * (dlon / 2).apply(math.sin) ** 2
    )
    c = 2 * a.apply(lambda value: math.asin(math.sqrt(value)))
    return earth_radius_km * c


def find_closest_district(latitude: float, longitude: float, soil_df: pd.DataFrame) -> pd.Series:
    enriched = soil_df.copy()
    enriched["distance_km"] = haversine_km(
        latitude,
        longitude,
        enriched["Latitude"],
        enriched["Longitude"],
    )
    return enriched.loc[enriched["distance_km"].idxmin()]


def parse_percentage(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    cleaned = str(value).strip().replace("%", "")
    if not cleaned:
        return 0.0
    return float(cleaned)


def calculate_expected_nutrient(row: pd.Series, prefix: str, vals: dict[str, float], limits: tuple[float, float]) -> dict[str, Any]:
    """
    Calculate the expected value based on percentages of Low, Medium, and High.
    Normalizes the percentages first, then weights them by the median values for each tier.
    Clips to known training dataset boundaries.
    """
    raw_low = parse_percentage(row[f"{prefix} - Low"])
    raw_med = parse_percentage(row[f"{prefix} - Medium"])
    raw_high = parse_percentage(row[f"{prefix} - High"])
    
    total = raw_low + raw_med + raw_high
    if total == 0:
        total = 1.0  # fallback to avoid division by zero
        
    prob_low = raw_low / total
    prob_med = raw_med / total
    prob_high = raw_high / total
    
    expected_val = (prob_low * vals["Low"]) + (prob_med * vals["Medium"]) + (prob_high * vals["High"])
    expected_val = max(limits[0], min(limits[1], expected_val))
    
    # Still determine dominant for summary context
    scores = {"Low": raw_low, "Medium": raw_med, "High": raw_high}
    dominant_level = max(scores, key=scores.get)
    dominant_pct = scores[dominant_level]
    
    return {
        "value": expected_val,
        "dominant_level": dominant_level,
        "dominant_pct": dominant_pct
    }


def extract_soil_features(row: pd.Series) -> dict[str, float | str]:
    n_info = calculate_expected_nutrient(row, "Nitrogen", N_VALS, N_LIMITS)
    p_info = calculate_expected_nutrient(row, "Phosphorous", P_VALS, P_LIMITS)
    k_info = calculate_expected_nutrient(row, "Potassium", K_VALS, K_LIMITS)

    # Inject slight gaussian variance to prevent static pH from overfitting predictions
    ph_val = random.gauss(PH_DEFAULT, 0.3)
    ph_val = max(3.5, min(9.9, ph_val))

    return {
        "N": n_info["value"],
        "P": p_info["value"],
        "K": k_info["value"],
        "pH": round(ph_val, 3),
        "nitrogen_level": n_info["dominant_level"],
        "nitrogen_pct": round(n_info["dominant_pct"], 2),
        "phosphorous_level": p_info["dominant_level"],
        "phosphorous_pct": round(p_info["dominant_pct"], 2),
        "potassium_level": k_info["dominant_level"],
        "potassium_pct": round(k_info["dominant_pct"], 2),
    }


def _num_list(values: list[Any] | None) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for x in values:
        if x is None:
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out


def fetch_weather_summary(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Pull temperature, rainfall, and humidity from Open-Meteo for this point.

    Uses ``past_days`` + ``forecast_days`` so totals/averages reflect recent
    weather plus the near-term outlook. Humidity uses daily mean RH and current
    RH when available (blended).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "models": "best_match",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
    }
    response = requests.get(OPEN_METEO_FORECAST, params=params, timeout=30)
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo request failed with status {response.status_code}.",
        )

    body = response.json()
    daily = body.get("daily") or {}
    current = body.get("current") or {}

    temps_daily = _num_list(daily.get("temperature_2m_mean"))
    rain_daily = _num_list(daily.get("precipitation_sum"))
    rh_daily = _num_list(daily.get("relative_humidity_2m_mean"))

    if not temps_daily or not rain_daily:
        raise HTTPException(
            status_code=502,
            detail="Open-Meteo response missing daily temperature and/or precipitation arrays.",
        )

    temp_daily_mean = sum(temps_daily) / len(temps_daily)
    total_rainfall = sum(rain_daily)
    rh_daily_mean = sum(rh_daily) / len(rh_daily) if rh_daily else None

    cur_temp = current.get("temperature_2m")
    cur_rh = current.get("relative_humidity_2m")

    try:
        cur_temp_f = float(cur_temp) if cur_temp is not None else None
    except (TypeError, ValueError):
        cur_temp_f = None
    try:
        cur_rh_f = float(cur_rh) if cur_rh is not None else None
    except (TypeError, ValueError):
        cur_rh_f = None

    w = CURRENT_BLEND_WEIGHT
    if cur_temp_f is not None:
        temperature = w * cur_temp_f + (1.0 - w) * temp_daily_mean
    else:
        temperature = temp_daily_mean

    if cur_rh_f is not None and rh_daily_mean is not None:
        humidity = w * cur_rh_f + (1.0 - w) * rh_daily_mean
    elif cur_rh_f is not None:
        humidity = cur_rh_f
    elif rh_daily_mean is not None:
        humidity = rh_daily_mean
    else:
        humidity = FALLBACK_HUMIDITY

    humidity = max(0.0, min(100.0, humidity))

    meta = {
        "source": "open-meteo.com (v1/forecast)",
        "model_preference": "best_match",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "aggregation_days_count": len(temps_daily),
        "rainfall_mm_period_total": round(total_rainfall, 2),
        "rainfall_note": f"Sum of daily precipitation_sum over {len(rain_daily)} days "
        f"({PAST_DAYS} past + up to {FORECAST_DAYS} forecast).",
        "temperature_c": round(temperature, 2),
        "temperature_note": "Blend of current temperature_2m and mean of daily temperature_2m_mean "
        f"(current weight {CURRENT_BLEND_WEIGHT}).",
        "humidity_pct": round(humidity, 2),
        "humidity_note": "Blend of current relative_humidity_2m and mean of daily "
        "relative_humidity_2m_mean when both exist.",
        "current_temperature_2m": cur_temp_f,
        "current_relative_humidity_2m": cur_rh_f,
        "daily_mean_temperature_2m": round(temp_daily_mean, 2),
        "daily_mean_relative_humidity_2m": round(rh_daily_mean, 2) if rh_daily_mean is not None else None,
    }

    return {
        "temperature": round(temperature, 2),
        "rainfall": round(total_rainfall, 2),
        "humidity": round(humidity, 2),
        "meta": meta,
    }


def build_feature_frame(
    *,
    n: float,
    p: float,
    k: float,
    ph: float,
    rainfall: float,
    temperature: float,
    humidity: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [[n, p, k, ph, rainfall, temperature, humidity]],
        columns=FEATURE_ORDER,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_crop(payload: PredictRequest) -> PredictResponse:
    try:
        soil_df = load_soil_dataset()
        model = load_model()
        label_encoder = load_label_encoder()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    nearest = find_closest_district(payload.latitude, payload.longitude, soil_df)
    soil_features = extract_soil_features(nearest)
    weather = fetch_weather_summary(payload.latitude, payload.longitude)

    feature_frame = build_feature_frame(
        n=float(soil_features["N"]),
        p=float(soil_features["P"]),
        k=float(soil_features["K"]),
        ph=float(soil_features["pH"]),
        rainfall=float(weather["rainfall"]),
        temperature=float(weather["temperature"]),
        humidity=float(weather["humidity"]),
    )

    try:
        encoded_prediction = model.predict(feature_frame)
        predicted_crop = label_encoder.inverse_transform(encoded_prediction)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    context = {
        "nearest_district": str(nearest.get("Address") or nearest.get("Formatted address") or ""),
        "district_coordinates": {
            "latitude": float(nearest["Latitude"]),
            "longitude": float(nearest["Longitude"]),
            "distance_km": round(float(nearest["distance_km"]), 3),
        },
        "model_features": {
            "N": int(soil_features["N"]),
            "P": int(soil_features["P"]),
            "K": int(soil_features["K"]),
            "pH": float(soil_features["pH"]),
            "rainfall": float(weather["rainfall"]),
            "temperature": float(weather["temperature"]),
            "humidity": float(weather["humidity"]),
        },
        "soil_summary": {
            "nitrogen": {
                "dominant_level": str(soil_features["nitrogen_level"]),
                "dominant_percentage": float(soil_features["nitrogen_pct"]),
            },
            "phosphorous": {
                "dominant_level": str(soil_features["phosphorous_level"]),
                "dominant_percentage": float(soil_features["phosphorous_pct"]),
            },
            "potassium": {
                "dominant_level": str(soil_features["potassium_level"]),
                "dominant_percentage": float(soil_features["potassium_pct"]),
            },
        },
        "weather_meta": weather.get("meta") or {},
    }

    return PredictResponse(recommended_crop=str(predicted_crop), context=context)
