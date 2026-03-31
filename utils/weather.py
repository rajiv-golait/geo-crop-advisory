"""Geocoding + current weather.

Primary: **Open-Meteo** (no API key; real lat/lon + live conditions).

Optional: **OpenWeather** only if ``OPENWEATHER_API_KEY`` is set and
``USE_OPENWEATHER=1`` — then we try OpenWeather first and fall back to
Open-Meteo on any failure (keys, 401, quota).
"""

from __future__ import annotations

import os
from typing import Any

import requests

OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

OW_GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
OW_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# Identify this app for Nominatim usage policy.
NOMINATIM_UA = "MDM-CropAdvisory/1.0 (university project; educational use)"


class WeatherError(Exception):
    """Raised when location or weather cannot be fetched."""


def _clean_api_key(raw: str) -> str:
    k = raw.strip()
    if len(k) >= 2 and k[0] == k[-1] and k[0] in "\"'":
        k = k[1:-1].strip()
    if k.startswith("\ufeff"):
        k = k.lstrip("\ufeff").strip()
    return k


def _openweather_key() -> str | None:
    k = _clean_api_key(os.environ.get("OPENWEATHER_API_KEY", ""))
    return k or None


def _use_openweather_first() -> bool:
    v = os.environ.get("USE_OPENWEATHER", "").strip().lower()
    return v in ("1", "true", "yes")


# --- Open-Meteo (default) -----------------------------------------------------


def _om_search_variants(place: str) -> list[str]:
    p = place.strip()
    if not p:
        return []
    out = [p]
    lower = p.lower()
    if "," not in p:
        out.extend([f"{p}, India", f"{p}, Maharashtra, IN"])
    else:
        if "india" not in lower and "maharashtra" not in lower:
            out.append(f"{p}, India")
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _pick_om_result(results: list[dict[str, Any]], hint: str) -> dict[str, Any]:
    hint_root = hint.strip().lower().split(",")[0].strip()
    for r in results:
        if str(r.get("country_code") or "").upper() == "IN":
            return r
    for r in results:
        name = str(r.get("name") or "").lower()
        if hint_root and (hint_root in name or name in hint_root):
            return r
    return results[0]


def geocode_open_meteo(place: str) -> tuple[float, float, str]:
    place = place.strip()
    if not place:
        raise WeatherError("Location is empty.")

    for name in _om_search_variants(place):
        results: list[dict[str, Any]] = []
        for country_code in ("IN", None):
            params: dict[str, Any] = {
                "name": name,
                "count": 10,
                "language": "en",
                "format": "json",
            }
            if country_code:
                params["countryCode"] = country_code
            r = requests.get(OPEN_METEO_GEO, params=params, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            results = data.get("results") or []
            if results:
                break
        if not results:
            continue
        row = _pick_om_result(results, place)
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        nm = row.get("name") or place.split(",")[0].strip()
        admin1 = row.get("admin1") or ""
        country = row.get("country") or row.get("country_code") or ""
        parts = [nm]
        if admin1:
            parts.append(admin1)
        if country:
            parts.append(country)
        display = ", ".join(parts)
        return lat, lon, display

    # Nominatim fallback (no API key; be gentle with usage)
    q = place if "," in place else f"{place}, India"
    r = requests.get(
        NOMINATIM_SEARCH,
        params={
            "q": q,
            "format": "json",
            "limit": 5,
            "addressdetails": 0,
        },
        headers={"User-Agent": NOMINATIM_UA},
        timeout=20,
    )
    if r.status_code != 200:
        raise WeatherError(
            "Location not found. Try adding state (e.g. 'Akola, Maharashtra') or check spelling."
        )
    rows = r.json()
    if not rows:
        raise WeatherError(
            "Location not found. Try adding state (e.g. 'Akola, Maharashtra') or check spelling."
        )
    row = rows[0]
    lat, lon = float(row["lat"]), float(row["lon"])
    display = row.get("display_name") or place
    return lat, lon, display


def fetch_weather_open_meteo(lat: float, lon: float) -> tuple[float, float, float]:
    r = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "timezone": "auto",
        },
        timeout=20,
    )
    if r.status_code != 200:
        raise WeatherError(f"Open-Meteo forecast failed (HTTP {r.status_code}).")
    body = r.json()
    cur = body.get("current") or {}
    try:
        temp = float(cur["temperature_2m"])
        humidity = float(cur["relative_humidity_2m"])
    except (KeyError, TypeError, ValueError) as e:
        raise WeatherError("Open-Meteo response missing temperature or humidity.") from e
    precip = cur.get("precipitation")
    rainfall = float(precip) if precip is not None else 0.0
    return temp, humidity, max(0.0, rainfall)


# --- OpenWeather (optional) ---------------------------------------------------


def _ow_geocode_queries(place: str) -> list[str]:
    p = place.strip()
    if not p:
        return []
    queries = [p]
    lower = p.lower()
    if "," not in p:
        queries.extend([f"{p}, IN", f"{p}, India", f"{p}, Maharashtra, IN"])
    else:
        if (
            "india" not in lower
            and not lower.endswith(", in")
            and "maharashtra" not in lower
        ):
            queries.append(f"{p}, IN")
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _fetch_ow_geocode_raw(query: str, api_key: str, *, limit: int = 5) -> list[dict[str, Any]]:
    r = requests.get(
        OW_GEOCODE_URL,
        params={"q": query, "limit": limit, "appid": api_key},
        timeout=15,
    )
    if r.status_code == 401:
        raise WeatherError(
            "OpenWeather returned 401 Unauthorized. Check OPENWEATHER_API_KEY or disable USE_OPENWEATHER."
        )
    if r.status_code != 200:
        raise WeatherError(f"OpenWeather geocoding failed (HTTP {r.status_code}).")
    data = r.json()
    if isinstance(data, dict):
        msg = str(data.get("message") or data.get("cod") or data)
        raise WeatherError(f"OpenWeather geocoding error: {msg}")
    return data  # type: ignore[return-value]


def _pick_ow_row(rows: list[dict[str, Any]], original_place: str) -> dict[str, Any] | None:
    if not rows:
        return None
    for row in rows:
        if row.get("country") == "IN":
            return row
    original = original_place.strip().lower().split(",")[0].strip()
    for row in rows:
        name = str(row.get("name") or "").lower()
        if original and (original in name or name in original):
            return row
    return rows[0]


def geocode_openweather(place: str, api_key: str) -> tuple[float, float, str]:
    place = place.strip()
    if not place:
        raise WeatherError("Location is empty.")
    for q in _ow_geocode_queries(place):
        rows = _fetch_ow_geocode_raw(q, api_key, limit=5)
        row = _pick_ow_row(rows, place)
        if row is None:
            continue
        lat, lon = float(row["lat"]), float(row["lon"])
        name = row.get("name") or place.split(",")[0].strip()
        state = row.get("state") or ""
        country = row.get("country") or ""
        parts = [name]
        if state:
            parts.append(state)
        if country:
            parts.append(country)
        return lat, lon, ", ".join(parts)
    raise WeatherError("OpenWeather: location not found.")


def fetch_weather_openweather(lat: float, lon: float, api_key: str) -> tuple[float, float, float]:
    r = requests.get(
        OW_WEATHER_URL,
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=15,
    )
    if r.status_code != 200:
        raise WeatherError(f"OpenWeather weather failed (HTTP {r.status_code}).")
    body = r.json()
    try:
        temp = float(body["main"]["temp"])
        humidity = float(body["main"]["humidity"])
    except (KeyError, TypeError, ValueError) as e:
        raise WeatherError("OpenWeather response missing temperature or humidity.") from e
    rain = body.get("rain") or {}
    rainfall = 0.0
    if isinstance(rain, dict):
        if "1h" in rain:
            rainfall = float(rain["1h"])
        elif "3h" in rain:
            rainfall = float(rain["3h"]) / 3.0
    return temp, humidity, rainfall


# --- Public API ---------------------------------------------------------------


def resolve_place_and_weather(
    place: str,
) -> tuple[float, float, str, float, float, float, str]:
    """
    Return (lat, lon, display_name, temp_c, humidity_pct, rainfall_mm, provider_label).

    ``provider_label`` is ``"openweather"`` or ``"open-meteo"``.
    """
    ow_key = _openweather_key()
    if ow_key and _use_openweather_first():
        try:
            lat, lon, display = geocode_openweather(place, ow_key)
            temp, hum, rain = fetch_weather_openweather(lat, lon, ow_key)
            return lat, lon, display, temp, hum, rain, "openweather"
        except WeatherError:
            pass

    lat, lon, display = geocode_open_meteo(place)
    temp, hum, rain = fetch_weather_open_meteo(lat, lon)
    return lat, lon, display, temp, hum, rain, "open-meteo"


# Backwards-compatible names (used by tests or imports)

def geocode_location(place: str, **_kwargs: Any) -> tuple[float, float, str]:
    ow_key = _openweather_key()
    if ow_key and _use_openweather_first():
        try:
            return geocode_openweather(place, ow_key)
        except WeatherError:
            pass
    return geocode_open_meteo(place)


def fetch_weather(lat: float, lon: float) -> tuple[float, float, float]:
    ow_key = _openweather_key()
    if ow_key and _use_openweather_first():
        try:
            return fetch_weather_openweather(lat, lon, ow_key)
        except WeatherError:
            pass
    return fetch_weather_open_meteo(lat, lon)
