"""Geocoding + current weather via Open-Meteo."""

from __future__ import annotations

from typing import Any

import requests

OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

# Identify this app for Nominatim usage policy.
NOMINATIM_UA = "MDM-CropAdvisory/1.0 (university project; educational use)"


class WeatherError(Exception):
    """Raised when location or weather cannot be fetched."""


# --- Open-Meteo ---------------------------------------------------------------


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


# --- Public API ---------------------------------------------------------------


def resolve_place_and_weather(
    place: str,
) -> tuple[float, float, str, float, float, float, str]:
    """
    Return (lat, lon, display_name, temp_c, humidity_pct, rainfall_mm, provider_label).

    ``provider_label`` is always ``"open-meteo"``.
    """
    lat, lon, display = geocode_open_meteo(place)
    temp, hum, rain = fetch_weather_open_meteo(lat, lon)
    return lat, lon, display, temp, hum, rain, "open-meteo"


# Backwards-compatible names (used by tests or imports)

def geocode_location(place: str, **_kwargs: Any) -> tuple[float, float, str]:
    return geocode_open_meteo(place)


def fetch_weather(lat: float, lon: float) -> tuple[float, float, float]:
    return fetch_weather_open_meteo(lat, lon)
