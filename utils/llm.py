"""Groq (Llama) chat for crop advisory and optional translation."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from groq import Groq

LanguageCode = Literal["en", "hi", "mr"]

LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

AG_SYSTEM_EN = """You are a helpful agricultural assistant for smallholder farmers.
You ONLY answer questions about: crop recommendations, farming practices, soil, fertilizers (NPK), weather impacts on crops, irrigation, and related agronomy.
If the user asks anything unrelated (politics, coding, medical, jokes, etc.), politely refuse in one short sentence and remind them you only help with farming and crops.
Keep answers practical, short (2–5 sentences unless asked for detail), and farmer-friendly. Use simple words."""


def _client() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return Groq(api_key=key)


def _system_prompt_for_language(lang: LanguageCode) -> str:
    base = AG_SYSTEM_EN
    name = LANGUAGE_NAMES[lang]
    if lang == "en":
        return base + "\nAlways respond in English."
    return (
        base
        + f"\nAlways respond entirely in {name} (use Devanagari script)."
        + f"\nFor crop names, always use the common {name} name farmers recognize (e.g. cotton → standard Hindi/Marathi crop term). Do not leave crop names in English."
    )


def format_localized_prediction_card(
    lang: LanguageCode,
    *,
    api_payload: dict[str, Any],
    place_user_typed: str,
    resolved_place: str,
    latitude: float,
    longitude: float,
    model_label: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1200,
) -> str:
    """
    One farmer-facing markdown message in the UI language — no JSON/code blocks.
    Hindi/Marathi: crop names and prose fully localized.
    """
    crop_en = str(api_payload.get("recommended_crop", ""))
    ctx = api_payload.get("context") or {}
    dc = ctx.get("district_coordinates") or {}
    facts = {
        "recommended_crop_model_label_english": crop_en,
        "user_typed_place": place_user_typed,
        "geocoded_place": resolved_place,
        "latitude": latitude,
        "longitude": longitude,
        "nearest_district": ctx.get("nearest_district"),
        "district_match_distance_km": dc.get("distance_km"),
        "model_features_N_P_K_pH_rain_temp_humidity": ctx.get("model_features"),
        "dominant_soil_category_percentages": ctx.get("soil_summary"),
        "weather_computation_notes": ctx.get("weather_meta"),
    }
    blob = json.dumps(facts, ensure_ascii=False, indent=2)
    lang_name = LANGUAGE_NAMES[lang]

    if lang == "en":
        user_msg = f"""Using the JSON facts below, write ONE markdown message for a farmer.

Rules:
- No JSON, no ``` code fences, no raw braces in the output.
- Use ### for a title, then bullet lists for: recommended crop, location (what they typed + resolved place + lat/lon), nearest district and distance, N/P/K/pH and weather numbers (rainfall mm period total, °C, humidity %), soil categories in plain English (High/Medium/Low for N, P, K).
- Briefly explain how weather was derived using the notes inside weather_computation_notes (in simple English).
- End with one line: this is model guidance — verify with local experts / soil tests.

FACTS:
{blob}"""
    else:
        user_msg = f"""The facts below are JSON for your eyes only.

OUTPUT LANGUAGE: Write the entire answer in {lang_name} only. Use Devanagari script.

RULES:
- Do not output JSON, ``` code blocks, or raw braces.
- The field recommended_crop_model_label_english is the model's English label — translate it to the standard common {lang_name} crop name farmers use (e.g. cotton → proper local name). Do not leave the crop name only in English.
- Use ### title, then bullets: recommended crop (localized name), location (user typed + geocoded + lat/lon), nearest district and distance, N/P/K/pH, rainfall mm (period), temperature °C, humidity %, soil category summary in simple words.
- Summarize weather_computation_notes briefly in {lang_name}.
- Close with one line: this is model advice — confirm with local experts / soil tests.

FACTS:
{blob}"""

    return chat_farmer(lang, user_msg, model_label=model_label, max_tokens=max_tokens)


def summarize_api_prediction(
    lang: LanguageCode,
    *,
    recommended_crop: str,
    context: dict[str, Any],
    farmer_stated_soil: str | None = None,
    geocoded_place: str | None = None,
    model_label: str = "llama-3.3-70b-versatile",
) -> str:
    """Short farmer-friendly explanation after FastAPI `/predict` returns."""
    district = context.get("nearest_district", "")
    dist_coord = context.get("district_coordinates", {})
    feats = context.get("model_features", {})
    soil = context.get("soil_summary", {})
    soil_note = (
        f"\n- **Farmer selected soil type (for context):** {farmer_stated_soil}"
        " — note the model still uses district soil statistics for N/P/K, not this choice alone."
        if farmer_stated_soil
        else ""
    )
    place_note = (
        f"\n- **Location resolved from user text:** {geocoded_place}" if geocoded_place else ""
    )
    user_en = f"""A crop recommendation model returned this result for the farmer's GPS point:

- **Recommended crop:** {recommended_crop}
- **Nearest district (soil lookup):** {district} (match ~{dist_coord.get('distance_km', '?')} km from CSV coordinates)
- **Values fed to the model:** N={feats.get('N')}, P={feats.get('P')}, K={feats.get('K')}, pH={feats.get('pH')}, rainfall={feats.get('rainfall')} mm (period total), temperature={feats.get('temperature')} °C (mean), humidity={feats.get('humidity')}
- **Dominant soil categories in that district:** N={soil.get('nitrogen', {})}, P={soil.get('phosphorous', {})}, K={soil.get('potassium', {})}{soil_note}{place_note}

Write 2–4 sentences for the farmer: name the crop, mention district/soil/weather are approximations from data + forecast, and remind them to verify with local extension officers or soil testing. Do not sound mechanical."""
    return chat_farmer(lang, user_en, model_label=model_label)


def generate_advisory_reply(
    *,
    lang: LanguageCode,
    soil: str,
    location_display: str,
    temp_c: float,
    humidity: float,
    rainfall_mm: float,
    n: int,
    p: int,
    k: int,
    top_crops: list[tuple[str, float]],
    model_label: str = "llama-3.3-70b-versatile",
) -> str:
    """Natural-language summary after numeric prediction."""
    crops_line = ", ".join(f"{c.title()} ({prob:.0%})" for c, prob in top_crops)
    user_en = f"""Farmer context:
- Soil type: {soil}
- Location: {location_display}
- Approximate soil NPK used for model: N={n}, P={p}, K={k}
- Current weather: {temp_c:.1f}°C, humidity {humidity:.0f}%, rainfall indicator {rainfall_mm:.1f} mm (0 if not reported)

Model top crops (probability): {crops_line}

Write a short, encouraging message naming the top 3 crops in order and briefly why they fit these conditions. Do not claim certainty about real soil tests — note these are model suggestions."""
    return chat_farmer(lang, user_en, model_label=model_label)


def chat_farmer(
    lang: LanguageCode,
    user_message: str,
    *,
    model_label: str = "llama-3.3-70b-versatile",
    max_tokens: int = 512,
) -> str:
    """Single-turn advisory chat with domain restriction."""
    try:
        client = _client()
        sys_prompt = _system_prompt_for_language(lang)
        resp = client.chat.completions.create(
            model=model_label,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message.content
        return (choice or "").strip() or _fallback_message(lang)
    except Exception:
        return _fallback_message(lang)


def _fallback_message(lang: LanguageCode) -> str:
    if lang == "hi":
        return "माफ़ कीजिए, सलाह उत्पन्न करने में तकनीकी समस्या आई। कृपया बाद में फिर कोशिश करें।"
    if lang == "mr":
        return "क्षमस्व, सल्ला तयार करताना तांत्रिक अडचण आली. कृपया नंतर पुन्हा प्रयत्न करा."
    return "Sorry — the advice service hit a technical issue. Please try again in a moment."


def is_llm_failure_reply(text: str, lang: LanguageCode) -> bool:
    """True when Groq failed and chat_farmer returned the generic error string."""
    t = (text or "").strip()
    if not t:
        return True
    return t == _fallback_message(lang)
