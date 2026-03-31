"""
AI Crop Advisory — Streamlit UI: place name → geocode → FastAPI `/predict`.

Run API first:  uvicorn backend_api:app --reload
Run UI:         streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from utils.crop_i18n import localized_crop_name
from utils.i18n import LanguageCode, t
from utils.llm import chat_farmer, format_localized_prediction_card, is_llm_failure_reply
from utils.weather import WeatherError, geocode_open_meteo

APP_DIR = Path(__file__).resolve().parent
if (APP_DIR.parent / ".env").is_file():
    load_dotenv(APP_DIR.parent / ".env", encoding="utf-8-sig", override=False)
if (APP_DIR / ".env").is_file():
    load_dotenv(APP_DIR / ".env", encoding="utf-8-sig", override=True)


def _api_base() -> str:
    return os.environ.get("PREDICT_API_BASE", "http://127.0.0.1:8000").strip().rstrip("/")


def _groq_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def _strip_code_fences(text: str) -> str:
    """Remove ``` blocks if the model echoes JSON despite instructions."""
    return re.sub(r"```[\s\S]*?```", "", text).strip()


def _inject_chat_css() -> None:
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
  .app-header-wrap {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(31, 111, 84, 0.15);
  }
  .hdr {
    color: #145a46;
    font-weight: 700;
    font-size: 1.65rem;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
  }
  .sub {
    color: #4a6358;
    font-size: 0.95rem;
    line-height: 1.45;
    max-width: 36rem;
    margin: 0 auto;
  }
  div[data-testid="stChatMessage"] {
    background: linear-gradient(165deg, #f7fbf9 0%, #eef6f1 100%);
    border-radius: 14px;
    border: 1px solid #cfe5d8;
    box-shadow: 0 1px 3px rgba(20, 90, 70, 0.06);
  }
  div[data-testid="stChatMessage"] p,
  div[data-testid="stChatMessage"] li,
  div[data-testid="stChatMessage"] strong {
    color: #12271d !important;
  }
  div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #f4faf7 0%, #eef5f1 100%);
    padding-top: 0.75rem;
  }
  div[data-testid="stSidebarContent"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.55);
    border-radius: 10px;
    border: 1px solid rgba(20, 90, 70, 0.12);
  }
  div[data-testid="stSidebarContent"] .stCaption,
  div[data-testid="stSidebarContent"] [data-testid="stCaptionContainer"] p {
    color: #3d5348 !important;
    line-height: 1.45;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #1f8f6a 0%, #16664c 100%);
    border: none;
    font-weight: 600;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _render_messages() -> None:
    for m in st.session_state.messages:
        role = m["role"]
        avatar = "🧑‍🌾" if role == "user" else "🌾"
        with st.chat_message(role, avatar=avatar):
            st.markdown(m["content"])


def _init_state() -> None:
    if "flow_step" not in st.session_state:
        st.session_state.flow_step = "location"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False


def _reset_flow(lang: LanguageCode) -> None:
    st.session_state.flow_step = "location"
    st.session_state.messages = [
        {"role": "assistant", "content": t(lang, "welcome")},
        {"role": "assistant", "content": t(lang, "coords_help")},
    ]
    st.session_state.chat_started = True


def _soil_lines(soil: dict) -> list[str]:
    lines: list[str] = []
    for key in ("nitrogen", "phosphorous", "potassium"):
        block = soil.get(key) or {}
        lvl = block.get("dominant_level", "")
        pct = block.get("dominant_percentage", "")
        lines.append(f"- **{key.title()}:** {lvl} ({pct}%)")
    return lines


def _fallback_advisory_markdown(
    lang: LanguageCode,
    data: dict,
    *,
    place_text: str,
    resolved_place: str,
    lat: float,
    lon: float,
) -> str:
    """Readable summary without JSON when GROQ_API_KEY is missing."""
    crop = localized_crop_name(str(data.get("recommended_crop", "")), lang)
    ctx = data.get("context") or {}
    mf = ctx.get("model_features") or {}
    dc = ctx.get("district_coordinates") or {}
    district = ctx.get("nearest_district", "")
    soil = ctx.get("soil_summary") or {}
    wm = ctx.get("weather_meta") or {}

    parts = [
        f"### {t(lang, 'recommended_crop_header')}",
        f"**{crop}**",
        "",
        f"**{t(lang, 'fb_you_entered')}:** {place_text}",
        f"**{t(lang, 'fb_resolved')}:** {resolved_place}",
        f"**{t(lang, 'fb_coordinates')}:** `{lat:.5f}`, `{lon:.5f}`",
        "",
        f"**{t(lang, 'fb_district')}:** {district}",
        f"**{t(lang, 'fb_distance_km')}:** {dc.get('distance_km', '—')}",
        "",
        f"**{t(lang, 'fb_model_inputs')}**",
        f"- N={mf.get('N')}, P={mf.get('P')}, K={mf.get('K')}, pH={mf.get('pH')}",
        f"- {mf.get('rainfall')} mm rain (period total) · {mf.get('temperature')} °C · {mf.get('humidity')} % humidity",
        "",
        f"**{t(lang, 'fb_soil_categories')}**",
        *_soil_lines(soil),
        "",
        f"**{t(lang, 'fb_weather_info')}**",
    ]
    if wm.get("rainfall_note"):
        parts.append(f"- {wm['rainfall_note']}")
    if wm.get("temperature_note"):
        parts.append(f"- {wm['temperature_note']}")
    if wm.get("humidity_note"):
        parts.append(f"- {wm['humidity_note']}")
    parts.extend(
        [
            "",
            f"*{t(lang, 'fb_disclaimer')}*",
            "",
            f"💡 {t(lang, 'fallback_groq_hint')}",
        ]
    )
    return "\n".join(parts)


def _call_predict(lat: float, lon: float) -> dict:
    url = f"{_api_base()}/predict"
    r = requests.post(url, json={"latitude": lat, "longitude": lon}, timeout=60)
    if r.status_code != 200:
        detail = ""
        try:
            body = r.json()
            detail = str(body.get("detail", body))
        except Exception:
            detail = r.text[:500] if r.text else ""
        raise RuntimeError(f"HTTP {r.status_code}: {detail}".strip())
    return r.json()


def _run_prediction_pipeline(
    lang: LanguageCode,
    *,
    place_text: str,
    lat: float,
    lon: float,
    resolved_display: str,
) -> None:
    messages = st.session_state.messages
    if messages and "🔄" in messages[-1].get("content", ""):
        messages.pop()

    try:
        data = _call_predict(lat, lon)
    except requests.exceptions.RequestException as e:
        messages.append(
            {"role": "assistant", "content": f"{t(lang, 'error_predict_api')}\n\n`{e!s}`"}
        )
        st.session_state.flow_step = "location"
        return
    except RuntimeError as e:
        messages.append(
            {"role": "assistant", "content": f"{t(lang, 'error_predict_api')}\n\n`{e!s}`"}
        )
        st.session_state.flow_step = "location"
        return

    if _groq_configured():
        raw = format_localized_prediction_card(
            lang,
            api_payload=data,
            place_user_typed=place_text,
            resolved_place=resolved_display,
            latitude=lat,
            longitude=lon,
        )
        body = _strip_code_fences(raw)
        if is_llm_failure_reply(body, lang):
            body = _fallback_advisory_markdown(
                lang,
                data,
                place_text=place_text,
                resolved_place=resolved_display,
                lat=lat,
                lon=lon,
            )
    else:
        body = _fallback_advisory_markdown(
            lang,
            data,
            place_text=place_text,
            resolved_place=resolved_display,
            lat=lat,
            lon=lon,
        )

    messages.append({"role": "assistant", "content": body})
    st.session_state.flow_step = "chat"


def main() -> None:
    st.set_page_config(
        page_title="AI Crop Advisory",
        page_icon="🌱",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        _lang_label = st.session_state.get("lang_select", "en")
        lang_choice = st.selectbox(
            t(_lang_label, "language_label"),
            options=["en", "hi", "mr"],
            format_func=lambda c: {"en": "English", "hi": "हिन्दी", "mr": "मराठी"}[c],
            key="lang_select",
        )
        lang: LanguageCode = lang_choice  # type: ignore[assignment]

        st.caption(t(lang, "sidebar_short_tip"))
        if _groq_configured():
            st.caption(f"✓ {t(lang, 'sidebar_smart_on')}")
        else:
            st.caption(t(lang, "sidebar_smart_off"))

        st.divider()

        with st.expander(t(lang, "sidebar_dev_title"), expanded=False):
            st.markdown(t(lang, "sidebar_keys_hint"))
            st.caption("PREDICT_API_BASE")
            st.code(_api_base(), language=None)

        if st.button(t(lang, "restart"), use_container_width=True):
            _reset_flow(lang)
            st.rerun()

    _init_state()
    st.session_state.lang = lang

    if not st.session_state.chat_started:
        _reset_flow(lang)

    _inject_chat_css()
    st.markdown(
        f'<div class="app-header-wrap"><h1 class="hdr">{t(lang, "app_title")}</h1>'
        f'<p class="sub">{t(lang, "tagline")}</p></div>',
        unsafe_allow_html=True,
    )

    _render_messages()

    if st.session_state.flow_step == "location":
        st.divider()
        with st.container(border=True):
            with st.form("location_form", clear_on_submit=False):
                loc = st.text_input(
                    t(lang, "enter_location"),
                    placeholder=t(lang, "location_placeholder"),
                    key="location_field",
                )
                submit_location = st.form_submit_button(
                    t(lang, "get_advisory"),
                    type="primary",
                    use_container_width=True,
                )
            if submit_location:
                if not loc or not loc.strip():
                    st.warning(t(lang, "empty_location"))
                else:
                    place_clean = loc.strip()
                    try:
                        lat, lon, resolved = geocode_open_meteo(place_clean)
                    except WeatherError as e:
                        st.session_state.messages.append(
                            {"role": "user", "content": place_clean}
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"{t(lang, 'error_location')}\n\n`{e!s}`",
                            }
                        )
                        st.rerun()

                    st.session_state.messages.append({"role": "user", "content": place_clean})
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"🔄 {t(lang, 'fetching_api')}"}
                    )
                    _run_prediction_pipeline(
                        lang,
                        place_text=place_clean,
                        lat=lat,
                        lon=lon,
                        resolved_display=resolved,
                    )
                    st.rerun()

    elif st.session_state.flow_step == "chat":
        st.divider()
        with st.container(border=True):
            with st.form("followup_form", clear_on_submit=True):
                q = st.text_input(t(lang, "ask_more"), key="followup_q")
                send_followup = st.form_submit_button(
                    t(lang, "send"),
                    type="primary",
                    use_container_width=True,
                )
            if send_followup and q.strip():
                st.session_state.messages.append({"role": "user", "content": q.strip()})
                reply = _strip_code_fences(chat_farmer(lang, q.strip()))
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()


if __name__ == "__main__":
    main()
