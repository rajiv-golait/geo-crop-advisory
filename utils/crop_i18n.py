"""Local crop names when Groq is unavailable (fallback UI). Keys = English model labels (lowercase)."""

from __future__ import annotations

from utils.i18n import LanguageCode

# Common farmer-facing names (approximate; LLM localizes better when API is set).
CROP_LOCAL: dict[str, dict[str, str]] = {
    "apple": {"hi": "सेब", "mr": "सफरचंद"},
    "banana": {"hi": "केला", "mr": "केळ"},
    "barley": {"hi": "जौ", "mr": "बार्ली / जव"},
    "blackgram": {"hi": "उड़द", "mr": "उडीद"},
    "chickpea": {"hi": "चना", "mr": "हरभरा / चणा"},
    "coconut": {"hi": "नारियल", "mr": "नारळ"},
    "coffee": {"hi": "कॉफ़ी", "mr": "कॉफी"},
    "cotton": {"hi": "कपास", "mr": "कापूस"},
    "grapes": {"hi": "अंगूर", "mr": "द्राक्ष"},
    "jute": {"hi": "जूट", "mr": "ज्यूट"},
    "kidneybeans": {"hi": "राजमा", "mr": "राजमा"},
    "lentil": {"hi": "मसूर", "mr": "मसूर"},
    "maize": {"hi": "मक्का", "mr": "मका"},
    "mango": {"hi": "आम", "mr": "आंबा"},
    "mothbeans": {"hi": "मोठ", "mr": "मटकी"},
    "mungbean": {"hi": "मूंग", "mr": "मूग"},
    "muskmelon": {"hi": "खरबूजा", "mr": "खरबूज"},
    "orange": {"hi": "संतरा", "mr": "संत्रे"},
    "papaya": {"hi": "पपीता", "mr": "पपई"},
    "pigeonpeas": {"hi": "अरहर / तूर", "mr": "तूर"},
    "pomegranate": {"hi": "अनार", "mr": "डाळिंब"},
    "rice": {"hi": "धान / चावल", "mr": "भात / तांदूळ"},
    "watermelon": {"hi": "तरबूज", "mr": "कलिंगड"},
}


def localized_crop_name(crop_en: str, lang: LanguageCode) -> str:
    key = crop_en.strip().lower()
    if lang == "en":
        return crop_en.replace("_", " ").strip().title() or crop_en
    row = CROP_LOCAL.get(key)
    if row and lang in row:
        return row[lang]
    return crop_en.replace("_", " ").strip().title() or crop_en
