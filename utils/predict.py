"""Load crop classifier and return top-k crop recommendations."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Default pH when only N,P,K + weather are collected (matches typical mid-range in dataset).
DEFAULT_SOIL_PH = 6.5

_DEFAULT_MODEL_FILE = "best_crop_model.pkl"
_FALLBACK_MODEL_FILE = "crop_rf_model.pkl"
_LABEL_FILE = "crop_label_encoder.pkl"

# sklearn LabelEncoder.classes_ = sorted unique labels — matches common Crop_recommendation.csv
# (use CSV or crop_label_encoder.pkl when available for an exact match to your training data).
_STANDARD_LABELS: tuple[str, ...] = (
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon",
)

_MODEL = None
_ENCODER: LabelEncoder | None = None


def _models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def _dataset_csv_paths() -> list[Path]:
    """Prefer packaged data copy: crop_advisory_app/data/Crop_recommendation.csv"""
    base = _models_dir().parent
    return [
        base / "data" / "Crop_recommendation.csv",
        base / "Crop_recommendation.csv",
        base.parent / "Crop_recommendation.csv",
    ]


def _build_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    for csv_path in _dataset_csv_paths():
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            le.fit(df["label"])
            return le
    le.fit(list(_STANDARD_LABELS))
    return le


def load_artifacts() -> tuple[object, LabelEncoder]:
    global _MODEL, _ENCODER
    if _MODEL is not None and _ENCODER is not None:
        return _MODEL, _ENCODER

    models_dir = _models_dir()
    override = os.environ.get("CROP_MODEL_FILE", "").strip()
    name = override or _DEFAULT_MODEL_FILE
    mpath = models_dir / name
    if not mpath.is_file() and name == _DEFAULT_MODEL_FILE:
        fallback = models_dir / _FALLBACK_MODEL_FILE
        if fallback.is_file():
            mpath = fallback

    if not mpath.is_file():
        raise FileNotFoundError(
            f"Missing model pickle: expected {models_dir / _DEFAULT_MODEL_FILE} "
            f"(or set CROP_MODEL_FILE to another .pkl filename in models/)."
        )

    epath = models_dir / _LABEL_FILE
    _MODEL = joblib.load(mpath)
    if epath.is_file():
        _ENCODER = joblib.load(epath)
    else:
        _ENCODER = _build_label_encoder()
    return _MODEL, _ENCODER


def predict_top_crops(
    n: float,
    p: float,
    k: float,
    temperature: float,
    humidity: float,
    rainfall: float,
    *,
    ph: float = DEFAULT_SOIL_PH,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Build feature row [N, P, K, temperature, humidity, ph, rainfall] and return
    [(crop_name, probability), ...] sorted by probability descending.
    """
    model, le = load_artifacts()
    X = pd.DataFrame(
        [[n, p, k, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    proba = model.predict_proba(X)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    classes = le.classes_
    return [(str(classes[i]), float(proba[i])) for i in top_idx]
