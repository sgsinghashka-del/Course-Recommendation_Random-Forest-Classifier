from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "content_model.pkl"
VALID_DOMAINS = ["Data Science", "Web Development", "Backend", "Cloud"]


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _normalize_input(user_input: dict) -> dict:
    if not isinstance(user_input, dict):
        raise TypeError("user_input must be a dictionary.")

    normalized = {
        "age": int(user_input.get("age", 0)),
        "experience": int(user_input.get("experience", 0)),
        "interest_level": int(user_input.get("interest_level", 0)),
        "preferred_domain": str(user_input.get("preferred_domain", "")).strip(),
    }

    if normalized["preferred_domain"] not in VALID_DOMAINS:
        raise ValueError(
            f"Unsupported preferred_domain '{normalized['preferred_domain']}'. "
            f"Allowed values: {VALID_DOMAINS}"
        )

    return normalized


def recommend_courses(user_input: dict):
    model = _load_model()
    normalized_input = _normalize_input(user_input)

    df = pd.DataFrame([normalized_input])
    df = pd.get_dummies(df, columns=["preferred_domain"], prefix="preferred_domain")

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError("The loaded model does not include feature names.")

    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    df = df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(df)[0]

    return {"recommended_course": str(prediction)}
