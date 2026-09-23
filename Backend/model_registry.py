from __future__ import annotations

from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_REGISTRY = {
    "v1": BASE_DIR / "models" / "content_model.pkl",
    "v2": BASE_DIR / "models" / "content_model_v2.pkl",
}


def load_model(version: str = "v1"):
    if version not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model version: {version}")

    model_path = MODEL_REGISTRY[version]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)
