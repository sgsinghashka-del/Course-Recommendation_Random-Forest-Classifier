from __future__ import annotations

import random

from Backend.model_registry import load_model


def select_model():
    return "v1" if random.random() < 0.5 else "v2"


def ab_predict(user_input):
    if not isinstance(user_input, dict):
        raise TypeError("user_input must be a dictionary.")

    version = select_model()
    model = load_model(version)

    # Accept either a raw dict or a DataFrame-like structure.
    if hasattr(user_input, "to_dict"):
        payload = user_input.to_dict(orient="records")
    else:
        payload = [user_input]

    prediction = model.predict(payload)[0]

    return {
        "model_version": version,
        "recommendation": prediction,
    }
