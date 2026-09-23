from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_FILE = BASE_DIR / "Data" / "feedback.csv"


def store_feedback(user_input: dict, recommended_course: str, feedback: str):
    if not isinstance(user_input, dict):
        raise TypeError("user_input must be a dictionary.")

    row = {
        **user_input,
        "recommended_course": recommended_course,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat(),
    }

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    if FEEDBACK_FILE.exists():
        existing = pd.read_csv(FEEDBACK_FILE)
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        combined.to_csv(FEEDBACK_FILE, index=False)
    else:
        pd.DataFrame([row]).to_csv(FEEDBACK_FILE, index=False)

    return {"status": "success"}
