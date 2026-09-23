# Backend/recommender.py

import joblib
import pandas as pd

MODEL_PATH = "models/content_model.pkl"

model = joblib.load(MODEL_PATH)
MODEL_FEATURES = model.feature_names_in_


def recommend_courses(user_input: dict):
    # Convert to DataFrame
    df = pd.DataFrame([user_input])

    # One-hot encode preferred_domain (same as training)
    df = pd.get_dummies(df, columns=["preferred_domain"])

    # Add missing columns
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Keep correct column order
    df = df[MODEL_FEATURES]

    prediction = model.predict(df)[0]

    return {
        "recommended_course": prediction
    }