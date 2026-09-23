import random
from Backend.model_registry import load_model

def select_model():
    return "v1" if random.random() < 0.5 else "v2"

def ab_predict(user_df):
    version = select_model()
    model = load_model(version)

    prediction = model.predict(user_df)[0]

    return {
        "model_version": version,
        "recommendation": prediction
    }