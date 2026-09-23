import joblib

MODEL_REGISTRY = {
    "v1": "models/content_model.pkl",
    "v2": "models/content_model_v2.pkl"  # future
}

def load_model(version="v1"):
    return joblib.load(MODEL_REGISTRY[version])