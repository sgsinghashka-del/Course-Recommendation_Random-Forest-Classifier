import numpy as np
from Backend.recommender import model

def predict_with_confidence(X):
    probs = model.predict_proba(X)
    pred = model.classes_[np.argmax(probs)]
    confidence = np.max(probs)
    return pred, round(confidence, 3)