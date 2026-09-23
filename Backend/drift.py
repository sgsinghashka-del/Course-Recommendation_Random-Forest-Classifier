import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(train_df, inference_df):
    drift_report = {}

    for col in train_df.columns:
        stat, p = ks_2samp(train_df[col], inference_df[col])
        drift_report[col] = {
            "p_value": p,
            "drift_detected": p < 0.05
        }

    return drift_report