import pandas as pd
import json
import os

DATA_PATH = "Data/user_feedback.csv"
OUTPUT_PATH = "model/baseline_stats.json"

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_PATH)

if df.empty:
    raise ValueError("user_feedback.csv is empty. Cannot compute baseline stats.")

# ---------------------------
# Define feature columns (ALREADY ENCODED)
# ---------------------------
feature_columns = [
    "preferred_domain_AI",
    "preferred_domain_ML",
    "preferred_domain_DS",
    "experience_years"
]

# ---------------------------
# Ensure all columns exist
# ---------------------------
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0

# ---------------------------
# Compute baseline stats safely
# ---------------------------
means = df[feature_columns].mean().fillna(0)
stds = df[feature_columns].std().replace(0, 1).fillna(1)

baseline_stats = {
    "mean": means.to_dict(),
    "std": stds.to_dict()
}

# ---------------------------
# Save JSON (VALID, NO NaN)
# ---------------------------
os.makedirs("model", exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(baseline_stats, f, indent=2)

print("✅ Baseline stats created successfully")