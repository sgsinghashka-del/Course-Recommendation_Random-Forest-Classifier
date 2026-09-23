import json
import pandas as pd
import sqlite3
import numpy as np

BASELINE = json.load(open("model/baseline_stats.json"))
DB = "users.db"

def check_drift():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()

    alerts = []
    for col, stats in BASELINE.items():
        if col in df:
            mean_now = df[col].mean()
            if abs(mean_now - stats["mean"]) > 2.5 * stats["std"]:
                alerts.append(col)

    if alerts:
        print("⚠️ Drift detected in:", alerts)
    else:
        print("✅ No drift detected")

if __name__ == "__main__":
    check_drift()