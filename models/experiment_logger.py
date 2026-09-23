import csv
from datetime import datetime
import os

LOG_FILE = "model/experiments.csv"

def log_experiment(version, accuracy, rows, notes=""):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["time","version","rows","accuracy","notes"])
        writer.writerow([datetime.now(), version, rows, accuracy, notes])

if __name__ == "__main__":
    log_experiment("v1.0", 0.82, 1200, "baseline random forest")