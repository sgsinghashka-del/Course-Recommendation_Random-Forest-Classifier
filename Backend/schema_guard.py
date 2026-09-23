import json
import pandas as pd

SCHEMA_PATH = "model/feature_schema.json"

def enforce_schema(input_dict):
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)["features"]

    df = pd.DataFrame([input_dict])

    # add missing columns
    for col in schema:
        if col not in df.columns:
            df[col] = 0

    # keep only schema columns & correct order
    df = df[schema]
    return df