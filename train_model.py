from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "courses.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "rf_model.joblib"


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    df = pd.read_csv(path)
    required_columns = [
        "age",
        "experience",
        "interest_level",
        "preferred_domain",
        "course_label",
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    return df


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("course_recommendation")

    df = load_training_data(DATA_PATH)

    features = ["age", "experience", "interest_level", "preferred_domain"]
    X = pd.get_dummies(df[features], columns=["preferred_domain"], prefix="preferred_domain")
    y = df["course_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    with mlflow.start_run(run_name="random_forest_baseline") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tags(
            {
                "model_type": "RandomForestClassifier",
                "stage": "baseline",
                "dataset": "courses.csv",
                "version": "v1",
            }
        )

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="course_recommender")

        print(f"Training complete | accuracy={accuracy:.4f} | run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
