import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("course_recommendation")

# ----------------------
# Data
# ----------------------
X = pd.DataFrame({
    "experience_years": [0, 1, 2, 3, 4, 5],
    "preferred_domain_AI": [1, 0, 0, 1, 0, 1],
    "preferred_domain_ML": [0, 1, 0, 0, 1, 0],
    "preferred_domain_DS": [0, 0, 1, 0, 0, 0]
})

y = [0, 1, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=50, random_state=42)

with mlflow.start_run():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ---- Logging ----
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)

    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("stage", "baseline")
    mlflow.set_tag("version", "v1")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/rf_model.joblib")

    mlflow.sklearn.log_model(model, name="course_recommender")

print("✅ Training complete | Accuracy:", acc)