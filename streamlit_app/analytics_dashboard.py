import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

st.set_page_config(page_title="MLflow Analytics", layout="wide")
st.title("📊 Course Recommendation – MLflow Dashboard")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiments = client.search_experiments()

if not experiments:
    st.warning("No experiments logged yet.")
    st.stop()

exp = experiments[0]
runs = client.search_runs(exp.experiment_id)

if not runs:
    st.warning("No runs found.")
    st.stop()

data = []
for r in runs:
    row = {
        "Run ID": r.info.run_id,
        "Accuracy": r.data.metrics.get("accuracy"),
        "n_estimators": r.data.params.get("n_estimators"),
        "Stage": r.data.tags.get("stage"),
        "Version": r.data.tags.get("version"),
    }
    data.append(row)

df = pd.DataFrame(data)

st.subheader("📌 Latest Run Summary")
st.metric("Latest Accuracy", df.iloc[0]["Accuracy"])

st.subheader("📈 Accuracy Across Runs")
st.line_chart(df["Accuracy"])

st.subheader("📋 Experiment Runs")
st.dataframe(df)