import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

st.set_page_config(page_title="MLflow Dashboard", layout="wide")
st.title("📊 MLflow Experiment Dashboard")

# ---------------------------------------
# ABSOLUTE PATH (MANDATORY)
# ---------------------------------------
MLFLOW_DB = r"C:\Users\Hp\Desktop\Course Recommendation_Random Forest Classifier\mlflow.db"

if not os.path.exists(MLFLOW_DB):
    st.error(f"❌ mlflow.db NOT FOUND:\n{MLFLOW_DB}")
    st.stop()

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
client = MlflowClient()

# ---------------------------------------
# LOAD EXPERIMENTS (NEW API)
# ---------------------------------------
experiments = client.search_experiments()

if not experiments:
    st.error("❌ No experiments found in MLflow database")
    st.stop()

st.success("Experiments detected:")
for e in experiments:
    st.write(f"• {e.name}")

# ---------------------------------------
# TARGET EXPERIMENT
# ---------------------------------------
exp = client.get_experiment_by_name("course_recommendation")

if exp is None:
    st.error("❌ 'course_recommendation' experiment not found")
    st.stop()

# ---------------------------------------
# LOAD RUNS
# ---------------------------------------
runs = client.search_runs([exp.experiment_id])

if not runs:
    st.warning("⚠ Experiment exists but no runs logged yet")
    st.stop()

# ---------------------------------------
# DISPLAY RUN DATA
# ---------------------------------------
rows = []
for r in runs:
    row = {
        "run_id": r.info.run_id,
        "status": r.info.status
    }
    row.update(r.data.params)
    row.update(r.data.metrics)
    rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

st.success("✅ MLflow data loaded successfully")