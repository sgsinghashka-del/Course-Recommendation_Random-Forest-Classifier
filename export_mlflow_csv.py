# export_mlflow_csv.py
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("course_recommendation")
if exp:
    runs = client.search_runs(exp.experiment_id)
    import pandas as pd
    df = pd.DataFrame([
        {
            "run_id": r.info.run_id,
            "metrics": r.data.metrics,
            "params": r.data.params
        }
        for r in runs
    ])
    df.to_csv("mlflow_runs.csv", index=False)
    print("Saved mlflow_runs.csv")
else:
    print("Experiment not found")
    