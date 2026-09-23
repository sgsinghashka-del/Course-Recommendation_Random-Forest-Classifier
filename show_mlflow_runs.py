import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("course_recommendation")

if exp is None:
    print("❌ Experiment not found")
else:
    runs = client.search_runs(exp.experiment_id)
    print(f"Found {len(runs)} runs\n")
    for r in runs:
        print("Run ID:", r.info.run_id)
        print("Metrics:", r.data.metrics)
        print("Params:", r.data.params)
        print("-" * 40)