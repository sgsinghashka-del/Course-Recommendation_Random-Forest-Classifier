# Course Recommendation System

A machine learning-powered course recommendation platform that suggests the most suitable learning path based on user profile attributes such as age, experience, interest level, and preferred domain. The project combines a `RandomForestClassifier`, a FastAPI backend, and a Streamlit web interface to provide both API and interactive user experiences.

## Overview

This repository demonstrates a complete end-to-end recommendation workflow:

- Data preparation using a small domain-aware dataset
- Training a Random Forest model for course classification
- Saving and loading ML models for inference
- Exposing a REST API for recommendation requests
- Capturing user feedback and model monitoring signals
- Tracking experiments with MLflow
- Visualizing results through a Streamlit dashboard

The project is designed as a practical ML system prototype for educational recommendations, but the architecture is general enough to adapt to other recommendation domains.

## Project Goals

- Recommend a course based on a learner's background and interests
- Provide a simple user-friendly interface for testing recommendations
- Support versioned model experimentation and deployment workflows
- Log model metrics and training metadata with MLflow
- Enable behavioral feedback collection and monitoring

## Architecture

The project is split into the following components:

- `Backend/` — FastAPI application and machine learning support modules
- `streamlit_app/` — Streamlit UI and dashboards
- `Data/` — training and feedback datasets
- `models/` and `model/` — serialized model artifacts and baseline statistics
- Root scripts — training, DB initialization, and MLflow export utilities

## Core Features

- Random Forest-based recommendation engine
- FastAPI REST API with endpoints for recommendations and feedback
- Streamlit app for learner interaction
- User preference input mapping to course labels
- Feedback storage for improving the recommendation system
- Experiment tracking with MLflow
- Drift monitoring utilities and baseline stats
- A/B testing scaffold for model version comparison

## Data Model

The sample dataset uses a synthetic learning profile table:

- `age`
- `experience`
- `interest_level`
- `preferred_domain`
- `course_label`

Example training data is stored in `Data/courses.csv`.

Example rows:

```csv
age,experience,interest_level,preferred_domain,course_label
21,0,8,Data Science,Data Analyst
23,2,9,Web Development,Frontend Developer
25,3,9,Backend,Backend Developer
27,4,7,Cloud,Cloud Engineer
```

## Model

The recommendation model is trained using a `RandomForestClassifier` from scikit-learn.

The training script initializes a simple MLflow experiment and logs:

- model parameters
- accuracy metric
- experiment tags
- serialized model artifact

Model training logic is implemented in:

- `train_model.py`
- `Backend/recommender.py`

The model artifact is stored in:

- `model/rf_model.joblib`
- `models/content_model.pkl`

## API

The backend is implemented with FastAPI and can be launched using uvicorn.

### Endpoints

- `GET /` — health check
- `POST /recommend` — returns a recommended course for a learner profile
- `POST /feedback` — stores feedback for a recommendation
- `POST /recommend-ab` — A/B testing inference route

### Example request

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 24,
    "experience": 2,
    "interest_level": 8,
    "preferred_domain": "Web Development"
  }'
```

### Example response

```json
{
  "recommended_course": "Frontend Developer"
}
```

## Web Interface

The Streamlit application provides a simple UI for entering learner preferences and requesting course suggestions.

Launch:

```bash
streamlit run streamlit_app/app.py
```

This app sends a request to the FastAPI service and displays the returned recommendations.

## MLflow Integration

MLflow is used to track experiments and model runs.

Files:

- `train_model.py` — creates the experiment and logs metrics
- `show_mlflow_runs.py` — prints logged runs
- `export_mlflow_csv.py` — exports MLflow run metadata to a CSV
- `mlflow.db` — local MLflow tracking database

Example:

```bash
python train_model.py
python show_mlflow_runs.py
```

## Monitoring and Governance

The repository includes prototype components for model monitoring and governance:

- `Backend/drift_monitor.py` — checks feature drift against a stored baseline
- `Backend/schema_guard.py` — intended for input validation and schema checks
- `Backend/ab_testing.py` — demonstrates versioned model selection logic
- `Backend/model_registry.py` — manages model registry entries

These are useful for building more robust production-grade ML systems.

## Repository Structure

```text
Course-Recommendation_Random-Forest-Classifier/
├── Backend/
│   ├── __init__.py
│   ├── ab_testing.py
│   ├── drift.py
│   ├── drift_monitor.py
│   ├── feedback.py
│   ├── main.py
│   ├── model_registry.py
│   ├── recommender.py
│   ├── recommender_confidence.py
│   ├── schema_guard.py
│   └── __pycache__/
├── Data/
│   ├── courses.csv
│   ├── feedback.csv
│   ├── inference_logs.csv
│   └── user_feedback.csv
├── model/
│   ├── baseline_stats.json
│   ├── encoders.joblib
│   ├── metrics.joblib
│   ├── rf_model.joblib
│   └── metrics.json
├── models/
│   ├── baseline_stats.json
│   ├── content_model.pkl
│   ├── content_model_v1.pkl
│   ├── content_model_v2.pkl
│   ├── course_recommender.pkl
│   ├── create_baseline.py
│   ├── experiment_logger.py
│   ├── feature_columns.pkl
│   ├── feature_schema.json
│   └── ...
├── streamlit_app/
│   ├── analytics_dashboard.py
│   ├── app.py
│   └── mlflow_dashboard.py
├── init_db.py
├── train_model.py
├── export_mlflow_csv.py
├── show_mlflow_runs.py
├── requirements.txt
├── mlflow.db
├── users.db
├── mlflow_runs.csv
├── mlruns/
├── README.md
└── ...
```

## Setup Instructions

### 1) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Initialize local database

```bash
python init_db.py
```

### 4) Train the model

```bash
python train_model.py
```

This creates the MLflow experiment and saves model artifacts.

## Running the Project

### Start the FastAPI backend

```bash
uvicorn Backend.main:app --reload
```

Then open the Swagger docs:

```text
http://localhost:8000/docs
```

### Start the Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

### Optional: view MLflow experiment dashboard

```bash
streamlit run streamlit_app/mlflow_dashboard.py
```

Note: some dashboard paths are environment-specific and may require adjustment depending on your local file location.

## Technologies Used

- Python
- FastAPI
- Streamlit
- scikit-learn
- pandas
- joblib
- MLflow
- SQLite
- SQLAlchemy
- NumPy

## Use Cases

This project is particularly useful for:

- educational recommendation systems
- learning-path personalization
- prototype ML product demos
- model monitoring experiments
- API-driven recommendation workflows

## Strengths of the Project

- Clear separation between model logic, API layer, and UI
- Demonstrates a realistic ML pipeline
- Includes experiment tracking and model artifact management
- Provides a simple and testable user flow
- Good starter template for expanding into a larger recommender system

## Limitations / Improvement Opportunities

- The dataset is synthetic and small; production-scale use would require real learner data
- Some paths and scripts are environment-specific (for example, hardcoded Windows paths)
- Model artifact naming is slightly inconsistent across directories
- A/B testing and drift monitoring are implemented as prototypes, not full production services
- Feedback persistence and model retraining workflows can be further automated

## Future Enhancements

- Replace synthetic data with real user-course interaction data
- Add authentication and user profiles
- Support multiple recommendation models and evaluation benchmarks
- Build an automated retraining pipeline
- Deploy the API and dashboard to a cloud environment
- Add logging, monitoring, and alerting for model health

## Summary

This project is a practical machine learning recommendation system that demonstrates how to train, expose, monitor, and interact with a course recommendation model using modern Python tools. It combines classic ML with deployment patterns that are commonly used in real-world AI applications.

The combination of a Random Forest model, FastAPI backend, and Streamlit interface makes it a strong educational example for building recommendation systems end-to-end.

## License

This project does not appear to include an explicit license file. If you intend to share or publish it publicly, consider adding a license such as MIT or Apache 2.0.
