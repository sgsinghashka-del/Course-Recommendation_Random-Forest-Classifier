# Course Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow" alt="MLflow" />
</p>

<p align="center">
  <strong>Random Forest-powered recommendation engine for course suggestions</strong>
</p>

A machine learning-powered course recommendation platform that suggests the most suitable learning path based on learner profile attributes such as age, experience, interest level, and preferred domain.

## Demo Screenshots

<p align="center">
  <img src="screenshots/course%20streamlit.png" alt="Course recommendation Streamlit app" width="900" />
</p>

<p align="center">
  <img src="screenshots/course%20docs%20output.png" alt="FastAPI docs output" width="900" />
</p>

## Overview

This repository demonstrates a complete end-to-end recommendation workflow:

- Data preparation using a small domain-aware dataset
- Training a Random Forest classifier for course prediction
- Saving and loading ML models for inference
- Exposing a REST API for recommendation requests
- Capturing user feedback and monitoring signals
- Tracking experiments with MLflow
- Visualizing results in a Streamlit dashboard

The project is designed as a practical ML system prototype for educational recommendations, while also being flexible enough to adapt to other recommendation domains.

## Project Goals

- Recommend a course based on a learner's background and interests
- Provide a simple user-friendly interface for testing recommendations
- Support model experimentation and versioned deployment workflows
- Log model metrics and metadata with MLflow
- Enable behavioral feedback collection and monitoring

## Architecture

The system is split into the following components:

- `Backend/` — FastAPI application and ML support modules
- `streamlit_app/` — Streamlit UI and dashboards
- `Data/` — training and feedback datasets
- `model/` and `models/` — serialized models and baseline statistics
- Root scripts — training, database initialization, and MLflow utilities

## Key Features

- Random Forest-based recommendation engine
- FastAPI REST API with endpoints for recommendations and feedback
- Streamlit app for learner interaction
- Preference mapping to course labels
- Feedback storage for improving recommendations
- Experiment tracking with MLflow
- Drift monitoring utilities and baseline stats
- A/B testing scaffolding for model comparison

## Dataset

The sample dataset uses a synthetic learning profile table with:

- `age`
- `experience`
- `interest_level`
- `preferred_domain`
- `course_label`

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

Training logs include:

- model parameters
- accuracy metric
- experiment tags
- serialized model artifact

Core training logic is implemented in:

- `train_model.py`
- `Backend/recommender.py`

Model artifacts are stored in:

- `model/rf_model.joblib`
- `models/content_model.pkl`

## API

The backend is implemented with FastAPI and can be launched with Uvicorn.

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

The Streamlit application provides a simple UI for entering learner preferences and requesting suggestions.

Launch it with:

```bash
streamlit run streamlit_app/app.py
```

This app sends a request to the FastAPI service and displays the returned recommendations.

## MLflow Integration

MLflow is used to track experiments and model runs.

Relevant files:

- `train_model.py` — creates the experiment and logs metrics
- `show_mlflow_runs.py` — prints logged runs
- `export_mlflow_csv.py` — exports run metadata to CSV
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
├── screenshots/
│   ├── course 1.png
│   ├── course docs output.png
│   ├── course streamlit.png
│   └── init.py
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

### Optional: view MLflow dashboard

```bash
streamlit run streamlit_app/mlflow_dashboard.py
```

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

## Strengths

- Clear separation between model logic, API layer, and UI
- Demonstrates a realistic ML pipeline
- Includes experiment tracking and model artifact management
- Provides a testable user flow
- Good starter template for broader recommender systems

## Improvement Opportunities

- The dataset is synthetic and small; real data would improve realism
- Some paths and scripts are environment-specific
- Model artifact naming is slightly inconsistent across directories
- A/B testing and drift monitoring are prototype-level implementations
- Feedback persistence and retraining workflows could be automated

## Future Enhancements

- Replace synthetic data with real user-course interactions
- Add authentication and user profiles
- Support multiple recommendation models and evaluation benchmarks
- Build an automated retraining pipeline
- Deploy the API and dashboard to a cloud environment
- Add logging, monitoring, and alerting for model health

## Summary

This project is a practical machine learning recommendation system that demonstrates how to train, expose, monitor, and interact with a course recommendation model using modern Python tools.

The combination of a Random Forest model, FastAPI backend, and Streamlit interface makes it a strong educational example for building recommendation systems end-to-end.

---

<p align="center">
  Made with ❤️ for ML-powered learning recommendations
</p>
