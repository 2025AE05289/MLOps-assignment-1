# Heart Disease Prediction — End-to-End MLOps Solution

A scalable, reproducible, and monitored machine learning solution for predicting heart disease risk using the **UCI Heart Disease Dataset**, engineered in accordance with modern MLOps production standards.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Architecture & Workflow](#architecture--workflow)
4. [Quick Start & Setup](#quick-start--setup)
5. [Assignment Tasks Implementation](#assignment-tasks-implementation)
   - [Task 1: Data Acquisition & EDA](#task-1-data-acquisition--eda)
   - [Task 2: Feature Engineering & Model Training](#task-2-feature-engineering--model-training)
   - [Task 3: Experiment Tracking with MLflow](#task-3-experiment-tracking-with-mlflow)
   - [Task 4: Model Packaging & Reproducibility](#task-4-model-packaging--reproducibility)
   - [Task 5: CI/CD Pipeline & Automated Testing](#task-5-cicd-pipeline--automated-testing)
   - [Task 6: Model Containerization & API Serving](#task-6-model-containerization--api-serving)
6. [API Documentation](#api-documentation)
7. [Running Tests & Linting](#running-tests--linting)

---

## Project Overview

- **Dataset**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) (Cleveland database: 303 patient records, 13 clinical features).
- **Problem Statement**: Predict the presence of heart disease (binary classification: 0 = No Disease, 1 = Disease present) using clinical indicators.
- **Key Highlights**:
  - Reproducible preprocessing pipeline with Scikit-learn `ColumnTransformer`.
  - Multiple algorithms compared: Logistic Regression (L2 regularized) and Random Forest (tuned ensemble).
  - Experiment tracking with **MLflow** (metrics, parameters, confusion matrix, ROC curve, model artifacts).
  - Production-ready REST API built with **FastAPI** and **Pydantic v2**.
  - Containerized with **Docker** (slim Python base, health checks).
  - Automated **CI/CD** on GitHub Actions with 3 stages: Linting, Unit Tests, and Training & Packaging.

---

## Repository Structure

```
MLOps-Assignment/
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions 4-stage CI/CD pipeline
├── data/
│   ├── download_data.py             # Automated dataset downloader (ucimlrepo / web fallback)
│   ├── raw/                         # Raw unprocessed dataset
│   └── processed/                   # Preprocessed & binarized dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Cleaning, missing-value imputation, target binarization
│   ├── eda.py                       # Exploratory Data Analysis & plot generation
│   ├── feature_engineering.py       # Scikit-learn scaling & one-hot encoding pipelines
│   └── train.py                     # Model training, cross-validation, MLflow tracking
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py   # Unit tests for preprocessing & cleaning
│   ├── test_feature_engineering.py  # Unit tests for transformers & pipelines
│   └── test_train.py                # Unit tests for evaluation, CV, and artifacts
├── output/
│   ├── figures/                     # EDA visualizations & model evaluation plots
│   ├── models/                      # Saved final model, pipeline, and metadata
│   └── mlruns/                      # MLflow experiment tracking database
├── Dockerfile                       # Production Docker container definition
├── .dockerignore                    # Build optimization filter
├── app.py                           # FastAPI model serving REST API
├── test_api.sh                      # Shell script for endpoint testing
├── requirements.txt                 # Pinned project dependencies
└── README.md                        # Project documentation
```

---

## Quick Start & Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd MLOps-Assignment

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Data Acquisition & EDA

```bash
# Download the dataset
python data/download_data.py

# Run Exploratory Data Analysis (generates plots in output/figures/)
python src/eda.py
```

### 3. Train Models & Track with MLflow

```bash
# Run training pipeline
python src/train.py

# Launch MLflow Dashboard
mlflow ui --backend-store-uri file://$(pwd)/output/mlruns
# Visit http://127.0.0.1:5000 in your browser
```

### 4. Start the Prediction API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Interactive documentation available at: http://localhost:8000/docs
```

---

## Assignment Tasks Implementation

### Task 1: Data Acquisition & EDA
- **Acquisition**: `data/download_data.py` programmatically fetches the dataset using `ucimlrepo` with fallback to direct UCI HTTP mirror.
- **Preprocessing**: `src/data_preprocessing.py` handles missing values (median for continuous, mode for categorical), maps target values (0 = healthy, 1–4 = heart disease).
- **EDA Visualizations**: `src/eda.py` outputs high-resolution figures to `output/figures/`:
  - `01_target_distribution.png`: Class balance.
  - `02_numerical_histograms.png`: Distributions of numerical features.
  - `03_correlation_heatmap.png`: Feature correlation matrix.
  - `04_boxplots_by_target.png`: Feature distributions by diagnosis.
  - `05_categorical_distributions.png`: Frequency breakdown of categorical features.
  - `06_pairplot.png`: Pairwise relationship matrix.

### Task 2: Feature Engineering & Model Development
- **Pipelines**: `src/feature_engineering.py` implements a Scikit-Learn `ColumnTransformer`:
  - Standard scaling for continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
  - One-hot encoding for multi-category features (`cp`, `restecg`, `slope`, `thal`).
  - Binary pass-through for binary indicators (`sex`, `fbs`, `exang`).
- **Models**: `src/train.py` trains two classification algorithms:
  - **Logistic Regression**: Scaled inputs, L2 regularization, LBFGS solver.
  - **Random Forest**: 200 estimators, balanced class weights, tuned max depth.
- **Validation**: 5-Fold Stratified Cross-Validation evaluating Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Task 3: Experiment Tracking with MLflow
- Integrated in `src/train.py` with custom tracking URI (`output/mlruns`):
  - Parameters logged: Hyperparameters, CV fold count, feature dimensions.
  - Metrics logged: Cross-validation mean & standard deviation, test set metrics.
  - Artifacts logged: ROC curves, confusion matrices, classification reports, feature importance, and serialized models.

### Task 4: Model Packaging & Reproducibility
- Preprocessor pipeline saved as `output/models/preprocessing_pipeline.joblib`.
- Champion model saved as `output/models/final_model.joblib`.
- Model metadata saved as `output/models/model_metadata.json`.
- Complete dependencies tracked in `requirements.txt`.

### Task 5: CI/CD Pipeline & Automated Testing
- **Test Suite**: 55 unit tests covering data preprocessing, feature engineering pipelines, and model training/evaluation (`tests/test_*.py`).
- **Pipeline**: GitHub Actions (`.github/workflows/ci.yml`) executes on every `push` and `pull_request` to `main`/`master`:
  1. **Lint Code**: Enforces PEP8 standards via `flake8`.
  2. **Unit Tests**: Runs `pytest` with coverage report generation and artifact upload.
  3. **Train & Package**: Executes training pipeline and publishes model/EDA artifacts.

### Task 6: Model Containerization & API Serving
- **FastAPI API**: Implemented in `app.py` with Pydantic v2 data validation schemas.
- **Docker**: `Dockerfile` uses `python:3.12-slim` with health check instructions and minimal layer sizing via `.dockerignore`.

---

## API Documentation

### Run with Docker

```bash
# Build the Docker image
docker build -t heart-disease-api:latest .

# Run the container
docker run -d --name heart-api -p 8000:8000 heart-disease-api:latest

# Check logs
docker logs -f heart-api
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and links to docs |
| `GET` | `/health` | Health status and model loaded status |
| `POST` | `/predict` | Predict heart disease risk |
| `GET` | `/docs` | Swagger interactive UI |

### Sample Request (`POST /predict`)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54,
    "sex": 1,
    "cp": 3,
    "trestbps": 131,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 149,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3
  }'
```

### Sample Response

```json
{
  "prediction": 0,
  "label": "No Disease",
  "confidence": 0.8489,
  "probabilities": {
    "no_disease": 0.8489,
    "disease": 0.1511
  },
  "model_name": "Logistic Regression"
}
```

### Automated Script Testing

```bash
bash test_api.sh http://localhost:8000
```

---

## Running Tests & Linting

```bash
# Run code linter
flake8 src/ app.py tests/ --max-line-length=120 --statistics --count --ignore=E501,W503,E203,W293,E402

# Run full test suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```
