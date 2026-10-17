# Heart Disease Prediction — End-to-End MLOps Solution

A scalable, reproducible, containerized, and monitored machine learning solution for predicting heart disease risk using the **UCI Heart Disease Dataset**, engineered in accordance with modern MLOps production standards.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start & Setup](#quick-start--setup)
4. [Assignment Tasks Implementation](#assignment-tasks-implementation)
   - [Task 1: Data Acquisition & EDA](#task-1-data-acquisition--eda)
   - [Task 2: Feature Engineering & Model Training](#task-2-feature-engineering--model-training)
   - [Task 3: Experiment Tracking with MLflow](#task-3-experiment-tracking-with-mlflow)
   - [Task 4: Model Packaging & Reproducibility](#task-4-model-packaging--reproducibility)
   - [Task 5: CI/CD Pipeline & Automated Testing](#task-5-cicd-pipeline--automated-testing)
   - [Task 6: Model Containerization & API Serving](#task-6-model-containerization--api-serving)
   - [Task 7: Production Deployment (Kubernetes on AKS / Docker Desktop)](#task-7-production-deployment)
   - [Task 8: Monitoring & Logging (Prometheus + Grafana)](#task-8-monitoring--logging)
5. [API Documentation](#api-documentation)
6. [One-Line Pipeline Execution](#one-line-pipeline-execution)
7. [Makefile Shortcuts](#makefile-shortcuts)

---

## Project Overview

- **Dataset**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) (Cleveland database: 303 patient records, 13 clinical features).
- **Problem Statement**: Predict heart disease risk (binary classification: 0 = No Disease, 1 = Disease present) using clinical health indicators.
- **Key MLOps Features**:
  - Reproducible preprocessing pipeline with Scikit-learn `ColumnTransformer`.
  - Logistic Regression (L2 regularized) vs. Random Forest (tuned ensemble) compared via 5-fold Stratified CV.
  - Experiment tracking with **MLflow** (parameters, metrics, confusion matrix, ROC curve, model artifacts).
  - Production REST API built with **FastAPI** and **Pydantic v2**.
  - Containerized with **Docker** (slim Python base, health checks).
  - Automated 3-stage **CI/CD** on GitHub Actions (Linting, Unit Tests, and Training & Packaging).
  - **Kubernetes Production Deployment**: Declarative manifests (`k8s/`) with replicas, health probes, and LoadBalancer.
  - **Monitoring & Logging**: Structured request logging (`logs/api.log`), Prometheus `/metrics`, and pre-configured **Grafana dashboard**.

---

## Repository Structure

```
MLOps-Assignment/
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions 3-stage CI/CD pipeline
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
├── k8s/
│   ├── deployment.yaml              # Kubernetes Deployment (2 replicas, health probes, limits)
│   ├── service.yaml                 # Kubernetes LoadBalancer Service (Public IP)
│   └── hpa.yaml                     # HorizontalPodAutoscaler (CPU autoscale 2-5 pods)
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml           # Prometheus scrape config (/metrics)
│   └── grafana/
│       ├── provisioning/            # Auto-provisioned datasources & dashboard providers
│       └── dashboards/definitions/
│           └── mlops_dashboard.json # Pre-built MLOps Monitoring Dashboard
├── Dockerfile                       # Production Docker container definition
├── docker-compose.monitoring.yml    # Prometheus (9090) + Grafana (3000) stack
├── deploy_azure.sh                  # Quota-safe AKS automated deploy & destroy script
├── Makefile                         # Single-command workflow shortcuts
├── main.py                          # One-line end-to-end pipeline orchestrator
├── app.py                           # Monitored FastAPI prediction REST API
├── test_api.sh                      # Shell script for endpoint verification
├── requirements.txt                 # Pinned project dependencies
└── README.md                        # Project documentation
```

---

## Quick Start & Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Mr-ZeelPatel/MLOps-assignment-1.git
cd MLOps-assignment-1

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
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
- **View Dashboard**:
  ```bash
  mlflow ui --backend-store-uri sqlite:///output/mlruns/mlflow.db --port 5001
  ```
  Open **http://127.0.0.1:5001** to view.

### Task 4: Model Packaging & Reproducibility
- Preprocessor pipeline saved as `output/models/preprocessing_pipeline.joblib`.
- Champion model saved as `output/models/final_model.joblib`.
- Model metadata saved as `output/models/model_metadata.json`.
- Complete dependencies tracked in `requirements.txt`.

### Task 5: CI/CD Pipeline & Automated Testing
- **Test Suite**: 55 unit tests covering data preprocessing, feature engineering pipelines, and model training/evaluation (`tests/test_*.py`).
- **Pipeline**: GitHub Actions (`.github/workflows/ci.yml`) executes on every `push` and `pull_request` to `main`:
  1. **Lint Code**: Enforces PEP8 standards via `flake8`.
  2. **Unit Tests**: Runs `pytest` with coverage report generation and artifact upload.
  3. **Train & Package**: Executes training pipeline and publishes model/EDA artifacts.

### Task 6: Model Containerization & API Serving
- **FastAPI API**: Implemented in `app.py` with Pydantic v2 data validation schemas.
- **Docker**: `Dockerfile` uses `python:3.12-slim` with health check instructions and minimal layer sizing via `.dockerignore`.
- **Run Container**:
  ```bash
  docker build -t heart-disease-api:latest .
  docker run -d --name heart-api -p 8000:8000 heart-disease-api:latest
  bash test_api.sh http://localhost:8000
  ```

---

### Task 7: Production Deployment

Declarative Kubernetes manifests are located in `k8s/`:
- **`k8s/deployment.yaml`**: Deploys 2 pod replicas with resource requests/limits (`cpu: 100m-500m`, `memory: 256Mi-512Mi`), `livenessProbe` and `readinessProbe` checking `/health`.
- **`k8s/service.yaml`**: Exposes the pods via a `LoadBalancer` allocating an external public IP on port 80.
- **`k8s/hpa.yaml`**: Configures automated autoscaling from 2 to 5 pods when CPU utilization exceeds 70%.

#### Option A: Deploy to Azure Kubernetes Service (AKS)
Optimized for **Azure for Students ($100 credit, 4 vCPU quota)** using 1 node of `Standard_B2s` (~$0.04/hr):

```bash
# 1. Login to Azure (if using local terminal) or use Azure Cloud Shell (shell.azure.com)
az login

# 2. Deploy cluster and manifests automatically
bash deploy_azure.sh deploy

# 3. Check deployment & obtain Public IP
bash deploy_azure.sh status

# 4. Test live prediction on Azure Public IP
bash deploy_azure.sh test <PUBLIC_IP>

# 5. IMPORTANT: Destroy cluster when done to preserve your $100 credit!
bash deploy_azure.sh destroy
```

#### Option B: Deploy Locally on Docker Desktop Kubernetes / Minikube
```bash
# 1. Apply Kubernetes manifests
kubectl apply -f k8s/

# 2. Check pod and service status
kubectl get pods,svc -o wide

# 3. Test locally
curl http://localhost/health
curl -X POST http://localhost/predict -H "Content-Type: application/json" -d @sample_patient.json

# 4. Cleanup
kubectl delete -f k8s/
```

---

### Task 8: Monitoring & Logging

The application includes production-grade observability:

1. **Structured Request Logging**:
   - `app.py` logs client IP, timestamp, HTTP method, path, status code, and latency (ms) for every request.
   - Logs inference results: predicted class, confidence score, and input features summary.
   - Logged to standard output (`stdout`) and saved persistently to `logs/api.log`.

2. **Prometheus Metrics Endpoint (`GET /metrics`)**:
   - HTTP traffic metrics: `http_requests_total`, `http_request_duration_seconds` (latency histogram).
   - Custom ML business metrics:
     - `model_predictions_total{prediction_label="Disease|No Disease"}`
     - `model_prediction_confidence` (distribution histogram)
     - `model_inference_duration_seconds` (pure ML inference latency)

3. **Prometheus + Grafana Monitoring Stack**:
   Start Prometheus and Grafana with one command:
   ```bash
   make monitoring
   # OR: docker compose -f docker-compose.monitoring.yml up -d
   ```

   - **Prometheus UI**: **[http://localhost:9090](http://localhost:9090)**
   - **Grafana Dashboard**: **[http://localhost:3000](http://localhost:3000)** (Username: `admin`, Password: `admin`)

   The **Heart Disease MLOps Monitoring Dashboard** is pre-configured with:
   - **Total Predictions Served** (Counter)
   - **Average Prediction Confidence** (Gauge)
   - **Predictions by Class** (Donut chart: Disease vs. Healthy)
   - **API Request Rate (req/sec)** (Time Series)
   - **API Latency (p50, p95, p99)** (Time Series)
   - **HTTP Status Code Breakdown** (Time Series)

   To stop monitoring:
   ```bash
   make monitoring-down
   ```

---

## API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and links to docs |
| `GET` | `/health` | Health status and model loaded status |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `POST` | `/predict` | Predict heart disease risk |
| `GET` | `/docs` | Swagger interactive UI |

### Sample Prediction Request (`POST /predict`)

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

### Sample Response (`200 OK`)

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

---

## One-Line Pipeline Execution

Run the complete pipeline from data download to testing in a single command:

```bash
python main.py
```

Optional flags:
- `python main.py --skip-eda`: Skip generating exploratory analysis plots.
- `python main.py --skip-tests`: Skip running the unit test suite.
- `python main.py --serve`: Automatically start the API server upon pipeline completion.
- `python main.py --mlflow`: Launch the MLflow UI upon pipeline completion.

---

## Makefile Shortcuts

| Command | Action |
|---|---|
| `make pipeline` | Run full pipeline (`main.py`) |
| `make train` | Train models, log to MLflow & package model |
| `make test` | Run 55 unit tests with coverage report |
| `make lint` | Check code with `flake8` |
| `make serve` | Start local FastAPI server on port 8000 |
| `make mlflow` | Start MLflow UI on port 5001 |
| `make docker-build` | Build Docker container |
| `make docker-run` | Run Docker container on port 8000 |
| `make monitoring` | Launch Prometheus (9090) and Grafana (3000) |
| `make monitoring-down` | Stop Prometheus and Grafana |
| `make k8s-apply` | Apply Kubernetes deployment & service |
| `make k8s-status` | View running pods and LoadBalancer IP |
| `make k8s-delete` | Delete Kubernetes deployment |
