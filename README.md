# Heart Disease Classification — End-to-End MLOps Solution

[![CI/CD Pipeline](https://github.com/2025AE05289/MLOps-assignment-1/actions/workflows/ci.yml/badge.svg)](https://github.com/2025AE05289/MLOps-assignment-1/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-AKS-326CE5.svg)](https://azure.microsoft.com/en-us/products/kubernetes-service/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800.svg)](https://grafana.com/)

A production-grade, cloud-native Machine Learning Operations (MLOps) pipeline for **Heart Disease Classification** using the **UCI Heart Disease Dataset**. The system automates data acquisition, feature engineering, experiment tracking, continuous integration, containerization, Kubernetes orchestration, and full-stack observability.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Live Cloud Services](#-live-cloud-services-azure-aks)
- [Project Structure](#-project-structure)
- [Machine Learning Lifecycle](#-machine-learning-lifecycle)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Quickstart Guide](#-quickstart-guide)
- [API Usage & Endpoints](#-api-usage--endpoints)
- [Monitoring & Observability](#-monitoring--observability)
- [Author & Academic Context](#-author--academic-context)

---

## 🏛 System Architecture

The solution follows modern cloud-native MLOps architecture patterns, decoupling offline experimentation from online serving and continuous delivery:

```
[Raw UCI Data] ──> [Data Cleaning & EDA] ──> [Feature Pipeline]
                                                    │
                                                    ▼
[MLflow Server] <─── [5-Fold CV & Metrics] <── [Model Training]
 (Remote :5000)                                     │
                                                    ▼
                                           [Champion Model]
                                                    │
[GitHub Actions CI] ───> [Docker Image] ───> [Azure ACR]
 (Lint + Test + Train)                              │
                                                    ▼
                                           [Azure AKS Cluster]
                                           ├── API Pods (HPA 2-5)
                                           ├── NGINX Gateway (:80, :3000, :5000)
                                           ├── Prometheus Scraper
                                           └── Grafana Dashboard
```

---

## 🌐 Live Cloud Services (Azure AKS)

All services are deployed to **Azure Kubernetes Service (AKS)** in the `centralindia` region under a single LoadBalancer Public IP (`4.224.167.2`):

| Service | Public URL | Port | Description |
| :--- | :--- | :---: | :--- |
| **FastAPI Swagger UI** | [http://4.224.167.2/docs](http://4.224.167.2/docs) | `80` | Interactive OpenAPI documentation and test client |
| **Health Check** | [http://4.224.167.2/health](http://4.224.167.2/health) | `80` | Kubernetes liveness and readiness probe |
| **Inference API** | `POST http://4.224.167.2/predict` | `80` | Real-time prediction endpoint |
| **Prometheus Metrics** | [http://4.224.167.2/metrics](http://4.224.167.2/metrics) | `80` | Live operational and ML metric exposition |
| **Grafana Dashboard** | [http://4.224.167.2:3000](http://4.224.167.2:3000) | `3000` | Real-time monitoring (`admin` / `admin`) |
| **MLflow UI** | [http://4.224.167.2:5000](http://4.224.167.2:5000) | `5000` | Experiment runs, metric comparison, model registry |

---

## 📂 Project Structure

```text
├── .github/workflows/       # GitHub Actions CI/CD workflows
│   └── ci.yml               # 3-stage pipeline: Lint, Test, Train & Cloud Sync
├── data/                    # Dataset directory
│   ├── download_data.py     # UCI dataset downloader script
│   ├── raw/                 # Raw UCI dataset
│   └── processed/           # Imputed, cleaned, and binarized data
├── k8s/                     # Kubernetes manifests (AKS)
│   ├── deployment.yaml      # FastAPI deployment with resource limits & probes
│   ├── hpa.yaml             # Horizontal Pod Autoscaler (2–5 replicas @ 70% CPU)
│   ├── gateway.yaml         # NGINX LoadBalancer gateway (ports 80, 3000, 5000)
│   ├── mlflow.yaml          # MLflow tracking server on Kubernetes
│   ├── monitoring-prometheus.yaml # Prometheus server & scrape configuration
│   └── monitoring-grafana.yaml    # Grafana dashboard & datasource provisioning
├── monitoring/              # Local Docker Compose observability stack
│   ├── docker-compose.yml
│   └── prometheus/
├── output/                  # Generated artifacts
│   ├── figures/             # EDA visualisations & evaluation curves
│   └── models/              # Champion model, preprocessor, and metadata
├── screenshots/             # Deployment and dashboard verification images
├── scripts/                 # Utility scripts (testing, docx generation)
│   ├── test_api.sh          # Automated API test script
│   └── generate_docx.py     # Markdown-to-Word report generator
├── src/                     # Core ML source code
│   ├── data_preprocessing.py # Missing value imputation & target binarization
│   ├── eda.py               # Exploratory data analysis & figure generator
│   ├── feature_engineering.py# ColumnTransformer: StandardScaler + OneHotEncoder
│   └── train.py             # Model training, 5-fold CV, MLflow tracking & packaging
├── tests/                   # Pytest test suite (55 unit tests, 100% pass)
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_train.py
├── app.py                   # Production FastAPI inference application
├── Dockerfile               # Production container image definition
├── Makefile                 # Developer task automation
├── main.py                  # Single-command pipeline orchestrator
└── requirements.txt         # Production dependencies
```

---

## 🤖 Machine Learning Lifecycle

### 1. Data Cleaning & Feature Engineering
- **Missing Values**: Handled discrete feature missingness via mode imputation (`ca=0.0`, `thal=3.0`).
- **Target Binarization**: Converted multiclass clinical stage (`0..4`) into binary classification (`0`: Healthy, `1`: Heart Disease Present).
- **Preprocessing Pipeline**: Built with Scikit-Learn `ColumnTransformer` featuring `StandardScaler` for 5 numerical features and `OneHotEncoder(drop='first')` for 8 categorical features. Fitted strictly on training splits to prevent data leakage.

### 2. Model Evaluation & Benchmark Results
Evaluated using **5-Fold Stratified Cross-Validation** and a held-out test split (20%):

| Metric | Logistic Regression (CV) | Logistic Regression (Test) | Random Forest (CV) | Random Forest (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.8429 ± 0.0168 | **0.8689** | 0.8016 ± 0.0215 | **0.9016** |
| **Precision** | 0.8658 ± 0.0680 | **0.8333** | 0.8045 ± 0.0435 | **0.8438** |
| **Recall (Sensitivity)** | 0.7921 ± 0.0751 | **0.8929** | 0.7561 ± 0.0633 | **0.9643** |
| **F1-Score** | 0.8214 ± 0.0215 | **0.8621** | 0.7767 ± 0.0284 | **0.9000** |
| **ROC-AUC** | **0.9018 ± 0.0172** | **0.9621** | 0.8828 ± 0.0251 | **0.9481** |

> **Champion Model**: **Logistic Regression** was chosen due to higher generalization discrimination (**ROC-AUC 0.9621**), lower fold variance, and superior clinical interpretability.

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) executes automatically on every push to `main`:

1. **Lint Job**: Enforces PEP8 standards using `flake8` across `src/`, `app.py`, and `tests/`.
2. **Unit Test Job**: Executes **55 unit tests** via `pytest` with code coverage (`pytest-cov`).
3. **Train & Packaging Job**: Trains the champion model, packages artifacts, and syncs experiment metrics with the remote cloud MLflow server.

---

## 🚀 Quickstart Guide

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/2025AE05289/MLOps-assignment-1.git
cd MLOps-assignment-1

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the entire pipeline (data -> eda -> train -> lint -> test)
python main.py
# or using Makefile:
make pipeline
```

### Running Services Locally

```bash
# Start FastAPI inference server (http://localhost:8000)
make serve

# Start local MLflow UI (http://localhost:5001)
make mlflow

# Start local Prometheus & Grafana stack (http://localhost:3000)
make monitoring
```

---

## 📡 API Usage & Endpoints

### Single Patient Prediction (`POST /predict`)

```bash
curl -X POST "http://4.224.167.2/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 63,
       "sex": 1,
       "cp": 3,
       "trestbps": 145,
       "chol": 233,
       "fbs": 1,
       "restecg": 0,
       "thalach": 150,
       "exang": 0,
       "oldpeak": 2.3,
       "slope": 0,
       "ca": 0,
       "thal": 1
     }'
```

#### Sample Response:
```json
{
  "prediction": 0,
  "label": "No Disease",
  "confidence": 0.8489,
  "probability": {
    "No Disease": 0.8489,
    "Disease": 0.1511
  },
  "model_version": "1.0.0"
}
```

---

## 📊 Monitoring & Observability

- **Structured Request Logs**: Every inference request logs timestamp, client IP, latency, prediction, and confidence to [`logs/api.log`](logs/api.log).
- **Prometheus Metrics (`/metrics`)**: Exposes request rates, response times, model confidence distributions, and prediction counts by class.
- **Grafana Dashboard**: Auto-configured panels for throughput, p95 latency, model confidence gauge, and real-time class distribution.

---

## 👨‍💻 Author & Academic Context

- **Author**: Patel Zeel Dharmendrakumar
- **BITS ID**: `2025AE05289`
- **Email**: `2025ae05289@wilp.bits-pilani.ac.in`
- **Course**: MLOps
- **GitHub**: [@2025AE05289](https://github.com/2025AE05289)

