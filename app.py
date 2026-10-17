"""
Heart Disease Prediction API — FastAPI Application with Monitoring & Logging.

Serves the trained ML model via a REST API with endpoints:
- GET  /          → API info and welcome message
- GET  /health    → Health check
- GET  /metrics   → Prometheus metrics endpoint
- POST /predict   → Predict heart disease risk from patient features
"""

import os
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "output", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "final_model.joblib")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "api.log")

# ── Logging Setup (stdout + file) ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("heart_disease_api")

# ── Prometheus Metrics Setup ───────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTIONS_COUNT = Counter(
    "model_predictions_total",
    "Total ML model predictions grouped by outcome",
    ["prediction_label", "model_name"],
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Prediction confidence score distribution",
    ["model_name"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Pure ML model inference time (feature transform + prediction)",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

# ── Load model and pipeline ───────────────────────────────────────────
model = None
preprocessor = None
metadata = None


def load_artifacts():
    """Load model, preprocessor, and metadata at startup."""
    global model, preprocessor, metadata

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline not found at: {PIPELINE_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PIPELINE_PATH)

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    else:
        metadata = {"model_name": "Unknown", "model_class": "Unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on application startup."""
    try:
        load_artifacts()
        logger.info("Model artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
    yield


# ── Pydantic Schemas ───────────────────────────────────────────────────

class PatientInput(BaseModel):
    """Input schema: patient health features for prediction."""
    age: float = Field(..., description="Age in years", ge=0, le=120, examples=[54])
    sex: float = Field(..., description="Sex (1 = male, 0 = female)", ge=0, le=1, examples=[1])
    cp: float = Field(..., description="Chest pain type (1-4)", ge=1, le=4, examples=[3])
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)", ge=50, le=300, examples=[131])
    chol: float = Field(..., description="Serum cholesterol (mg/dl)", ge=50, le=700, examples=[246])
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", ge=0, le=1, examples=[0])
    restecg: float = Field(..., description="Resting ECG results (0-2)", ge=0, le=2, examples=[1])
    thalach: float = Field(..., description="Maximum heart rate achieved", ge=50, le=250, examples=[149])
    exang: float = Field(..., description="Exercise-induced angina (1 = yes, 0 = no)", ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., description="ST depression induced by exercise", ge=0, le=10, examples=[1.0])
    slope: float = Field(..., description="Slope of peak exercise ST segment (1-3)", ge=1, le=3, examples=[2])
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)", ge=0, le=3, examples=[0])
    thal: float = Field(..., description="Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)", examples=[3])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 54, "sex": 1, "cp": 3, "trestbps": 131, "chol": 246,
                "fbs": 0, "restecg": 1, "thalach": 149, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
            }
        }
    )


class PredictionOutput(BaseModel):
    """Output schema: prediction result with confidence."""
    prediction: int = Field(..., description="0 = No Disease, 1 = Disease")
    label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Prediction confidence (probability)")
    probabilities: dict = Field(..., description="Class probabilities")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str


# ── FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "A monitored, production machine learning API that predicts heart disease "
        "risk with Prometheus metrics export and structured request logging."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Logging & Metrics Middleware ───────────────────────────────────────
@app.middleware("http")
async def log_and_meter_requests(request: Request, call_next):
    """Middleware to log request details and update Prometheus metrics."""
    start_time = time.time()
    endpoint = request.url.path
    method = request.method
    client_host = request.client.host if request.client else "unknown"

    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        logger.error(f"{method} {endpoint} unhandled error: {e}", exc_info=True)
        raise e
    finally:
        latency = time.time() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
        logger.info(
            f"{client_host} - '{method} {endpoint}' {status_code} - Latency: {latency * 1000:.2f}ms"
        )


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """API welcome endpoint with basic information."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "metrics": "GET /metrics",
            "predict": "POST /predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API is healthy and model is loaded."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=metadata.get("model_name", "Unknown") if metadata else "Not loaded",
    )


@app.get("/metrics", tags=["Monitoring"], include_in_schema=True)
async def metrics():
    """Prometheus metrics endpoint for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict heart disease risk for a patient.

    Accepts patient health features as JSON input and returns:
    - **prediction**: Binary prediction (0 = No Disease, 1 = Disease)
    - **label**: Human-readable label
    - **confidence**: Prediction confidence score
    - **probabilities**: Probability for each class
    - **model_name**: The model used for prediction
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the server logs.",
        )

    try:
        inf_start = time.time()

        # Convert input to DataFrame (preprocessor expects DataFrame)
        input_data = pd.DataFrame([patient.model_dump()])

        # Transform features using the saved preprocessing pipeline
        X = preprocessor.transform(input_data)

        # Get prediction and probabilities
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        inf_duration = time.time() - inf_start

        label = "Disease" if prediction == 1 else "No Disease"
        model_name = metadata.get("model_name", "Unknown") if metadata else "Unknown"

        # Record custom MLOps Prometheus metrics
        PREDICTIONS_COUNT.labels(prediction_label=label, model_name=model_name).inc()
        PREDICTION_CONFIDENCE.labels(model_name=model_name).observe(confidence)
        INFERENCE_LATENCY.labels(model_name=model_name).observe(inf_duration)

        logger.info(
            f"[PREDICTION] Result: {label} (pred={prediction}) | Confidence: {confidence:.4f} | "
            f"Inference: {inf_duration * 1000:.2f}ms | Age: {patient.age}, Sex: {patient.sex}, CP: {patient.cp}"
        )

        return PredictionOutput(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
            probabilities={
                "no_disease": round(float(probabilities[0]), 4),
                "disease": round(float(probabilities[1]), 4),
            },
            model_name=model_name,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ── Run with uvicorn ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
