"""
Heart Disease Prediction API — FastAPI Application.

Serves the trained ML model via a REST API with endpoints:
- GET  /          → API info and welcome message
- GET  /health    → Health check
- POST /predict   → Predict heart disease risk from patient features
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "output", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "final_model.joblib")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

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
        print("[INFO] Model artifacts loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model artifacts: {e}")
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
        "A machine learning API that predicts the risk of heart disease "
        "based on patient health data. Built with FastAPI and scikit-learn."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/", tags=["Info"])
async def root():
    """API welcome endpoint with basic information."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
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
        # Convert input to DataFrame (preprocessor expects DataFrame)
        input_data = pd.DataFrame([patient.model_dump()])

        # Transform features using the saved preprocessing pipeline
        X = preprocessor.transform(input_data)

        # Get prediction and probabilities
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        label = "Disease" if prediction == 1 else "No Disease"

        return PredictionOutput(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
            probabilities={
                "no_disease": round(float(probabilities[0]), 4),
                "disease": round(float(probabilities[1]), 4),
            },
            model_name=metadata.get("model_name", "Unknown") if metadata else "Unknown",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ── Run with uvicorn ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
