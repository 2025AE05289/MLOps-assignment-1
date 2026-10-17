#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# test_api.sh — Test the Heart Disease Prediction API
#
# Usage:
#   1. Start the API:  uvicorn app:app --host 0.0.0.0 --port 8000
#   2. Run this script: bash test_api.sh
# ──────────────────────────────────────────────────────────────────

BASE_URL="${1:-http://localhost:8000}"

echo "================================================"
echo "  Testing Heart Disease Prediction API"
echo "  Base URL: $BASE_URL"
echo "================================================"

# ── Test 1: Root endpoint ──
echo -e "\n[TEST 1] GET / (Root)"
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""

# ── Test 2: Health check ──
echo "[TEST 2] GET /health (Health Check)"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# ── Test 3: Predict — healthy patient (low risk) ──
echo "[TEST 3] POST /predict — Low risk patient"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "sex": 0,
    "cp": 1,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 180,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 0,
    "thal": 3
  }' | python3 -m json.tool
echo ""

# ── Test 4: Predict — high risk patient ──
echo "[TEST 4] POST /predict — High risk patient"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "sex": 1,
    "cp": 4,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 7
  }' | python3 -m json.tool
echo ""

# ── Test 5: Predict — average patient ──
echo "[TEST 5] POST /predict — Average patient"
curl -s -X POST "$BASE_URL/predict" \
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
  }' | python3 -m json.tool

echo ""
echo "================================================"
echo "  All API tests completed!"
echo "  Swagger docs available at: $BASE_URL/docs"
echo "================================================"
