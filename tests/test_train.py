"""
Unit tests for the model training module.

Tests cover:
- Model evaluation metrics
- Cross-validation
- Model comparison / best model selection
- Saved model artifacts existence
"""

import os
import sys
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.train import (
    evaluate_model,
    cross_validate_model,
    select_best_model,
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def binary_classification_data():
    """Generate synthetic binary classification data."""
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10,
        n_classes=2, random_state=42
    )
    return X, y


@pytest.fixture
def trained_lr(binary_classification_data):
    """Train a Logistic Regression model on synthetic data."""
    X, y = binary_classification_data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X[:160], y[:160])
    return model, X[160:], y[160:]


@pytest.fixture
def trained_rf(binary_classification_data):
    """Train a Random Forest model on synthetic data."""
    X, y = binary_classification_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X[:160], y[:160])
    return model, X[160:], y[160:]


# ── Test: evaluate_model ───────────────────────────────────────────────

class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_metrics_dict(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        assert isinstance(metrics, dict)

    def test_all_expected_metrics_present(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_in_valid_range(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} = {value} is out of [0, 1]"

    def test_predictions_are_binary(self, trained_lr):
        model, X_test, y_test = trained_lr
        _, y_pred, _ = evaluate_model(model, X_test, y_test)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_probabilities_are_valid(self, trained_lr):
        model, X_test, y_test = trained_lr
        _, _, y_prob = evaluate_model(model, X_test, y_test)
        assert (y_prob >= 0).all() and (y_prob <= 1).all()

    def test_works_with_random_forest(self, trained_rf):
        model, X_test, y_test = trained_rf
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        assert isinstance(metrics, dict)
        assert len(y_pred) == len(y_test)


# ── Test: cross_validate_model ─────────────────────────────────────────

class TestCrossValidateModel:
    """Tests for the cross_validate_model function."""

    def test_returns_dict(self, binary_classification_data):
        X, y = binary_classification_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_results = cross_validate_model(model, X, y, cv=3)
        assert isinstance(cv_results, dict)

    def test_all_metrics_present(self, binary_classification_data):
        X, y = binary_classification_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_results = cross_validate_model(model, X, y, cv=3)
        expected = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert set(cv_results.keys()) == expected

    def test_each_metric_has_mean_and_std(self, binary_classification_data):
        X, y = binary_classification_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_results = cross_validate_model(model, X, y, cv=3)
        for metric, vals in cv_results.items():
            assert "mean" in vals, f"{metric} missing 'mean'"
            assert "std" in vals, f"{metric} missing 'std'"
            assert "scores" in vals, f"{metric} missing 'scores'"

    def test_cv_scores_count_matches_folds(self, binary_classification_data):
        X, y = binary_classification_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_results = cross_validate_model(model, X, y, cv=5)
        for metric, vals in cv_results.items():
            assert len(vals["scores"]) == 5

    def test_means_in_valid_range(self, binary_classification_data):
        X, y = binary_classification_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_results = cross_validate_model(model, X, y, cv=3)
        for metric, vals in cv_results.items():
            assert 0.0 <= vals["mean"] <= 1.0


# ── Test: select_best_model ────────────────────────────────────────────

class TestSelectBestModel:
    """Tests for the select_best_model function."""

    def test_selects_higher_roc_auc(self):
        results = {
            "Model A": {"test_metrics": {"roc_auc": 0.85, "accuracy": 0.80}},
            "Model B": {"test_metrics": {"roc_auc": 0.92, "accuracy": 0.78}},
        }
        best_name, best_result = select_best_model(results)
        assert best_name == "Model B"

    def test_returns_name_and_result(self):
        results = {
            "Model A": {"test_metrics": {"roc_auc": 0.90}},
        }
        best_name, best_result = select_best_model(results)
        assert isinstance(best_name, str)
        assert isinstance(best_result, dict)


# ── Test: saved artifacts ──────────────────────────────────────────────

class TestSavedArtifacts:
    """Tests to verify that model artifacts were saved correctly."""

    def test_final_model_exists(self):
        path = os.path.join(PROJECT_ROOT, "output", "models", "final_model.joblib")
        assert os.path.exists(path), "Final model file should exist"

    def test_preprocessing_pipeline_exists(self):
        path = os.path.join(PROJECT_ROOT, "output", "models", "preprocessing_pipeline.joblib")
        assert os.path.exists(path), "Preprocessing pipeline file should exist"

    def test_model_metadata_exists(self):
        path = os.path.join(PROJECT_ROOT, "output", "models", "model_metadata.json")
        assert os.path.exists(path), "Model metadata file should exist"

    def test_model_metadata_valid_json(self):
        import json
        path = os.path.join(PROJECT_ROOT, "output", "models", "model_metadata.json")
        if os.path.exists(path):
            with open(path) as f:
                metadata = json.load(f)
            assert "model_name" in metadata
            assert "model_class" in metadata

    def test_model_comparison_plot_exists(self):
        path = os.path.join(PROJECT_ROOT, "output", "figures", "model_comparison.png")
        assert os.path.exists(path), "Model comparison plot should exist"
