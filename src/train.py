"""
Model Training Script with MLflow Experiment Tracking.

Covers:
- Task 2: Model Development (Logistic Regression + Random Forest)
         Cross-validation, metrics (accuracy, precision, recall, ROC-AUC)
- Task 3: MLflow experiment tracking (params, metrics, artifacts, plots)
- Task 4: Model packaging & reproducibility (saved model, pipeline, requirements)

Usage:
    python src/train.py
"""

import os
import sys

# Ensure MLflow records user as Zeel
os.environ["MLFLOW_USER"] = "Zeel"
os.environ["LOGNAME"] = "Zeel"
os.environ["USER"] = "Zeel"

import json
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MLRUNS_DIR = os.path.join(OUTPUT_DIR, "mlruns")

for d in [MODELS_DIR, FIGURES_DIR, MLRUNS_DIR]:
    os.makedirs(d, exist_ok=True)

from src.data_preprocessing import preprocess_data
from src.feature_engineering import (
    prepare_features,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


# ── Plot Helpers ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                linewidths=1, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_prob, model_name):
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, f"roc_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, model_name):
    """Generate and save feature importance plot (for tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 15 features
    top_n = min(15, len(importances))
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if feature_names is not None and len(feature_names) == len(importances):
        names = [feature_names[i] for i in top_indices]
    else:
        names = [f"Feature {i}" for i in top_indices]
    
    ax.barh(range(top_n), importances[top_indices], color="#2563EB", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, f"importance_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results):
    """Generate comparison bar chart across models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    colors = ["#2563EB", "#7C3AED"]
    
    for i, model_name in enumerate(model_names):
        values = [results[model_name]["test_metrics"][m] for m in metrics_to_plot]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)],
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Training ───────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Compute all evaluation metrics on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    
    return metrics, y_pred, y_prob


def cross_validate_model(model, X, y, cv=5):
    """Perform stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(
        model, X, y, cv=skf, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )
    
    cv_summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        cv_summary[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        }
    
    return cv_summary


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test,
                        params, feature_names=None):
    """
    Train a model, evaluate it, and log everything to MLflow.
    
    Parameters
    ----------
    model : sklearn estimator
    model_name : str
    X_train, X_test, y_train, y_test : arrays
    params : dict
    feature_names : list, optional
    
    Returns
    -------
    dict : Results including metrics and model
    """
    print(f"\n{'─' * 60}")
    print(f"  Training: {model_name}")
    print(f"{'─' * 60}")
    
    with mlflow.start_run(run_name=model_name):
        # Set user tags explicitly to Zeel
        mlflow.set_tag("mlflow.user", "Zeel")
        mlflow.set_tag("user_id", "Zeel")
        mlflow.set_tag("author", "Zeel Patel")
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)
        
        # Train
        model.fit(X_train, y_train)
        
        # Cross-validation on training data
        print("\n  [CV] Running 5-fold stratified cross-validation...")
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)
        
        print("\n  Cross-Validation Results:")
        for metric, vals in cv_results.items():
            print(f"    {metric:>12}: {vals['mean']:.4f} ± {vals['std']:.4f}")
            mlflow.log_metric(f"cv_{metric}_mean", vals["mean"])
            mlflow.log_metric(f"cv_{metric}_std", vals["std"])
        
        # Test set evaluation
        test_metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        print("\n  Test Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"    {metric:>12}: {value:.4f}")
            mlflow.log_metric(f"test_{metric}", value)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
        print(f"\n  Classification Report:\n{report}")
        
        # Save classification report as artifact
        report_path = os.path.join(FIGURES_DIR, f"report_{model_name.lower().replace(' ', '_')}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report — {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        # Generate plots
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_path = plot_roc_curve(y_test, y_prob, model_name)
        imp_path = plot_feature_importance(model, feature_names, model_name)

        # Log artifacts with MLflow (graceful for remote/CI environments)
        try:
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            if imp_path:
                mlflow.log_artifact(imp_path)
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                serialization_format="pickle",
            )
        except Exception as e:
            print(f"  [INFO] Remote artifact upload notice: {e}")
        
        print("\n  [MLflow] Run logged successfully.")
    
    return {
        "model": model,
        "model_name": model_name,
        "params": params,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def select_best_model(results):
    """Select the best model based on ROC-AUC score."""
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["roc_auc"])
    best = results[best_name]
    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best_name}")
    print(f"  ROC-AUC: {best['test_metrics']['roc_auc']:.4f}")
    print(f"{'=' * 60}")
    return best_name, best


def save_final_model(model, preprocessor, model_name):
    """
    Save the final model and preprocessor for reproducibility (Task 4).
    
    Saves:
    - Model as pickle (.joblib)
    - Preprocessor pipeline as pickle (.joblib)
    - Model metadata as JSON
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    joblib.dump(model, model_path)
    print(f"[SAVED] Final model: {model_path}")
    
    # Save preprocessor
    pipeline_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, pipeline_path)
    print(f"[SAVED] Preprocessing pipeline: {pipeline_path}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_class": type(model).__name__,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": sys.version,
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Model metadata: {meta_path}")
    
    return model_path, pipeline_path


# ── Main ───────────────────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("  MLOps HEART DISEASE CLASSIFIER — Training Pipeline")
    print("=" * 60)
    
    # ── Step 1: Load and preprocess data ──
    print("\n[STEP 1] Loading and preprocessing data...")
    df = preprocess_data()
    
    # ── Step 2: Feature engineering ──
    print("\n[STEP 2] Feature engineering...")
    X, y, preprocessor = prepare_features(df)
    
    # Get feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = None
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")
    
    # ── Step 3: Setup MLflow ──
    print("\n[STEP 3] Setting up MLflow experiment tracking...")
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow_db_path = os.path.join(MLRUNS_DIR, "mlflow.db")

    # Safeguard: if an existing DB references paths from a different machine, reset it
    if os.path.exists(mlflow_db_path):
        try:
            temp_client = mlflow.tracking.MlflowClient(f"sqlite:///{mlflow_db_path}")
            exp = temp_client.get_experiment_by_name("heart_disease_classification")
            if exp and not exp.artifact_location.startswith(PROJECT_ROOT) and not os.path.exists(exp.artifact_location):
                print(f"[WARN] Stale artifact location detected ({exp.artifact_location}). Resetting database...")
                os.remove(mlflow_db_path)
        except Exception:
            pass

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{mlflow_db_path}")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = mlflow.tracking.MlflowClient(tracking_uri)
        exp_name = "heart_disease_classification"
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            if tracking_uri.startswith("sqlite"):
                artifacts_dir = os.path.join(MLRUNS_DIR, "artifacts")
                os.makedirs(artifacts_dir, exist_ok=True)
                client.create_experiment(exp_name, artifact_location=artifacts_dir)
            else:
                client.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
    except Exception as e:
        print(f"[WARN] Remote MLflow setup note: {e}. Falling back to default experiment.")
        mlflow.set_experiment("heart_disease_classification")

    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    
    # ── Step 4: Train models ──
    print("\n[STEP 4] Training models...")
    
    results = {}
    
    # ── Model 1: Logistic Regression ──
    lr_params = {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "penalty": "l2",
        "random_state": 42,
    }
    lr_model = LogisticRegression(**lr_params)
    results["Logistic Regression"] = train_and_log_model(
        lr_model, "Logistic Regression",
        X_train, X_test, y_train, y_test,
        lr_params, feature_names
    )
    
    # ── Model 2: Random Forest ──
    rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    rf_model = RandomForestClassifier(**rf_params)
    results["Random Forest"] = train_and_log_model(
        rf_model, "Random Forest",
        X_train, X_test, y_train, y_test,
        rf_params, feature_names
    )
    
    # ── Step 5: Compare models ──
    print("\n[STEP 5] Comparing models...")
    comparison_path = plot_model_comparison(results)
    print(f"  [SAVED] {comparison_path}")
    
    # ── Step 6: Select best & save (Task 4) ──
    print("\n[STEP 6] Selecting best model and packaging...")
    best_name, best_result = select_best_model(results)
    
    model_path, pipeline_path = save_final_model(
        best_result["model"], preprocessor, best_name
    )
    
    # Log best model selection to MLflow
    with mlflow.start_run(run_name="Best Model Selection"):
        mlflow.set_tag("mlflow.user", "Zeel")
        mlflow.set_tag("user_id", "Zeel")
        mlflow.set_tag("author", "Zeel Patel")
        mlflow.log_param("best_model", best_name)
        for metric, value in best_result["test_metrics"].items():
            mlflow.log_metric(f"best_{metric}", value)
        try:
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(pipeline_path)
            mlflow.log_artifact(comparison_path)
            
            # Log requirements.txt for reproducibility
            req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
            if os.path.exists(req_path):
                mlflow.log_artifact(req_path)
        except Exception as e:
            print(f"  [INFO] Remote artifact upload notice: {e}")
    
    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  Best Model: {best_name}")
    print("\n  Test Metrics:")
    for m, v in best_result["test_metrics"].items():
        print(f"    {m:>12}: {v:.4f}")
    print("\n  Artifacts:")
    print(f"    Model:        {model_path}")
    print(f"    Pipeline:     {pipeline_path}")
    print(f"    MLflow runs:  {MLRUNS_DIR}")
    print(f"    Figures:      {FIGURES_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
