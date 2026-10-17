"""
Heart Disease Prediction — End-to-End MLOps Pipeline Orchestrator.

One-line entry point to execute the complete workflow:
1. Data Acquisition (download dataset)
2. Exploratory Data Analysis (generate visualizations)
3. Feature Engineering & Model Training (MLflow tracking + model packaging)
4. Automated Testing (pytest unit test suite)

Usage:
    python main.py                  # Run full pipeline
    python main.py --skip-eda       # Skip EDA plot generation
    python main.py --skip-tests     # Skip running pytest
    python main.py --serve          # Start API server after pipeline completion
"""

import sys
import time
import argparse
import subprocess


def print_banner(step_num, title):
    print("\n" + "=" * 65)
    print(f"  [STAGE {step_num}] {title.upper()}")
    print("=" * 65 + "\n")


def run_command(cmd, stage_name):
    start = time.time()
    print(f"[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    duration = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] Stage '{stage_name}' exited with error code {result.returncode} ({duration:.1f}s)")
        sys.exit(result.returncode)

    print(f"[SUCCESS] Stage '{stage_name}' completed in {duration:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End MLOps Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA visualization generation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pytest unit tests")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI server after pipeline finishes")
    parser.add_argument("--mlflow", action="store_true", help="Launch MLflow UI dashboard after pipeline finishes")
    args = parser.parse_args()

    pipeline_start = time.time()
    print("\n" + "█" * 65)
    print("  STARTING END-TO-END MLOps PIPELINE")
    print("█" * 65)

    step = 1

    # Stage 1: Data Acquisition
    print_banner(step, "Data Acquisition (Download Dataset)")
    run_command([sys.executable, "data/download_data.py"], "Data Download")
    step += 1

    # Stage 2: Exploratory Data Analysis (optional)
    if not args.skip_eda:
        print_banner(step, "Exploratory Data Analysis (Visualizations)")
        run_command([sys.executable, "src/eda.py"], "EDA Generation")
        step += 1
    else:
        print("[INFO] Skipping EDA generation (--skip-eda)")

    # Stage 3: Feature Engineering, Model Training & Packaging
    print_banner(step, "Feature Engineering, Model Training & MLflow Tracking")
    run_command([sys.executable, "src/train.py"], "Model Training & Packaging")
    step += 1

    # Stage 4: Automated Testing (optional)
    if not args.skip_tests:
        print_banner(step, "Automated Testing (Pytest Suite & Coverage)")
        run_command(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--cov=src", "--cov-report=term-missing"],
            "Unit Tests",
        )
        step += 1
    else:
        print("[INFO] Skipping unit tests (--skip-tests)")

    total_time = time.time() - pipeline_start
    print("█" * 65)
    print(f"  END-TO-END PIPELINE COMPLETED SUCCESSFULLY ({total_time:.1f}s)")
    print("█" * 65 + "\n")
    print("Summary of outputs generated:")
    print("  • Raw dataset:          data/raw/heart_disease.csv")
    print("  • Clean dataset:        data/processed/heart_disease_clean.csv")
    print("  • Best model:           output/models/final_model.joblib")
    print("  • Preprocessor:         output/models/preprocessing_pipeline.joblib")
    print("  • Metadata:             output/models/model_metadata.json")
    print("  • MLflow runs:          output/mlruns/mlflow.db")
    print("  • Evaluation plots:     output/figures/")
    print()

    # Optional launches
    if args.serve:
        print("[INFO] Launching FastAPI prediction server at http://127.0.0.1:8000...")
        subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"])

    if args.mlflow:
        print("[INFO] Launching MLflow UI at http://127.0.0.1:5001...")
        subprocess.run([
            "mlflow", "ui",
            "--backend-store-uri", "sqlite:///output/mlruns/mlflow.db",
            "--port", "5001",
        ])


if __name__ == "__main__":
    main()
