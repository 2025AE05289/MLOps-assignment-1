# Screenshot Storage Directory (`docs/screenshots/`)

This directory stores the visual evidence for **Task 9 (Documentation & Reporting)** embedded directly in [`REPORT.md`](../../REPORT.md).

## Screenshot Inventory & Submission Guidelines

| # | Filename | Assignment Task | What to Capture | Source URL / Terminal Command |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `01_github_actions_ci.png` | **Task 5**: CI/CD Automation | GitHub Actions run overview showing all 3 jobs green: `lint` (flake8), `test` (55 unit tests passing), and `train`. | `https://github.com/Mr-ZeelPatel/MLOps-assignment-1/actions` |
| **2** | `02_fastapi_swagger_docs.png` | **Task 6 & 7**: API Serving | Interactive OpenAPI Swagger UI showing `/predict`, `/health`, `/metrics` routes and the `PatientInput` schema. | `http://4.224.167.2/docs` |
| **3** | `03_kubernetes_cluster_status.png` | **Task 7**: AKS Cloud Deployment | Terminal output showing all 6 pods in `1/1 Running` state, the LoadBalancer External IP `4.224.167.2`, and HPA status. | `kubectl get pods,svc,hpa -o wide` |
| **4** | `04_mlflow_tracking_runs.png` | **Task 3**: Experiment Tracking | MLflow UI showing experiment `heart_disease_classification`, runs for `logistic_regression` and `random_forest`, metrics, and author `Zeel`. | `http://4.224.167.2:5000/#/experiments/1` |
| **5** | `05_grafana_monitoring_dashboard.png` | **Task 8**: Monitoring & Logging | Grafana dashboard with 6 panels: Total Predictions, Average Confidence gauge, Class breakdown donut, Request Rate, Latency P95, Status Codes. | `http://4.224.167.2:3000/d/heart-disease-mlops` |

### How to Replace or Add Screenshots:
Simply save or overwrite the `.png` files in this directory with the exact filenames listed above. [`REPORT.md`](../../REPORT.md) will automatically display them upon viewing on GitHub or exporting to PDF.
