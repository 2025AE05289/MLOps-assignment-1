.PHONY: help pipeline data eda train test lint serve mlflow docker-build docker-run monitoring monitoring-down k8s-apply k8s-status k8s-delete clean

help:
	@echo "Available commands:"
	@echo "  make pipeline       - Run complete end-to-end pipeline (data -> eda -> train -> test)"
	@echo "  make data           - Download raw dataset"
	@echo "  make eda            - Generate exploratory data analysis plots"
	@echo "  make train          - Train models, log to MLflow, and package champion model"
	@echo "  make test           - Run unit tests with code coverage"
	@echo "  make lint           - Check PEP8 formatting with flake8"
	@echo "  make serve          - Start FastAPI prediction server locally on port 8000"
	@echo "  make mlflow         - Launch MLflow UI dashboard on port 5001"
	@echo "  make docker-build   - Build Docker container image"
	@echo "  make docker-run     - Run Docker container on port 8000"
	@echo "  make test-api       - Run sample prediction tests against API (scripts/test_api.sh)"
	@echo "  make monitoring     - Start Prometheus (9090) and Grafana (3000) stack"
	@echo "  make monitoring-down- Stop Prometheus and Grafana stack"
	@echo "  make k8s-apply      - Deploy to Kubernetes (apply deployment and service)"
	@echo "  make k8s-status     - Check Kubernetes pod and service status"
	@echo "  make k8s-delete     - Delete Kubernetes resources"

pipeline:
	python main.py

test-api:
	bash scripts/test_api.sh http://localhost:8000

data:
	python data/download_data.py

eda:
	python src/eda.py

train:
	python src/train.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ app.py tests/ --max-line-length=120 --statistics --count --ignore=E501,W503,E203,W293,E402

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --backend-store-uri sqlite:///output/mlruns/mlflow.db --port 5001

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run -d --name heart-api -p 8000:8000 heart-disease-api:latest

monitoring:
	docker compose -f monitoring/docker-compose.yml up -d

monitoring-down:
	docker compose -f monitoring/docker-compose.yml down

k8s-apply:
	kubectl apply -f k8s/

k8s-status:
	kubectl get pods,svc,hpa -o wide

k8s-delete:
	kubectl delete -f k8s/
