.PHONY: setup install generate train monitor dashboard api api-prod load-test test test-fast mlflow-ui run bms-demo clean help

PYTHON = python
PIP    = pip

setup: install
	@echo "Setup complete. Run: make run"

install:
	$(PIP) install -r requirements.txt

generate:
	@echo "Generating reference + production datasets..."
	$(PYTHON) -m src.data.data_generator

train:
	@echo "Training baseline model..."
	$(PYTHON) -m src.models.train_model

monitor:
	@echo "Running monitoring loop across all batches..."
	$(PYTHON) -m src.monitoring.monitoring_loop

dashboard:
	@echo "Launching Dash dashboard at http://localhost:8050"
	$(PYTHON) -m src.dashboard.dashboard

api:
	@echo "Starting FastAPI inference server at http://localhost:8000"
	@echo "  API docs: http://localhost:8000/docs"
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	@echo "Starting FastAPI server (production mode, 4 workers)"
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4

load-test:
	@echo "Running load test against http://localhost:8000"
	$(PYTHON) scripts/load_test.py --url http://localhost:8000 --n 200 --workers 10

run: generate train monitor
	@echo ""
	@echo "Monitoring complete!"
	@echo "  Launch dashboard:   make dashboard"
	@echo "  Launch API server:  make api"

bms-demo:
	@echo "Running BMS-first monitoring demo..."
	$(PYTHON) -m src.monitoring.bms_monitoring_loop --demo --batches 10 --chemistry NMC

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

clean:
	rm -rf data/ models/ mlruns/ reports/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

help:
	@echo ""
	@echo "Battery ML Monitoring Platform --- Available Commands"
	@echo "  make install      Install all dependencies"
	@echo "  make generate     Generate synthetic datasets"
	@echo "  make train        Train baseline XGBoost model"
	@echo "  make monitor      Run monitoring loop (all 15 batches)"
	@echo "  make dashboard    Launch Plotly Dash dashboard    (port 8050)"
	@echo "  make bms-demo     Run the battery-first monitoring demo"
	@echo "  make api          Launch FastAPI inference server  (port 8000)"
	@echo "  make api-prod     FastAPI production mode (4 workers)"
	@echo "  make load-test    Benchmark API latency (start server first)"
	@echo "  make mlflow-ui    Open MLflow experiment tracker  (port 5000)"
	@echo "  make run          Full pipeline: generate + train + monitor"
	@echo "  make test         Run full test suite with coverage"
	@echo "  make clean        Remove all generated data/models"
	@echo ""
