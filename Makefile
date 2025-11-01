# Makefile for NeuroDrive DMS project
# Simplifies common development and deployment tasks

.PHONY: help install test train export-model build up down logs clean lint format

# Variables
PYTHON := python3
PIP := pip
DOCKER_COMPOSE := docker compose
VENV := .venv

help:
	@echo "NeuroDrive DMS - Available commands:"
	@echo ""
	@echo "  make install          - Install Python dependencies"
	@echo "  make train            - Train ML model and export to ONNX"
	@echo "  make export-model     - Export PyTorch model to ONNX"
	@echo "  make prepare-data     - Generate synthetic training data"
	@echo "  make test             - Run all tests"
	@echo "  make test-backend     - Run backend tests"
	@echo "  make lint             - Run linting (flake8)"
	@echo "  make format           - Format code (black)"
	@echo "  make build            - Build Docker images"
	@echo "  make up               - Start all services with docker-compose"
	@echo "  make down             - Stop all services"
	@echo "  make logs             - View logs from docker-compose"
	@echo "  make clean            - Clean generated files and caches"
	@echo "  make dev-backend      - Run backend in development mode"
	@echo "  make dev-frontend     - Run frontend in development mode"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cd frontend && npm install

prepare-data:
	$(PYTHON) scripts/prepare_sample_data.py --num-telemetry 10000 --num-keypoints 1000 --output-dir data

train:
	@echo "Training ML model..."
	$(PYTHON) scripts/train_model.py --train-data data/train_data.csv --test-data data/test_data.csv \
		--output-dir backend/models --epochs 50 --export-onnx

export-model:
	$(PYTHON) scripts/export_model.py --model-path backend/models/risk_model.pth \
		--output backend/models/risk_model.onnx

test:
	$(PYTHON) -m pytest tests/ -v --cov=backend --cov-report=html --cov-report=term

test-backend:
	$(PYTHON) -m pytest tests/backend/ -v

lint:
	flake8 backend/ scripts/ --max-line-length=120 --ignore=E203,W503
	cd frontend && npm run lint

format:
	black backend/ scripts/ --line-length=120
	cd frontend && npm run format

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
	rm -rf backend/models/*.pth backend/models/*.onnx data/*.csv data/*.json 2>/dev/null || true

dev-backend:
	export DEV_MODE=true && \
	export NEURODRIVE_CONFIG=$$(pwd)/config.yaml && \
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev


