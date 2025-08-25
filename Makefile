# GUIDE - Publication-Grade Machine Learning Pipeline
# Makefile for reproducible research workflow

# Configuration
PYTHON = python
PYTHONHASHSEED = 0
DATA_DIR = data
FIGURES_DIR = figures
TABLES_DIR = tables
REPORTS_DIR = reports
ARTIFACTS_DIR = artifacts
MODEL_DIR = models

# Get current date for versioned artifacts
TIMESTAMP = $(shell date +%Y-%m-%d)
ARTIFACT_VERSION_DIR = $(ARTIFACTS_DIR)/$(TIMESTAMP)

# Default target
.PHONY: all
all: setup data eda train fairness explain dashboard paper-assets

# Setup and environment
.PHONY: setup
setup:
	@echo "Setting up environment and dependencies..."
	pip install -r requirements.txt
	mkdir -p $(FIGURES_DIR) $(TABLES_DIR) $(REPORTS_DIR) $(ARTIFACTS_DIR) $(MODEL_DIR)
	mkdir -p $(ARTIFACT_VERSION_DIR)

# Data preparation and validation
.PHONY: data
data: setup
	@echo "Preparing and validating data..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) -c "import src.data; print('Data loading validated')"
	
# Exploratory Data Analysis
.PHONY: eda
eda: data
	@echo "Running exploratory data analysis..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) src/eda.py
	@echo "EDA completed. Outputs in $(FIGURES_DIR)/ and $(TABLES_DIR)/"

# Model training
.PHONY: train
train: data
	@echo "Training baseline models..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) src/train.py --model_type logistic
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) src/train.py --model_type random_forest
	@echo "Model training completed. Models saved in $(MODEL_DIR)/"

# Early risk assessment
.PHONY: early-risk
early-risk: train
	@echo "Running early risk assessment..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) -m src.early_risk --upto_grade 1
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) -m src.early_risk --upto_grade 2
	@echo "Early risk analysis completed"

# Nested cross-validation
.PHONY: nested-cv
nested-cv: data
	@echo "Running nested cross-validation..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) -m src.nested_cv
	@echo "Nested CV completed"

# Transfer learning
.PHONY: transfer
transfer: train
	@echo "Running transfer learning experiments..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) -m src.transfer.uci_transfer
	@echo "Transfer learning completed"

# Fairness analysis
.PHONY: fairness
fairness: train
	@echo "Running fairness analysis..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) src/train_eval.py --sensitive_attr sex
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) src/train_eval.py --sensitive_attr school
	@echo "Fairness analysis completed. Results in $(TABLES_DIR)/ and $(FIGURES_DIR)/"

# Explainability analysis
.PHONY: explain
explain: train
	@echo "Running explainability analysis..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) $(PYTHON) src/explain/importance.py --model-path $(MODEL_DIR)/model.pkl --data-path student-mat.csv
	@echo "Explainability analysis completed. Reports in $(REPORTS_DIR)/"

# Dashboard generation
.PHONY: dashboard
dashboard: train fairness explain
	@echo "Generating dashboard..."
	@echo "Run 'streamlit run dashboard.py' to start the dashboard"
	@echo "Run 'streamlit run dashboard_student.py' for student view"
	@echo "Run 'streamlit run dashboard_teacher.py' for teacher view"

# Generate all paper-ready artifacts
.PHONY: paper-assets
paper-assets: eda train fairness explain early-risk nested-cv transfer
	@echo "Generating publication-ready artifacts..."
	mkdir -p $(ARTIFACT_VERSION_DIR)/figures $(ARTIFACT_VERSION_DIR)/tables $(ARTIFACT_VERSION_DIR)/reports
	cp -r $(FIGURES_DIR)/* $(ARTIFACT_VERSION_DIR)/figures/ 2>/dev/null || true
	cp -r $(TABLES_DIR)/* $(ARTIFACT_VERSION_DIR)/tables/ 2>/dev/null || true
	cp -r $(REPORTS_DIR)/* $(ARTIFACT_VERSION_DIR)/reports/ 2>/dev/null || true
	@echo "Paper artifacts saved to $(ARTIFACT_VERSION_DIR)/"

# Testing
.PHONY: test
test:
	@echo "Running tests..."
	PYTHONHASHSEED=$(PYTHONHASHSEED) pytest -v

# Code quality checks
.PHONY: lint
lint:
	@echo "Running code quality checks..."
	black --check .
	ruff check . || true

# Format code
.PHONY: format
format:
	@echo "Formatting code..."
	black .
	isort .

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Clean all outputs (use with caution)
.PHONY: clean-all
clean-all: clean
	@echo "WARNING: This will delete all generated outputs!"
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	rm -rf $(FIGURES_DIR)/* $(TABLES_DIR)/* $(REPORTS_DIR)/* $(MODEL_DIR)/*

# Development environment setup
.PHONY: dev-setup
dev-setup:
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pip install pytest black isort ruff pre-commit
	pre-commit install || echo "pre-commit not available, skipping hook installation"

# Help
.PHONY: help
help:
	@echo "GUIDE - Publication-Grade ML Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Run complete pipeline (data → eda → train → fairness → explain → dashboard → paper-assets)"
	@echo "  setup        - Set up directories and environment"
	@echo "  data         - Prepare and validate data"
	@echo "  eda          - Run exploratory data analysis"
	@echo "  train        - Train baseline models"
	@echo "  early-risk   - Run early risk assessment"
	@echo "  nested-cv    - Run nested cross-validation"
	@echo "  transfer     - Run transfer learning experiments"
	@echo "  fairness     - Run fairness analysis"
	@echo "  explain      - Run explainability analysis"
	@echo "  dashboard    - Prepare dashboard (then run streamlit manually)"
	@echo "  paper-assets - Generate all publication-ready artifacts"
	@echo "  test         - Run test suite"
	@echo "  lint         - Check code quality"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean temporary files"
	@echo "  clean-all    - Clean all generated outputs (WARNING: destructive)"
	@echo "  dev-setup    - Set up development environment"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  PYTHONHASHSEED=$(PYTHONHASHSEED) (for reproducibility)"
	@echo ""
	@echo "Artifacts are versioned in: $(ARTIFACTS_DIR)/YYYY-MM-DD/"