# Command Line Interface Guide

Complete reference for all CLI commands and options in the GUIDE pipeline.

## Overview

The GUIDE pipeline provides several command-line interfaces:

- **Makefile targets** - High-level pipeline orchestration
- **Python modules** - Detailed analysis components  
- **Reproduction scripts** - One-shot paper reproduction

## Makefile Targets

The primary interface for running analyses. All targets respect the `PYTHONHASHSEED=0` environment variable for reproducibility.

### Pipeline Targets

```bash
# Complete pipeline
make all                 # Run data → eda → train → fairness → explain → paper-assets

# Individual stages  
make setup              # Create directories and install dependencies
make data               # Validate data and check schemas
make eda                # Exploratory data analysis
make train              # Train baseline models
make early-risk         # Early warning system analysis
make nested-cv          # Nested cross-validation
make transfer           # Transfer learning experiments
make fairness           # Bias analysis and mitigation
make explain            # Model explainability analysis
make dashboard          # Prepare dashboard components
make paper-assets       # Generate publication-ready artifacts
```

### Development Targets

```bash
# Code quality
make lint               # Check code style with black and ruff
make format             # Auto-format code with black and isort
make test               # Run test suite

# Environment management
make dev-setup          # Set up development environment
make clean              # Clean temporary files
make clean-all          # Clean all outputs (WARNING: destructive)

# Help
make help               # Show all available targets
```

### Example Workflows

```bash
# Quick development cycle
make format && make test && make eda

# Full analysis for paper
make clean && make all

# Debug specific component
make setup && make data && make train
```

## Python Modules

### Training and Evaluation

#### Basic Training
```bash
# Train logistic regression (default)
python src/train.py

# Train specific model types
python src/train.py --model_type random_forest
python src/train.py --model_type logistic
python src/train.py --model_type stacking
python src/train.py --model_type bagging

# Classification vs regression
python src/train.py --task classification
python src/train.py --task regression

# Custom pass threshold
python src/train.py --pass_threshold 12  # Default is 10
```

#### Advanced Training Options
```bash
# Ensemble methods
python src/train.py --model_type stacking --estimators logistic random_forest
python src/train.py --model_type bagging --base_estimator decision_tree

# Sequence models
python src/train.py --sequence_model lstm --hidden_size 64 --epochs 100
python src/train.py --sequence_model gru --learning_rate 0.001

# Fairness mitigation during training
python src/train.py --mitigation reweighing
python src/train.py --mitigation threshold_opt
```

#### Hyperparameter Search
```bash
# Grid search
python src/train.py --param_grid default
python src/train.py --param_grid extensive

# No hyperparameter search
python src/train.py --param_grid none
```

### Fairness Analysis

#### Comprehensive Fairness Evaluation
```bash
# Train with fairness analysis
python src/train_eval.py --sensitive_attr sex
python src/train_eval.py --sensitive_attr school
python src/train_eval.py --sensitive_attr age_group

# Multiple models with fairness
python src/train_eval.py --model logistic --sensitive_attr sex
python src/train_eval.py --model random_forest --sensitive_attr school

# Custom dataset splits
python src/train_eval.py --dataset data/custom.parquet --split-file data/splits.json
```

#### Bias Mitigation
```bash
# Preprocessing mitigation
python src/train_eval.py --sensitive_attr sex --mitigation reweighing

# Postprocessing mitigation  
python src/train_eval.py --sensitive_attr sex --mitigation threshold_opt

# Compare before/after mitigation
python src/train_eval.py --sensitive_attr sex --mitigation threshold_opt --save_comparison
```

### Early Risk Assessment

```bash
# Risk assessment using partial grade information
python -m src.early_risk --upto_grade 1    # First period only
python -m src.early_risk --upto_grade 2    # First two periods

# Custom thresholds and models
python -m src.early_risk --upto_grade 1 --model random_forest
python -m src.early_risk --upto_grade 1 --threshold 0.3

# Generate risk reports
python -m src.early_risk --upto_grade 1 --save_predictions
```

### Model Explainability

#### Feature Importance and Explanations
```bash
# Basic explainability analysis
python src/explain/importance.py --model-path models/model.pkl --data-path student-mat.csv

# Custom output directories
python src/explain/importance.py \
    --model-path models/model.pkl \
    --data-path student-mat.csv \
    --figures-dir custom_figures \
    --reports-dir custom_reports

# Specific sensitive attribute analysis
python src/explain/importance.py \
    --model-path models/model.pkl \
    --data-path student-mat.csv \
    --sensitive-attr sex
```

#### Advanced Explanation Options
```bash
# Include local explanations for specific instances
python src/explain/importance.py \
    --model-path models/model.pkl \
    --data-path student-mat.csv \
    --n-samples 100  # Number of background samples for SHAP

# Generate stability analysis
python src/explain/importance.py \
    --model-path models/model.pkl \
    --data-path student-mat.csv \
    --stability-analysis
```

### Cross-Validation and Model Selection

#### Nested Cross-Validation
```bash
# Basic nested CV
python -m src.nested_cv

# Custom CV configuration
python -m src.nested_cv --outer_cv 5 --inner_cv 3
python -m src.nested_cv --n_jobs 4  # Parallel processing

# Specific models only
python -m src.nested_cv --models logistic random_forest
```

#### Model Comparison
```bash
# Compare multiple models with statistical tests
python -m src.nested_cv --statistical_tests
python -m src.nested_cv --save_predictions  # For McNemar tests
```

### Transfer Learning

```bash
# Cross-dataset transfer
python -m src.transfer.uci_transfer

# Bidirectional transfer analysis
python -m src.transfer.uci_transfer --bidirectional

# Domain adaptation experiments
python -m src.transfer.uci_transfer --domain_adapt
```

### Data Processing and EDA

#### Exploratory Data Analysis
```bash
# Basic EDA with all default plots
python src/eda.py

# Custom output directory
python src/eda.py --output_dir custom_eda

# Focus on specific features
python src/eda.py --features G1 G2 G3 sex school
```

#### OULAD Dataset Processing
```bash
# Build OULAD dataset
python -m src.oulad.build_dataset \
    --raw_dir data/oulad/raw \
    --output data/oulad/processed.parquet

# Create stratified splits
python -m src.oulad.splits \
    --dataset data/oulad/processed.parquet \
    --output data/oulad/splits.json

# Train regression models on OULAD
python src/oulad/train_regression.py \
    --dataset data/oulad/processed.parquet \
    --splits data/oulad/splits.json
```

## Environment Variables

### Reproducibility
```bash
export PYTHONHASHSEED=0              # Deterministic hashing
export PYTHONDONTWRITEBYTECODE=1     # Disable .pyc files
```

### Performance
```bash
export OMP_NUM_THREADS=4             # OpenMP parallelism
export MKL_NUM_THREADS=4             # Intel MKL threads
export NUMBA_NUM_THREADS=4           # Numba parallelism
```

### Paths and Configuration
```bash
export FIGURES_DIR=custom_figures    # Override default figure directory
export TABLES_DIR=custom_tables      # Override default table directory
export REPORTS_DIR=custom_reports    # Override default reports directory
```

## Output Directory Structure

All commands produce outputs in standardized locations:

```
├── figures/          # Publication-ready plots
│   ├── eda_*.png
│   ├── roc_*.png
│   ├── fairness_*.png
│   └── shap_*.png
├── tables/           # Data tables and metrics
│   ├── classification_summary.csv
│   ├── fairness_*.csv
│   └── feature_importance.csv
├── reports/          # Narrative reports
│   ├── eda_narrative_summary.md
│   ├── explainability_report.md
│   └── fairness_summary.md
├── models/           # Trained model artifacts
│   ├── model.pkl
│   └── pipeline.pkl
└── artifacts/        # Versioned complete outputs
    └── YYYY-MM-DD/   # Timestamped collections
```

## Common Patterns

### Batch Processing
```bash
# Multiple sensitive attributes
for attr in sex school age_group; do
    python src/train_eval.py --sensitive_attr $attr
done

# Multiple models
for model in logistic random_forest; do
    python src/train.py --model_type $model
done
```

### Pipeline Combinations
```bash
# Custom analysis pipeline
make data
python src/train.py --model_type random_forest
python src/train_eval.py --sensitive_attr sex
python src/explain/importance.py --model-path models/model.pkl --data-path student-mat.csv
make paper-assets
```

### Debugging and Development
```bash
# Verbose output
python src/train.py --verbose
python -m src.nested_cv --debug

# Dry run (check parameters without execution)
python src/train.py --dry_run

# Small subset for testing
python src/train.py --n_samples 100
```

## Error Handling

### Common Issues and Solutions

#### Import Errors
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check module installation
python -c "import src.train; print('OK')"
```

#### Memory Issues
```bash
# Reduce batch size or samples
python src/train.py --batch_size 16
python -m src.nested_cv --n_jobs 1

# Monitor memory usage
htop  # or top
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Check directory permissions
ls -la figures/ tables/ reports/
```

## Performance Optimization

### Parallel Processing
```bash
# Use multiple cores
make -j4                    # Parallel make targets
python -m src.nested_cv --n_jobs -1  # All available cores

# GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0
python src/train.py --device cuda
```

### Memory Management
```bash
# Reduce memory footprint
python src/train.py --low_memory
python -m src.nested_cv --batch_size 32

# Monitor resource usage
time make train             # Time execution
/usr/bin/time -v make train # Detailed resource usage
```

## Help and Documentation

Every script provides help information:

```bash
python src/train.py --help
python src/explain/importance.py --help
python -m src.early_risk --help
make help
```

For more detailed information, see:
- **[Quick Start Guide](quickstart.md)** - Basic usage
- **[Dashboard Guide](dashboard_guide.md)** - Interactive interfaces
- **[Data Card](data_card_student_performance.md)** - Dataset documentation