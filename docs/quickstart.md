# Quick Start Guide

Get up and running with the GUIDE student performance analysis pipeline in 5 minutes.

## Prerequisites

- Python 3.8+ 
- Git

## Installation

### Option 1: Standard Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/MustafaHameed/GUIDE.git
cd GUIDE

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
PYTHONHASHSEED=0 pytest
```

### Option 2: Conda Environment (For Reproducibility)

```bash
# Clone the repository
git clone https://github.com/MustafaHameed/GUIDE.git
cd GUIDE

# Create conda environment (CPU version)
conda env create -f envs/environment-cpu.yml
conda activate guide-cpu

# Or GPU version if you have CUDA
conda env create -f envs/environment-gpu.yml
conda activate guide-gpu
```

### Option 3: Exact Paper Reproduction

```bash
# Clone the repository
git clone https://github.com/MustafaHameed/GUIDE.git
cd GUIDE

# Install exact versions used in paper
pip install -r requirements-paper.txt

# Verify reproducibility
PYTHONHASHSEED=0 pytest
```

## Quick Demo

### 5-Minute Pipeline

Run the complete analysis pipeline:

```bash
# Set up environment and run everything
make all
```

This will:
1. Prepare data and validate schemas
2. Generate exploratory data analysis 
3. Train baseline models
4. Run fairness analysis
5. Generate explainability reports
6. Create publication-ready artifacts

### Individual Components

Run specific analyses:

```bash
# Exploratory data analysis
make eda

# Train models
make train

# Fairness analysis
make fairness

# Explainability
make explain

# Generate paper artifacts
make paper-assets
```

## Key Outputs

After running the pipeline, you'll find:

### Figures (`figures/`)
- `eda_*.png` - Exploratory data analysis plots
- `roc_*.png` - ROC curves for model performance
- `fairness_*.png` - Bias analysis visualizations
- `shap_*.png` - Feature importance explanations

### Tables (`tables/`)
- `classification_summary.csv` - Model performance metrics
- `fairness_*.csv` - Bias metrics across demographic groups
- `feature_importance.csv` - Ranked feature contributions

### Reports (`reports/`)
- `eda_narrative_summary.md` - Automated data insights
- `explainability_report.md` - Model explanation summary
- `fairness_summary.md` - Bias analysis findings

## Interactive Dashboards

Launch interactive web applications:

```bash
# Student-focused dashboard
streamlit run dashboard_student.py

# Teacher/administrator dashboard  
streamlit run dashboard_teacher.py

# General analysis dashboard
streamlit run dashboard.py
```

## Common Tasks

### Predict Student Risk

```bash
# Early risk assessment using only first period grades
python -m src.early_risk --upto_grade 1

# Using first two periods
python -m src.early_risk --upto_grade 2
```

### Fairness Analysis

```bash
# Analyze bias by gender
python src/train_eval.py --sensitive_attr sex

# Analyze bias by school
python src/train_eval.py --sensitive_attr school

# Multiple attributes
python src/train_eval.py --sensitive_attr ses  # socioeconomic status
```

### Model Explanation

```bash
# Explain a trained model
python src/explain/importance.py --model-path models/model.pkl --data-path student-mat.csv
```

### Cross-Dataset Transfer

```bash
# Transfer learning between Portuguese and Math datasets
python -m src.transfer.uci_transfer
```

## One-Shot Paper Reproduction

For complete paper reproduction with versioned artifacts:

```bash
./scripts/reproduce_paper.sh
```

This creates a timestamped directory in `artifacts/` with all publication-ready materials.

## Environment Variables

For maximum reproducibility:

```bash
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
```

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/GUIDE
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Test Failures
```bash
# Run tests with verbose output
PYTHONHASHSEED=0 pytest -v

# Run specific test file
PYTHONHASHSEED=0 pytest tests/test_data.py -v
```

### Dependency Issues
```bash
# Clean install
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Or use conda for better dependency resolution
conda env create -f envs/environment-cpu.yml --force
```

## Next Steps

- **[CLI Reference](cli_guide.md)** - Detailed command documentation
- **[Dashboard Guide](dashboard_guide.md)** - Interactive analysis tutorials  
- **[Data Card](data_card_student_performance.md)** - Complete dataset information
- **[GitHub Issues](https://github.com/MustafaHameed/GUIDE/issues)** - Report bugs or request features

## Performance Tips

- Use `make -j4` for parallel execution of independent targets
- Run on GPU for faster deep learning components (if available)
- Use `--help` flag with any script for detailed options
- Monitor `artifacts/` directory size - clean old versions as needed

Happy analyzing! 🎓📊