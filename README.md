# GUIDE: Student Performance Dataset Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/MustafaHameed/GUIDE/workflows/CI/badge.svg)](https://github.com/MustafaHameed/GUIDE/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://zenodo.org/)

**Publication-grade machine learning pipeline for student performance prediction with comprehensive fairness analysis and explainability.**

## Quick Links

📚 **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 5 minutes  
🔧 **[CLI Reference](docs/cli_guide.md)** - Complete command-line interface documentation  
📊 **[Dashboard Guide](docs/dashboard_guide.md)** - Interactive visualization and analysis  
📋 **[Data Card](docs/data_card_student_performance.md)** - Comprehensive dataset documentation  

## Overview

This repository contains the `student-mat.csv` dataset with 395 records of Portuguese secondary school students and 33 features. The dataset includes demographic information, study habits and the final grade columns `G1`, `G2` and `G3`.

### Key Features
- 🔄 **Reproducible Pipeline**: Deterministic results with versioned artifacts
- ⚖️ **Fairness Analysis**: Comprehensive bias detection and mitigation
- 🔍 **Explainable AI**: SHAP, LIME, and counterfactual explanations  
- 📊 **Interactive Dashboards**: Student and teacher-focused views
- 🧪 **Advanced Modeling**: Early risk assessment, transfer learning, nested CV
- 📖 **Publication Ready**: Complete documentation and artifact generation

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

For stricter reproducibility, you may use a tool like `pip-tools` or provide a `conda` environment file to pin exact versions.
## Dataset Description

Each row in the CSV describes one student with the following columns:

- `school`: student's school (`GP` or `MS`)
- `sex`: student's sex (`F` or `M`)
- `age`: age in years (15–22)
- `address`: home address type (`U` for urban, `R` for rural)
- `famsize`: family size (`LE3` ≤ 3 or `GT3` > 3)
- `Pstatus`: parents' cohabitation status (`T` together or `A` apart)
- `Medu`: mother's education level (0 none to 4 higher education)
- `Fedu`: father's education level (0 none to 4 higher education)
- `Mjob`: mother's job (`teacher`, `health`, `services`, `at_home`, `other`)
- `Fjob`: father's job (`teacher`, `health`, `services`, `at_home`, `other`)
- `reason`: reason to choose this school (`home`, `reputation`, `course`, `other`)
- `guardian`: student's guardian (`mother`, `father`, `other`)
- `traveltime`: home to school travel time (1 <15 min, 2 15–30 min, 3 30–60 min, 4 >1 h)
- `studytime`: weekly study time (1 <2 h, 2 2–5 h, 3 5–10 h, 4 >10 h)
- `failures`: number of past class failures (n if 1≤n<3, else 4)
- `schoolsup`: extra educational support (`yes` or `no`)
- `famsup`: family educational support (`yes` or `no`)
- `paid`: extra paid classes (`yes` or `no`)
- `activities`: extra-curricular activities (`yes` or `no`)
- `nursery`: attended nursery school (`yes` or `no`)
- `higher`: wants to take higher education (`yes` or `no`)
- `internet`: Internet access at home (`yes` or `no`)
- `romantic`: with a romantic relationship (`yes` or `no`)
- `famrel`: quality of family relationships (1 very bad to 5 excellent)
- `freetime`: free time after school (1 very low to 5 very high)
- `goout`: going out with friends (1 very low to 5 very high)
- `Dalc`: workday alcohol consumption (1 very low to 5 very high)
- `Walc`: weekend alcohol consumption (1 very low to 5 very high)
- `health`: current health status (1 very bad to 5 very good)
- `absences`: number of school absences (0–93)
- `G1`: first period grade (0–20)
- `G2`: second period grade (0–20)
- `G3`: final grade (0–20)

## Key Data Insights

Based on exploratory data analysis of 395 students, here are the main findings:

### Academic Performance
- **Pass Rate**: 67.1% of students achieve passing grades (≥10)
- **Grade Progression**: Strong correlation exists between consecutive periods (G1→G2: 0.85, G2→G3: 0.91)
- **Early Prediction**: First period grades (G1) are highly predictive of final performance (correlation: 0.80)

### Influential Factors
- **Previous Failures**: Students with past failures average 7.3 vs 11.3 for those without
- **School Support**: Students receiving educational support show different grade patterns
- **Study Patterns**: Moderate correlation between study time and final grades (0.10)

### Risk Factors
- Students with >10 absences tend to have lower performance
- Previous academic failures are strong negative predictors
- Certain demographic and family factors show associations with academic outcomes

*For detailed analysis, see `reports/eda_narrative_summary.md` after running the EDA.*

## Usage

Load the data with [pandas](https://pandas.pydata.org/):

```python
import pandas as pd

df = pd.read_csv('student-mat.csv')
print(df.head())
```

### Run Exploratory Data Analysis

Generate comprehensive EDA reports with enhanced categorical analysis:

```bash
python src/eda.py
```

This creates:
- **Publication-quality figures** in `figures/` with standardized naming (`eda_*`)
- **Analysis tables** in `tables/` including categorical correlations and feature importance
- **Narrative summary** in `reports/eda_narrative_summary.md` with key insights

The enhanced EDA includes:
- Cramér's V correlation analysis for categorical variables
- Mutual information feature importance for categorical predictors
- Chi-square tests for categorical associations
- Comprehensive visualizations with consistent styling

### Train a baseline model

The project code now lives in the `src/` package:

- `data.py` for loading and preparing the dataset
- `preprocessing.py` for building the feature-processing pipeline
- `model.py` for configuring classifiers such as logistic regression, random
  forest, and multilayer perceptrons (MLP)
- `train.py` as the script entry point

Run the training script to see a hold‑out evaluation and 5‑fold cross‑validation score (requires `pandas` and `scikit-learn`):

```bash
python -m src.train --task classification
```

The `--task` flag also enables a regression mode:

```bash
python -m src.train --task regression
```

Regression predicts the final grade and reports RMSE, MAE and R² scores. Use
`--model-type` (e.g. `linear`, `random_forest`) to select the regressor.

To compute fairness metrics for specific demographic groups (classification only), supply the column names:

```bash
python -m src.train --task classification --group-cols sex school
```

Each `fairness_<column>.csv` contains the group's positive rate, disparity, and
false/true positive and negative rates (FPR, FNR, TPR, TNR).
Group-specific reports will be written to `reports/` and figures to `figures/`.

### Running tests

```bash
pip install -r requirements.txt
PYTHONHASHSEED=0 pytest
```

Setting `PYTHONHASHSEED` ensures deterministic hashing for reproducible results.

After training, a feature-importance ranking is saved to `reports/feature_importance.csv`
and a corresponding plot to `figures/feature_importance.png`. The script uses
[`shap`](https://shap.readthedocs.io/) if available, otherwise falling back to
permutation importance.
Partial dependence and individual conditional expectation (ICE) plots for the top
features are also written to `figures/` as `pdp_<feature>.png` and
`ice_<feature>.png`.
LIME explanations for up to three misclassified or representative students are
exported to `figures/` as `lime_<index>.html` and `lime_<index>.png`.

### Explain stored models

Run the explainability CLI on a saved model and dataset:

```bash
python src/explain/importance.py --model-path models/model.pkl --data-path data.parquet
```

SHAP summary plots and PDP/ICE curves are written to `figures/`, while
LIME HTML explanations and a markdown report citing SHAP and LIME are
written to `reports/`.
### Early risk assessment

To assess student risk using only early grade information, run:

```bash
python -m src.early_risk --upto_grade 1
```

The script outputs classification metrics, ROC curves, and risk probabilities. It
also computes feature importances via SHAP when available (otherwise permutation
importance), saving a ranked CSV to
`reports/early_feature_importance_G1.csv` and a plot to
`figures/early_feature_importance_G1.png`.

### Sequence models

The training script can also evaluate grade sequences with an RNN or HMM:

```bash
python -m src.train --sequence-model rnn
```

The RNN implementation depends on the optional `torch` and `captum`
packages. Install them separately if you plan to train the sequence model or
compute importance scores.

When using the RNN, the hidden size, number of epochs, and learning rate can be
adjusted via ``--hidden-size``, ``--epochs``, and ``--learning-rate``. The
defaults are 8, 50, and 0.01 respectively.

### Cross-dataset transfer learning

Transfer learning experiments can evaluate how models trained on one dataset
perform on another. The `src/transfer/uci_transfer.py` script supports logistic
regression, random forests, and a small PyTorch-based multilayer perceptron
(MLP) that is pretrained on the source data and fine-tuned on the target.

```bash
python -m src.transfer.uci_transfer --models logistic random_forest mlp
```

The CLI accepts a `--models` list to run a subset of the available models.
### Exploratory data analysis

An exploratory analysis script generates publication-ready figures and summary tables:

```bash
python src/eda.py
```

Outputs are written to `figures/` and `tables/` directories.
### Nested cross-validation and interpretation

The `nested_cv.py` script evaluates regression models using nested cross-
validation, generates baseline and ablation comparisons, and produces
interpretation plots. Supported models include random forest, linear regression,
Lasso, support vector regression, k-nearest neighbors, bagging, gradient
boosting, multilayer perceptrons, stacking and optionally XGBoost:

```bash
python src/nested_cv.py
```

Results are saved to `tables/` and figures to `figures/`. The optional
dependencies [`shap`](https://shap.readthedocs.io/) and

[`statsmodels`](https://www.statsmodels.org/) are used for SHAP feature
importance plots and LOESS smoothing, respectively. If these packages are
missing, the script will skip the associated plots.

### Streamlit dashboard

Interactive dashboards built with Streamlit can display the generated tables and
figures. First make sure the analysis scripts above have produced outputs in the
`tables/` and `figures/` directories. Then launch one of the apps:

```bash
streamlit run dashboard.py          # project overview
streamlit run dashboard_teacher.py  # class-level view for instructors
streamlit run dashboard_student.py  # personalized view for a single student
```
The main `dashboard.py` sidebar lets you switch between exploratory plots,
model performance summaries, and per‑group fairness metrics. The teacher
dashboard focuses on class summaries, at-risk students, and fairness by
subgroup, while the student dashboard reports an individual learner’s predicted
risk and recommended resources.

To secure the overview or teacher dashboards (e.g. on Streamlit Community
Cloud), set a `DASHBOARD_PASSWORD` environment variable. The app will prompt for
the token on startup and only continue when it matches. The student dashboard
supports optional token-based auth via a `STUDENT_TOKENS` environment variable
formatted as comma-separated `id:token` pairs, e.g. `STUDENT_TOKENS="42:alpha,84:beta"`.
Each token grants access to the corresponding student ID’s view.

These basic auth features are optional but recommended for shared deployments.
## File

- `student-mat.csv` – raw dataset sourced from the UCI Machine Learning Repository.

## Data provenance and licenses

- **OULAD** – The Open University Learning Analytics Dataset is distributed under a [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/) [source](https://analyse.kmi.open.ac.uk/open-dataset/).
- **UCI Student Performance** – Data obtained from the UCI Machine Learning Repository and released under a [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode) [source](https://archive.ics.uci.edu/ml/datasets/student+performance).
- **SAM** – Student Action Mining event log schema provided in this repository with no explicit license; confirm licensing with the data owner before use.

## Bibliography

- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. https://jmlr.csail.mit.edu/papers/v11/cawley10a.html
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. https://doi.org/10.1109/TPAMI.2017.2992093
- Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. https://doi.org/10.1109/5.18626
- scikit-learn documentation. https://scikit-learn.org/stable/
- Fairlearn documentation. https://fairlearn.org/
- SHAP documentation. https://shap.readthedocs.io/
- LIME documentation. https://lime-ml.readthedocs.io/
