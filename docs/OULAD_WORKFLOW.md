# OULAD Dataset Ingestion and Preprocessing Workflow

This guide provides a complete step-by-step workflow for ingesting and preprocessing the Open University Learning Analytics Dataset (OULAD) in this repository.

## Quick Start

For users who want to get started immediately:

```bash
# 1. Download and preprocess OULAD data
make oulad-setup

# 2. Build the ML dataset
make oulad-build

# 3. Validate the processed data
make oulad-validate
```

## Detailed Workflow

### Step 1: Environment Setup

Ensure you have the required dependencies:

```bash
# Install core dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/oulad/{raw,processed,configs,reports}
```

### Step 2: Data Download

Download the raw OULAD dataset:

```bash
# Automated download (recommended)
python scripts/oulad_download.py

# Manual download alternative:
# 1. Visit https://analyse.kmi.open.ac.uk/open-dataset
# 2. Download the ZIP file to data/oulad/raw/
# 3. Extract CSV files
```

**Expected files after download:**
- `data/oulad/raw/studentInfo.csv` - Student demographics
- `data/oulad/raw/studentVle.csv` - VLE interaction logs  
- `data/oulad/raw/vle.csv` - VLE resource metadata
- `data/oulad/raw/studentRegistration.csv` - Registration and outcomes
- `data/oulad/raw/studentAssessment.csv` - Assessment submissions
- `data/oulad/raw/assessments.csv` - Assessment metadata

### Step 3: Data Preprocessing Configuration

Choose or create a preprocessing configuration:

```bash
# Use default configuration
cp configs/oulad_default.json data/oulad/configs/

# Or create custom configuration
python src/oulad/create_config.py --template default --output data/oulad/configs/my_config.json
```

**Configuration options:**
- Feature engineering parameters
- Temporal window settings
- Missing data handling strategies
- Output format preferences

### Step 4: Build ML Dataset

Run the preprocessing pipeline:

```bash
# Basic preprocessing
python src/oulad/build_dataset.py \
    --raw-dir data/oulad/raw \
    --output data/oulad/processed/oulad_ml.parquet

# With graph construction
python src/oulad/build_dataset.py \
    --raw-dir data/oulad/raw \
    --output data/oulad/processed/oulad_ml.parquet \
    --include-graph \
    --graph-output data/oulad/processed/oulad_graph.pt

# With custom configuration
python src/oulad/build_dataset.py \
    --config data/oulad/configs/my_config.json
```

### Step 5: Data Quality Validation

Validate the processed dataset:

```bash
# Run validation checks
python src/oulad/validate_dataset.py \
    --input data/oulad/processed/oulad_ml.parquet \
    --reports-dir data/oulad/reports

# Generate quality report
python src/oulad/data_quality_report.py \
    --input data/oulad/processed/oulad_ml.parquet \
    --output data/oulad/reports/quality_report.html
```

### Step 6: Exploratory Data Analysis

Generate EDA reports:

```bash
# Basic EDA
python src/oulad/eda.py \
    --input data/oulad/processed/oulad_ml.parquet \
    --output-dir data/oulad/reports/eda

# Fairness-aware EDA
python src/oulad/fairness_eda.py \
    --input data/oulad/processed/oulad_ml.parquet \
    --sensitive-attrs sex,age_band,imd_band \
    --output-dir data/oulad/reports/fairness
```

## Output Structure

After successful preprocessing, your data directory will contain:

```
data/oulad/
├── raw/                          # Original CSV files
│   ├── studentInfo.csv
│   ├── studentVle.csv
│   ├── vle.csv
│   ├── studentRegistration.csv
│   ├── studentAssessment.csv
│   └── assessments.csv
├── processed/                    # Processed ML datasets
│   ├── oulad_ml.parquet         # Main ML dataset
│   ├── oulad_graph.pt           # Student-VLE graph (optional)
│   ├── group_counts.csv         # Fairness group statistics
│   └── metadata.json           # Processing metadata
├── configs/                     # Configuration files
│   └── processing_config.json
└── reports/                     # Quality and EDA reports
    ├── data_quality.html
    ├── eda/
    └── fairness/
```

## Feature Engineering Details

The preprocessing pipeline creates the following feature categories:

### VLE Interaction Features
- **Total engagement**: `vle_total_clicks`, `vle_days_active`
- **Temporal patterns**: `vle_first4_clicks`, `vle_last4_clicks`
- **Intensity metrics**: `vle_mean_clicks`, `vle_max_clicks`
- **Engagement trajectory**: `vle_cumulative_clicks`

### Assessment Features
- **Performance**: `assessment_mean_score`, `assessment_last_score`
- **Behavior**: `assessment_count`, `assessment_ontime_rate`

### Labels and Sensitive Attributes
- **Target variables**: `label_pass`, `label_fail_or_withdraw`
- **Fairness attributes**: `sex`, `age_band`, `highest_education`, `imd_band`
- **Intersection features**: `sex_x_age` for intersectional analysis

## Common Use Cases

### Academic Performance Prediction
```bash
# Focus on early prediction
python src/oulad/build_dataset.py \
    --config configs/early_prediction.json \
    --temporal-cutoff 4  # First 4 weeks only
```

### Fairness Analysis
```bash
# Include all sensitive attributes
python src/oulad/build_dataset.py \
    --include-fairness-features \
    --sensitive-attrs sex,age_band,highest_education,imd_band
```

### Sequence Modeling
```bash
# Include temporal sequence data
python src/oulad/build_dataset.py \
    --include-sequences \
    --sequence-length 35  # Weekly sequences
```

## Troubleshooting

### Common Issues

1. **Download fails**: Check internet connection and try manual download
2. **Missing files**: Verify all 6 CSV files are in `data/oulad/raw/`
3. **Memory issues**: Use chunked processing for large datasets
4. **Feature mismatches**: Check configuration file format

### Data Quality Checks

The pipeline includes automatic validation for:
- Required columns presence
- Data type consistency
- Missing value patterns
- Temporal alignment
- Label distribution
- Sensitive attribute completeness

### Getting Help

- Check logs in the console output for detailed error messages
- Review the data quality report for dataset-specific issues
- Consult `README_OULAD.md` for detailed feature descriptions
- Run validation script for comprehensive checks

## Advanced Configuration

### Custom Feature Engineering

Create custom preprocessing configurations:

```json
{
  "vle_features": {
    "temporal_windows": [4, 8, 12],
    "aggregation_methods": ["sum", "mean", "max"],
    "include_sequences": false
  },
  "assessment_features": {
    "score_thresholds": [40, 50, 60],
    "timing_features": true
  },
  "missing_data": {
    "strategy": "median",
    "drop_threshold": 0.9
  }
}
```

### Batch Processing

For processing multiple configurations:

```bash
# Process multiple scenarios
python scripts/batch_process_oulad.py \
    --config-dir configs/oulad/ \
    --output-dir data/oulad/processed/batch/
```

## Integration with GUIDE Pipeline

The processed OULAD dataset integrates seamlessly with the main GUIDE pipeline:

```bash
# Train models on OULAD data
python src/train.py \
    --data data/oulad/processed/oulad_ml.parquet \
    --model_type random_forest \
    --fairness-aware

# Run fairness evaluation
python src/train_eval.py \
    --dataset data/oulad/processed/oulad_ml.parquet \
    --sensitive-attr sex \
    --postprocess equalized_odds
```