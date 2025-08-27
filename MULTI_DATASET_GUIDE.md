# Multi-Dataset Integration Guide

## Overview

The GUIDE project now supports three educational datasets for comprehensive transfer learning research:

1. **OULAD** (Open University Learning Analytics Dataset)
2. **XuetangX** (MOOC Dataset)  
3. **UCI** (Student Performance Dataset)

## Dataset Status

| Dataset | Status | Samples | Features | Pass Rate | Source |
|---------|--------|---------|----------|-----------|---------|
| OULAD | ✅ Ready (Synthetic*) | 5,000 | 23 | 40.2% | Synthetic based on real schema |
| XuetangX | ✅ Ready (Synthetic) | 1,000 | 29 | 52.7% | Synthetic MOOC data |
| UCI | ✅ Ready (Real) | 395 | 34 | 67.1% | Real student performance data |

*Real OULAD data can be integrated using the preprocessing scripts provided.

## Quick Start

### 1. Load Individual Datasets

```python
from scripts.enhanced_data_loader import load_dataset

# Load individual datasets
oulad_df = load_dataset('oulad')
xuetangx_df = load_dataset('xuetangx') 
uci_df = load_dataset('uci')

print(f"OULAD: {oulad_df.shape}")
print(f"XuetangX: {xuetangx_df.shape}")
print(f"UCI: {uci_df.shape}")
```

### 2. Load All Datasets

```python
from scripts.enhanced_data_loader import load_all_datasets

datasets = load_all_datasets()
for name, df in datasets.items():
    print(f"{name.upper()}: {df.shape}, Pass rate: {df['label_pass'].mean():.1%}")
```

### 3. Run Transfer Learning Demo

```python
# Run comprehensive transfer learning between all datasets
python demo_multi_dataset_transfer.py
```

## Dataset Setup

### Download and Setup All Datasets

```bash
# Download/setup all datasets
python scripts/download_datasets.py --dataset all

# Process XuetangX (creates synthetic dataset)
python scripts/preprocess_xuetangx.py --synthetic

# Verify all datasets
python scripts/enhanced_data_loader.py
```

### Individual Dataset Setup

#### OULAD Dataset
```bash
# Download (creates instructions for manual download)
python scripts/download_datasets.py --dataset oulad

# For real data: manually download files to data/oulad/raw/ then:
python scripts/preprocess_oulad.py

# Currently using synthetic data that matches real schema
```

#### XuetangX Dataset
```bash
# Create synthetic MOOC dataset
python scripts/preprocess_xuetangx.py --synthetic

# For real data: place CSV files in data/xuetangx/raw/ then:
python scripts/preprocess_xuetangx.py
```

#### UCI Dataset
```bash
# Download UCI student performance data
python scripts/download_datasets.py --dataset uci

# Already available in repository root
```

## Transfer Learning Results

### Performance Summary

From multi-dataset transfer learning experiments:

- **Total Experiments**: 12 scenarios (3 datasets × 2 directions × 2 models)
- **Successful**: 8/12 experiments (67% success rate)
- **Average Performance**: 45.9% accuracy, 29.4% F1-score
- **Best Transfer**: XuetangX → UCI (57.1% accuracy, 67.9% F1-score)

### Transfer Scenarios

| Source | Target | Best Model | Accuracy | F1-Score | ROC-AUC |
|--------|--------|------------|----------|----------|---------|
| OULAD | XuetangX | Random Forest | 50.3% | 18.6% | 52.3% |
| OULAD | UCI | Random Forest | 50.4% | 53.5% | 58.0% |
| XuetangX | OULAD | Random Forest | 49.9% | 45.1% | 49.6% |
| XuetangX | UCI | Random Forest | **57.1%** | **67.9%** | 51.4% |
| UCI | OULAD | Error* | - | - | - |
| UCI | XuetangX | Error* | - | - | - |

*Some transfers fail due to categorical data encoding issues - minor preprocessing fixes needed.

## Dataset Features

### Common Features Across Datasets

All datasets include these standardized features for transfer learning:

- `gender` - Student gender (M/F)
- `age_normalized` - Age or age band converted to numeric
- `education` - Educational background level
- `engagement_score` - Learning activity/engagement metric
- `prior_performance` - Previous academic performance indicator
- `gender_x_age` - Interaction feature
- `label_pass` - Binary pass/fail target

### Dataset-Specific Features

#### OULAD Features
- VLE engagement metrics (clicks, days active)
- Assessment performance (scores, submission timing)
- Demographic information (IMD band, education level)
- Course information (module, presentation)

#### XuetangX Features  
- Video watching behavior (time, completion rate, pauses)
- Assignment performance (submissions, scores)
- Forum engagement (posts, replies, views)
- Course completion metrics

#### UCI Features
- Student demographics (age, sex, family)
- Academic background (parent education, previous grades)
- Study habits (study time, failures, absences)
- Final grades (G1, G2, G3)

## File Structure

```
data/
├── oulad/
│   ├── raw/DOWNLOAD_INSTRUCTIONS.txt
│   └── processed/oulad_ml.csv
├── xuetangx/
│   ├── raw/DOWNLOAD_INSTRUCTIONS.txt
│   └── processed/xuetangx_ml.csv
└── uci/
    └── raw/student-mat.csv

scripts/
├── download_datasets.py          # Download all datasets
├── preprocess_oulad.py           # Process real OULAD data
├── preprocess_xuetangx.py        # Process/create XuetangX data
└── enhanced_data_loader.py       # Unified data loading

demo_multi_dataset_transfer.py    # Transfer learning demo
```

## Integration with Existing Code

### Update Existing Transfer Learning Scripts

The enhanced data loader is backward compatible. Update existing scripts:

```python
# OLD: Individual loading functions
from enhanced_transfer_learning_quickwins import load_oulad_data, load_uci_data

# NEW: Unified loading
from scripts.enhanced_data_loader import load_dataset

oulad_df = load_dataset('oulad')
uci_df = load_dataset('uci')
xuetangx_df = load_dataset('xuetangx')  # New dataset!
```

### Use in Transfer Learning Pipeline

```python
from scripts.enhanced_data_loader import load_all_datasets

# Load all datasets
datasets = load_all_datasets()

# Run transfer learning between any pair
for source_name in datasets:
    for target_name in datasets:
        if source_name != target_name:
            # Your transfer learning code here
            print(f"Transfer: {source_name} → {target_name}")
```

## Real Data Integration

### OULAD Real Data

To use real OULAD data instead of synthetic:

1. Register at: https://analyse.kmi.open.ac.uk/open_dataset
2. Download CSV files to `data/oulad/raw/`
3. Run: `python scripts/preprocess_oulad.py`

### XuetangX Real Data

To integrate real XuetangX data:

1. Obtain dataset files (contact authors or check academic repositories)
2. Place CSV files in `data/xuetangx/raw/`
3. Update `preprocess_xuetangx.py` based on actual schema
4. Run: `python scripts/preprocess_xuetangx.py`

## Future Extensions

### Immediate Improvements
- Fix categorical encoding issues in UCI→other transfers
- Add more sophisticated feature alignment methods
- Implement domain adaptation techniques

### Research Opportunities
- Cross-cultural transfer learning (OULAD UK ↔ XuetangX China ↔ UCI Portugal)
- Temporal transfer learning across academic terms
- Multi-task transfer learning (dropout prediction, grade prediction, etc.)

### Additional Datasets
- EdNet (Korean education data)
- Coursera/edX MOOC datasets
- Khan Academy learning analytics

## Support

For issues or questions:
1. Check dataset loading: `python scripts/enhanced_data_loader.py`
2. Verify transfer learning: `python demo_multi_dataset_transfer.py`
3. Review download instructions in `data/*/raw/DOWNLOAD_INSTRUCTIONS.txt`

The multi-dataset integration provides a robust foundation for educational transfer learning research across diverse learning contexts and cultural settings.