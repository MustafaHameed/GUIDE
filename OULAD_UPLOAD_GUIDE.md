# OULAD Dataset Upload Guide

This guide helps you upload your OULAD dataset files from your local computer to this repository.

## Quick Start

If you have OULAD dataset files on your computer at `C:\Users\MyName\Documents\Github\10-Aug-25\data\oulad\raw`, follow these steps:

### Step 1: Upload Helper Script (Recommended)

Use the upload helper script to validate and copy your files:

```bash
# On Windows:
python scripts/upload_oulad_dataset.py --source-dir "C:\Users\MyName\Documents\Github\10-Aug-25\data\oulad\raw"

# On Linux/Mac:
python scripts/upload_oulad_dataset.py --source-dir "/path/to/your/oulad/files"

# Dry run to validate first:
python scripts/upload_oulad_dataset.py --source-dir "C:\Users\MyName\Documents\Github\10-Aug-25\data\oulad\raw" --dry-run
```

### Step 2: Manual Upload (Alternative)

If you prefer to copy files manually:

1. Copy these CSV files from your local directory to `data/oulad/raw/`:
   - `studentInfo.csv`
   - `studentRegistration.csv`
   - `studentAssessment.csv`
   - `studentVle.csv`
   - `vle.csv`
   - `assessments.csv`
   - `courses.csv`

2. Validate the format:
   ```bash
   python scripts/validate_oulad_format.py
   ```

### Step 3: Process the Data

Once files are uploaded, run preprocessing:

```bash
python scripts/preprocess_oulad.py
```

This creates the ML-ready dataset in `data/oulad/processed/oulad_ml.csv`.

### Step 4: Commit to Repository

The .gitignore has been updated to allow OULAD CSV files. Commit them:

```bash
git add data/oulad/raw/*.csv data/oulad/processed/
git commit -m "Add OULAD dataset and processed files"
git push
```

## Expected File Structure

```
data/oulad/
├── raw/
│   ├── studentInfo.csv         # Student demographics
│   ├── studentRegistration.csv # Course registrations and outcomes  
│   ├── studentAssessment.csv   # Assessment submissions
│   ├── studentVle.csv          # VLE interaction logs
│   ├── vle.csv                 # VLE resource metadata
│   ├── assessments.csv         # Assessment metadata
│   └── courses.csv             # Course information
└── processed/
    ├── oulad_ml.csv           # ML-ready dataset
    └── dataset_summary.json   # Dataset statistics
```

## File Format Requirements

The upload script validates that your files have the correct columns:

- **studentInfo.csv**: `code_module`, `code_presentation`, `id_student`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `num_of_prev_attempts`, `studied_credits`, `disability`
- **studentRegistration.csv**: `code_module`, `code_presentation`, `id_student`, `date_registration`, `date_unregistration`, `final_result`
- **studentAssessment.csv**: `id_assessment`, `id_student`, `date`, `is_banked`, `score`
- **studentVle.csv**: `code_module`, `code_presentation`, `id_student`, `id_site`, `date`, `sum_click`
- **vle.csv**: `id_site`, `code_module`, `code_presentation`, `activity_type`, `week_from`, `week_to`
- **assessments.csv**: `code_module`, `code_presentation`, `id_assessment`, `assessment_type`, `date`, `weight`
- **courses.csv**: `code_module`, `code_presentation`, `module_presentation_length`

## Troubleshooting

### Column Name Issues
If you get errors about missing columns, the validator script will suggest fixes for common issues like `date_submitted` vs `date`.

### Large Files
The OULAD dataset can be large. If you encounter issues with file sizes, consider:
- Using Git LFS for large files
- Compressing files before upload
- Using the `--dry-run` option first to validate

### Processing Errors
If preprocessing fails:
1. Check the log output for specific error messages
2. Validate your file format with `python scripts/validate_oulad_format.py`
3. Ensure all required files are present

## Next Steps

After successful upload and processing:
1. The processed dataset is ready for machine learning experiments
2. Run existing OULAD analysis scripts in the repository
3. Use the data with the transfer learning framework

## Getting Help

- Check `data/oulad/raw/DOWNLOAD_INSTRUCTIONS.txt` for additional information
- Review the processed dataset summary in `data/oulad/processed/dataset_summary.json`
- Consult the main README and documentation for usage examples