# Automated Experiment Reporting

This directory contains an automated experiment reporting system that aggregates results from machine learning experiments and generates statistical reports with visualizations.

## Quick Start

1. **Save experiment results** in the `results/` directory using this structure:
   ```
   results/
     experiment_name/
       run_001/
         metrics.json    # {"accuracy": 0.85, "f1": 0.82, ...}
         config.json     # optional: experiment configuration
       run_002/
         metrics.json
   ```

2. **Generate a report**:
   ```bash
   python scripts/aggregate_results.py --baseline baseline_experiment
   ```

3. **View the report** at `docs/experiment_report.md` with plots in `docs/figures/`

## Features

- **Statistical Analysis**: Bootstrap confidence intervals, Cohen's d effect sizes, p-values
- **Baseline Comparisons**: Automatic statistical comparison against a baseline experiment
- **Visualizations**: Bar charts with confidence intervals, violin plots showing distributions
- **Multiple Formats**: Supports JSON and CSV metric files
- **Auto-detection**: Automatically detects primary metrics (accuracy, f1, rmse, etc.)
- **GitHub Integration**: Automatic reporting via GitHub Actions on every push/PR

## Example Usage

### Basic Report Generation
```bash
# Generate report with auto-detected primary metric
python scripts/aggregate_results.py

# Specify metric and baseline
python scripts/aggregate_results.py --metric f1 --baseline baseline_experiment

# Custom output paths
python scripts/aggregate_results.py --out custom_report.md --plots-dir custom_figures/
```

### File Formats

**JSON format** (preferred):
```json
{
  "accuracy": 0.85,
  "f1": 0.82,
  "precision": 0.87,
  "recall": 0.78
}
```

**CSV format**:
```csv
metric,value
accuracy,0.85
f1,0.82
precision,0.87
recall,0.78
```

### Integration with Experiments

Use the example script to see how to save results:
```bash
python scripts/generate_sample_experiments.py
```

### Converting Existing Results

Convert existing CSV files to the new format:
```bash
python scripts/convert_existing_results.py tables/model_performance.csv
```

## GitHub Actions Integration

The system automatically runs on every push/PR via `.github/workflows/experiment-report.yml`:

- Installs dependencies
- Runs `aggregate_results.py`
- Uploads reports as artifacts
- Shows summary in job output

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | `results` | Directory containing experiment results |
| `--out` | `docs/experiment_report.md` | Output path for Markdown report |
| `--plots-dir` | `docs/figures` | Directory for generated plots |
| `--metric` | auto-detect | Primary metric to analyze |
| `--baseline` | auto-detect | Baseline experiment for comparisons |
| `--alpha` | `0.05` | Significance level for confidence intervals |
| `--bootstrap` | `10000` | Bootstrap iterations for statistical tests |
| `--min-runs` | `3` | Minimum runs per experiment for statistics |

## Statistical Methods

- **Confidence Intervals**: Bootstrap sampling (default: 95% CI)
- **Effect Sizes**: Cohen's d for practical significance
- **Hypothesis Testing**: Bootstrap permutation tests for p-values
- **Multiple Experiments**: Automatic handling of multiple experimental conditions

## Best Practices

1. **Run Multiple Seeds**: Use at least 5 runs per experiment for stable statistics
2. **Consistent Splits**: Ensure train/test splits are consistent across experiments
3. **Metric Direction**: The system automatically handles higher-is-better vs lower-is-better metrics
4. **Baseline Selection**: Choose a simple, well-understood baseline for comparison
5. **Documentation**: Use the experiment report template in `docs/EXPERIMENT_REPORT_TEMPLATE.md`

## Troubleshooting

- **No results found**: Check that `results/` directory has the correct structure
- **Metric not found**: Verify metric names match between runs, or specify `--metric`
- **Not enough runs**: Reduce `--min-runs` or add more experimental runs
- **Plot errors**: Ensure matplotlib backend is available in your environment