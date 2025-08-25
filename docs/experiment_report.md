# Experiment Report

- Primary metric: accuracy
- Confidence level: 95%

## Summary Statistics

| Experiment | N runs | Mean | Std | CI low | CI high |
|---|---:|---:|---:|---:|---:|
| sample_experiment | 3 | 0.7633 | 0.0153 | 0.7500 | 0.7800 |

## Plots

![Mean with 95% CI](docs/figures/accuracy_bar_ci.png)

![Per-run distribution](docs/figures/accuracy_violin.png)

## Notes
- Ensure at least 5 runs per experiment for stable intervals.
- Check that the metric direction matches your task (higher-is-better vs lower-is-better).
- Confirm dataset splits and seeds are consistent across runs.