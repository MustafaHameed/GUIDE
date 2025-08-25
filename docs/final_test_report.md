# Experiment Report

- Primary metric: accuracy
- Confidence level: 95%
- Baseline: baseline

## Summary Statistics

| Experiment | N runs | Mean | Std | CI low | CI high |
|---|---:|---:|---:|---:|---:|
| improved_model | 5 | 0.8225 | 0.0247 | 0.8071 | 0.8437 |
| neural_network | 4 | 0.7974 | 0.0255 | 0.7803 | 0.8225 |
| sample_experiment | 3 | 0.7633 | 0.0153 | 0.7500 | 0.7800 |
| baseline | 5 | 0.7225 | 0.0247 | 0.7071 | 0.7437 |

## Plots

![Mean with 95% CI](docs/figures/accuracy_bar_ci.png)

![Per-run distribution](docs/figures/accuracy_violin.png)

## Improvements over Baseline

| Experiment | Improvement | CI low | CI high | Cohen's d | p-value | n(exp) | n(base) |
|---|---:|---:|---:|---:|---:|---:|---:|
| improved_model | 0.1000 | 0.0726 | 0.1274 | 4.052 | 0.0000 | 5 | 5 |
| neural_network | 0.0749 | 0.0464 | 0.1036 | 2.991 | 0.0000 | 4 | 5 |
| sample_experiment | 0.0408 | 0.0159 | 0.0631 | 1.855 | 0.0008 | 3 | 5 |

_Improvement is defined so that positive values favor the non-baseline model._

## Notes
- Ensure at least 5 runs per experiment for stable intervals.
- Check that the metric direction matches your task (higher-is-better vs lower-is-better).
- Confirm dataset splits and seeds are consistent across runs.