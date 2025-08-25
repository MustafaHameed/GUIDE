# Experiment Report

- Primary metric: rmse
- Confidence level: 95%
- Baseline: baseline

## Summary Statistics

| Experiment | N runs | Mean | Std | CI low | CI high |
|---|---:|---:|---:|---:|---:|
| sample_experiment | 3 | 0.4367 | 0.0153 | 0.4200 | 0.4500 |
| baseline | 5 | 0.2678 | 0.0849 | 0.1983 | 0.3331 |
| neural_network | 4 | 0.1788 | 0.0849 | 0.1109 | 0.2467 |
| improved_model | 5 | 0.1678 | 0.0849 | 0.0983 | 0.2331 |

## Plots

![Mean with 95% CI](docs/figures/rmse_bar_ci.png)

![Per-run distribution](docs/figures/rmse_violin.png)

## Improvements over Baseline

| Experiment | Improvement | CI low | CI high | Cohen's d | p-value | n(exp) | n(base) |
|---|---:|---:|---:|---:|---:|---:|---:|
| improved_model | 0.1000 | 0.0073 | 0.1927 | -1.177 | 0.0440 | 5 | 5 |
| neural_network | 0.0890 | -0.0127 | 0.1858 | -1.049 | 0.1060 | 4 | 5 |
| sample_experiment | -0.1688 | -0.2413 | -0.1035 | 2.415 | 0.0000 | 3 | 5 |

_Improvement is defined so that positive values favor the non-baseline model._

## Notes
- Ensure at least 5 runs per experiment for stable intervals.
- Check that the metric direction matches your task (higher-is-better vs lower-is-better).
- Confirm dataset splits and seeds are consistent across runs.