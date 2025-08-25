# Experiment Report Template

Use this template to standardize reporting across experiments. Replace placeholders with your content.

## Abstract
A 2–4 sentence summary of the purpose, method, key results, and takeaway.

## Experimental Setup
- Task and objective: what are you optimizing and why?
- Datasets: name, version, source, license, preprocessing steps.
- Splits: train/validation/test scheme; stratification and leakage checks.
- Models/methods: architectures, algorithms, or procedures.
- Hyperparameters: ranges, search strategy, selected values.
- Compute: hardware (GPU/CPU), runtime, parallelization, and budget.

## Evaluation Protocol
- Metrics: definitions, direction (higher vs lower is better), and why chosen.
- Repetition: number of runs; seeds used; aggregation approach.
- Statistical testing: CI level, hypothesis tests, and multiple-comparison handling (if applicable).

## Baselines
- Describe baseline(s) and rationale.
- Link configs for exact reproducibility.

## Results
- Summary table with mean ± std and 95% CI.
- Improvements vs baseline with effect sizes and p-values.
- Plots: mean with CI bars, distribution plots.

## Robustness and Sensitivity
- Sensitivity to key hyperparameters and data variations.
- Ablations: what components matter and by how much.
- Out-of-distribution or stress tests (if relevant).

## Error Analysis
- Qualitative failure modes with examples.
- Class- or subgroup-wise performance (fairness), if applicable.
- Calibration (for probabilistic outputs).

## Limitations
- Known caveats, trade-offs, and threats to validity.

## Reproducibility Checklist
- [ ] Fixed random seeds
- [ ] Environment and dependency lock (e.g., requirements.txt)
- [ ] Versioned data and code
- [ ] Deterministic operations (where possible)
- [ ] Configs and scripts to fully reproduce

## Appendix
- Full configs, command lines, and additional plots/tables.