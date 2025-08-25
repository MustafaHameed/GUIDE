#!/usr/bin/env python3
"""
Aggregate experiment results into a Markdown report with confidence intervals,
baseline comparisons, and plots.

Expected directory structure:
  results/
    <experiment_name>/
      <run_id>/
        metrics.json  # preferred format (dict of metric -> value) or {"metrics": {...}}
        config.json   # optional, arbitrary dict for run metadata
      <run_id2>/...
    <experiment2>/...

Alternatively, metrics.csv per run is supported if it includes columns like:
  metric,value   OR   columns named by each metric with a single row.

Usage:
  python scripts/aggregate_results.py \
    --input-dir results \
    --out docs/experiment_report.md \
    --plots-dir docs/figures \
    --metric accuracy \
    --baseline baseline_experiment \
    --alpha 0.05 \
    --bootstrap 10000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


PRIMARY_METRIC_CANDIDATES = [
    "accuracy",
    "acc",
    "f1",
    "f1_macro",
    "f1_micro",
    "auc",
    "auroc",
    "rmse",
    "mae",
    "mape",
    "mse",
    "loss",
    "precision",
    "recall",
    "bleu",
    "rouge",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate experiment results and produce a Markdown report."
    )
    p.add_argument(
        "--input-dir",
        default="results",
        help="Root directory containing experiment results.",
    )
    p.add_argument(
        "--out",
        default="docs/experiment_report.md",
        help="Path to write the Markdown report.",
    )
    p.add_argument(
        "--plots-dir", default="docs/figures", help="Directory to write plots."
    )
    p.add_argument(
        "--metric",
        default=None,
        help="Primary metric to report. Auto-detect if omitted.",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Name of the baseline experiment to compare against.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Bootstrap iterations for CIs and p-values.",
    )
    p.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum runs per experiment to include in stats.",
    )
    return p.parse_args()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_metrics_json(p: Path) -> Dict[str, float]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "metrics" in data and isinstance(data["metrics"], dict):
            return {k: float(v) for k, v in data["metrics"].items()}
        # flatten one level if needed
        return {
            k: float(v) for k, v in data.items() if isinstance(v, (int, float, str))
        }
    raise ValueError(f"Unsupported metrics.json format at {p}")


def load_metrics_csv(p: Path) -> Dict[str, float]:
    df = pd.read_csv(p)
    if "metric" in df.columns and "value" in df.columns:
        return {str(m): float(v) for m, v in zip(df["metric"], df["value"])}
    # single-row wide format
    if len(df) == 1:
        row = df.iloc[0].to_dict()
        # cast numeric-like
        out = {}
        for k, v in row.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    raise ValueError(f"Unsupported metrics.csv format at {p}")


def discover_runs(root: Path) -> List[Tuple[str, str, Path]]:
    out: List[Tuple[str, str, Path]] = []
    if not root.exists():
        return out
    for experiment_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        experiment = experiment_dir.name
        for run_dir in sorted([p for p in experiment_dir.iterdir() if p.is_dir()]):
            out.append((experiment, run_dir.name, run_dir))
    return out


def get_primary_metric(all_metrics: List[str], user_metric: Optional[str]) -> str:
    if user_metric:
        # permissive matching
        low = user_metric.lower()
        for m in all_metrics:
            if m.lower() == low:
                return m
        # try contains
        for m in all_metrics:
            if low in m.lower():
                return m
        # fallback to user_metric as-is if not found
        return user_metric
    # auto-detect
    lowers = {m.lower(): m for m in all_metrics}
    for cand in PRIMARY_METRIC_CANDIDATES:
        if cand in lowers:
            return lowers[cand]
    # pick the most frequent metric
    return all_metrics[0] if all_metrics else "metric"


def bootstrap_ci(
    a: np.ndarray, alpha: float = 0.05, iters: int = 10000, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    # If all values same, CI is that value
    if np.allclose(a, a[0]):
        return (float(a[0]), float(a[0]))
    means = []
    for _ in range(iters):
        sample = rng.choice(a, size=n, replace=True)
        means.append(sample.mean())
    low = np.percentile(means, 100 * (alpha / 2))
    high = np.percentile(means, 100 * (1 - alpha / 2))
    return float(low), float(high)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    # Handle unequal n and variance
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    s_pooled = math.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if s_pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / s_pooled


@dataclass
class ExperimentStats:
    experiment: str
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def summarize(
    df: pd.DataFrame, metric: str, alpha: float, bootstrap_iters: int
) -> List[ExperimentStats]:
    stats_list: List[ExperimentStats] = []
    for exp, grp in df.groupby("experiment"):
        vals = grp[metric].dropna().to_numpy()
        if len(vals) == 0:
            continue
        ci_low, ci_high = bootstrap_ci(vals, alpha=alpha, iters=bootstrap_iters)
        stats_list.append(
            ExperimentStats(
                experiment=exp,
                n=len(vals),
                mean=float(np.mean(vals)),
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return stats_list


def load_all(input_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    metric_names = set()
    for experiment, run_id, run_dir in discover_runs(input_dir):
        metrics: Dict[str, float] = {}
        mjson = run_dir / "metrics.json"
        mcsv = run_dir / "metrics.csv"
        try:
            if mjson.exists():
                metrics = load_metrics_json(mjson)
            elif mcsv.exists():
                metrics = load_metrics_csv(mcsv)
            else:
                continue
        except Exception as e:
            print(f"[WARN] Skipping {run_dir} due to parse error: {e}", file=sys.stderr)
            continue
        # coerce numeric
        cleaned = {}
        for k, v in metrics.items():
            try:
                cleaned[k] = float(v)
            except Exception:
                pass
        if not cleaned:
            continue
        metric_names.update(cleaned.keys())
        rows.append({"experiment": experiment, "run_id": run_id, **cleaned})
    if not rows:
        return pd.DataFrame(), sorted(metric_names)
    df = pd.DataFrame(rows)
    # Order columns: experiment, run_id, metrics...
    cols = ["experiment", "run_id"] + [
        c for c in df.columns if c not in ("experiment", "run_id")
    ]
    df = df[cols]
    return df, sorted(metric_names)


def make_plots(
    df: pd.DataFrame, metric: str, stats_list: List[ExperimentStats], plots_dir: Path
) -> Dict[str, str]:
    safe_mkdir(plots_dir)
    paths: Dict[str, str] = {}
    # Bar + CI
    bar_df = pd.DataFrame(
        [
            {
                "experiment": s.experiment,
                "mean": s.mean,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
            }
            for s in stats_list
        ]
    )
    if not bar_df.empty:
        plt.figure(figsize=(10, 5))
        order = bar_df.sort_values("mean", ascending=False)["experiment"]
        sns.barplot(data=bar_df, x="experiment", y="mean", order=order, color="#4472C4")
        # error bars
        for i, row in enumerate(bar_df.set_index("experiment").loc[order].itertuples()):
            plt.plot([i, i], [row.ci_low, row.ci_high], color="black")
            plt.plot([i - 0.1, i + 0.1], [row.ci_low, row.ci_low], color="black")
            plt.plot([i - 0.1, i + 0.1], [row.ci_high, row.ci_high], color="black")
        plt.ylabel(metric)
        plt.xlabel("Experiment")
        plt.title(f"{metric} with 95% CI")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        bar_path = plots_dir / f"{metric.replace(' ', '_')}_bar_ci.png"
        plt.savefig(bar_path, dpi=180)
        plt.close()
        paths["bar_ci"] = str(bar_path)

    # Violin/box per run
    if not df.empty and metric in df.columns:
        plt.figure(figsize=(10, 5))
        sns.violinplot(data=df, x="experiment", y=metric, inner="box", cut=0)
        plt.ylabel(metric)
        plt.xlabel("Experiment")
        plt.title(f"Per-run distribution of {metric}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        viol_path = plots_dir / f"{metric.replace(' ', '_')}_violin.png"
        plt.savefig(viol_path, dpi=180)
        plt.close()
        paths["violin"] = str(viol_path)

    return paths


def relative_improvements(
    df: pd.DataFrame, metric: str, baseline: str, alpha: float, bootstrap_iters: int
) -> pd.DataFrame:
    # Higher-is-better heuristic for metric naming; allow loss to be lower-better
    higher_is_better = not (metric.lower() in {"loss", "rmse", "mae", "mape", "mse"})
    base_vals = df[df["experiment"] == baseline][metric].dropna().to_numpy()
    rows = []
    for exp, grp in df.groupby("experiment"):
        if exp == baseline:
            continue
        vals = grp[metric].dropna().to_numpy()
        if len(vals) == 0 or len(base_vals) == 0:
            continue
        # Define "improvement" as exp - base (if higher is better), else base - exp (so positive is better)
        diff = (
            (vals.mean() - base_vals.mean())
            if higher_is_better
            else (base_vals.mean() - vals.mean())
        )
        d = cohen_d(vals, base_vals)
        # bootstrap p-value: proportion of bootstrap differences <= 0 (one-sided), convert to two-sided
        rng = np.random.default_rng(123)
        n1, n2 = len(vals), len(base_vals)
        diffs = []
        for _ in range(bootstrap_iters):
            s1 = rng.choice(vals, size=n1, replace=True)
            s2 = rng.choice(base_vals, size=n2, replace=True)
            diffs.append(
                (s1.mean() - s2.mean()) if higher_is_better else (s2.mean() - s1.mean())
            )
        diffs = np.array(diffs)
        p_one_sided = np.mean(diffs <= 0.0)
        p_two_sided = 2 * min(p_one_sided, 1 - p_one_sided)
        ci_low, ci_high = np.percentile(
            diffs, [100 * (alpha / 2), 100 * (1 - alpha / 2)]
        )
        rows.append(
            {
                "experiment": exp,
                "improvement": float(diff),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "cohens_d": float(d),
                "p_value": float(p_two_sided),
                "n_exp": int(n1),
                "n_base": int(n2),
            }
        )
    return pd.DataFrame(rows)


def render_markdown(
    out_path: Path,
    metric: str,
    alpha: float,
    stats_list: List[ExperimentStats],
    plots: Dict[str, str],
    baseline: Optional[str],
    improvements: Optional[pd.DataFrame],
) -> None:
    safe_mkdir(out_path.parent)
    lines: List[str] = []
    lines.append(f"# Experiment Report")
    lines.append("")
    lines.append(f"- Primary metric: {metric}")
    lines.append(f"- Confidence level: {int((1 - alpha) * 100)}%")
    if baseline:
        lines.append(f"- Baseline: {baseline}")
    lines.append("")
    if stats_list:
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append("| Experiment | N runs | Mean | Std | CI low | CI high |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for s in sorted(stats_list, key=lambda x: x.mean, reverse=True):
            lines.append(
                f"| {s.experiment} | {s.n} | {s.mean:.4f} | {s.std:.4f} | {s.ci_low:.4f} | {s.ci_high:.4f} |"
            )
        lines.append("")
    if plots:
        lines.append("## Plots")
        lines.append("")
        if "bar_ci" in plots:
            lines.append(f"![Mean with 95% CI]({plots['bar_ci']})")
        if "violin" in plots:
            lines.append("")
            lines.append(f"![Per-run distribution]({plots['violin']})")
        lines.append("")
    if baseline and improvements is not None and not improvements.empty:
        lines.append("## Improvements over Baseline")
        lines.append("")
        lines.append(
            "| Experiment | Improvement | CI low | CI high | Cohen's d | p-value | n(exp) | n(base) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in improvements.sort_values("improvement", ascending=False).iterrows():
            lines.append(
                f"| {r['experiment']} | {r['improvement']:.4f} | {r['ci_low']:.4f} | {r['ci_high']:.4f} | {r['cohens_d']:.3f} | {r['p_value']:.4f} | {int(r['n_exp'])} | {int(r['n_base'])} |"
            )
        lines.append("")
        lines.append(
            "_Improvement is defined so that positive values favor the non-baseline model._"
        )
        lines.append("")
    lines.append("## Notes")
    lines.append("- Ensure at least 5 runs per experiment for stable intervals.")
    lines.append(
        "- Check that the metric direction matches your task (higher-is-better vs lower-is-better)."
    )
    lines.append("- Confirm dataset splits and seeds are consistent across runs.")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    plots_dir = Path(args.plots_dir)

    df, metric_names = load_all(input_dir)
    if df.empty:
        print(
            f"[INFO] No result files found under {input_dir}. Skipping report.",
            file=sys.stderr,
        )
        safe_mkdir(out_path.parent)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# Experiment Report\n\n_No results found._\n")
        return 0

    primary_metric = get_primary_metric(metric_names, args.metric)
    if primary_metric not in df.columns:
        print(
            f"[WARN] Primary metric '{primary_metric}' not found; available: {metric_names}",
            file=sys.stderr,
        )
        # try fallback to first available metric column
        metric_cols = [c for c in df.columns if c not in ("experiment", "run_id")]
        if not metric_cols:
            print("[ERROR] No metric columns found.", file=sys.stderr)
            return 1
        primary_metric = metric_cols[0]
        print(f"[INFO] Falling back to '{primary_metric}'", file=sys.stderr)

    # Filter experiments with enough runs
    counts = df.groupby("experiment")[primary_metric].count().rename("n")
    keep = counts[counts >= args.min_runs].index.tolist()
    if not keep:
        print(
            f"[WARN] No experiments meet min-runs={args.min_runs}. Using all experiments.",
            file=sys.stderr,
        )
        keep = sorted(df["experiment"].unique().tolist())
    df_use = df[df["experiment"].isin(keep)].copy()

    stats_list = summarize(
        df_use[["experiment", primary_metric]],
        primary_metric,
        args.alpha,
        args.bootstrap,
    )
    plots = make_plots(
        df_use[["experiment", primary_metric, "run_id"]],
        primary_metric,
        stats_list,
        plots_dir,
    )

    improvements_df: Optional[pd.DataFrame] = None
    baseline = args.baseline
    if baseline is None:
        # auto-pick baseline if an experiment named 'baseline' exists
        if "baseline" in set(df_use["experiment"].unique()):
            baseline = "baseline"
    if baseline is not None and baseline in set(df_use["experiment"].unique()):
        improvements_df = relative_improvements(
            df_use[["experiment", primary_metric]],
            primary_metric,
            baseline,
            args.alpha,
            args.bootstrap,
        )
    elif baseline is not None:
        print(
            f"[WARN] Baseline '{baseline}' not found among experiments.",
            file=sys.stderr,
        )

    render_markdown(
        out_path,
        primary_metric,
        args.alpha,
        stats_list,
        plots,
        baseline,
        improvements_df,
    )
    print(f"[OK] Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
