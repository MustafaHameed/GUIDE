from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install via: pip install matplotlib") from e


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Sort by score desc
    order = np.argsort(-y_score, kind="mergesort")
    y = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    denom = np.maximum(tp + fp, 1e-12)
    precision = tp / denom
    P = float(np.sum(y))
    recall = tp / max(P, 1e-12)
    # prepend (0,1) style optional — not necessary for plotting
    return recall, precision


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    P = tp[-1] if tp.size else 0.0
    N = fp[-1] if fp.size else 0.0
    tpr = tp / max(P, 1e-12)
    fpr = fp / max(N, 1e-12)
    return fpr, tpr


def plot_pr_roc(val_preds_path: str, out_dir: str) -> List[str]:
    matplotlib = _import_matplotlib()
    import matplotlib.pyplot as plt

    df = pd.read_csv(val_preds_path)
    if "y_true" not in df.columns:
        raise ValueError("val preds CSV must contain column 'y_true'")
    y_true = df["y_true"].to_numpy()
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("val preds CSV must contain one or more 'prob_*' columns")

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    # PR curves
    plt.figure(figsize=(7, 5))
    for c in prob_cols:
        r, p = pr_curve(y_true, df[c].to_numpy())
        plt.plot(r, p, label=c)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (validation)")
    plt.legend()
    pr_path = str(Path(out_dir) / "pr_curves.png")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    paths.append(pr_path)

    # ROC curves
    plt.figure(figsize=(7, 5))
    for c in prob_cols:
        f, t = roc_curve(y_true, df[c].to_numpy())
        plt.plot(f, t, label=c)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (validation)")
    plt.legend()
    roc_path = str(Path(out_dir) / "roc_curves.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    paths.append(roc_path)
    return paths


def plot_early_warning(curve_csv: str, out_dir: str) -> str:
    _import_matplotlib()
    import matplotlib.pyplot as plt

    df = pd.read_csv(curve_csv)
    if "prefix_steps" not in df.columns:
        raise ValueError("early_warning CSV must contain 'prefix_steps'")
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    if "auprc" in df.columns:
        plt.plot(df["prefix_steps"], df["auprc"], marker="o", label="AUPRC")
    if "roc_auc" in df.columns:
        plt.plot(df["prefix_steps"], df["roc_auc"], marker="s", label="ROC-AUC")
    if "logloss" in df.columns:
        plt.plot(df["prefix_steps"], df["logloss"], marker="^", label="LogLoss")
    plt.xlabel("Prefix steps")
    plt.title("Early-warning curve (validation)")
    plt.grid(alpha=0.3)
    plt.legend()
    out_path = str(Path(out_dir) / "early_warning.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_per_course(test_preds: str, out_dir: str, top_k: int = 20) -> List[str]:
    _import_matplotlib()
    import matplotlib.pyplot as plt

    df = pd.read_csv(test_preds)
    if "pred_dropout_prob" not in df.columns:
        raise ValueError("test predictions CSV must contain 'pred_dropout_prob'")
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    # Histogram of probabilities
    plt.figure(figsize=(7, 5))
    plt.hist(df["pred_dropout_prob"], bins=50, color="#4C72B0", alpha=0.8)
    plt.xlabel("pred_dropout_prob")
    plt.ylabel("Count")
    plt.title("Prediction probability distribution (Test)")
    p1 = str(Path(out_dir) / "prob_hist.png")
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    paths.append(p1)

    # Per-course mean (top_k)
    if "course_id" in df.columns:
        means = (
            df.groupby("course_id")["pred_dropout_prob"].mean().sort_values(ascending=False).head(top_k)
        )
        plt.figure(figsize=(max(8, top_k * 0.4), 5))
        means.plot(kind="bar", color="#55A868")
        plt.ylabel("Mean predicted prob")
        plt.title(f"Per-course mean prob (top {top_k})")
        plt.xticks(rotation=45, ha="right")
        p2 = str(Path(out_dir) / "per_course_means.png")
        plt.tight_layout()
        plt.savefig(p2)
        plt.close()
        paths.append(p2)

        # Per-course boxplot (top_k by count)
        counts = df.groupby("course_id")["pred_dropout_prob"].count().sort_values(ascending=False).head(top_k)
        top_courses = counts.index.tolist()
        data = [df[df["course_id"] == c]["pred_dropout_prob"].to_numpy() for c in top_courses]
        plt.figure(figsize=(max(8, top_k * 0.4), 5))
        plt.boxplot(data, labels=top_courses, showfliers=False)
        plt.ylabel("pred_dropout_prob")
        plt.title(f"Per-course probability distribution (top {top_k} by count)")
        plt.xticks(rotation=45, ha="right")
        p3 = str(Path(out_dir) / "per_course_boxplot.png")
        plt.tight_layout()
        plt.savefig(p3)
        plt.close()
        paths.append(p3)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot PR/ROC, early-warning, and per-course distributions")
    p.add_argument("--val_preds", default="", help="CSV from eval_models.py --save_val_preds (must contain y_true and prob_* columns)")
    p.add_argument("--early_csv", default=str(Path("models/xuetangx/early_warning_curve.csv")))
    p.add_argument("--test_preds", default=str(Path("data/xuetangx/processed/Test_predictions_ensemble.csv")))
    p.add_argument("--out_dir", default=str(Path("models/xuetangx/plots")))
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Plots] Saving to {out_dir}")
    if args.val_preds:
        paths = plot_pr_roc(args.val_preds, out_dir)
        for p in paths:
            print("[Saved]", p)
    if args.early_csv and os.path.exists(args.early_csv):
        pth = plot_early_warning(args.early_csv, out_dir)
        print("[Saved]", pth)
    if args.test_preds and os.path.exists(args.test_preds):
        paths = plot_per_course(args.test_preds, out_dir, args.top_k)
        for p in paths:
            print("[Saved]", p)


if __name__ == "__main__":
    main()

