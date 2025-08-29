from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_KEYS = ["username", "course_id", "session_id", "timestamp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply stacking calibrator to per-model prediction CSVs")
    p.add_argument("--inputs", nargs="+", required=True, help="List of per-model prediction CSVs (GRU/LSTM/Transformer/TCN)")
    p.add_argument("--calibrator", required=True, help="Path to calibrator.json from stack_calibrator.py")
    p.add_argument("--output", default=str(Path("data/xuetangx/processed/Test_predictions_stacked.csv")))
    p.add_argument("--on", nargs="*", default=None, help="Join keys (default auto-detect common keys)")
    p.add_argument("--keep_individual", action="store_true")
    return p.parse_args()


def detect_name(path: str) -> str:
    s = os.path.basename(path).lower()
    for name in ["gru", "lstm", "transformer", "tcn"]:
        if name in s:
            return name
    return os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    args = parse_args()
    with open(args.calibrator, "r", encoding="utf-8") as f:
        cal = json.load(f)
    names: List[str] = cal.get("names", [])
    w = np.asarray(cal.get("weights", []), dtype=np.float64)
    b = float(cal.get("bias", 0.0))
    if len(names) != len(w):
        raise ValueError("calibrator weights length mismatch")

    dfs = [pd.read_csv(p, low_memory=False) for p in args.inputs]
    model_names = [detect_name(p) for p in args.inputs]
    # Determine join keys
    keys = args.on
    if not keys:
        keys = [k for k in DEFAULT_KEYS if all(k in df.columns for df in dfs)]
    if not keys:
        raise RuntimeError("No common join keys among inputs; pass --on ...")

    # Merge all inputs
    out = dfs[0][keys + ["pred_dropout_prob"]].rename(columns={"pred_dropout_prob": f"prob_{model_names[0]}"})
    for i in range(1, len(dfs)):
        di = dfs[i][keys + ["pred_dropout_prob"]].rename(columns={"pred_dropout_prob": f"prob_{model_names[i]}"})
        out = out.merge(di, on=keys, how="inner", validate="one_to_one")

    # Build matrix in calibrator name order; fill missing with zeros
    X = np.zeros((len(out), len(names)), dtype=np.float64)
    for j, n in enumerate(names):
        col = f"prob_{n}"
        if col in out.columns:
            X[:, j] = out[col].to_numpy(dtype=np.float64)
        else:
            X[:, j] = 0.0
    z = X @ w.reshape(-1, 1) + b
    p = 1.0 / (1.0 + np.exp(-z))
    out["pred_dropout_prob"] = p.astype(np.float64)
    if not args.keep_individual:
        for n in model_names:
            c = f"prob_{n}"
            if c in out.columns:
                out = out.drop(columns=[c])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[Saved] {args.output} rows={len(out):,}")


if __name__ == "__main__":
    main()

