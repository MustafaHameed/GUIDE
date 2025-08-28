from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


DEFAULT_KEYS = ["username", "course_id", "session_id", "timestamp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble predictions by averaging probabilities across CSVs")
    p.add_argument("--inputs", nargs="+", required=True, help="List of prediction CSVs to ensemble")
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights (same length as inputs). Defaults to equal weights",
    )
    p.add_argument(
        "--on",
        nargs="*",
        default=None,
        help="Join keys (e.g., username course_id session_id). Defaults to auto-detect common keys or fallback to row order",
    )
    p.add_argument(
        "--output",
        default=os.path.join("data", "xuetangx", "processed", "Test_predictions_ensemble.csv"),
        help="Output CSV path",
    )
    p.add_argument("--keep_individual", action="store_true", help="Keep individual model columns in output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dfs: List[pd.DataFrame] = [pd.read_csv(p, low_memory=False) for p in args.inputs]
    n = len(dfs)
    for i, (p, df) in enumerate(zip(args.inputs, dfs)):
        if "pred_dropout_prob" not in df.columns:
            raise ValueError(f"Missing 'pred_dropout_prob' in {p}")
        print(f"[Load] {p} rows={len(df):,}")

    # Weights
    if args.weights is None:
        w = np.ones(n, dtype=np.float64) / n
    else:
        if len(args.weights) != n:
            raise ValueError("--weights length must match --inputs")
        w = np.asarray(args.weights, dtype=np.float64)
        w = w / np.sum(w)
    print(f"[Weights] {w}")

    # Determine join keys
    keys = args.on
    if not keys:
        keys = [k for k in DEFAULT_KEYS if all(k in df.columns for df in dfs)]
        if keys:
            print(f"[Join] Auto keys: {keys}")
        else:
            print("[Join] No common keys found; will align by row order")

    use_index = False
    if not keys:
        use_index = True
    else:
        for i, df in enumerate(dfs):
            if df.duplicated(keys).any():
                print(f"[Warn] Duplicates found in keys for input {i}; fallback to row-order align")
                use_index = True
                break

    # Build combined frame
    pred_cols: List[str] = []
    if use_index:
        base = pd.DataFrame(index=range(len(dfs[0])))
        for i, df in enumerate(dfs):
            if len(df) != len(base):
                raise ValueError("Inputs have different row counts; cannot align by index")
            col = f"pred_{i}"
            base[col] = df["pred_dropout_prob"].to_numpy()
            pred_cols.append(col)
        # carry over any of the known keys if present in first df
        carry = [k for k in DEFAULT_KEYS if k in dfs[0].columns]
        if carry:
            base = pd.concat([dfs[0][carry], base], axis=1)
        out = base
    else:
        out = dfs[0][keys + ["pred_dropout_prob"]].rename(columns={"pred_dropout_prob": "pred_0"})
        pred_cols.append("pred_0")
        for i in range(1, n):
            di = dfs[i][keys + ["pred_dropout_prob"]].rename(columns={"pred_dropout_prob": f"pred_{i}"})
            out = out.merge(di, on=keys, how="inner", validate="one_to_one")
            pred_cols.append(f"pred_{i}")

    # Weighted average
    preds_mat = out[pred_cols].to_numpy(dtype=np.float64)
    if preds_mat.shape[1] != len(w):
        raise RuntimeError("Internal: pred columns do not match weights")
    ensemble = np.clip(np.sum(preds_mat * w.reshape(1, -1), axis=1), 0.0, 1.0)
    out["pred_dropout_prob"] = ensemble.astype(np.float32)
    if not args.keep_individual:
        out = out.drop(columns=pred_cols)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[Saved] {args.output} rows={len(out):,}")


if __name__ == "__main__":
    main()

