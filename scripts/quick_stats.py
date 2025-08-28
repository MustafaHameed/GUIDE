from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Quick stats for predictions CSV")
    p.add_argument(
        "--path",
        default=os.path.join("data", "xuetangx", "processed", "Test_predictions.csv"),
        help="Path to predictions CSV with column 'pred_dropout_prob'",
    )
    p.add_argument(
        "--group",
        nargs="*",
        default=["course_id"],
        help="Columns to group by (e.g., course_id or username course_id). Empty to skip",
    )
    p.add_argument("--top", type=int, default=5, help="Show top/bottom N groups by mean prob")
    p.add_argument(
        "--save_grouped",
        default="",
        help="Optional path to save grouped means CSV (file or directory)",
    )
    p.add_argument("--head", type=int, default=5, help="Rows to show in sample head")
    args = p.parse_args()

    df = pd.read_csv(args.path, low_memory=False)
    print(f"[File] {args.path} rows={len(df):,}")

    col = "pred_dropout_prob"
    if col not in df.columns:
        print(f"[Error] Missing column '{col}'. Available columns: {list(df.columns)[:20]}")
        return

    print("[Overall] describe:")
    print(df[col].describe())

    qs = [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0]
    quant = df[col].quantile(qs)
    print("[Quantiles]")
    for q in qs:
        val = float(quant.loc[q])
        print(f"  q{q:>4.2f}: {val:.6f}")

    keep_cols: List[str] = [c for c in ["username", "course_id", "session_id", "timestamp", col] if c in df.columns]
    if keep_cols:
        print(f"[Sample head {args.head}] {keep_cols}")
        print(df[keep_cols].head(args.head).to_string(index=False))

    if args.group:
        missing = [g for g in args.group if g not in df.columns]
        if missing:
            print(f"[Group] Skip: missing columns {missing}")
        else:
            agg = (
                df.groupby(args.group, dropna=False)[col]
                .mean()
                .reset_index()
                .rename(columns={col: "mean_pred"})
                .sort_values("mean_pred", ascending=False)
            )
            print(f"[Top {args.top}] by {args.group} (mean_pred desc):")
            print(agg.head(args.top).to_string(index=False))
            print(f"[Bottom {args.top}] by {args.group} (mean_pred asc):")
            print(agg.tail(args.top).to_string(index=False))

            if args.save_grouped:
                out_path = args.save_grouped
                if os.path.isdir(out_path) or out_path.endswith(os.sep):
                    os.makedirs(out_path, exist_ok=True)
                    out_path = os.path.join(out_path, "grouped_stats.csv")
                else:
                    dirn = os.path.dirname(out_path)
                    if dirn:
                        os.makedirs(dirn, exist_ok=True)
                agg.to_csv(out_path, index=False)
                print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()

