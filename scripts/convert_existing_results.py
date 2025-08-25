#!/usr/bin/env python3
"""
Helper script to convert existing CSV results to the aggregate_results.py format.

This can help migrate existing experimental results to the new automated reporting system.
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def convert_model_performance_csv(csv_path: Path, output_dir: Path) -> None:
    """
    Convert model_performance.csv to individual experiment runs.

    Expected format:
    model_type,accuracy_mean,accuracy_std,f1_mean,f1_std
    logistic,0.91,0.03,0.93,0.02
    ...
    """
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        model_type = row["model_type"]

        # Extract mean values as the main metrics
        metrics = {}
        for col in df.columns:
            if col.endswith("_mean"):
                metric_name = col.replace("_mean", "")
                metrics[metric_name] = float(row[col])

        # Create experiment directory
        exp_dir = output_dir / f"model_{model_type}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # For aggregated results, create a single run with the mean values
        run_dir = exp_dir / "aggregated_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        # Save config with standard deviation info
        config = {"model_type": model_type}
        for col in df.columns:
            if col.endswith("_std"):
                metric_name = col.replace("_std", "")
                config[f"{metric_name}_std"] = float(row[col])

        config_path = run_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        print(f"Converted {model_type} model results")


def convert_nested_cv_results(csv_path: Path, output_dir: Path) -> None:
    """
    Convert nested cross-validation results to experiment format.

    Expected format includes columns like:
    outer_fold,inner_fold,model,accuracy,f1,precision,recall
    """
    df = pd.read_csv(csv_path)

    if "model" in df.columns:
        # Group by model type
        for model_type, model_df in df.groupby("model"):
            exp_dir = output_dir / f"cv_{model_type}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Create runs for each fold combination
            for i, (_, row) in enumerate(model_df.iterrows()):
                run_id = f"fold_{i+1:03d}"
                run_dir = exp_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                # Extract metrics (skip non-numeric columns)
                metrics = {}
                for col in row.index:
                    if col not in [
                        "model",
                        "outer_fold",
                        "inner_fold",
                    ] and pd.api.types.is_numeric_dtype(type(row[col])):
                        try:
                            metrics[col] = float(row[col])
                        except (ValueError, TypeError):
                            continue

                # Save metrics
                metrics_path = run_dir / "metrics.json"
                with metrics_path.open("w") as f:
                    json.dump(metrics, f, indent=2)

                # Save fold info in config
                config = {
                    "model": model_type,
                    "outer_fold": int(row.get("outer_fold", i)),
                    "inner_fold": int(row.get("inner_fold", 1)),
                }

                config_path = run_dir / "config.json"
                with config_path.open("w") as f:
                    json.dump(config, f, indent=2)

            print(f"Converted {len(model_df)} CV runs for {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing CSV results to aggregate_results.py format"
    )
    parser.add_argument("csv_path", type=Path, help="Path to CSV file to convert")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for converted results",
    )
    parser.add_argument(
        "--format",
        choices=["model_performance", "nested_cv", "auto"],
        default="auto",
        help="Format of the input CSV",
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: {args.csv_path} does not exist")
        return 1

    # Auto-detect format based on columns
    df = pd.read_csv(args.csv_path)
    columns = set(df.columns)

    if args.format == "auto":
        if "model_type" in columns and any(col.endswith("_mean") for col in columns):
            args.format = "model_performance"
        elif "model" in columns and (
            "outer_fold" in columns or "inner_fold" in columns
        ):
            args.format = "nested_cv"
        else:
            print("Could not auto-detect format. Available formats:")
            print(
                "- model_performance: requires 'model_type' column and '*_mean' columns"
            )
            print("- nested_cv: requires 'model' column and fold information")
            return 1

    print(f"Converting {args.csv_path} using format: {args.format}")

    if args.format == "model_performance":
        convert_model_performance_csv(args.csv_path, args.output_dir)
    elif args.format == "nested_cv":
        convert_nested_cv_results(args.csv_path, args.output_dir)

    print(f"\nConversion complete! Results saved to {args.output_dir}")
    print("Generate a report with:")
    print(f"python scripts/aggregate_results.py --input-dir {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
