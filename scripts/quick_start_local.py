from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[Run]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local quick start: train GRU/LSTM/Transformer/TCN, infer, ensemble, risk list")
    p.add_argument("--train_csv", default=str(Path("data/xuetangx/raw/Train.csv")))
    p.add_argument("--test_csv", default=str(Path("data/xuetangx/raw/Test.csv")))
    p.add_argument("--model_dir", default=str(Path("models/xuetangx")))
    p.add_argument("--out_dir", default=str(Path("data/xuetangx/processed")))
    p.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    p.add_argument("--epochs_gru", type=int, default=5)
    p.add_argument("--epochs_lstm", type=int, default=5)
    p.add_argument("--epochs_tr", type=int, default=5)
    p.add_argument("--epochs_tcn", type=int, default=5)
    p.add_argument("--batch_size_train", type=int, default=64)
    p.add_argument("--batch_size_infer", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--skip_train_if_exists", action="store_true")
    p.add_argument("--include_lstm", action="store_true", help="Also train LSTM baseline")
    p.add_argument("--include_tcn", action="store_true", help="Also train TCN baseline")
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional ensemble weights in order of available models (gru lstm transformer tcn)",
    )
    p.add_argument("--risk_criterion", choices=["f1", "youden"], default="f1")
    p.add_argument("--threshold_grouped", action="store_true", help="Choose threshold over grouped validation averages")
    p.add_argument("--demographics_csv", default=str(Path("data/xuetangx/raw/user_info (1).csv")), help="Optional demographics CSV with 'username' and fields like sex, education_level, country, age")
    return p.parse_args()


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    py = sys.executable
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_gru = model_dir / "model_gru.pt"
    ckpt_lstm = model_dir / "model_lstm.pt"
    ckpt_tr = model_dir / "model_transformer.pt"
    ckpt_tcn = model_dir / "model_tcn.pt"

    # 1) Train GRU (skip if requested and exists)
    if not (args.skip_train_if_exists and ckpt_gru.exists()):
        run(
            [
                py,
                str(Path("scripts") / "train_dropout.py"),
                "--train_csv",
                args.train_csv,
                "--model",
                "gru",
                "--loss",
                "weighted_bce",
                "--pos_weight",
                "auto",
                "--early_stop_metric",
                "auprc",
                "--patience",
                "4",
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
                "--batch_size",
                str(args.batch_size_train),
                "--epochs",
                str(args.epochs_gru),
                "--save_dir",
                args.model_dir,
                "--demographics_csv",
                args.demographics_csv,
            ]
        )
    else:
        print(f"[Skip] Training GRU (found {ckpt_gru})")

    # 2) Train Transformer (skip if requested and exists)
    if not (args.skip_train_if_exists and ckpt_tr.exists()):
        run(
            [
                py,
                str(Path("scripts") / "train_dropout.py"),
                "--train_csv",
                args.train_csv,
                "--model",
                "transformer",
                "--loss",
                "focal",
                "--focal_alpha",
                "0.25",
                "--focal_gamma",
                "2.0",
                "--early_stop_metric",
                "auprc",
                "--patience",
                "4",
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
                "--batch_size",
                str(args.batch_size_train),
                "--epochs",
                str(args.epochs_tr),
                "--save_dir",
                args.model_dir,
                "--demographics_csv",
                args.demographics_csv,
            ]
        )
    else:
        print(f"[Skip] Training Transformer (found {ckpt_tr})")

    # 2b) Train LSTM (optional)
    if args.include_lstm:
        if not (args.skip_train_if_exists and ckpt_lstm.exists()):
            run(
                [
                    py,
                    str(Path("scripts") / "train_dropout.py"),
                    "--train_csv",
                    args.train_csv,
                    "--model",
                    "lstm",
                    "--loss",
                    "weighted_bce",
                    "--pos_weight",
                    "auto",
                    "--early_stop_metric",
                    "auprc",
                    "--patience",
                    "4",
                    "--device",
                    args.device,
                    "--num_workers",
                    str(args.num_workers),
                    "--batch_size",
                    str(args.batch_size_train),
                    "--epochs",
                    str(args.epochs_lstm),
                    "--save_dir",
                    args.model_dir,
                    "--demographics_csv",
                    args.demographics_csv,
                ]
            )
        else:
            print(f"[Skip] Training LSTM (found {ckpt_lstm})")

    # 2c) Train TCN (optional)
    if args.include_tcn:
        if not (args.skip_train_if_exists and ckpt_tcn.exists()):
            run(
                [
                    py,
                    str(Path("scripts") / "train_dropout.py"),
                    "--train_csv",
                    args.train_csv,
                    "--model",
                    "tcn",
                    "--loss",
                    "focal",
                    "--focal_alpha",
                    "0.25",
                    "--focal_gamma",
                    "2.0",
                    "--early_stop_metric",
                    "auprc",
                    "--patience",
                    "4",
                    "--device",
                    args.device,
                    "--num_workers",
                    str(args.num_workers),
                    "--batch_size",
                    str(args.batch_size_train),
                    "--epochs",
                    str(args.epochs_tcn),
                    "--save_dir",
                    args.model_dir,
                    "--demographics_csv",
                    args.demographics_csv,
                ]
            )
        else:
            print(f"[Skip] Training TCN (found {ckpt_tcn})")

    # 3) Inference for all available models and ensemble
    infer_cmd = [
        py,
        str(Path("scripts") / "run_infer_and_ensemble.py"),
        "--test_csv",
        args.test_csv,
        "--model_dir",
        args.model_dir,
        "--out_dir",
        args.out_dir,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size_infer),
        "--num_workers",
        str(args.num_workers),
        "--demographics_csv",
        args.demographics_csv,
    ]
    if args.weights is not None and len(args.weights) > 0:
        infer_cmd += ["--weights"] + [str(w) for w in args.weights]
    run(infer_cmd)

    # 4) Risk list from ensemble predictions with threshold from validation
    ensemble_csv = str(Path(args.out_dir) / "Test_predictions_ensemble.csv")
    risk_cmd = [
        py,
        str(Path("scripts") / "make_risk_list.py"),
        "--train_csv",
        args.train_csv,
        "--model_dir",
        args.model_dir,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size_infer),
        "--num_workers",
        str(args.num_workers),
        "--test_preds",
        ensemble_csv,
        "--out_risk",
        str(Path(args.out_dir) / "Test_risk_by_user_course_ensemble.csv"),
        "--criterion",
        args.risk_criterion,
        "--demographics_csv",
        args.demographics_csv,
    ]
    if args.threshold_grouped:
        risk_cmd.append("--threshold_grouped")
    run(risk_cmd)

    print("[Done] Quick start pipeline completed.")


if __name__ == "__main__":
    main()
