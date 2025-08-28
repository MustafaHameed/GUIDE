from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GRU/LSTM/Transformer/TCN inferences and write an ensemble CSV")
    p.add_argument("--test_csv", default=str(Path("data/xuetangx/raw/Test.csv")))
    p.add_argument("--model_dir", default=str(Path("models/xuetangx")))
    p.add_argument("--out_dir", default=str(Path("data/xuetangx/processed")))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights in order of available models (gru lstm transformer tcn). If omitted, tries ensemble.json, else equal",
    )
    p.add_argument("--skip_if_exists", action="store_true", help="Skip inference if per-model outputs already exist")
    p.add_argument("--demographics_csv", default=str(Path("data/xuetangx/raw/user_info (1).csv")), help="Optional demographics CSV with 'username' and fields like sex, education_level, country, age")
    return p.parse_args()


def run_infer(model_dir: str, test_csv: str, ckpt: str, out_csv: str, device: str, batch_size: int, num_workers: int, demographics_csv: str) -> None:
    cmd = [
        sys.executable,
        os.path.join("scripts", "infer_dropout.py"),
        "--test_csv",
        test_csv,
        "--model_dir",
        model_dir,
        "--ckpt",
        ckpt,
        "--output_csv",
        out_csv,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--demographics_csv",
        demographics_csv,
    ]
    print("[Run]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_gru = model_dir / "model_gru.pt"
    ckpt_lstm = model_dir / "model_lstm.pt"
    ckpt_tr = model_dir / "model_transformer.pt"
    ckpt_tcn = model_dir / "model_tcn.pt"
    if not ckpt_gru.exists() and not ckpt_lstm.exists() and not ckpt_tr.exists() and not ckpt_tcn.exists():
        raise FileNotFoundError("No checkpoints found in model_dir")

    out_gru = out_dir / "Test_predictions_gru.csv"
    out_lstm = out_dir / "Test_predictions_lstm.csv"
    out_tr = out_dir / "Test_predictions_transformer.csv"
    out_tcn = out_dir / "Test_predictions_tcn.csv"

    if ckpt_gru.exists() and (not args.skip_if_exists or not out_gru.exists()):
        run_infer(str(model_dir), args.test_csv, str(ckpt_gru), str(out_gru), args.device, args.batch_size, args.num_workers, args.demographics_csv)
    if ckpt_tr.exists() and (not args.skip_if_exists or not out_tr.exists()):
        run_infer(str(model_dir), args.test_csv, str(ckpt_tr), str(out_tr), args.device, args.batch_size, args.num_workers, args.demographics_csv)
    if ckpt_lstm.exists() and (not args.skip_if_exists or not out_lstm.exists()):
        run_infer(str(model_dir), args.test_csv, str(ckpt_lstm), str(out_lstm), args.device, args.batch_size, args.num_workers, args.demographics_csv)
    if ckpt_tcn.exists() and (not args.skip_if_exists or not out_tcn.exists()):
        run_infer(str(model_dir), args.test_csv, str(ckpt_tcn), str(out_tcn), args.device, args.batch_size, args.num_workers, args.demographics_csv)

    # Determine weights
    # Determine weights for any subset of models (gru, transformer, tcn) in that order
    avail_names: List[str] = []
    if ckpt_gru.exists():
        avail_names.append("gru")
    if ckpt_lstm.exists():
        avail_names.append("lstm")
    if ckpt_tr.exists():
        avail_names.append("transformer")
    if ckpt_tcn.exists():
        avail_names.append("tcn")

    w: Optional[np.ndarray] = None
    if args.weights is not None and len(args.weights) > 0:
        if len(args.weights) != len(avail_names):
            raise ValueError(f"--weights length ({len(args.weights)}) must match number of available models ({len(avail_names)}): {avail_names}")
        w = np.asarray(args.weights, dtype=np.float64)
        print(f"[Weights] from CLI for {avail_names}: {w}")
    else:
        ens_path = model_dir / "ensemble.json"
        if ens_path.exists():
            with open(ens_path, "r", encoding="utf-8") as f:
                ens = json.load(f)
            weight_map = ens.get("weights", {})  # expected like {"gru": 0.4, "transformer": 0.6, "tcn": 0.2}
            if isinstance(weight_map, dict) and weight_map:
                w = np.array([float(weight_map.get(name, 1.0)) for name in avail_names], dtype=np.float64)
                print(f"[Weights] from ensemble.json for {avail_names}: {w}")
    if w is None:
        w = np.ones(len(avail_names), dtype=np.float64)
        print(f"[Weights] default equal for {avail_names}: {w}")
    w = w / w.sum()

    # Load predictions and ensemble
    dfs: List[pd.DataFrame] = []
    names: List[str] = []
    if ckpt_gru.exists():
        dfs.append(pd.read_csv(out_gru))
        names.append("gru")
    if ckpt_tr.exists():
        dfs.append(pd.read_csv(out_tr))
        names.append("transformer")
    if ckpt_lstm.exists():
        dfs.append(pd.read_csv(out_lstm))
        names.append("lstm")
    if ckpt_tcn.exists():
        dfs.append(pd.read_csv(out_tcn))
        names.append("tcn")
    if len(dfs) < 2:
        print("[Warn] Only one model predictions available; copying to ensemble output")
        single = out_gru if ckpt_gru.exists() else (out_tr if ckpt_tr.exists() else out_tcn)
        out_final = out_dir / "Test_predictions_ensemble.csv"
        pd.read_csv(single).to_csv(out_final, index=False)
        print(f"[Saved] {out_final}")
        return

    # Align by common keys
    keys = [k for k in ["username", "course_id", "session_id", "timestamp"] if all(k in df.columns for df in dfs)]
    if not keys:
        raise RuntimeError("Cannot ensemble because no common alignment keys present")
    ref = dfs[0][keys + ["pred_dropout_prob"]].rename(columns={"pred_dropout_prob": f"pred_{names[0]}"})
    out = ref
    for i in range(1, len(dfs)):
        di = dfs[i][keys + ["pred_dropout_prob"]].rename(columns={"pred_dropout_prob": f"pred_{names[i]}"})
        out = out.merge(di, on=keys, how="inner", validate="one_to_one")

    # Weighted average across all available models
    pred_cols = [f"pred_{n}" for n in names]
    mat = out[pred_cols].to_numpy(dtype=float)
    # arrange weights per model name order
    if len(w) != len(names):
        # adjust to available names
        print(f"[Warn] weight vector length {len(w)} != models {len(names)}; resizing to equal weights")
        w = np.ones(len(names), dtype=np.float64)
    w_vec = np.array([float(w[i]) for i in range(len(names))], dtype=float)
    out["pred_dropout_prob"] = np.clip((mat * (w_vec / w_vec.sum())).sum(axis=1), 0.0, 1.0)
    out = out.drop(columns=pred_cols)
    out_final = out_dir / "Test_predictions_ensemble.csv"
    out.to_csv(out_final, index=False)
    print(f"[Saved] {out_final} rows={len(out):,}")


if __name__ == "__main__":
    main()
