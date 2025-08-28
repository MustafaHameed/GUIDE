from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make repo root importable when running as a script (python scripts/infer_dropout.py)
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.seq_models import GRUClassifier, TimeAwareTransformer, TCNClassifier, LSTMClassifier
from scripts.data_utils import (
    apply_standardizer,
    build_sequences_from_df,
    load_meta,
    prepare_dataframe,
    PackedSequenceDataset,
    collate_padded,
    make_demog_features_infer,
)


def build_model_from_meta(
    meta: Dict[str, object], input_dim: int, course_vocab: int, device: str, model_type: str | None = None
) -> torch.nn.Module:
    h = meta.get("hparams", {})
    if model_type is None:
        model_type = meta["model_type"]
    if model_type == "gru":
        model = GRUClassifier(
            input_dim=input_dim,
            hidden_dim=int(h.get("hidden_dim", 128)),
            num_layers=int(h.get("num_layers", 2)),
            dropout=float(h.get("dropout", 0.1)),
            course_vocab=course_vocab,
            course_emb_dim=int(h.get("course_emb_dim", 16)),
        )
    elif model_type == "lstm":
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=int(h.get("hidden_dim", 128)),
            num_layers=int(h.get("num_layers", 2)),
            dropout=float(h.get("dropout", 0.1)),
            course_vocab=course_vocab,
            course_emb_dim=int(h.get("course_emb_dim", 16)),
        )
    elif model_type == "transformer":
        model = TimeAwareTransformer(
            input_dim=input_dim,
            d_model=int(h.get("d_model", 128)),
            nhead=int(h.get("nhead", 4)),
            num_layers=int(h.get("num_layers", 2)),
            dim_feedforward=int(h.get("dim_feedforward", 256)),
            dropout=float(h.get("dropout", 0.1)),
            course_vocab=course_vocab,
            course_emb_dim=int(h.get("course_emb_dim", 16)),
            time_freqs=int(h.get("time_freqs", 8)),
        )
    else:  # tcn
        model = TCNClassifier(
            input_dim=input_dim,
            d_model=int(h.get("d_model", 128)),
            num_layers=int(h.get("num_layers", 2)),
            kernel_size=int(h.get("tcn_kernel", 3)),
            dropout=float(h.get("dropout", 0.1)),
            course_vocab=course_vocab,
            course_emb_dim=int(h.get("course_emb_dim", 16)),
        )
    return model.to(device)


def main() -> None:
    p = argparse.ArgumentParser(description="Inference for XuetangX dropout prediction on Test.csv")
    p.add_argument("--test_csv", default=str(Path("data/xuetangx/raw/Test.csv")))
    p.add_argument("--model_dir", default=str(Path("models/xuetangx")))
    p.add_argument("--output_csv", default=str(Path("data/xuetangx/processed/Test_predictions.csv")))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", default="", help="Optional explicit checkpoint path to load (overrides auto)")
    p.add_argument("--demographics_csv", default=str(Path("data/xuetangx/raw/user_info (1).csv")))

    args = p.parse_args()

    meta_path = Path(args.model_dir) / "meta.json"
    meta = load_meta(str(meta_path))
    feature_names: List[str] = meta["feature_names"]
    course_to_idx: Dict[str, int] = meta["course_to_idx"]
    mean = np.asarray(meta["standardizer"]["mean"], dtype=np.float32)
    std = np.asarray(meta["standardizer"]["std"], dtype=np.float32)

    # Resolve checkpoint: prefer explicit, else match meta['model_type'], else fallback
    ckpt_path: Path | None = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        mt = meta.get("model_type")
        expected = (
            "model_gru.pt"
            if mt == "gru"
            else (
                "model_lstm.pt"
                if mt == "lstm"
                else ("model_transformer.pt" if mt == "transformer" else "model_tcn.pt")
            )
        )
        p_expected = Path(args.model_dir) / expected
        if p_expected.exists():
            ckpt_path = p_expected
        else:
            for name in ["model_gru.pt", "model_lstm.pt", "model_transformer.pt", "model_tcn.pt"]:
                pth = Path(args.model_dir) / name
                if pth.exists():
                    ckpt_path = pth
                    print(f"[Warn] Expected {expected} but found {name}. Using {name}.")
                    break
    if ckpt_path is None:
        raise FileNotFoundError("No model checkpoint found (model_gru.pt/model_lstm.pt/model_transformer.pt/model_tcn.pt)")

    print(f"[Load Test] {args.test_csv}")
    df = pd.read_csv(args.test_csv, low_memory=False)
    df = prepare_dataframe(df, drop_leaky=True)
    # Demographics: reproduce same feature columns used in training if present
    demog_cols = meta.get("demog", {}).get("feature_cols", []) if isinstance(meta.get("demog"), dict) else []
    if demog_cols:
        try:
            demog_tbl = make_demog_features_infer(args.demographics_csv, df["username"].astype(str), demog_cols)
            df = df.merge(demog_tbl, on="username", how="left")
            missing = [c for c in demog_cols if c not in df.columns]
            for c in missing:
                df[c] = 0.0
            print(f"[Demog] Merged demographics with {len(demog_cols)} columns")
        except Exception as e:
            print(f"[Demog] Skipped demographics on inference due to error: {e}")
    # Build sequences without labels
    # Enforce same feature order as training (feature_names includes 'log_dt' at end)
    action_cols = [c for c in feature_names if c != "log_dt"]
    # Identify extra (demog) feature cols as those not starting with action_
    extra_cols = [c for c in action_cols if not c.startswith("action_")]
    action_only = [c for c in action_cols if c.startswith("action_")]
    items, _, _ = build_sequences_from_df(
        df,
        action_cols=action_only,
        label_col=None,
        course_to_idx=course_to_idx,
        extra_feature_cols=extra_cols if extra_cols else None,
    )
    apply_standardizer(items, mean, std)

    ds = PackedSequenceDataset(items)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_padded,
        pin_memory=(args.device == "cuda"),
    )

    # Infer model type from explicit ckpt name if provided, else use meta
    ckpt_model_type: str | None = None
    if args.ckpt:
        n = ckpt_path.name.lower()
        if "gru" in n:
            ckpt_model_type = "gru"
        elif "lstm" in n:
            ckpt_model_type = "lstm"
        elif "transformer" in n:
            ckpt_model_type = "transformer"
        elif "tcn" in n:
            ckpt_model_type = "tcn"

    model = build_model_from_meta(
        meta,
        input_dim=len(feature_names),
        course_vocab=len(course_to_idx),
        device=args.device,
        model_type=ckpt_model_type,
    )
    model_type_used = ckpt_model_type if ckpt_model_type is not None else meta.get("model_type")
    # Safer load when supported (PyTorch >=2.4): weights_only=True
    try:
        state = torch.load(ckpt_path, map_location=args.device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    preds = np.zeros(len(df), dtype=np.float32)

    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(args.device)
            dt = batch["dt"].to(args.device)
            mask = batch["mask"].to(args.device)
            course_ids = batch["course_ids"].to(args.device)
            if model_type_used in ("gru", "lstm", "tcn"):
                logits = model(x=x, mask=mask, course_ids=course_ids)
            else:
                logits = model(x=x, dt=dt, mask=mask, course_ids=course_ids)
            prob = torch.sigmoid(logits)  # (B,T)
            # Scatter back to original row order
            for i in range(prob.size(0)):
                ridx = batch["row_ids"][i].numpy()
                pvals = prob[i, : len(ridx)].detach().cpu().numpy().astype(np.float32)
                preds[ridx] = pvals

    # Write output aligned with original order
    out = df.copy()
    keep_cols = [c for c in ["username", "course_id", "session_id", "timestamp"] if c in out.columns]
    out = out[keep_cols]
    out["pred_dropout_prob"] = preds
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
