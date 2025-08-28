from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.seq_models import GRUClassifier, LSTMClassifier, TimeAwareTransformer, TCNClassifier
from scripts.data_utils import (
    PackedSequenceDataset,
    apply_standardizer,
    build_sequences_from_df,
    load_meta,
    prepare_dataframe,
    collate_padded,
    truncate_items,
)
from scripts.metrics import average_precision, roc_auc, log_loss


def build_model_from_meta(meta: Dict[str, object], input_dim: int, course_vocab: int, device: str) -> torch.nn.Module:
    h = meta.get("hparams", {})
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
    else:
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


def eval_prefixes(model_dir: str, train_csv: str, prefixes: List[int], device: str, batch_size: int, num_workers: int) -> pd.DataFrame:
    meta_path = Path(model_dir) / "meta.json"
    meta = load_meta(str(meta_path))
    feature_names: List[str] = meta["feature_names"]
    course_to_idx: Dict[str, int] = meta["course_to_idx"]
    mean = np.asarray(meta["standardizer"]["mean"], dtype=np.float32)
    std = np.asarray(meta["standardizer"]["std"], dtype=np.float32)

    # Load data and build sequences
    df = pd.read_csv(train_csv, low_memory=False)
    df = prepare_dataframe(df)
    items, _, _ = build_sequences_from_df(df, action_cols=[c for c in feature_names if c != "log_dt"], label_col="truth", course_to_idx=course_to_idx)
    apply_standardizer(items, mean, std)

    # Load model
    model = build_model_from_meta(meta, input_dim=len(feature_names), course_vocab=len(course_to_idx), device=device)
    ckpt = None
    mt = meta.get("model_type")
    expected = (
        "model_gru.pt" if mt == "gru" else ("model_lstm.pt" if mt == "lstm" else ("model_transformer.pt" if mt == "transformer" else "model_tcn.pt"))
    )
    p_expected = Path(model_dir) / expected
    if p_expected.exists():
        ckpt = p_expected
    else:
        for name in ["model_gru.pt", "model_lstm.pt", "model_transformer.pt", "model_tcn.pt"]:
            pth = Path(model_dir) / name
            if pth.exists():
                ckpt = pth
                break
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found in model_dir")
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    for K in prefixes:
        items_K = truncate_items(items, max_steps=K)
        ds = PackedSequenceDataset(items_K)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_padded,
            pin_memory=(device == "cuda"),
        )
        y_true_all, y_prob_all = [], []
        with torch.no_grad():
            for batch in dl:
                x = batch["x"].to(device)
                dt = batch["dt"].to(device)
                y = batch["y"].to(device)
                mask = batch["mask"].to(device)
                course_ids = batch["course_ids"].to(device)
                logits = (
                    model(x=x, mask=mask, course_ids=course_ids)
                    if meta["model_type"] in ("gru", "lstm", "tcn")
                    else model(x=x, dt=dt, mask=mask, course_ids=course_ids)
                )
                prob = torch.sigmoid(logits)
                valid = (mask > 0.5)
                y_true_all.append((y[valid]).detach().cpu().numpy().astype(np.float32))
                y_prob_all.append((prob[valid]).detach().cpu().numpy().astype(np.float32))
        y_true = np.concatenate(y_true_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)
        ap = average_precision(y_true, y_prob)
        roc = roc_auc(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        rows.append({"prefix_steps": K, "auprc": ap, "roc_auc": roc, "logloss": ll})

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate early-warning performance vs. prefix length")
    p.add_argument("--train_csv", default=str(Path("data/xuetangx/raw/Train.csv")))
    p.add_argument("--model_dir", default=str(Path("models/xuetangx")))
    p.add_argument("--prefixes", nargs="*", type=int, default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_csv", default=str(Path("models/xuetangx/early_warning_curve.csv")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = eval_prefixes(args.model_dir, args.train_csv, args.prefixes, args.device, args.batch_size, args.num_workers)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("[Saved]", out_path)
    print(df)


if __name__ == "__main__":
    main()

