from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local imports
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
import sys

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.seq_models import GRUClassifier, TimeAwareTransformer, TCNClassifier, LSTMClassifier
from scripts.data_utils import (
    PackedSequenceDataset,
    apply_standardizer,
    build_sequences_from_df,
    load_meta,
    prepare_dataframe,
    split_items,
    collate_padded,
    make_demog_features_infer,
)
from scripts.metrics import average_precision, roc_auc, log_loss


def build_model_from_meta(meta: Dict[str, object], model_type: str, input_dim: int, course_vocab: int, device: str) -> torch.nn.Module:
    h = meta.get("hparams", {})
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


def collect_probs(
    model: torch.nn.Module,
    model_type: str,
    dl: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            dt = batch["dt"].to(device)
            y = batch["y"].to(device) if batch["y"] is not None else None
            mask = batch["mask"].to(device)
            course_ids = batch["course_ids"].to(device)
            if model_type in ("gru", "lstm", "tcn"):
                logits = model(x=x, mask=mask, course_ids=course_ids)
            else:
                logits = model(x=x, dt=dt, mask=mask, course_ids=course_ids)
            prob = torch.sigmoid(logits)
            valid = (mask > 0.5)
            y_true_all.append((y[valid]).detach().cpu().numpy().astype(np.float32))
            y_prob_all.append((prob[valid]).detach().cpu().numpy().astype(np.float32))
    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    return y_true, y_prob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GRU/Transformer and ensemble on validation split; pick ensemble weights")
    p.add_argument("--train_csv", default=str(Path("data/xuetangx/raw/Train.csv")))
    p.add_argument("--model_dir", default=str(Path("models/xuetangx")))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=["auprc", "logloss"], default="auprc")
    p.add_argument("--grid", type=float, default=0.05, help="Weight grid step for ensemble (0..1)")
    p.add_argument("--save_json", default="", help="Optional path to save chosen weights (defaults to model_dir/ensemble.json)")
    p.add_argument("--save_val_preds", default="", help="Optional CSV path to save validation probs for each model")
    p.add_argument("--demographics_csv", default=str(Path("data/xuetangx/raw/user_info (1).csv")), help="Optional demographics CSV with 'username' and fields like sex, education_level, country, age")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = Path(args.model_dir) / "meta.json"
    meta = load_meta(str(meta_path))
    feature_names: List[str] = meta["feature_names"]
    course_to_idx: Dict[str, int] = meta["course_to_idx"]
    mean = np.asarray(meta["standardizer"]["mean"], dtype=np.float32)
    std = np.asarray(meta["standardizer"]["std"], dtype=np.float32)

    # Load and build sequences; split
    df = pd.read_csv(args.train_csv, low_memory=False)
    df = prepare_dataframe(df)
    # Demographics merge if present in meta
    demog_cols = meta.get("demog", {}).get("feature_cols", []) if isinstance(meta.get("demog"), dict) else []
    if demog_cols:
        try:
            demog_tbl = make_demog_features_infer(args.demographics_csv, df["username"].astype(str), demog_cols)
            df["username"] = df["username"].astype(str)  # Ensure consistent string type for merge
            df = df.merge(demog_tbl, on="username", how="left")
            for c in demog_cols:
                if c not in df.columns:
                    df[c] = 0.0
        except Exception:
            pass
    action_cols = [c for c in feature_names if c != "log_dt"]
    extra_cols = [c for c in action_cols if not c.startswith("action_")]
    action_only = [c for c in action_cols if c.startswith("action_")]
    items, _, _ = build_sequences_from_df(df, action_cols=action_only, label_col="truth", course_to_idx=course_to_idx, extra_feature_cols=extra_cols if extra_cols else None)
    train_items, val_items = split_items(items, val_ratio=args.val_ratio, seed=args.seed)
    apply_standardizer(val_items, mean, std)

    val_ds = PackedSequenceDataset(val_items)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_padded,
        pin_memory=(args.device == "cuda"),
    )

    # Evaluate available models
    ckpts = []
    ckpt_gru = Path(args.model_dir) / "model_gru.pt"
    ckpt_lstm = Path(args.model_dir) / "model_lstm.pt"
    ckpt_tr = Path(args.model_dir) / "model_transformer.pt"
    ckpt_tcn = Path(args.model_dir) / "model_tcn.pt"
    if ckpt_gru.exists():
        ckpts.append(("gru", ckpt_gru))
    if ckpt_lstm.exists():
        ckpts.append(("lstm", ckpt_lstm))
    if ckpt_tr.exists():
        ckpts.append(("transformer", ckpt_tr))
    if ckpt_tcn.exists():
        ckpts.append(("tcn", ckpt_tcn))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in model_dir")

    per_model: Dict[str, Dict[str, float]] = {}
    probs_map: Dict[str, np.ndarray] = {}
    y_true_ref: Optional[np.ndarray] = None
    for mtype, ckpt in ckpts:
        model = build_model_from_meta(meta, mtype, input_dim=len(feature_names), course_vocab=len(course_to_idx), device=args.device)
        try:
            state = torch.load(ckpt, map_location=args.device, weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            state = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(state)
        y_true, y_prob = collect_probs(model, mtype, val_dl, args.device)
        y_true_ref = y_true if y_true_ref is None else y_true_ref
        assert y_true_ref is not None and np.array_equal(y_true_ref, y_true), "y alignment mismatch"
        probs_map[mtype] = y_prob
        per_model[mtype] = {
            "auprc": average_precision(y_true, y_prob),
            "roc_auc": roc_auc(y_true, y_prob),
            "logloss": log_loss(y_true, y_prob),
        }
        print(f"[{mtype}] AUPRC={per_model[mtype]['auprc']:.4f} ROC-AUC={per_model[mtype]['roc_auc']:.4f} LogLoss={per_model[mtype]['logloss']:.4f}")

    # Ensemble grid search on simplex for K models (K>=2). Uses step=args.grid.
    names = [m for m, _ in ckpts]
    best_metric = -float("inf") if args.metric == "auprc" else float("inf")
    best_weights_vec: np.ndarray | None = None
    if len(names) >= 2:
        y_true = y_true_ref  # type: ignore
        # Stack probs matrix: (N,K)
        P = np.stack([probs_map[n] for n in names], axis=1).astype(np.float64)
        # Normalize grid so it partitions [0,1] exactly
        step_raw = max(args.grid, 1e-6)
        n_steps = int(round(1.0 / step_raw))
        step = 1.0 / max(n_steps, 1)

        K = len(names)
        def eval_weights(w: np.ndarray) -> float:
            w = w / max(w.sum(), 1e-12)
            pe = (P * w.reshape(1, -1)).sum(axis=1)
            if args.metric == "auprc":
                return average_precision(y_true, pe)
            else:
                return -log_loss(y_true, pe)  # negate to keep maximize convention

        if K == 2:
            for i in range(n_steps + 1):
                w0 = i * step
                w = np.array([w0, 1.0 - w0], dtype=np.float64)
                m = eval_weights(w)
                if m > best_metric:
                    best_metric = m
                    best_weights_vec = w
        elif K == 3:
            for i in range(n_steps + 1):
                w0 = i * step
                rem1 = 1.0 - w0
                sub_steps = int(round(rem1 / step))
                for j in range(sub_steps + 1):
                    w1 = j * step
                    w2 = 1.0 - (w0 + w1)
                    if w2 < -1e-9:
                        continue
                    w = np.array([w0, w1, max(w2, 0.0)], dtype=np.float64)
                    m = eval_weights(w)
                    if m > best_metric:
                        best_metric = m
                        best_weights_vec = w
        else:  # K >= 4 (search first 3, assign remainder to last)
            for i in range(n_steps + 1):
                w0 = i * step
                rem1 = 1.0 - w0
                s1 = int(round(rem1 / step))
                for j in range(s1 + 1):
                    w1 = j * step
                    rem2 = 1.0 - (w0 + w1)
                    if rem2 < -1e-12:
                        continue
                    s2 = int(round(max(rem2, 0.0) / step))
                    for k in range(s2 + 1):
                        w2 = k * step
                        w_rest = 1.0 - (w0 + w1 + w2)
                        if w_rest < -1e-9:
                            continue
                        # Distribute remainder entirely to last component; remaining (if any) to zeros of middle if K>4
                        w = np.zeros(K, dtype=np.float64)
                        w[0] = w0
                        w[1] = w1
                        w[2] = w2
                        w[3] = max(w_rest, 0.0)
                        m = eval_weights(w)
                        if m > best_metric:
                            best_metric = m
                            best_weights_vec = w
        # Convert best_metric back if logloss (we maximized negative logloss)
        final_metric = best_metric if args.metric == "auprc" else -best_metric
        if best_weights_vec is None:
            print("[Ensemble] Grid search failed to find weights; defaulting to equal")
            best_weights_vec = np.ones(len(names), dtype=np.float64) / len(names)
        print(f"[Ensemble] best weights {dict(zip(names, np.round(best_weights_vec, 3)))} metric({args.metric})={final_metric:.4f}")
        best_metric = final_metric
    else:
        print("[Ensemble] Only one model available; skipping weight search")

    # Save JSON
    if args.save_json == "":
        out_json = Path(args.model_dir) / "ensemble.json"
    else:
        out_json = Path(args.save_json)
    payload = {
        "per_model": per_model,
        "metric": args.metric,
        "best_metric": best_metric,
        "weights": None,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "note": "weights is a mapping over model names that sum to 1",
        "names": names,
    }
    if len(names) >= 2 and best_weights_vec is not None:
        payload["weights"] = {n: float(w) for n, w in zip(names, (best_weights_vec / best_weights_vec.sum()))}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {out_json}")

    # Optional save validation probs
    if args.save_val_preds:
        df_out = pd.DataFrame({"y_true": y_true_ref})
        for name in names:
            df_out[f"prob_{name}"] = probs_map[name]
        df_out.to_csv(args.save_val_preds, index=False)
        print(f"[Saved] {args.save_val_preds}")


if __name__ == "__main__":
    main()
