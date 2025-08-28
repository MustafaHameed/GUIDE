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

# Ensure local imports work when executed directly
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


def collect_probs(model: torch.nn.Module, model_type: str, dl: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect per-timestep labels, probabilities, and row_ids aligned to valid timesteps.

    Returns:
      y_true: (N,)
      y_prob: (N,)
      row_ids: (N,) integer indices referencing the temporary sub-DataFrame used to build sequences
    """
    y_true_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []
    row_ids_all: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            dt = batch["dt"].to(device)
            y = batch["y"].to(device) if batch["y"] is not None else None
            mask = batch["mask"].to(device)
            course_ids = batch["course_ids"].to(device)
            lengths = batch["lengths"].cpu().numpy()
            if model_type in ("gru", "lstm", "tcn"):
                logits = model(x=x, mask=mask, course_ids=course_ids)
            else:
                logits = model(x=x, dt=dt, mask=mask, course_ids=course_ids)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_np = y.cpu().numpy()
            # Gather per-sample valid prefix and associated row_ids
            for i in range(prob.shape[0]):
                L = int(lengths[i])
                if L <= 0:
                    continue
                y_true_all.append(y_np[i, :L].astype(np.float32))
                y_prob_all.append(prob[i, :L].astype(np.float32))
                ridx = batch["row_ids"][i]
                if ridx is not None:
                    row_ids_all.append(ridx.numpy()[:L].astype(np.int64))
                else:
                    # Fallback: create a dummy index range
                    if len(row_ids_all) == 0:
                        row_ids_all.append(np.arange(L, dtype=np.int64))
                    else:
                        row_ids_all.append(np.arange(row_ids_all[-1][-1] + 1, row_ids_all[-1][-1] + 1 + L, dtype=np.int64))
    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    row_ids = np.concatenate(row_ids_all, axis=0) if row_ids_all else np.arange(y_true.shape[0], dtype=np.int64)
    return y_true, y_prob, row_ids


def f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= thr).astype(np.float32)
    tp = float(np.sum((y_true > 0.5) & (y_pred > 0.5)))
    fp = float(np.sum((y_true <= 0.5) & (y_pred > 0.5)))
    fn = float(np.sum((y_true > 0.5) & (y_pred <= 0.5)))
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else (2 * tp) / denom


def youden_j(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= thr).astype(np.float32)
    tp = float(np.sum((y_true > 0.5) & (y_pred > 0.5)))
    tn = float(np.sum((y_true <= 0.5) & (y_pred <= 0.5)))
    fp = float(np.sum((y_true <= 0.5) & (y_pred > 0.5)))
    fn = float(np.sum((y_true > 0.5) & (y_pred <= 0.5)))
    tpr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    tnr = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
    return tpr + tnr - 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create risk list and choose operating threshold from validation")
    p.add_argument("--train_csv", default=str(Path("data/xuetangx/raw/Train.csv")))
    p.add_argument("--model_dir", default=str(Path("models/xuetangx")))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--criterion", choices=["f1", "youden"], default="f1")
    p.add_argument("--threshold_grouped", action="store_true", help="Choose threshold using grouped (username, course_id) averages on validation")
    p.add_argument("--grid", type=float, default=0.01, help="Threshold grid step in [0,1]")
    p.add_argument("--test_preds", default=str(Path("data/xuetangx/processed/Test_predictions_ensemble.csv")))
    p.add_argument("--out_risk", default=str(Path("data/xuetangx/processed/Test_risk_by_user_course.csv")))
    p.add_argument("--save_threshold", default="", help="Optional JSON path; defaults to model_dir/threshold.json")
    p.add_argument("--demographics_csv", default=str(Path("data/xuetangx/raw/user_info (1).csv")), help="Optional demographics CSV with 'username' and fields like sex, education_level, country, age")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load meta and (optional) ensemble weights
    meta = load_meta(str(Path(args.model_dir) / "meta.json"))
    feature_names: List[str] = meta["feature_names"]
    course_to_idx: Dict[str, int] = meta["course_to_idx"]
    mean = np.asarray(meta["standardizer"]["mean"], dtype=np.float32)
    std = np.asarray(meta["standardizer"]["std"], dtype=np.float32)

    ens_path = Path(args.model_dir) / "ensemble.json"
    weights = None
    names: List[str] = []
    if ens_path.exists():
        try:
            with open(ens_path, "r", encoding="utf-8") as f:
                ens = json.load(f)
            names = ens.get("names", [])
            wmap = ens.get("weights", {})
            if set(names) >= {"gru", "transformer"} and wmap:
                weights = np.array([wmap.get("gru", 0.5), wmap.get("transformer", 0.5)], dtype=np.float64)
        except Exception:
            pass
    if weights is None:
        weights = np.array([0.5, 0.5], dtype=np.float64)
        names = ["gru", "transformer"]

    # Build validation loader
    df = pd.read_csv(args.train_csv, low_memory=False)
    df = prepare_dataframe(df)
    # Merge demographics if available in meta
    demog_cols = meta.get("demog", {}).get("feature_cols", []) if isinstance(meta.get("demog"), dict) else []
    if demog_cols:
        try:
            demog_tbl = make_demog_features_infer(str(Path("data/xuetangx/raw/user_info (1).csv")), df["username"].astype(str), demog_cols)
            df = df.merge(demog_tbl, on="username", how="left")
            for c in demog_cols:
                if c not in df.columns:
                    df[c] = 0.0
        except Exception:
            pass
    action_cols_all = [c for c in feature_names if c != "log_dt"]
    extra_cols = [c for c in action_cols_all if not c.startswith("action_")]
    action_only = [c for c in action_cols_all if c.startswith("action_")]
    items, _, _ = build_sequences_from_df(
        df,
        action_cols=action_only,
        label_col="truth",
        course_to_idx=course_to_idx,
        extra_feature_cols=extra_cols if extra_cols else None,
    )
    _, val_items = split_items(items, val_ratio=args.val_ratio, seed=args.seed)
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

    # Available checkpoints
    ckpt_gru = Path(args.model_dir) / "model_gru.pt"
    ckpt_lstm = Path(args.model_dir) / "model_lstm.pt"
    ckpt_tr = Path(args.model_dir) / "model_transformer.pt"
    avail: List[Tuple[str, Path]] = []
    if ckpt_gru.exists():
        avail.append(("gru", ckpt_gru))
    if ckpt_lstm.exists():
        avail.append(("lstm", ckpt_lstm))
    if ckpt_tr.exists():
        avail.append(("transformer", ckpt_tr))
    if not avail:
        raise FileNotFoundError("No checkpoints found in model_dir")

    # Collect validation probabilities and ensemble
    y_true_ref: Optional[np.ndarray] = None
    probs: Dict[str, np.ndarray] = {}
    # Build a map from row_id -> (username, course_id) for grouping later
    rowid_to_group: Dict[int, Tuple[str, str]] = {}
    for it in items:
        if it.row_ids is None:
            continue
        for rid in it.row_ids.tolist():
            rowid_to_group[int(rid)] = (it.username, it.course_id)

    for mtype, ckpt in avail:
        model = build_model_from_meta(meta, mtype, input_dim=len(feature_names), course_vocab=len(course_to_idx), device=args.device)
        try:
            state = torch.load(ckpt, map_location=args.device, weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            state = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(state)
        y_true, y_prob, row_ids = collect_probs(model, mtype, val_dl, args.device)
        y_true_ref = y_true if y_true_ref is None else y_true_ref
        probs[mtype] = y_prob

    y_true = y_true_ref if y_true_ref is not None else np.zeros(1, dtype=np.float32)
    # Build ensemble vector in the order [gru, transformer] when available
    comp: List[np.ndarray] = []
    wvec: List[float] = []
    if "gru" in probs:
        comp.append(probs["gru"]) ; wvec.append(float(weights[0]))
    if "transformer" in probs:
        comp.append(probs["transformer"]) ; wvec.append(float(weights[1]))
    if len(comp) == 1:
        y_prob_ens = comp[0]
    else:
        W = np.array(wvec, dtype=np.float64)
        W = W / W.sum()
        y_prob_ens = np.clip(np.sum(np.stack(comp, axis=1) * W.reshape(1, -1), axis=1), 0.0, 1.0)

    # Search threshold (per-row or grouped)
    grid = np.linspace(0.0, 1.0, int(1.0 / max(args.grid, 1e-6)) + 1)
    best_thr = 0.5
    best_score = -1.0

    if args.threshold_grouped:
        # Aggregate by (username, course_id)
        sums_p: Dict[Tuple[str, str], float] = {}
        sums_y: Dict[Tuple[str, str], float] = {}
        counts: Dict[Tuple[str, str], int] = {}
        for rid, yt, yp in zip(row_ids.tolist(), y_true.tolist(), y_prob_ens.tolist()):
            grp = rowid_to_group.get(int(rid))
            if grp is None:
                continue
            sums_p[grp] = sums_p.get(grp, 0.0) + float(yp)
            sums_y[grp] = sums_y.get(grp, 0.0) + float(yt)
            counts[grp] = counts.get(grp, 0) + 1
        if not counts:
            raise RuntimeError("No groups available for grouped thresholding")
        prob_avg = np.array([sums_p[g] / counts[g] for g in counts.keys()], dtype=np.float32)
        y_avg = np.array([sums_y[g] / counts[g] for g in counts.keys()], dtype=np.float32)
        y_bin = (y_avg >= 0.5).astype(np.float32)
        for thr in grid:
            if args.criterion == "f1":
                s = f1_at_threshold(y_bin, prob_avg, thr)
            else:
                s = youden_j(y_bin, prob_avg, thr)
            if s > best_score:
                best_score = s
                best_thr = float(thr)
    else:
        for thr in grid:
            if args.criterion == "f1":
                s = f1_at_threshold(y_true, y_prob_ens, thr)
            else:
                s = youden_j(y_true, y_prob_ens, thr)
            if s > best_score:
                best_score = s
                best_thr = float(thr)
    # Save threshold JSON
    thr_path = Path(args.save_threshold) if args.save_threshold else (Path(args.model_dir) / "threshold.json")
    thr_payload = {
        "threshold": best_thr,
        "criterion": args.criterion,
        "grid": args.grid,
        "weights": {"gru": float(weights[0]), "transformer": float(weights[1])},
        "note": "Apply to per-row or per-user-course averaged probs",
    }
    thr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thr_payload, f, indent=2)
    print(f"[Threshold] {best_thr:.3f} ({args.criterion}, score={best_score:.4f}) -> saved {thr_path}")

    # Build risk list from test predictions (ensemble by default)
    test_path = Path(args.test_preds)
    if not test_path.exists():
        # fallback to the GRU file name used earlier
        alt = Path("data/xuetangx/processed/Test_predictions.csv")
        if alt.exists():
            test_path = alt
        else:
            raise FileNotFoundError(f"Test predictions not found: {args.test_preds}")

    tdf = pd.read_csv(test_path, low_memory=False)
    if "pred_dropout_prob" not in tdf.columns:
        raise ValueError(f"Missing 'pred_dropout_prob' in {test_path}")
    keys = [k for k in ["username", "course_id"] if k in tdf.columns]
    if not keys:
        raise ValueError("Predictions file lacks grouping keys (username, course_id)")
    agg = (
        tdf.groupby(keys, dropna=False)["pred_dropout_prob"]
        .agg(prob_avg="mean", prob_max="max", n_rows="count")
        .reset_index()
    )
    agg["high_risk"] = (agg["prob_avg"] >= best_thr).astype(int)
    out_path = Path(args.out_risk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"[Saved] {out_path} rows={len(agg):,}")


if __name__ == "__main__":
    main()
