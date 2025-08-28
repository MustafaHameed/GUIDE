from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make repo root importable when running as a script (python scripts/train_dropout.py)
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.seq_models import (
    GRUClassifier,
    TimeAwareTransformer,
    TCNClassifier,
    LSTMClassifier,
    masked_bce_with_logits,
)
from scripts.data_utils import (
    apply_standardizer,
    build_sequences_from_df,
    fit_standardizer,
    prepare_dataframe,
    split_items,
)


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_type: str, input_dim: int, course_vocab: int, args: argparse.Namespace) -> torch.nn.Module:
    if model_type == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            course_vocab=course_vocab,
            course_emb_dim=args.course_emb_dim,
        )
    elif model_type == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            course_vocab=course_vocab,
            course_emb_dim=args.course_emb_dim,
        )
    elif model_type == "transformer":
        return TimeAwareTransformer(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            course_vocab=course_vocab,
            course_emb_dim=args.course_emb_dim,
            time_freqs=args.time_freqs,
        )
    elif model_type == "tcn":
        return TCNClassifier(
            input_dim=input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            kernel_size=args.tcn_kernel,
            dropout=args.dropout,
            course_vocab=course_vocab,
            course_emb_dim=args.course_emb_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train GRU/Transformer/TCN baselines for XuetangX dropout prediction (raw)")
    p.add_argument("--train_csv", default=str(Path("data/xuetangx/raw/Train.csv")), help="Path to Train.csv")
    p.add_argument("--save_dir", default=str(Path("models/xuetangx")), help="Directory to save model and meta")
    p.add_argument("--model", choices=["gru", "lstm", "transformer", "tcn"], default="gru")

    # Model hyperparams
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--course_emb_dim", type=int, default=16)
    p.add_argument("--time_freqs", type=int, default=8)
    p.add_argument("--tcn_kernel", type=int, default=3)

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Imbalance + early stopping
    p.add_argument("--loss", choices=["bce", "weighted_bce", "focal"], default="bce")
    p.add_argument("--pos_weight", default="auto", help='For weighted_bce: "auto" or a float like 3.5')
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--early_stop_metric", choices=["val_loss", "auprc"], default="auprc")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=0.0)

    # Progress logging
    p.add_argument("--log_every", type=int, default=0, help="Log every N training batches (0=epoch only)")
    p.add_argument("--build_progress", action="store_true", help="Show progress while building sequences")
    p.add_argument("--build_log_every", type=int, default=2000, help="Groups per progress print while building")
    # Pretraining init (Transformer only)
    p.add_argument("--init_from_pretrained", default="", help="Optional path to a pretrained state_dict (Transformer encoder); loaded with strict=False")

    args = p.parse_args()
    set_seed(args.seed)

    print(f"[Load] {args.train_csv}")
    df = pd.read_csv(args.train_csv, low_memory=False)
    df = prepare_dataframe(df)

    # Build sequences and detect action columns dynamically
    items, feature_names, course_to_idx = build_sequences_from_df(
        df,
        progress=args.build_progress,
        log_every=args.build_log_every,
    )
    del df
    print(f"[Info] Groups: {len(items)} | Features: {len(feature_names)} | Courses: {len(course_to_idx)}")

    train_items, val_items = split_items(items, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[Split] Train groups: {len(train_items)} | Val groups: {len(val_items)}")

    # Fit standardizer on training only
    mean, std = fit_standardizer(train_items)
    apply_standardizer(train_items, mean, std)
    apply_standardizer(val_items, mean, std)

    # Datasets / loaders
    from scripts.data_utils import PackedSequenceDataset, collate_padded

    train_ds = PackedSequenceDataset(train_items)
    val_ds = PackedSequenceDataset(val_items)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_padded,
        pin_memory=(args.device == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_padded,
        pin_memory=(args.device == "cuda"),
    )

    # Model
    input_dim = len(feature_names)
    model = build_model(args.model, input_dim, course_vocab=len(course_to_idx), args=args)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optional: initialize from pretrained encoder (Transformer)
    if args.init_from_pretrained:
        if args.model != "transformer":
            print(f"[Init] --init_from_pretrained is intended for transformer; ignoring for {args.model}")
        else:
            pp = Path(args.init_from_pretrained)
            if not pp.exists():
                print(f"[Init] Pretrained path not found: {pp}")
            else:
                try:
                    state = torch.load(pp, map_location=args.device)
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    print(f"[Init] Loaded pretrained with strict=False. Missing keys={len(missing)} unexpected={len(unexpected)}")
                except Exception as e:
                    print(f"[Init] Failed to load pretrained: {e}")

    # Compute class weight if requested
    pos_weight_value: Optional[float] = None
    if args.loss == "weighted_bce":
        if isinstance(args.pos_weight, str) and args.pos_weight.lower() == "auto":
            pos_count = 0.0
            neg_count = 0.0
            for it in train_items:
                if it.y is None:
                    continue
                y = it.y.astype(np.float32)
                pos_count += float(y.sum())
                neg_count += float(y.shape[0] - y.sum())
            if pos_count <= 0:
                pos_weight_value = 1.0
            else:
                pos_weight_value = max(neg_count / max(pos_count, 1e-6), 1.0)
            print(f"[ClassWeight] auto pos_weight=neg/pos={neg_count:.0f}/{pos_count:.0f} -> {pos_weight_value:.4f}")
        else:
            try:
                pos_weight_value = float(args.pos_weight)
            except Exception:
                raise ValueError("--pos_weight must be 'auto' or a float value")
            print(f"[ClassWeight] manual pos_weight={pos_weight_value}")

    def masked_focal_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
        # logits, targets, mask: (B,T)
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        # Numerically stable CE
        ce_pos = torch.nn.functional.softplus(-logits)  # = -log(sigmoid)
        ce_neg = torch.nn.functional.softplus(logits)   # = -log(1 - sigmoid)
        ce = targets * ce_pos + (1 - targets) * ce_neg
        # Focal modulation and alpha weighting
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        loss = (alpha_t * (1 - pt) ** gamma) * ce
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def average_precision_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
        # Computes Average Precision (area under PR curve) without sklearn
        if y_true.ndim != 1:
            y_true = y_true.ravel()
        if y_score.ndim != 1:
            y_score = y_score.ravel()
        pos_total = float((y_true > 0.5).sum())
        if pos_total == 0:
            return 0.0
        order = np.argsort(-y_score, kind="mergesort")
        y_sorted = (y_true[order] > 0.5).astype(np.float32)
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1.0 - y_sorted)
        precision = tp / np.maximum(tp + fp, 1e-12)
        # AP is mean precision at ranks of positives
        ap = (precision[y_sorted == 1].sum()) / pos_total
        return float(ap)

    def run_epoch(split: str) -> Dict[str, float]:
        is_train = split == "train"
        model.train(is_train)
        total_loss = 0.0
        total_count = 0
        total_correct = 0
        total_labels = 0
        y_true_all: list[np.ndarray] = []
        y_score_all: list[np.ndarray] = []
        loader = train_dl if is_train else val_dl
        n_batches = len(loader)
        start_time = time.time()
        for batch in loader:
            step_start = time.time()
            x = batch["x"].to(args.device)
            dt = batch["dt"].to(args.device)
            y = batch["y"].to(args.device) if batch["y"] is not None else None
            mask = batch["mask"].to(args.device)
            course_ids = batch["course_ids"].to(args.device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if args.model in ("gru", "lstm", "tcn"):
                logits = model(x=x, mask=mask, course_ids=course_ids)
            else:
                logits = model(x=x, dt=dt, mask=mask, course_ids=course_ids)

            if args.loss == "bce":
                loss = masked_bce_with_logits(logits, y, mask)
            elif args.loss == "weighted_bce":
                loss = masked_bce_with_logits(logits, y, mask, pos_weight=pos_weight_value)
            else:  # focal
                loss = masked_focal_loss(logits, y, mask, alpha=args.focal_alpha, gamma=args.focal_gamma)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                total_loss += float(loss.item()) * mask.sum().item()
                total_count += mask.sum().item()
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct = ((preds == (y >= 0.5).float()) * mask).sum().item()
                total_correct += correct
                total_labels += mask.sum().item()
                if not is_train:
                    valid = (mask > 0.5)
                    y_true_all.append((y[valid]).detach().cpu().numpy().astype(np.float32))
                    y_score_all.append((probs[valid]).detach().cpu().numpy().astype(np.float32))

            # Logging per batch
            if is_train and args.log_every and (total_labels // 1) >= 0:
                # step index from loop counter by checking progress on dataloader
                # We can't directly access loop index here without enumerate; switch to enumerate for accurate step
                pass

        avg_loss = total_loss / max(total_count, 1)
        acc = total_correct / max(total_labels, 1)
        metrics = {"loss": avg_loss, "acc": acc}
        if not is_train and len(y_true_all) > 0:
            y_true_np = np.concatenate(y_true_all, axis=0)
            y_score_np = np.concatenate(y_score_all, axis=0)
            ap = average_precision_from_scores(y_true_np, y_score_np)
            metrics["auprc"] = ap
        return metrics

    best_state = None
    best_metric = -float("inf") if args.early_stop_metric == "auprc" else float("inf")
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        # Train with per-batch logging if requested
        if args.log_every:
            is_train = True
            model.train(True)
            total_loss = 0.0
            total_count = 0
            total_correct = 0
            total_labels = 0
            start_time = time.time()
            n_batches = len(train_dl)
            for i, batch in enumerate(train_dl, start=1):
                x = batch["x"].to(args.device)
                dt = batch["dt"].to(args.device)
                y = batch["y"].to(args.device)
                mask = batch["mask"].to(args.device)
                course_ids = batch["course_ids"].to(args.device)
                optimizer.zero_grad(set_to_none=True)
                if args.model in ("gru", "lstm", "tcn"):
                    logits = model(x=x, mask=mask, course_ids=course_ids)
                else:
                    logits = model(x=x, dt=dt, mask=mask, course_ids=course_ids)
                if args.loss == "bce":
                    loss = masked_bce_with_logits(logits, y, mask)
                elif args.loss == "weighted_bce":
                    loss = masked_bce_with_logits(logits, y, mask, pos_weight=pos_weight_value)
                else:
                    loss = masked_focal_loss(logits, y, mask, alpha=args.focal_alpha, gamma=args.focal_gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    total_loss += float(loss.item()) * mask.sum().item()
                    total_count += mask.sum().item()
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    correct = ((preds == (y >= 0.5).float()) * mask).sum().item()
                    total_correct += correct
                    total_labels += mask.sum().item()

                if i % args.log_every == 0 or i == n_batches:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / max(total_count, 1)
                    acc = total_correct / max(total_labels, 1)
                    speed = i / max(elapsed, 1e-9)
                    eta = (n_batches - i) / max(speed, 1e-9)
                    print(f"[Epoch {epoch:02d}][{i}/{n_batches}] loss={avg_loss:.4f} acc={acc:.4f} speed={speed:.1f} it/s eta={eta/60:.1f}m", flush=True)
            tr = {"loss": total_loss / max(total_count, 1), "acc": total_correct / max(total_labels, 1)}
        else:
            tr = run_epoch("train")
        va = run_epoch("val")
        msg = f"[Epoch {epoch:02d}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | val loss={va['loss']:.4f} acc={va['acc']:.4f}"
        if "auprc" in va:
            msg += f" auprc={va['auprc']:.4f}"
        print(msg)

        # Select metric
        current = va["loss"] if args.early_stop_metric == "val_loss" else va.get("auprc", float("nan"))
        if args.early_stop_metric == "auprc":
            is_better = (current > best_metric + args.min_delta)
        else:
            is_better = (current < best_metric - args.min_delta)

        if is_better:
            best_metric = current
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[EarlyStop] No improvement in {args.patience} epochs on {args.early_stop_metric}. Stopping.")
                break

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"model_{args.model}.pt"
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, ckpt_path)

    meta = {
        "model_type": args.model,
        "feature_names": feature_names,
        "course_to_idx": course_to_idx,
        "standardizer": {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()},
        "hparams": {
            "hidden_dim": args.hidden_dim,
            "d_model": args.d_model,
            "dim_feedforward": args.dim_feedforward,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "course_emb_dim": args.course_emb_dim,
            "time_freqs": args.time_freqs,
            "tcn_kernel": args.tcn_kernel,
        },
    }
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] {ckpt_path} and {save_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
