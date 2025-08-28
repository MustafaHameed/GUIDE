from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
import os, sys

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.seq_models import TimeAwareTransformer
from scripts.data_utils import (
    PackedSequenceDataset,
    apply_standardizer,
    build_sequences_from_df,
    fit_standardizer,
    prepare_dataframe,
    collate_padded,
    save_meta,
)


class MaskedStepModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, course_vocab: int, course_emb_dim: int, time_freqs: int):
        super().__init__()
        self.encoder = TimeAwareTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            course_vocab=course_vocab,
            course_emb_dim=course_emb_dim,
            time_freqs=time_freqs,
        )
        # Replace classification head with reconstruction head to predict input features
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor, course_ids: torch.Tensor) -> torch.Tensor:
        # Use internal encoder up to logits, then project to feature dim
        # We need to access the internal flow of TimeAwareTransformer; reuse forward then swap head
        # Re-implement minimal forward using its modules
        h = self.encoder.input_proj(x)
        pos = torch.arange(h.size(1), device=h.device, dtype=h.dtype).view(1, -1).expand(h.size(0), -1)
        tenc = self.encoder.time_enc(dt=dt, pos=pos)
        h = h + tenc
        if self.encoder.course_emb is not None and course_ids is not None:
            ce = self.encoder.course_emb(course_ids)
            h = h + ce.unsqueeze(1)
        key_padding_mask = (mask == 0) if mask is not None else None
        h = self.encoder.encoder(h, src_key_padding_mask=key_padding_mask)
        recon = self.recon_head(h)
        return recon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-supervised pretraining by masked step reconstruction (Transformer)")
    p.add_argument("--train_csv", default=str(Path("data/xuetangx/raw/Train.csv")))
    p.add_argument("--save_dir", default=str(Path("models/xuetangx_pretrain")))
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Model hparams
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--course_emb_dim", type=int, default=16)
    p.add_argument("--time_freqs", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Load and build sequences
    df = pd.read_csv(args.train_csv, low_memory=False)
    df = prepare_dataframe(df)
    items, feature_names, course_to_idx = build_sequences_from_df(df)
    del df
    # Standardize features
    mean, std = fit_standardizer(items)
    apply_standardizer(items, mean, std)

    ds = PackedSequenceDataset(items)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_padded,
        pin_memory=(args.device == "cuda"),
    )

    input_dim = len(feature_names)
    model = MaskedStepModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        course_vocab=len(course_to_idx),
        course_emb_dim=args.course_emb_dim,
        time_freqs=args.time_freqs,
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss(reduction="none")

    def train_epoch(ep: int) -> float:
        model.train()
        total = 0.0
        count = 0
        for batch in dl:
            x = batch["x"].to(args.device)
            dt = batch["dt"].to(args.device)
            mask = batch["mask"].to(args.device)
            course_ids = batch["course_ids"].to(args.device)
            B, T, F = x.shape
            # Create mask of timesteps to reconstruct: sample per-position bernoulli on valid positions
            bern = torch.rand(B, T, device=args.device)
            step_mask = (bern < args.mask_ratio).float() * mask  # (B,T)
            # Input corruption: zero out masked steps
            x_corrupt = x * (1.0 - step_mask.unsqueeze(-1))
            optim.zero_grad(set_to_none=True)
            recon = model(x_corrupt, dt, mask, course_ids)  # (B,T,F)
            mse = loss_fn(recon, x)  # (B,T,F)
            # Only count masked steps
            loss = (mse.mean(dim=-1) * step_mask).sum() / (step_mask.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.item()) * float(step_mask.sum().item())
            count += float(step_mask.sum().item())
        return total / max(count, 1.0)

    for ep in range(1, args.epochs + 1):
        l = train_epoch(ep)
        print(f"[Pretrain] epoch={ep} masked-MSE={l:.6f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save only encoder backbone weights (compatible with TimeAwareTransformer)
    torch.save(model.encoder.state_dict(), save_dir / "pretrained_encoder.pt")
    meta = {
        "feature_names": feature_names,
        "course_to_idx": course_to_idx,
        "standardizer": {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()},
        "hparams": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "course_emb_dim": args.course_emb_dim,
            "time_freqs": args.time_freqs,
        },
    }
    with open(save_dir / "pretrain_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] {save_dir / 'pretrained_encoder.pt'} and {save_dir / 'pretrain_meta.json'}")


if __name__ == "__main__":
    main()

