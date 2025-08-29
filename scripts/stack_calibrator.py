from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a logistic stacking calibrator on validation predictions")
    p.add_argument("--val_preds", required=True, help="CSV with columns y_true and prob_* from eval_models.py --save_val_preds")
    p.add_argument("--out", default=str(Path("models/xuetangx/calibrator.json")))
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3, help="L2 regularization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.val_preds)
    if "y_true" not in df.columns:
        raise ValueError("val_preds must contain y_true column")
    cols = [c for c in df.columns if c.startswith("prob_")]
    if not cols:
        raise ValueError("val_preds must contain one or more prob_* columns")
    names = [c[len("prob_") :] for c in cols]
    y = torch.tensor(df["y_true"].to_numpy(dtype=np.float32))
    X = torch.tensor(df[cols].to_numpy(dtype=np.float32))
    # Add bias in parameter vector
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=args.lr)
    for ep in range(args.epochs):
        opt.zero_grad()
        logits = X.matmul(w) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + args.l2 * (w.pow(2).sum())
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                # simple logloss metric
                ll = -(y * (p + 1e-7).log() + (1 - y) * (1 - p + 1e-7).log()).mean()
                print(f"[Cal] ep={ep+1} loss={float(loss):.5f} logloss={float(ll):.5f}")

    payload = {
        "names": names,
        "weights": w.detach().cpu().numpy().tolist(),
        "bias": float(b.detach().cpu().item()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()

