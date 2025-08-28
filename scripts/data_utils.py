from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ACTION_PREFIX = "action_"


def detect_action_columns(columns: Sequence[str]) -> List[str]:
    cols = [c for c in columns if c.startswith(ACTION_PREFIX)]
    # Deterministic order
    return sorted(cols)


def prepare_dataframe(df: pd.DataFrame, drop_leaky: bool = True) -> pd.DataFrame:
    df = df.copy()
    # Required columns guard
    required = {"username", "course_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.float64)

    # Drop known leaky/global aggregate columns if present
    if drop_leaky:
        for col in ["unique_session_count", "avg_nActions_per_session"]:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Sort to define sequence order
    df = df.sort_values(["username", "course_id", "timestamp"], kind="mergesort")
    return df


@dataclass
class SequenceItem:
    username: str
    course_id: str
    course_idx: int
    x: np.ndarray  # (T, F)
    dt: np.ndarray  # (T,)
    y: Optional[np.ndarray]  # (T,) or None for test
    row_ids: Optional[np.ndarray]  # indices in original file order for reconstructing predictions


def build_sequences_from_df(
    df: pd.DataFrame,
    action_cols: Optional[List[str]] = None,
    label_col: Optional[str] = "truth",
    course_to_idx: Optional[Dict[str, int]] = None,
    progress: bool = False,
    log_every: int = 2000,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[List[SequenceItem], List[str], Dict[str, int]]:
    """Group by (username, course_id) and build sequences.

    Returns: (items, feature_names, course_to_idx)
    Feature construction: log1p of action counts; adds log1p(delta_t) as last feature.
    """
    if action_cols is None:
        action_cols = detect_action_columns(df.columns)
    used_cols = ["username", "course_id", "timestamp"] + action_cols
    if label_col is not None and label_col in df.columns:
        used_cols += [label_col]
    if "session_id" in df.columns:
        used_cols += ["session_id"]

    sub = df[used_cols].copy()
    # Keep original row order id to reconstruct test predictions
    sub["__row_id__"] = np.arange(len(sub))

    # Course id mapping
    if course_to_idx is None:
        unique_courses = sub["course_id"].astype(str).unique().tolist()
        course_to_idx = {c: i for i, c in enumerate(sorted(unique_courses))}

    feature_names = action_cols + ["log_dt"]

    items: List[SequenceItem] = []
    gb = sub.groupby(["username", "course_id"], sort=False)
    total_groups: Optional[int] = None
    if progress:
        try:
            total_groups = gb.ngroups
        except Exception:
            total_groups = None
    printer: Callable[[str], None] = (logger if logger is not None else (lambda s: print(s, flush=True)))

    for idx, ((user, course), g) in enumerate(gb, start=1):
        g = g.sort_values("timestamp", kind="mergesort")
        # Build dt
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        dt = np.zeros_like(ts)
        if len(ts) > 1:
            dt[1:] = np.diff(ts)
            # Clamp negatives to zero
            dt = np.maximum(dt, 0)
        # Features
        X = g[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        # log1p counts
        X = np.log1p(X)
        log_dt = np.log1p(dt.astype(np.float32)).reshape(-1, 1)
        X = np.concatenate([X, log_dt], axis=1)

        y = None
        if label_col is not None and label_col in g.columns:
            y = pd.to_numeric(g[label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        row_ids = g["__row_id__"].to_numpy(dtype=np.int64)

        items.append(
            SequenceItem(
                username=str(user),
                course_id=str(course),
                course_idx=course_to_idx[str(course)],
                x=X,
                dt=dt.astype(np.float32),
                y=y,
                row_ids=row_ids,
            )
        )
        if progress and log_every and (idx % log_every == 0):
            if total_groups:
                printer(f"[Build] Groups {idx}/{total_groups} ({idx/total_groups:.1%})")
            else:
                printer(f"[Build] Groups processed: {idx}")

    return items, feature_names, course_to_idx


class PackedSequenceDataset(Dataset):
    def __init__(self, items: List[SequenceItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        it = self.items[idx]
        return {
            "x": torch.from_numpy(it.x),  # (T,F)
            "dt": torch.from_numpy(it.dt),  # (T,)
            "y": None if it.y is None else torch.from_numpy(it.y),  # (T,)
            "course_idx": torch.tensor(it.course_idx, dtype=torch.long),
            "length": torch.tensor(it.x.shape[0], dtype=torch.long),
            "row_ids": None if it.row_ids is None else torch.from_numpy(it.row_ids),
            "username": it.username,
            "course_id": it.course_id,
        }


def collate_padded(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([b["length"] for b in batch])  # (B,)
    B = len(batch)
    T = int(lengths.max().item())
    F = batch[0]["x"].shape[1]
    x = torch.zeros(B, T, F, dtype=torch.float32)
    dt = torch.zeros(B, T, dtype=torch.float32)
    y = None
    if batch[0]["y"] is not None:
        y = torch.zeros(B, T, dtype=torch.float32)
    mask = torch.zeros(B, T, dtype=torch.float32)
    course_ids = torch.stack([b["course_idx"] for b in batch])
    row_ids_list = []
    for i, b in enumerate(batch):
        L = int(b["length"].item())
        x[i, :L] = b["x"]
        dt[i, :L] = b["dt"]
        if y is not None:
            y[i, :L] = b["y"]
        mask[i, :L] = 1.0
        row_ids_list.append(b["row_ids"])  # tensor or None

    return {
        "x": x,
        "dt": dt,
        "y": y,
        "mask": mask,
        "course_ids": course_ids,
        "lengths": lengths,
        "row_ids": row_ids_list,
    }


def split_items(items: List[SequenceItem], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[SequenceItem], List[SequenceItem]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = int(len(items) * val_ratio)
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, it in enumerate(items):
        (val if i in val_idx else train).append(it)
    return train, val


def fit_standardizer(train_items: List[SequenceItem]) -> Tuple[np.ndarray, np.ndarray]:
    # Compute mean/std over all valid timesteps across training sequences
    sums = None
    sums2 = None
    count = 0
    for it in train_items:
        X = it.x
        if sums is None:
            F = X.shape[1]
            sums = np.zeros(F, dtype=np.float64)
            sums2 = np.zeros(F, dtype=np.float64)
        sums += X.sum(axis=0)
        sums2 += (X * X).sum(axis=0)
        count += X.shape[0]
    mean = (sums / max(count, 1)).astype(np.float32)
    var = (sums2 / max(count, 1)) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


def apply_standardizer(items: List[SequenceItem], mean: np.ndarray, std: np.ndarray) -> None:
    for it in items:
        it.x = (it.x - mean) / std


def save_meta(path: str, meta: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_meta(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_items(items: List[SequenceItem], max_steps: int) -> List[SequenceItem]:
    if max_steps <= 0:
        return items
    out: List[SequenceItem] = []
    for it in items:
        L = min(max_steps, it.x.shape[0])
        out.append(
            SequenceItem(
                username=it.username,
                course_id=it.course_id,
                course_idx=it.course_idx,
                x=it.x[:L].copy(),
                dt=it.dt[:L].copy(),
                y=None if it.y is None else it.y[:L].copy(),
                row_ids=None if it.row_ids is None else it.row_ids[:L].copy(),
            )
        )
    return out
