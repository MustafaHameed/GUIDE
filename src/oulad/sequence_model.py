"""Sequence-based modelling utilities for the OULAD dataset.

This module focuses on using sequences of daily virtual learning environment (VLE)
`sum_click` counts to predict whether a student passes the course (``label_pass``).

It provides helpers to
1. load and construct sequences from raw OULAD tables,
2. pad or trim sequences to a fixed length,
3. split sequences into train/validation/test sets using the same student id
   splits as produced by :mod:`src.oulad.splits`, and
4. train an attention-based LSTM or Transformer model that logs accuracy and
   exports per-time-step attention weights for interpretability.

The intention is to keep the implementation lightweight for unit testing while
still exercising core deep-learning components.  The models are intentionally
small and train for only a few epochs so that tests remain fast.

The attention weights are averaged across the evaluation split and stored under
``tables/oulad_attention_weights.csv`` by default.

Note
----
This module requires :mod:`torch`.  The repository's ``requirements.txt`` already
includes the dependency, but when running the tests in isolation make sure that
PyTorch is installed.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading and preparation utilities
# ---------------------------------------------------------------------------


def load_click_sequences(raw_dir: Path) -> Tuple[List[List[int]], List[int], List[int]]:
    """Load per-student daily ``sum_click`` sequences and pass labels.

    Parameters
    ----------
    raw_dir:
        Directory containing the original OULAD CSV tables.  Only the
        ``studentVle.csv`` and ``studentRegistration.csv`` tables are required.

    Returns
    -------
    sequences:
        List of click-count sequences, one per student.
    labels:
        Binary pass labels aligned with ``sequences``.
    student_ids:
        Student identifiers aligned with ``sequences``.
    """
    student_vle = pd.read_csv(raw_dir / "studentVle.csv")
    registration = pd.read_csv(raw_dir / "studentRegistration.csv")

    registration["label_pass"] = (registration["final_result"] == "Pass").astype(int)
    vle_reg = student_vle.merge(
        registration[["id_student", "label_pass"]], on="id_student", how="inner"
    )

    sequences: List[List[int]] = []
    labels: List[int] = []
    student_ids: List[int] = []

    for sid, group in vle_reg.groupby("id_student"):
        seq = group.sort_values("date")["sum_click"].astype(int).tolist()
        sequences.append(seq)
        labels.append(int(group["label_pass"].iloc[0]))
        student_ids.append(int(sid))

    return sequences, labels, student_ids


def pad_sequences(sequences: Iterable[Iterable[int]], max_len: int) -> np.ndarray:
    """Pad or truncate sequences to a fixed length.

    Parameters
    ----------
    sequences:
        Iterable of sequences (each an iterable of numbers).
    max_len:
        Desired sequence length.  Sequences longer than ``max_len`` are trimmed
        from the beginning and shorter sequences are left-padded with zeros.

    Returns
    -------
    ndarray
        Array of shape ``(n_sequences, max_len)`` containing the padded sequences.
    """
    arr = np.zeros((len(sequences), max_len), dtype=float)
    for i, seq in enumerate(sequences):
        seq = list(seq)[-max_len:]  # trim from the left if longer
        arr[i, -len(seq) :] = np.asarray(seq, dtype=float)
    return arr


def split_by_ids(
    X: np.ndarray,
    y: np.ndarray,
    student_ids: List[int],
    splits: Dict[str, Iterable[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/validation/test using provided student id splits.

    Parameters
    ----------
    X, y, student_ids:
        Arrays and ids produced by :func:`load_click_sequences`.
    splits:
        Dictionary with keys ``train``, ``val`` and ``test`` each containing an
        iterable of student ids.

    Returns
    -------
    Tuple of ``X_train, y_train, X_val, y_val, X_test, y_test``.
    """
    id_to_index = {sid: i for i, sid in enumerate(student_ids)}

    def _select(ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        idx = [id_to_index[i] for i in ids if i in id_to_index]
        return X[idx], y[idx]

    X_train, y_train = _select(splits.get("train", []))
    X_val, y_val = _select(splits.get("val", []))
    X_test, y_test = _select(splits.get("test", []))
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LSTMAttention(nn.Module):
    """Minimal LSTM with attention over time steps."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        logits = self.fc(context)
        return logits, weights


class TransformerModel(nn.Module):
    """Very small Transformer encoder for sequence classification."""

    def __init__(
        self, input_size: int, hidden_size: int, nhead: int = 2, num_layers: int = 1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.input_proj(x)
        out = self.encoder(out)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        logits = self.fc(context)
        return logits, weights


def _train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    epochs: int = 5,
    lr: float = 1e-3,
    attention_path: Path | None = None,
) -> float:
    """Train model and optionally export attention weights.

    Returns evaluation accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device).unsqueeze(-1)
    y_eval_t = torch.tensor(y_eval, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        logits, _ = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits, weights = model(X_eval_t)
        preds = logits.argmax(dim=1)
        acc = (preds == y_eval_t).float().mean().item()

    if attention_path is not None:
        weights_np = weights.detach().cpu().numpy()
        avg = weights_np.mean(axis=0)
        df = pd.DataFrame({"timestep": np.arange(1, len(avg) + 1), "attention": avg})
        attention_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(attention_path, index=False)

    return acc


def train_sequence_model(
    raw_dir: Path,
    splits: Dict[str, Iterable[int]],
    *,
    max_seq_len: int = 100,
    model_type: str = "lstm",
    hidden_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    attention_output: Path | None = Path("tables/oulad_attention_weights.csv"),
) -> float:
    """Train a sequence model on OULAD click sequences.

    Parameters
    ----------
    raw_dir:
        Directory with raw OULAD tables.
    splits:
        Mapping with ``train``, ``val`` and ``test`` student id lists.
    max_seq_len:
        Maximum sequence length after padding/trim.
    model_type:
        ``"lstm"`` or ``"transformer"``.
    hidden_size:
        Size of hidden representations.
    epochs:
        Number of training epochs (kept small for tests).
    lr:
        Learning rate.
    attention_output:
        Where to save averaged attention weights.  Set ``None`` to skip saving.

    Returns
    -------
    float
        Accuracy on the test split.
    """
    sequences, labels, student_ids = load_click_sequences(raw_dir)
    X = pad_sequences(sequences, max_seq_len)
    y = np.asarray(labels)

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_ids(
        X, y, student_ids, splits
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train and test splits must contain at least one sequence each"
        )

    if model_type == "lstm":
        model: nn.Module = LSTMAttention(1, hidden_size)
    elif model_type == "transformer":
        model = TransformerModel(1, hidden_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    acc = _train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        attention_path=attention_output,
    )
    logger.info("Sequence model (%s) accuracy: %.3f", model_type, acc)
    return acc
