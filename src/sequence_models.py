"""Sequence models for grade prediction.

References
----------
- Attention mechanism: https://arxiv.org/abs/1409.0473
- Integrated Gradients: https://doi.org/10.1109/TPAMI.2017.2992093
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .utils import ensure_dir


def _prepare_sequences(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load data and construct sequential inputs.

    Parameters
    ----------
    csv_path : str
        Path to the original student performance CSV file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X_seq`` with shape ``(n_samples, 2, 2)`` containing ``(G1, studytime)``
        and ``(G2, studytime)`` for each student, and binary target ``y`` where
        1 indicates a passing final grade (``G3`` >= 10).
    """
    df = pd.read_csv(csv_path)
    df["pass"] = (df["G3"] >= 10).astype(int)
    # Two time steps: (G1, studytime) and (G2, studytime)
    sequences = np.stack(
        [
            df[["G1", "studytime"]].to_numpy(),
            df[["G2", "studytime"]].to_numpy(),
        ],
        axis=1,
    )
    y = df["pass"].to_numpy()
    return sequences, y


def _train_rnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_size: int = 8,
    epochs: int = 50,
    learning_rate: float = 0.01,
    save_importance: bool = False,
) -> float:
    """Train an attention-based RNN classifier and return accuracy.

    Requires optional dependencies ``torch`` and ``captum``. If they are not
    available, a message is logged and ``float('nan')`` is returned.

    When ``save_importance`` is True, per-step importance scores are computed
    using Integrated Gradients and stored under ``tables/`` with a
    corresponding bar plot in ``figures/``.

    References
    ----------
    - Attention mechanism: https://arxiv.org/abs/1409.0473
    - Integrated Gradients: https://doi.org/10.1109/TPAMI.2017.2992093
    """
    import logging

    logger = logging.getLogger(__name__)
    try:
        import torch
        from torch import nn
    except ImportError:
        logger.error(
            "PyTorch is required for `_train_rnn`. Install torch to use the RNN model."
        )
        return float("nan")
    try:
        from captum.attr import (
            IntegratedGradients,
        )  # Reference: https://doi.org/10.1109/TPAMI.2017.2992093
    except ImportError:
        logger.error(
            "captum is required for `_train_rnn`. Install captum to use this feature."
        )
        return float("nan")
    import matplotlib.pyplot as plt

    input_size = X_train.shape[2]

    class AttentionRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.attn = nn.Linear(hidden_size, 1)
            self.fc = nn.Linear(hidden_size, 2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out, _ = self.rnn(x)
            weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
            context = torch.sum(out * weights.unsqueeze(-1), dim=1)
            logits = self.fc(context)
            return logits, weights

    model = AttentionRNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for _ in range(epochs):
        optimizer.zero_grad()
        out, _ = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out, _ = model(torch.tensor(X_test, dtype=torch.float32))
        preds = out.argmax(dim=1)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        acc = (preds == y_test_t).float().mean().item()

    if save_importance:

        def forward_wrapper(x: torch.Tensor) -> torch.Tensor:
            return model(x)[0]

        ig = IntegratedGradients(forward_wrapper)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
        attrs = ig.attribute(X_test_t, target=1)
        step_importance = attrs.abs().sum(dim=2).mean(dim=0).detach().cpu().numpy()
        table_dir = Path("tables")
        ensure_dir(table_dir)
        df_imp = pd.DataFrame(
            {
                "step": np.arange(1, len(step_importance) + 1),
                "importance": step_importance,
            }
        )
        df_imp.to_csv(table_dir / "sequence_step_importance.csv", index=False)
        fig_dir = Path("figures")
        ensure_dir(fig_dir)
        plt.figure()
        plt.bar(df_imp["step"], df_imp["importance"])
        plt.xlabel("Time Step")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(fig_dir / "sequence_step_importance.png")
        plt.close()

    return acc


def _train_hmm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 1,
) -> float:
    """Train class-specific HMMs and return accuracy.

    References
    ----------
    - Hidden Markov models tutorial: https://doi.org/10.1109/5.18626
    """
    from hmmlearn.hmm import GaussianHMM  # Reference: https://doi.org/10.1109/5.18626

    models: dict[int, GaussianHMM] = {}
    for cls in np.unique(y_train):
        model = GaussianHMM(
            n_components=n_components, covariance_type="diag", n_iter=100
        )
        cls_seqs = X_train[y_train == cls].reshape(-1, X_train.shape[2])
        lengths = [X_train.shape[1]] * np.sum(y_train == cls)
        try:
            model.fit(cls_seqs, lengths)
        except ValueError:
            return float("nan")
        models[int(cls)] = model

    preds: list[int] = []
    for seq in X_test:
        seq2 = seq.reshape(-1, X_train.shape[2])
        try:
            scores = {cls: m.score(seq2) for cls, m in models.items()}
        except ValueError:
            return float("nan")
        preds.append(max(scores, key=scores.get))
    return accuracy_score(y_test, preds)


def evaluate_sequence_model(
    csv_path: str,
    model_type: str = "rnn",
    hidden_size: int = 8,
    epochs: int = 50,
    learning_rate: float = 0.01,
) -> pd.DataFrame:
    """Train on partial grade sequences and evaluate accuracy.

    Parameters
    ----------
    csv_path : str
        Path to the student performance CSV file.
    model_type : {'rnn', 'hmm'}
        Sequence model to use.
    hidden_size : int, default 8
        RNN hidden dimension when ``model_type='rnn'``.
    epochs : int, default 50
        Number of training epochs for the RNN.
    learning_rate : float, default 0.01
        Optimizer learning rate for the RNN.

    Returns
    -------
    pandas.DataFrame
        Accuracy for each prefix length of the grade sequence.
    """
    X_seq, y = _prepare_sequences(csv_path)
    results: list[dict[str, float | int]] = []

    for steps in range(1, X_seq.shape[1] + 1):
        X_part = X_seq[:, :steps, :]
        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y, test_size=0.2, stratify=y, random_state=42
        )
        if model_type == "rnn":
            acc = _train_rnn(
                X_train,
                y_train,
                X_test,
                y_test,
                save_importance=steps == X_seq.shape[1],
                hidden_size=hidden_size,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        elif model_type == "hmm":
            acc = _train_hmm(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        results.append({"steps": steps, "accuracy": acc})

    df = pd.DataFrame(results)
    table_dir = Path("tables")
    ensure_dir(table_dir)
    df.to_csv(table_dir / f"sequence_{model_type}_performance.csv", index=False)
    return df
