from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
) -> float:
    """Train a simple RNN classifier using PyTorch and return accuracy."""
    import torch
    from torch import nn

    input_size = X_train.shape[2]

    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    fc = nn.Linear(hidden_size, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=0.01)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for _ in range(epochs):
        optimizer.zero_grad()
        out, _ = rnn(X_train_t)
        out = fc(out[:, -1, :])
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    rnn.eval()
    with torch.no_grad():
        out, _ = rnn(torch.tensor(X_test, dtype=torch.float32))
        out = fc(out[:, -1, :])
        preds = out.argmax(dim=1)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        acc = (preds == y_test_t).float().mean().item()
    return acc


def _train_hmm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 1,
) -> float:
    """Train class-specific HMMs and return accuracy."""
    from hmmlearn.hmm import GaussianHMM

    models: dict[int, GaussianHMM] = {}
    for cls in np.unique(y_train):
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
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


def evaluate_sequence_model(csv_path: str, model_type: str = "rnn") -> pd.DataFrame:
    """Train on partial grade sequences and evaluate accuracy.

    Parameters
    ----------
    csv_path : str
        Path to the student performance CSV file.
    model_type : {'rnn', 'hmm'}
        Sequence model to use.

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
            acc = _train_rnn(X_train, y_train, X_test, y_test)
        elif model_type == "hmm":
            acc = _train_hmm(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        results.append({"steps": steps, "accuracy": acc})

    df = pd.DataFrame(results)
    table_dir = Path("tables")
    table_dir.mkdir(exist_ok=True)
    df.to_csv(table_dir / f"sequence_{model_type}_performance.csv", index=False)
    return df
