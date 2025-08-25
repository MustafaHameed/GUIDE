"""Graph neural network models for OULAD student outcome prediction.

This module defines a simple GraphSAGE classifier implemented with
`torch_geometric`.  The model operates on the heterogeneous graph
constructed via :func:`build_student_vle_graph` by converting it to a
homogeneous graph internally.  Utility functions are provided for model
training, evaluation against tabular baselines and computation of
fairness metrics using ``fairlearn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)


class StudentGraphSAGE(torch.nn.Module):
    """Two-layer GraphSAGE network for node classification."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


@dataclass
class EvaluationResult:
    """Accuracy and fairness metrics for a model."""

    accuracy: float
    demographic_parity: float
    equalized_odds: float


def train_graph_model(
    data: HeteroData,
    labels: Tensor,
    train_idx: Iterable[int],
    val_idx: Optional[Iterable[int]] = None,
    epochs: int = 50,
    lr: float = 0.01,
) -> StudentGraphSAGE:
    """Train a GraphSAGE model on the provided data."""

    homo = data.to_homogeneous()
    model = StudentGraphSAGE(homo.num_features, 32, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x, edge_index = homo.x, homo.edge_index
    y = labels

    train_idx = torch.as_tensor(list(train_idx), dtype=torch.long)
    val_idx = (
        torch.as_tensor(list(val_idx), dtype=torch.long)
        if val_idx is not None
        else None
    )

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(
    model: StudentGraphSAGE,
    data: HeteroData,
    labels: Tensor,
    sensitive: Iterable[int],
) -> EvaluationResult:
    """Compute accuracy and fairness metrics on the graph."""

    homo = data.to_homogeneous()
    logits = model(homo.x, homo.edge_index)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    dp = demographic_parity_difference(y_true, preds, sensitive_features=sensitive)
    eo = equalized_odds_difference(y_true, preds, sensitive_features=sensitive)

    return EvaluationResult(acc, dp, eo)


def tabular_baseline(
    features: Tensor,
    labels: Tensor,
    sensitive: Iterable[int],
    train_idx: Iterable[int],
    test_idx: Iterable[int],
) -> EvaluationResult:
    """Train a logistic regression baseline and evaluate fairness."""

    X = features.cpu().numpy()
    y = labels.cpu().numpy()

    clf = LogisticRegression(max_iter=200)
    clf.fit(X[list(train_idx)], y[list(train_idx)])
    preds = clf.predict(X[list(test_idx)])

    acc = accuracy_score(y[list(test_idx)], preds)
    dp = demographic_parity_difference(
        y[list(test_idx)], preds, sensitive_features=[sensitive[i] for i in test_idx]
    )
    eo = equalized_odds_difference(
        y[list(test_idx)], preds, sensitive_features=[sensitive[i] for i in test_idx]
    )

    return EvaluationResult(acc, dp, eo)
