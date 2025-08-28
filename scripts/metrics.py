from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
    ap = (precision[y_sorted == 1].sum()) / pos_total
    return float(ap)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Sklearn-like ROC AUC via ranking
    if y_true.ndim != 1:
        y_true = y_true.ravel()
    if y_score.ndim != 1:
        y_score = y_score.ravel()
    order = np.argsort(-y_score, kind="mergesort")
    y = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    tp = np.concatenate([[0.0], tp])
    fp = np.concatenate([[0.0], fp])
    # Normalize to TPR/FPR
    P = tp[-1]
    N = fp[-1]
    if P == 0.0 or N == 0.0:
        return 0.5
    tpr = tp / P
    fpr = fp / N
    # Trapezoidal rule over FPR
    auc = np.trapz(tpr, fpr)
    return float(auc)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), eps, 1.0 - eps)
    ce = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(ce))

