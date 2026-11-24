"""Evaluation metrics for the illicit (positive) class."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score,
)


def illicit_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    """Precision / recall / F1 / MCC at a threshold, plus threshold-free PR-AUC."""
    y_pred = (score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, score)) if y_true.any() else float("nan"),
    }


def per_step_f1(y_true: np.ndarray, score: np.ndarray, time: np.ndarray, threshold: float = 0.5) -> dict[int, float]:
    out = {}
    for t in np.unique(time):
        m = time == t
        out[int(t)] = float(f1_score(y_true[m], (score[m] >= threshold).astype(int), zero_division=0))
    return out


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 on a validation set (searched over a fixed grid)."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (score >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on the discordant pairs of two classifiers' hard predictions."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b01 = int((a_ok & ~b_ok).sum())  # A right, B wrong
    b10 = int((~a_ok & b_ok).sum())
    n = b01 + b10
    p = float(binomtest(b01, n, 0.5).pvalue) if n else 1.0
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value": p}
