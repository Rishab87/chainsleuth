"""Concept-drift measurement and adaptation over the test time steps."""

from __future__ import annotations

import numpy as np
from river.drift import ADWIN
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins taken from the reference sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift_by_step(X_ref: np.ndarray, X: np.ndarray, time: np.ndarray, steps, feat_idx, alpha: float = 0.01) -> dict:
    """Per step: fraction of features whose KS test rejects 'same distribution', and mean PSI over feat_idx."""
    out = {}
    for t in steps:
        Xt = X[time == t]
        ks_rej = np.mean([ks_2samp(X_ref[:, j], Xt[:, j]).pvalue < alpha for j in range(X.shape[1])])
        mean_psi = float(np.mean([psi(X_ref[:, j], Xt[:, j]) for j in feat_idx]))
        out[int(t)] = {"ks_reject_frac": float(ks_rej), "mean_psi_top": mean_psi}
    return out


def adwin_detections(errors: np.ndarray, time: np.ndarray, delta: float = 0.002) -> list[int]:
    """Feed a 0/1 error stream (ordered by time step) to ADWIN; return the time steps where drift fires."""
    det = ADWIN(delta=delta)
    hits = []
    for e, t in zip(errors, time):
        det.update(float(e))
        if det.drift_detected:
            hits.append(int(t))
    return hits


def em_prior_adjust(prob: np.ndarray, train_prior: float, n_iter: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, float]:
    """Saerens et al. (2002) EM: re-estimate the positive-class prior on unlabelled scores and rescale them.

    Uses only the model's scores on the new batch, never its labels.
    """
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    prior = train_prior
    for _ in range(n_iter):
        num = p * prior / train_prior
        den = num + (1 - p) * (1 - prior) / (1 - train_prior)
        adj = num / den
        new_prior = float(adj.mean())
        if abs(new_prior - prior) < tol:
            prior = new_prior
            break
        prior = new_prior
    num = p * prior / train_prior
    adj = num / (num + (1 - p) * (1 - prior) / (1 - train_prior))
    return adj, prior
