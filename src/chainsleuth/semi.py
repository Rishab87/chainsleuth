"""Self-training: use the unlabelled training-step transactions as confident pseudo-labels."""

from __future__ import annotations

import numpy as np

from chainsleuth.tabular import make_model


def self_train(name: str, X_lab, y_lab, X_unlab, seed: int = 0, rounds: int = 3,
               hi: float = 0.95, lo: float = 0.02, max_frac_per_round: float = 0.25):
    """Iteratively add unlabelled rows whose predicted class is very confident.

    Returns the final model and a per-round log of how many pseudo-labels were added.
    """
    X, y = X_lab, y_lab
    remaining = np.arange(len(X_unlab))
    log = []
    model = make_model(name, seed)
    model.fit(X, y)
    for r in range(rounds):
        if len(remaining) == 0:
            break
        p = model.predict_proba(X_unlab[remaining])[:, 1]
        conf = np.flatnonzero((p >= hi) | (p <= lo))
        cap = int(max_frac_per_round * len(X_lab))
        if len(conf) > cap:  # keep the most confident ones
            conf = conf[np.argsort(-np.abs(p[conf] - 0.5))[:cap]]
        if len(conf) == 0:
            break
        pick = remaining[conf]
        X = np.vstack([X, X_unlab[pick]])
        y = np.concatenate([y, (p[conf] >= 0.5).astype(int)])
        remaining = np.setdiff1d(remaining, pick)
        log.append({"round": r + 1, "added": int(len(pick)), "added_illicit": int((p[conf] >= 0.5).sum())})
        model = make_model(name, seed)
        model.fit(X, y)
    return model, log
