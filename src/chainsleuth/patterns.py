"""Association-rule mining (Apriori) over discretised features of labelled training transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def illicit_rules(X: pd.DataFrame, y: np.ndarray, top_features: list[str], min_support: float = 0.01,
                  min_confidence: float = 0.5, n_bins: int = 3) -> pd.DataFrame:
    """Rules of the form {feature=bin, ...} -> {illicit}, ranked by lift.

    Each feature is cut into quantile bins (low/mid/high); every transaction becomes a basket of
    'feature=bin' items plus a class item.
    """
    names = ["low", "mid", "high"][:n_bins] if n_bins == 3 else [f"q{i}" for i in range(n_bins)]
    items = {}
    for f in top_features:
        binned = pd.qcut(X[f].rank(method="first"), n_bins, labels=names)
        for b in names:
            items[f"{f}={b}"] = (binned == b).to_numpy()
    items["class=illicit"] = y == 1
    basket = pd.DataFrame(items)
    freq = apriori(basket, min_support=min_support, use_colnames=True, max_len=4)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["consequents"] == frozenset({"class=illicit"})]
    rules = rules.assign(antecedents=rules["antecedents"].map(lambda s: " AND ".join(sorted(s))))
    return rules.sort_values(["lift", "support"], ascending=False)[
        ["antecedents", "support", "confidence", "lift"]
    ].reset_index(drop=True)
