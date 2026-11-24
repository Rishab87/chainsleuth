"""Feature-only models: classical supervised baselines and unsupervised anomaly detectors."""

from __future__ import annotations

import sys

if "torch" in sys.modules:  # see chainsleuth/__init__.py
    raise ImportError("import chainsleuth.tabular before torch: XGBoost segfaults if torch's libomp loads first on macOS")

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, seed: int = 0, pos_weight: float = 1.0):
    """Fixed, untuned hyperparameters for every model so comparisons are not tuning contests."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    if name == "rf":
        # Weber et al. RF settings: 50 trees, 50 max features
        return RandomForestClassifier(n_estimators=50, max_features=50, n_jobs=-1, random_state=seed)
    if name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=seed, eval_metric="aucpr",
        )
    if name == "xgb_weighted":
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=seed, eval_metric="aucpr", scale_pos_weight=pos_weight,
        )
    if name == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed),
        )
    raise ValueError(f"unknown model {name}")


SUPERVISED = ("logreg", "rf", "xgb", "xgb_weighted", "lgbm", "mlp")


def fit_predict(name: str, X_tr, y_tr, X_te, seed: int = 0) -> tuple[object, np.ndarray]:
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = make_model(name, seed, pos_weight)
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_te)[:, 1]


def isolation_forest_scores(X_fit, X_score, seed: int = 0) -> np.ndarray:
    """Higher = more anomalous. Fitted without labels."""
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)
    iso.fit(X_fit)
    return -iso.score_samples(X_score)


def lof_scores(X_fit, X_score, seed: int = 0, n_fit: int = 20000) -> np.ndarray:
    """Novelty-mode LOF fitted on a random subsample (full LOF on 100k+ rows is slow)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_fit), size=min(n_fit, len(X_fit)), replace=False)
    scaler = StandardScaler().fit(X_fit[idx])
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)
    lof.fit(scaler.transform(X_fit[idx]))
    return -lof.score_samples(scaler.transform(X_score))


def top_k_threshold(scores: np.ndarray, frac: float) -> float:
    """Score threshold that flags the top `frac` of rows (used for unsupervised detectors)."""
    return float(np.quantile(scores, 1 - frac))
