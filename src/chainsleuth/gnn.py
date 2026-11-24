"""Graph neural networks (GCN, GraphSAGE, GAT) trained full-batch on the Elliptic graph.

Protocol: pick the epoch count on validation steps 30..34 while training on 1..29, then
retrain on all of 1..34 for that many epochs and score the test steps 35..49.
Because no edge crosses a time step, message passing for training nodes never sees
a test node: transductive and inductive training are the same computation on this graph.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

_TREE_LIBS_LOADED = "xgboost" in sys.modules or "lightgbm" in sys.modules

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import to_undirected

from chainsleuth.config import TRAIN_END, VAL_START
from chainsleuth.data import EllipticData

if _TREE_LIBS_LOADED:  # see chainsleuth/__init__.py: multithreaded torch would deadlock here
    torch.set_num_threads(1)


class GNN(nn.Module):
    def __init__(self, kind: str, in_dim: int, hidden: int = 64, dropout: float = 0.5, heads: int = 4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden)
        elif kind == "gat":
            self.c1 = GATConv(in_dim, hidden // heads, heads=heads)
            self.c2 = GATConv(hidden, hidden // heads, heads=heads)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden, 2)

    def embed(self, x, edge_index):
        h = F.dropout(x, self.dropout, self.training)
        h = F.relu(self.c1(h, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        return F.relu(self.c2(h, edge_index))

    def forward(self, x, edge_index):
        return self.head(F.dropout(self.embed(x, edge_index), self.dropout, self.training))


def shuffle_edges_within_step(edge_index: np.ndarray, time: np.ndarray, seed: int) -> np.ndarray:
    """Random rewiring that keeps each step's edge count and each node's out-degree.

    Destinations are permuted among the edges of the same time step, so the shuffled graph
    still has no cross-step edges (and therefore no new leakage path).
    """
    rng = np.random.default_rng(seed)
    src, dst = edge_index.copy()
    step = time[src]
    for t in np.unique(step):
        idx = np.flatnonzero(step == t)
        dst[idx] = dst[rng.permutation(idx)]
    return np.stack([src, dst])


@dataclass
class Prepared:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    time: np.ndarray
    labelled: np.ndarray


def prepare(d: EllipticData, feature_cols: list[str], edges: str = "real", seed: int = 0) -> Prepared:
    """Standardise features with training-step statistics and build an undirected edge index."""
    X = d.nodes[feature_cols].to_numpy(np.float32)
    tr_rows = d.time <= TRAIN_END
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    X = (X - mu) / sd
    if edges == "real":
        ei = d.edge_index
    elif edges == "shuffled":
        ei = shuffle_edges_within_step(d.edge_index, d.time, seed)
    elif edges == "none":
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        raise ValueError(edges)
    ei_t = to_undirected(torch.from_numpy(ei).long(), num_nodes=len(X))
    y = torch.from_numpy(np.where(d.label < 0, 0, d.label).astype(np.int64))
    return Prepared(torch.from_numpy(X), y, ei_t, d.time, d.labelled)


def _train(kind, p: Prepared, train_idx, epochs, seed, val_idx=None, lr=0.01, weight_decay=5e-4, eval_every=10):
    """Full-batch training. With val_idx, also tracks val PR-AUC and keeps the best val probabilities."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GNN(kind, p.x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ytr = p.y[train_idx]
    w = torch.tensor([0.3, 0.7])  # licit / illicit loss weights, as in Weber et al. (2019)
    best = {"epoch": epochs, "ap": -1.0, "val_prob": None}
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(p.x, p.edge_index)
        F.cross_entropy(out[train_idx], ytr, weight=w).backward()
        opt.step()
        if val_idx is not None and ep % eval_every == 0:
            prob = predict(model, p)[val_idx]
            ap = average_precision_score(p.y[val_idx].numpy(), prob)
            if ap > best["ap"]:
                best = {"epoch": ep, "ap": float(ap), "val_prob": prob}
    return model, best


@torch.no_grad()
def predict(model, p: Prepared) -> np.ndarray:
    model.eval()
    return torch.softmax(model(p.x, p.edge_index), dim=1)[:, 1].numpy()


@torch.no_grad()
def embeddings(model, p: Prepared) -> np.ndarray:
    model.eval()
    return model.embed(p.x, p.edge_index).numpy()


def fit(kind: str, p: Prepared, seed: int = 0, max_epochs: int = 300) -> dict:
    """Select epochs + decision threshold on val (train 1..29), then retrain on 1..34.

    Returns dict(model, prob (all nodes), best_epoch, threshold).
    """
    from chainsleuth.metrics import best_f1_threshold

    t, lab = p.time, p.labelled
    tr_small = torch.from_numpy(np.flatnonzero(lab & (t < VAL_START)))
    val = torch.from_numpy(np.flatnonzero(lab & (t >= VAL_START) & (t <= TRAIN_END)))
    _, best = _train(kind, p, tr_small, max_epochs, seed, val_idx=val)
    threshold = best_f1_threshold(p.y[val].numpy(), best["val_prob"])
    tr_full = torch.from_numpy(np.flatnonzero(lab & (t <= TRAIN_END)))
    model, _ = _train(kind, p, tr_full, best["epoch"], seed)
    return {"model": model, "prob": predict(model, p), "best_epoch": best["epoch"], "threshold": threshold}
