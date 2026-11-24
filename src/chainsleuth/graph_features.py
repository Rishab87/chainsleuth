"""Structural graph-mining features, computed independently inside each time step.

Elliptic has no edges between time steps, so every feature here only ever looks at
transactions from the same step. No labels are used, so nothing can leak across the
temporal train/test split.
"""

from __future__ import annotations

import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np
import pandas as pd

from chainsleuth.config import CACHE_DIR
from chainsleuth.data import EllipticData

GRAPH_COLS = [
    "g_in_deg", "g_out_deg", "g_pagerank", "g_clustering", "g_core",
    "g_betweenness", "g_comm_size", "g_comm_density", "g_nbr_mean_deg", "g_ego_size_2hop",
]


def _step_features(nodes_idx: np.ndarray, edges: np.ndarray, seed: int) -> pd.DataFrame:
    g = nx.DiGraph()
    g.add_nodes_from(nodes_idx.tolist())
    g.add_edges_from(edges.T.tolist())
    u = g.to_undirected()
    u.remove_edges_from(nx.selfloop_edges(u))

    pr = nx.pagerank(g, alpha=0.85)
    clust = nx.clustering(u)
    core = nx.core_number(u)
    k = min(256, len(u))  # sampled betweenness: exact is O(VE), too slow for 8k-node steps
    btw = nx.betweenness_centrality(u, k=k, seed=seed, normalized=True)
    part = community_louvain.best_partition(u, random_state=seed)
    comm = pd.Series(part)
    comm_size = comm.map(comm.value_counts())
    comm_density = {}
    for cid, members in comm.groupby(comm).groups.items():
        sub = u.subgraph(members)
        n = sub.number_of_nodes()
        comm_density[cid] = (2 * sub.number_of_edges() / (n * (n - 1))) if n > 1 else 0.0
    deg = dict(u.degree())
    nbr_mean_deg = {n: (np.mean([deg[v] for v in u[n]]) if deg[n] else 0.0) for n in u}
    ego2 = {n: len(nx.single_source_shortest_path_length(u, n, cutoff=2)) - 1 for n in u}

    idx = nodes_idx
    return pd.DataFrame({
        "g_in_deg": [g.in_degree(n) for n in idx],
        "g_out_deg": [g.out_degree(n) for n in idx],
        "g_pagerank": [pr[n] for n in idx],
        "g_clustering": [clust[n] for n in idx],
        "g_core": [core[n] for n in idx],
        "g_betweenness": [btw[n] for n in idx],
        "g_comm_size": [comm_size[n] for n in idx],
        "g_comm_density": [comm_density[part[n]] for n in idx],
        "g_nbr_mean_deg": [nbr_mean_deg[n] for n in idx],
        "g_ego_size_2hop": [ego2[n] for n in idx],
    }, index=idx)


def compute(d: EllipticData, seed: int = 0, use_cache: bool = True) -> pd.DataFrame:
    """Return a DataFrame of GRAPH_COLS aligned with d.nodes (same positional index)."""
    path = CACHE_DIR / f"graph_features_s{seed}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    time = d.time
    src_t = time[d.edge_index[0]]
    parts = []
    for t in np.unique(time):
        nodes_idx = np.flatnonzero(time == t)
        edges = d.edge_index[:, src_t == t]
        parts.append(_step_features(nodes_idx, edges, seed))
    out = pd.concat(parts).sort_index().astype("float32")
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
    return out
