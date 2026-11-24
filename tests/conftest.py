import chainsleuth.tabular  # noqa: F401  load tree libs before torch (see chainsleuth/__init__.py)
import numpy as np
import pandas as pd
import pytest

from chainsleuth.data import EllipticData


@pytest.fixture
def tiny() -> EllipticData:
    """Synthetic 6-step graph: 150 nodes per step, edges only within a step, illicit nodes have a shifted feature."""
    rng = np.random.default_rng(0)
    n_steps, per = 6, 150
    time = np.repeat(np.arange(1, n_steps + 1), per)
    n = len(time)
    label = rng.choice([1, 0, -1], size=n, p=[0.2, 0.4, 0.4])
    local = rng.normal(size=(n, 4)).astype(np.float32)
    local[:, 0] += 3 * (label == 1)
    agg = rng.normal(size=(n, 2)).astype(np.float32)
    nodes = pd.DataFrame({"tx_id": np.arange(1000, 1000 + n), "time": time.astype("int16"), "label": label.astype("int8")})
    local_cols, agg_cols = [f"local_{i}" for i in range(1, 5)], ["agg_1", "agg_2"]
    nodes[local_cols], nodes[agg_cols] = local, agg
    edges = []
    for t in range(1, n_steps + 1):
        idx = np.flatnonzero(time == t)
        for _ in range(220):
            a, b = rng.choice(idx, 2, replace=False)
            edges.append((a, b))
    edge_index = np.unique(np.array(edges), axis=0).T.astype(np.int64)  # Elliptic has no duplicate edges
    return EllipticData(nodes, edge_index, local_cols, agg_cols)
