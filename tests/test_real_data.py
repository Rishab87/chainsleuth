"""Checks against the real Elliptic data. Skipped if it has not been downloaded yet."""

import numpy as np
import pytest

from chainsleuth.config import CACHE_DIR, RAW_DIR

pytestmark = pytest.mark.skipif(not (CACHE_DIR / "nodes.parquet").exists() and not (RAW_DIR / "elliptic_txs_features.csv").exists(),
                                reason="Elliptic data not downloaded")


@pytest.fixture(scope="module")
def d():
    from chainsleuth import data
    return data.load()


def test_published_statistics(d):
    from chainsleuth.data import summary
    s = summary(d)
    assert (s["n_nodes"], s["n_edges"], s["n_features"]) == (203769, 234355, 166)
    assert (s["n_illicit"], s["n_licit"], s["n_unknown"]) == (4545, 42019, 157205)
    assert s["n_time_steps"] == 49


def test_no_edge_crosses_time_steps(d):
    """The property that makes per-step features and GNN message passing leakage-free."""
    assert (d.time[d.edge_index[0]] == d.time[d.edge_index[1]]).all()


def test_split_is_temporal_and_disjoint(d):
    assert d.time[d.train_mask].max() == 34 and d.time[d.test_mask].min() == 35
    assert not (d.train_mask & d.test_mask).any()
    assert d.labelled[d.train_mask].all() and d.labelled[d.test_mask].all()


def test_edge_index_in_range(d):
    assert d.edge_index.min() >= 0 and d.edge_index.max() < len(d.nodes)
    assert np.isfinite(d.nodes[d.af_cols].to_numpy()).all()
