import numpy as np
import pytest

from chainsleuth import drift, gnn, graph_features, metrics, patterns, semi, tabular


def test_metrics_perfect_and_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    r = metrics.illicit_metrics(y, s)
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["pr_auc"] == 1.0
    assert metrics.illicit_metrics(y, s, threshold=0.85)["recall"] == 0.5
    assert 0.2 < metrics.best_f1_threshold(y, s) <= 0.8


def test_per_step_f1_keys():
    y = np.array([1, 0, 1, 0])
    out = metrics.per_step_f1(y, np.array([0.9, 0.1, 0.1, 0.1]), np.array([35, 35, 36, 36]))
    assert out == {35: 1.0, 36: 0.0}


def test_mcnemar_symmetric():
    y = np.array([1, 1, 0, 0])
    r = metrics.mcnemar(y, np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1]))
    assert r["a_right_b_wrong"] == 2 and r["a_wrong_b_right"] == 0


def test_shuffle_keeps_steps_and_counts(tiny):
    ei = gnn.shuffle_edges_within_step(tiny.edge_index, tiny.time, seed=1)
    assert ei.shape == tiny.edge_index.shape
    assert (tiny.time[ei[0]] == tiny.time[ei[1]]).all(), "shuffle created a cross-step edge"
    np.testing.assert_array_equal(np.sort(ei[1]), np.sort(tiny.edge_index[1]))  # same in-degree multiset
    np.testing.assert_array_equal(ei[0], tiny.edge_index[0])  # out-degrees preserved
    assert not np.array_equal(ei, tiny.edge_index)


def test_graph_features_shape_and_no_nans(tiny):
    g = graph_features.compute(tiny, use_cache=False)
    assert list(g.columns) == graph_features.GRAPH_COLS
    assert len(g) == len(tiny.nodes) and not g.isna().any().any()
    # in-degree recomputed independently
    np.testing.assert_array_equal(g["g_in_deg"].to_numpy(), np.bincount(tiny.edge_index[1], minlength=len(tiny.nodes)))


def test_graph_features_are_step_local(tiny):
    """Changing edges in one step must not change features of another step."""
    base = graph_features.compute(tiny, use_cache=False)
    keep = tiny.time[tiny.edge_index[0]] != 6
    tiny.edge_index = tiny.edge_index[:, keep]
    after = graph_features.compute(tiny, use_cache=False)
    other = tiny.time != 6
    np.testing.assert_allclose(base[other].drop(columns=["g_comm_size", "g_comm_density"]).to_numpy(),
                               after[other].drop(columns=["g_comm_size", "g_comm_density"]).to_numpy())


@pytest.mark.parametrize("name", tabular.SUPERVISED)
def test_supervised_models_learn_signal(tiny, name):
    X = tiny.nodes[tiny.af_cols].to_numpy()
    lab = tiny.labelled
    tr, te = lab & (tiny.time <= 4), lab & (tiny.time > 4)
    _, s = tabular.fit_predict(name, X[tr], tiny.label[tr], X[te])
    assert s.shape == (te.sum(),) and ((0 <= s) & (s <= 1)).all()
    assert metrics.illicit_metrics(tiny.label[te], s)["pr_auc"] > 0.6


def test_unsupervised_scores(tiny):
    X = tiny.nodes[tiny.af_cols].to_numpy()
    for fn in (tabular.isolation_forest_scores, tabular.lof_scores):
        s = fn(X, X)
        assert s.shape == (len(X),) and np.isfinite(s).all()
    assert tabular.top_k_threshold(np.arange(100.0), 0.1) == pytest.approx(89.1)


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat"])
def test_gnn_trains_and_predicts(tiny, kind, monkeypatch):
    monkeypatch.setattr(gnn, "TRAIN_END", 4)
    monkeypatch.setattr(gnn, "VAL_START", 4)
    p = gnn.prepare(tiny, tiny.af_cols, "real")
    res = gnn.fit(kind, p, seed=0, max_epochs=30)
    assert res["prob"].shape == (len(tiny.nodes),)
    assert 0.05 <= res["threshold"] <= 0.95 and 10 <= res["best_epoch"] <= 30
    emb = gnn.embeddings(res["model"], p)
    assert emb.shape == (len(tiny.nodes), 64)


def test_gnn_no_edges_variant(tiny):
    p = gnn.prepare(tiny, tiny.af_cols, "none")
    assert p.edge_index.shape[1] == 0


def test_psi_and_ks():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert drift.psi(a, rng.normal(size=5000)) < 0.02
    assert drift.psi(a, rng.normal(2, 1, size=5000)) > 0.5
    X = np.vstack([rng.normal(size=(500, 3)), rng.normal(3, 1, size=(500, 3))])
    t = np.repeat([35, 36], 500)
    out = drift.feature_drift_by_step(rng.normal(size=(1000, 3)), X, t, [35, 36], [0, 1])
    assert out[35]["ks_reject_frac"] < out[36]["ks_reject_frac"] == 1.0


def test_adwin_fires_on_error_jump():
    errors = np.r_[np.zeros(2000), np.ones(2000)]
    t = np.r_[np.full(2000, 35), np.full(2000, 36)]
    assert 36 in drift.adwin_detections(errors, t)


def test_em_prior_recovers_shift():
    rng = np.random.default_rng(0)
    # a calibrated classifier trained at prior 0.2, applied to a batch whose true prior is 0.05
    y = rng.random(20000) < 0.05
    x = np.where(y, rng.normal(2, 1, y.size), rng.normal(-2, 1, y.size))
    llr = 4 * x  # exact log-likelihood ratio of N(2,1) vs N(-2,1)
    prob = 1 / (1 + np.exp(-(llr + np.log(0.2 / 0.8))))  # calibrated posterior under a 0.2 prior
    adj, est = drift.em_prior_adjust(prob, 0.2)
    assert abs(est - 0.05) < 0.02 and adj.mean() < prob.mean()


def test_self_training_adds_pseudo_labels(tiny):
    X = tiny.nodes[tiny.af_cols].to_numpy()
    lab = tiny.labelled
    model, log = semi.self_train("xgb", X[lab], tiny.label[lab], X[~lab], rounds=2, hi=0.8, lo=0.2)
    assert log and log[0]["added"] > 0
    assert model.predict_proba(X[:3]).shape == (3, 2)


def test_apriori_finds_planted_rule(tiny):
    X = tiny.nodes[tiny.af_cols]
    lab = tiny.labelled
    rules = patterns.illicit_rules(X[lab].reset_index(drop=True), tiny.label[lab], ["local_1", "local_2"],
                                   min_support=0.05, min_confidence=0.4)
    assert len(rules) and "local_1=high" in rules.iloc[0]["antecedents"]
