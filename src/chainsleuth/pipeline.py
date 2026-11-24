"""End-to-end experiment runner.

    uv run python -m chainsleuth.pipeline                 # every stage
    uv run python -m chainsleuth.pipeline --stages tabular gnn --seeds 3

Each stage writes reports/results/<stage>.json, so stages can be re-run alone or concurrently.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import subprocess
import sys

import numpy as np
import pandas as pd

# Only OpenMP-free modules at top level. Tree libraries and torch are imported inside the stage
# that needs them, and each stage runs in its own process (see chainsleuth/__init__.py).
from chainsleuth import data, graph_features, metrics
from chainsleuth.config import ARTIFACTS_DIR, REPORTS_DIR, SEEDS, TRAIN_END, VAL_START

RESULTS_DIR = REPORTS_DIR / "results"  # one JSON per stage, so stages can run concurrently
TEST_STEPS = list(range(TRAIN_END + 1, 50))


# ───────────────────────── helpers ─────────────────────────

def load_results() -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted(RESULTS_DIR.glob("*.json"))}


def save_results(section: str, payload) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{section}.json").write_text(json.dumps(payload, indent=2, default=float))


def agg(runs: list[dict]) -> dict:
    """Mean and std over seeds for every numeric metric."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in keys}


def feature_sets(d: data.EllipticData) -> dict[str, np.ndarray]:
    g = graph_features.compute(d)
    af = d.nodes[d.af_cols].to_numpy(np.float32)
    return {
        "LF": d.nodes[d.local_cols].to_numpy(np.float32),
        "AF": af,
        "AF+G": np.hstack([af, g.to_numpy(np.float32)]),
    }


def masks(d):
    t, lab = d.time, d.labelled
    return {
        "train": lab & (t <= TRAIN_END),
        "train_small": lab & (t < VAL_START),
        "val": lab & (t >= VAL_START) & (t <= TRAIN_END),
        "test": lab & (t > TRAIN_END),
    }


def eval_scores(y, score, time, threshold):
    """Metrics at the val-tuned threshold, plus F1 at 0.5. Per-step F1 is returned for both thresholds."""
    r = metrics.illicit_metrics(y, score, threshold)
    r["f1_at_0.5"] = metrics.illicit_metrics(y, score, 0.5)["f1"]
    r["threshold"] = threshold
    steps = {"at_0.5": metrics.per_step_f1(y, score, time, 0.5), "at_val_thr": metrics.per_step_f1(y, score, time, threshold)}
    return r, steps


def tabular_run(name, X, y, M, seed, time):
    """Fit on 1..29 to choose a threshold on val, refit on 1..34, evaluate on 35..49."""
    from chainsleuth import tabular

    _, s_val = tabular.fit_predict(name, X[M["train_small"]], y[M["train_small"]], X[M["val"]], seed)
    thr = metrics.best_f1_threshold(y[M["val"]], s_val)
    model, s_te = tabular.fit_predict(name, X[M["train"]], y[M["train"]], X[M["test"]], seed)
    r, steps = eval_scores(y[M["test"]], s_te, time[M["test"]], thr)
    return model, s_te, r, steps


# ───────────────────────── stages ─────────────────────────

def stage_eda(d, seeds):
    s = data.summary(d)
    per_step = d.nodes.groupby("time")["label"].agg(
        illicit=lambda x: int((x == 1).sum()), licit=lambda x: int((x == 0).sum()), unknown=lambda x: int((x == -1).sum())
    )
    nodes_per_step = d.nodes.groupby("time").size()
    s["nodes_per_step_min"], s["nodes_per_step_max"] = int(nodes_per_step.min()), int(nodes_per_step.max())
    M = masks(d)
    s["train_illicit"], s["train_licit"] = int((d.label[M["train"]] == 1).sum()), int((d.label[M["train"]] == 0).sum())
    s["test_illicit"], s["test_licit"] = int((d.label[M["test"]] == 1).sum()), int((d.label[M["test"]] == 0).sum())
    s["per_step"] = per_step.reset_index().to_dict(orient="records")
    save_results("eda", s)
    print(json.dumps({k: v for k, v in s.items() if k != "per_step"}, indent=1))


def stage_tabular(d, seeds):
    from chainsleuth import tabular

    y, time = d.label, d.time
    M = masks(d)
    out = {}
    for fs, X in feature_sets(d).items():
        for name in tabular.SUPERVISED:
            model_seeds = seeds if name != "logreg" else seeds[:1]  # logreg is deterministic
            runs, steps_all = [], []
            for seed in model_seeds:
                t0 = time_now()
                _, _, r, steps = tabular_run(name, X, y, M, seed, time)
                r["fit_seconds"] = time_now() - t0
                runs.append(r)
                steps_all.append(steps)
            key = f"{name}|{fs}"
            out[key] = {"metrics": agg(runs), "per_step_f1": mean_steps(steps_all), "n_seeds": len(runs)}
            m = out[key]["metrics"]
            print(f"{key:22s} F1 {m['f1']['mean']:.3f}±{m['f1']['std']:.3f}  F1@0.5 {m['f1_at_0.5']['mean']:.3f}  PR-AUC {m['pr_auc']['mean']:.3f}", flush=True)
    save_results("tabular", out)


def stage_unsupervised(d, seeds):
    from chainsleuth import tabular

    X = feature_sets(d)["AF"]
    y = d.label
    M = masks(d)
    fit_rows = d.time <= TRAIN_END  # every training-step transaction, labelled or not; no labels used
    prior = float(y[M["train"]].mean())
    out = {}
    for name, fn in [("isolation_forest", tabular.isolation_forest_scores), ("lof", tabular.lof_scores)]:
        s = fn(X[fit_rows], X[M["test"]], seed=seeds[0])
        thr = tabular.top_k_threshold(s, prior)  # flag the same fraction as the training illicit rate
        r = metrics.illicit_metrics(y[M["test"]], s, thr)
        r["flag_fraction"] = prior
        out[name] = r
        print(name, {k: round(v, 3) for k, v in r.items()})
    save_results("unsupervised", out)


GNN_KINDS = ("gcn", "sage", "gat")
GNN_EDGES = ("real", "shuffled", "none")
GNN_RUNS_DIR = ARTIFACTS_DIR / "gnn_runs"  # one JSON per (model, edges, seed): makes the stage resumable


def stage_gnn(d, seeds, kinds=GNN_KINDS):
    """Train every (model, edge variant, seed). Finished runs are cached, so a killed run resumes.

    Several processes can split the work with --gnn-kinds; each one re-aggregates whatever is cached,
    so the last process to finish writes the complete results file.
    """
    import torch

    from chainsleuth import gnn

    y, time = d.label, d.time
    M = masks(d)
    GNN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    probs_dir = ARTIFACTS_DIR / "gnn_test_probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        for edges in GNN_EDGES:
            for seed in seeds:
                cache = GNN_RUNS_DIR / f"{kind}_{edges}_s{seed}.json"
                if cache.exists():
                    continue
                t0 = time_now()
                p = gnn.prepare(d, d.af_cols, edges, seed=seed)
                res = gnn.fit(kind, p, seed=seed)
                r, steps = eval_scores(y[M["test"]], res["prob"][M["test"]], time[M["test"]], res["threshold"])
                r["best_epoch"], r["fit_seconds"] = res["best_epoch"], time_now() - t0
                np.save(probs_dir / f"{kind}_{edges}_s{seed}.npy", res["prob"][M["test"]].astype(np.float32))
                if kind == "sage" and edges == "real" and seed == seeds[0]:
                    # GraphSAGE model + embeddings for the hybrid stage, GNNExplainer and the dashboard
                    torch.save(res["model"].state_dict(), ARTIFACTS_DIR / "sage.pt")
                    np.save(ARTIFACTS_DIR / "sage_embeddings.npy", gnn.embeddings(res["model"], p))
                    np.save(ARTIFACTS_DIR / "sage_prob.npy", res["prob"])
                    (ARTIFACTS_DIR / "sage_meta.json").write_text(
                        json.dumps({"threshold": res["threshold"], "best_epoch": res["best_epoch"]}))
                cache.write_text(json.dumps({"metrics": r, "steps": steps}))
                print(f"  {kind}|{edges} seed {seed}: F1@0.5 {r['f1_at_0.5']:.3f}  PR-AUC {r['pr_auc']:.3f}  "
                      f"({r['fit_seconds']:.0f}s)", flush=True)
    aggregate_gnn(seeds)


def aggregate_gnn(seeds):
    out = {}
    for kind in GNN_KINDS:
        for edges in GNN_EDGES:
            files = [GNN_RUNS_DIR / f"{kind}_{edges}_s{s}.json" for s in seeds]
            if not all(f.exists() for f in files):
                continue
            loaded = [json.loads(f.read_text()) for f in files]
            steps_all = [{v: {int(t): f1 for t, f1 in st.items()} for v, st in x["steps"].items()} for x in loaded]
            key = f"{kind}|{edges}"
            out[key] = {"metrics": agg([x["metrics"] for x in loaded]), "per_step_f1": mean_steps(steps_all),
                        "n_seeds": len(loaded)}
            m = out[key]["metrics"]
            print(f"{key:15s} F1 {m['f1']['mean']:.3f}±{m['f1']['std']:.3f}  F1@0.5 {m['f1_at_0.5']['mean']:.3f}"
                  f"±{m['f1_at_0.5']['std']:.3f}  PR-AUC {m['pr_auc']['mean']:.3f}", flush=True)
    complete = len(out) == len(GNN_KINDS) * len(GNN_EDGES)
    print(f"[gnn: {len(out)}/{len(GNN_KINDS) * len(GNN_EDGES)} configurations complete]")
    if out:
        save_results("gnn", out)
    return complete


def stage_hybrid(d, seeds):
    """GraphSAGE embeddings (trained on 1..34 only) appended to AF, fed to tree models (Weber's AF+NE)."""
    emb_path = ARTIFACTS_DIR / "sage_embeddings.npy"
    if not emb_path.exists():
        raise SystemExit("run the gnn stage first")
    X = np.hstack([feature_sets(d)["AF"], np.load(emb_path)])
    y, time = d.label, d.time
    M = masks(d)
    out = {}
    for name in ("rf", "xgb", "lgbm"):
        runs, steps_all = [], []
        for seed in seeds:
            _, _, r, steps = tabular_run(name, X, y, M, seed, time)
            runs.append(r)
            steps_all.append(steps)
        key = f"{name}|AF+SAGE-emb"
        out[key] = {"metrics": agg(runs), "per_step_f1": mean_steps(steps_all), "n_seeds": len(runs)}
        m = out[key]["metrics"]
        print(f"{key:22s} F1 {m['f1']['mean']:.3f}±{m['f1']['std']:.3f}  PR-AUC {m['pr_auc']['mean']:.3f}", flush=True)
    save_results("hybrid", out)


def stage_semi(d, seeds):
    from chainsleuth import semi, tabular

    X = feature_sets(d)["AF"]
    y = d.label
    M = masks(d)
    unlab_train = (~d.labelled) & (d.time <= TRAIN_END)
    out = {"supervised": [], "self_training": [], "pseudo_label_log": None}
    for seed in seeds:
        # supervised reference, threshold 0.5 for both arms so only the training data differs
        _, s = tabular.fit_predict("xgb", X[M["train"]], y[M["train"]], X[M["test"]], seed)
        out["supervised"].append(metrics.illicit_metrics(y[M["test"]], s))
        model, log = semi.self_train("xgb", X[M["train"]], y[M["train"]], X[unlab_train], seed=seed)
        out["self_training"].append(metrics.illicit_metrics(y[M["test"]], model.predict_proba(X[M["test"]])[:, 1]))
        out["pseudo_label_log"] = log
    out["supervised"], out["self_training"] = agg(out["supervised"]), agg(out["self_training"])
    print("supervised   ", {k: round(v["mean"], 3) for k, v in out["supervised"].items()})
    print("self-training", {k: round(v["mean"], 3) for k, v in out["self_training"].items()})
    print("pseudo labels", out["pseudo_label_log"])
    save_results("semi", out)


def stage_drift(d, seeds):
    from chainsleuth import drift, tabular

    X = feature_sets(d)["AF"]
    y, time = d.label, d.time
    M = masks(d)
    seed = seeds[0]

    # static model and the features it relies on most. Threshold fixed at 0.5 to match the headline
    # tables: the val-tuned threshold (steps 30..34) itself transfers poorly, a symptom of the drift.
    thr = 0.5
    static, _ = tabular.fit_predict("xgb", X[M["train"]], y[M["train"]], X[:1], seed)
    top = np.argsort(-static.feature_importances_)[:10]

    # 1) input drift per test step vs the training distribution (labelled + unlabelled rows)
    ref = X[time <= TRAIN_END]
    feat_drift = drift.feature_drift_by_step(ref, X, time, TEST_STEPS, top)

    # 2) label prior per step
    prior = {int(t): float(y[M["test"] & (time == t)].mean()) for t in TEST_STEPS}
    train_prior = float(y[M["train"]].mean())

    # 3) ADWIN over the static model's error stream, ordered by time step
    te = M["test"]
    s_static = static.predict_proba(X[te])[:, 1]
    errors = ((s_static >= thr).astype(int) != y[te]).astype(int)
    order = np.argsort(time[te], kind="stable")
    adwin = drift.adwin_detections(errors[order], time[te][order])
    # label-free monitor: ADWIN on the alert indicator (did the model flag this transaction?)
    alerts = (s_static >= thr).astype(int)
    adwin_alerts = drift.adwin_detections(alerts[order], time[te][order])
    alert_rate = {int(t): float(alerts[time[te] == t].mean()) for t in TEST_STEPS}

    # 4) adaptation strategies, each scored step by step
    strategies = {"static": {}, "expanding_retrain": {}, "sliding_retrain_w10": {}, "em_prior_adjust": {},
                  "expanding_retrain_retuned_thr": {}}
    em_priors = {}
    for t in TEST_STEPS:
        m_t = M["test"] & (time == t)
        yt = y[m_t]
        s = static.predict_proba(X[m_t])[:, 1]
        strategies["static"][t] = metrics.illicit_metrics(yt, s, thr)
        # labels assumed to arrive one step late: at step t we may train on everything up to t-1
        for key, lo in (("expanding_retrain", 1), ("sliding_retrain_w10", max(1, t - 10))):
            tr = d.labelled & (time >= lo) & (time <= t - 1)
            _, s2 = tabular.fit_predict("xgb", X[tr], y[tr], X[m_t], seed)
            strategies[key][t] = metrics.illicit_metrics(yt, s2, thr)
        # retrain AND re-pick the threshold: fit on <= t-2, tune the threshold on step t-1, refit on <= t-1
        tr2, last = d.labelled & (time <= t - 2), d.labelled & (time == t - 1)
        _, s_last = tabular.fit_predict("xgb", X[tr2], y[tr2], X[last], seed)
        thr_t = metrics.best_f1_threshold(y[last], s_last) if y[last].any() else thr
        tr1 = d.labelled & (time <= t - 1)
        _, s3 = tabular.fit_predict("xgb", X[tr1], y[tr1], X[m_t], seed)
        strategies["expanding_retrain_retuned_thr"][t] = metrics.illicit_metrics(yt, s3, thr_t)
        adj, est_prior = drift.em_prior_adjust(s, train_prior)
        em_priors[t] = est_prior
        strategies["em_prior_adjust"][t] = metrics.illicit_metrics(yt, adj, thr)

    summary = {}
    for k, per in strategies.items():
        f1s = [per[t]["f1"] for t in TEST_STEPS]
        summary[k] = {
            "mean_step_f1": float(np.mean(f1s)),
            "mean_step_f1_35_42": float(np.mean(f1s[:8])),
            "mean_step_f1_43_49": float(np.mean(f1s[8:])),
        }
        print(f"{k:22s}", {kk: round(v, 3) for kk, v in summary[k].items()})
    print("ADWIN on error stream (needs labels):", sorted(set(adwin)))
    print("ADWIN on alert stream (label-free):  ", sorted(set(adwin_alerts)))
    print("alert rate per step:", {k: round(v, 4) for k, v in alert_rate.items()})
    save_results("drift", {
        "feature_drift": feat_drift, "illicit_prior": prior, "train_prior": train_prior,
        "adwin_steps": sorted(set(adwin)), "adwin_alert_steps": sorted(set(adwin_alerts)), "alert_rate": alert_rate, "strategies": {k: {t: v["f1"] for t, v in per.items()} for k, per in strategies.items()},
        "strategy_summary": summary, "em_estimated_prior": em_priors, "threshold": thr,
    })


def stage_patterns(d, seeds):
    from chainsleuth import patterns, tabular

    M = masks(d)
    model, _ = tabular.fit_predict("xgb", d.nodes.loc[M["train"], d.af_cols].to_numpy(), d.label[M["train"]], d.nodes[d.af_cols].to_numpy()[:1])
    top = [d.af_cols[i] for i in np.argsort(-model.feature_importances_)[:8]]
    Xtr = d.nodes.loc[M["train"], d.af_cols].reset_index(drop=True)
    rules = patterns.illicit_rules(Xtr, d.label[M["train"]], top, min_support=0.01, min_confidence=0.5)
    # check each rule on the unseen test steps: does the pattern still predict illicit?
    Xte = d.nodes.loc[M["test"], d.af_cols].reset_index(drop=True)
    yte = d.label[M["test"]]
    bins = {f: np.quantile(Xtr[f], [1 / 3, 2 / 3]) for f in top}

    def holds(df, ante):
        ok = np.ones(len(df), bool)
        for item in ante.split(" AND "):
            f, b = item.split("=")
            lo, hi = bins[f]
            v = df[f].to_numpy()
            ok &= (v <= lo) if b == "low" else (v > hi) if b == "high" else ((v > lo) & (v <= hi))
        return ok

    test_conf, test_support = [], []
    for ante in rules["antecedents"]:
        h = holds(Xte, ante)
        test_support.append(float(h.mean()))
        test_conf.append(float(yte[h].mean()) if h.any() else float("nan"))
    rules["test_support"], rules["test_confidence"] = test_support, test_conf
    print(f"{len(rules)} rules -> illicit; top 10 by lift:")
    print(rules.head(10).to_string())
    save_results("patterns", {"top_features": top, "n_rules": int(len(rules)), "rules": rules.head(25).to_dict(orient="records")})


def stage_explain(d, seeds):
    """Fit the dashboard's XGBoost model; save it with its scores and SHAP global importance."""
    import joblib

    from chainsleuth import explain, tabular

    X = d.nodes[d.af_cols].to_numpy(np.float32)
    y = d.label
    M = masks(d)
    _, s_val = tabular.fit_predict("xgb", X[M["train_small"]], y[M["train_small"]], X[M["val"]], seeds[0])
    thr = metrics.best_f1_threshold(y[M["val"]], s_val)
    model, prob_all = tabular.fit_predict("xgb", X[M["train"]], y[M["train"]], X, seeds[0])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "xgb_af.joblib")
    np.save(ARTIFACTS_DIR / "xgb_prob.npy", prob_all)
    (ARTIFACTS_DIR / "xgb_meta.json").write_text(json.dumps({"threshold": thr, "features": d.af_cols}))

    rng = np.random.default_rng(0)
    sample = rng.choice(np.flatnonzero(M["test"]), size=3000, replace=False)
    sv = explain.shap_values(model, X[sample])
    imp = pd.Series(np.abs(sv).mean(0), index=d.af_cols).sort_values(ascending=False)
    print("SHAP top-10:", list(imp.head(10).index))
    save_results("explain", {"shap_top20": imp.head(20).to_dict(), "xgb_threshold": thr,
                             "xgb_test": metrics.illicit_metrics(y[M["test"]], prob_all[M["test"]], thr)})


def stage_gnn_explain(d, seeds):
    """GNNExplainer on the highest-scoring true-illicit test transactions under the saved GraphSAGE."""
    import torch

    from chainsleuth import explain, gnn

    sage_path = ARTIFACTS_DIR / "sage.pt"
    if not sage_path.exists():
        raise SystemExit("run the gnn stage first")
    M = masks(d)
    y = d.label
    p = gnn.prepare(d, d.af_cols, "real")
    model_g = gnn.GNN("sage", p.x.shape[1])
    model_g.load_state_dict(torch.load(sage_path))
    sage_prob = np.load(ARTIFACTS_DIR / "sage_prob.npy")
    cand = np.flatnonzero(M["test"] & (y == 1))
    examples = []
    for node in cand[np.argsort(-sage_prob[cand])][:5]:
        subset, sub_ei, edge_mask, feat_mask = explain.gnn_explain_node(model_g, p.x, p.edge_index, int(node))
        examples.append({
            "node": int(node), "tx_id": int(d.nodes.tx_id.iloc[node]), "sage_prob": float(sage_prob[node]),
            "subgraph_nodes": int(len(subset)), "subgraph_edges": int(sub_ei.shape[1]),
            "top_features": [d.af_cols[i] for i in np.argsort(-feat_mask)[:5]],
            "max_edge_importance": float(edge_mask.max()) if len(edge_mask) else 0.0,
        })
        print(examples[-1], flush=True)
    save_results("gnn_explain", {"examples": examples})


def stage_significance(d, seeds):
    """Exact McNemar test: LightGBM (AF) vs each GNN on identical test transactions, threshold 0.5, seed 0."""
    from chainsleuth import tabular

    probs_dir = ARTIFACTS_DIR / "gnn_test_probs"
    if not probs_dir.exists():
        raise SystemExit("run the gnn stage first")
    X = feature_sets(d)["AF"]
    y = d.label
    M = masks(d)
    _, s_lgbm = tabular.fit_predict("lgbm", X[M["train"]], y[M["train"]], X[M["test"]], seeds[0])
    pred_lgbm = (s_lgbm >= 0.5).astype(int)
    out = {}
    for f in sorted(probs_dir.glob(f"*_real_s{seeds[0]}.npy")):
        kind = f.name.split("_")[0]
        pred_g = (np.load(f) >= 0.5).astype(int)
        r = metrics.mcnemar(y[M["test"]], pred_lgbm, pred_g)
        out[f"lgbm_vs_{kind}"] = r
        print(f"LightGBM vs {kind}: {r}")
    save_results("significance", out)


# ───────────────────────── misc ─────────────────────────

def time_now() -> float:
    return time.perf_counter()


def mean_steps(steps_all: list[dict]) -> dict:
    """Average per-step F1 over seeds, for each threshold variant."""
    return {variant: {int(t): float(np.mean([s[variant][t] for s in steps_all])) for t in steps_all[0][variant]}
            for variant in steps_all[0]}


STAGES = {
    "eda": stage_eda, "tabular": stage_tabular, "unsupervised": stage_unsupervised, "gnn": stage_gnn,
    "hybrid": stage_hybrid, "semi": stage_semi, "drift": stage_drift, "patterns": stage_patterns,
    "explain": stage_explain, "gnn_explain": stage_gnn_explain, "significance": stage_significance,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="number of random seeds per stochastic model")
    ap.add_argument("--in-process", action="store_true", help="run stages in this process (one stage only is safe)")
    ap.add_argument("--gnn-kinds", nargs="+", default=list(GNN_KINDS), choices=list(GNN_KINDS),
                    help="GNN architectures this process trains (split across processes to use more cores)")
    args = ap.parse_args()
    seeds = list(SEEDS[: args.seeds])

    if not args.in_process:  # one fresh interpreter per stage
        data.load()  # download / cache once up front
        for name in args.stages:
            cmd = [sys.executable, "-m", "chainsleuth.pipeline", "--stages", name, "--seeds", str(args.seeds), "--in-process",
                   "--gnn-kinds", *args.gnn_kinds]
            if subprocess.run(cmd).returncode != 0:
                raise SystemExit(f"stage {name} failed")
        return

    d = data.load()
    for name in args.stages:
        print(f"\n=== {name} ===", flush=True)
        t0 = time_now()
        if name == "gnn":
            stage_gnn(d, seeds, kinds=args.gnn_kinds)
        else:
            STAGES[name](d, seeds)
        print(f"[{name} done in {time_now() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
