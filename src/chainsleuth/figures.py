"""Static report figures (PNG) built from reports/results/*.json.

    uv run python -m chainsleuth.figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chainsleuth.config import FIG_DIR, TRAIN_END
from chainsleuth.pipeline import load_results

# validated categorical order (dataviz reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / name)


def labels_per_step(res):
    rows = res["eda"]["per_step"]
    t = np.array([r["time"] for r in rows])
    ill, lic = np.array([r["illicit"] for r in rows]), np.array([r["licit"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, vals, name, col in ((axes[0], lic, "Licit transactions per time step", C[0]),
                                (axes[1], ill, "Illicit transactions per time step", C[1])):
        ax.bar(t, vals, color=col, width=0.8)
        ax.axvline(TRAIN_END + 0.5, color=INK2, lw=1, ls="--")
        ax.set_title(name)
    axes[1].text(TRAIN_END + 0.8, ill.max() * 0.9, "test →", color=INK2)
    axes[1].text(TRAIN_END - 0.2, ill.max() * 0.9, "← train", color=INK2, ha="right")
    axes[1].set_xlabel("time step")
    _save(fig, "labels_per_step.png")


def model_comparison(res):
    entries = []
    for sec in ("tabular", "hybrid", "gnn"):
        for k, v in res.get(sec, {}).items():
            if sec == "gnn" and not k.endswith("|real"):
                continue
            entries.append((k, v["metrics"]["f1_at_0.5"]["mean"], v["metrics"]["f1_at_0.5"]["std"], sec))
    entries = [e for e in entries if e[0].split("|")[1] in ("AF", "AF+G", "AF+SAGE-emb", "real")]
    entries.sort(key=lambda e: e[1])
    color = {"tabular": C[0], "hybrid": C[2], "gnn": C[1]}
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(entries) + 1))
    y = np.arange(len(entries))
    ax.barh(y, [e[1] for e in entries], xerr=[e[2] for e in entries], color=[color[e[3]] for e in entries],
            height=0.65, error_kw={"ecolor": INK2, "lw": 1})
    ax.set_yticks(y, [e[0] for e in entries])
    for yi, e in zip(y, entries):
        ax.text(e[1] + e[2] + 0.012, yi, f"{e[1]:.3f}", va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("illicit F1 on test steps 35–49 (threshold 0.5, mean ± sd over seeds)")
    ax.set_title("Model comparison")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color[s], label=l) for s, l in
                       (("tabular", "feature-only"), ("gnn", "graph neural network"), ("hybrid", "trees + GNN embeddings"))],
              loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "model_comparison.png")


def gnn_ablation(res):
    g = res["gnn"]
    kinds, variants = ("gcn", "sage", "gat"), ("real", "shuffled", "none")
    names = {"real": "real edges", "shuffled": "shuffled edges", "none": "no edges"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, metric, title in ((axes[0], "f1_at_0.5", "Illicit F1 (threshold 0.5)"), (axes[1], "pr_auc", "PR-AUC")):
        w = 0.26
        for i, v in enumerate(variants):
            means = [g[f"{k}|{v}"]["metrics"][metric]["mean"] for k in kinds]
            sds = [g[f"{k}|{v}"]["metrics"][metric]["std"] for k in kinds]
            ax.bar(np.arange(3) + (i - 1) * w, means, w * 0.92, yerr=sds, color=C[i], label=names[v],
                   error_kw={"ecolor": INK2, "lw": 1})
        ax.set_xticks(range(3), ["GCN", "GraphSAGE", "GAT"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    _save(fig, "gnn_edge_ablation.png")


def per_step(res):
    picks = [("tabular", "lgbm|AF", "LightGBM (AF)"), ("tabular", "rf|AF", "Random Forest (AF)"),
             ("gnn", "sage|real", "GraphSAGE"), ("gnn", "gcn|real", "GCN")]
    rows = {r["time"]: r["illicit"] for r in res["eda"]["per_step"]}
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.3})
    for i, (sec, key, label) in enumerate(picks):
        if key not in res.get(sec, {}):
            continue
        ps = res[sec][key]["per_step_f1"]["at_0.5"]
        t = sorted(int(k) for k in ps)
        axes[0].plot(t, [ps[str(k)] for k in t], color=C[i], marker="o", ms=4, label=label)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("illicit F1")
    axes[0].set_title("Per-time-step illicit F1 on the test period (threshold 0.5)")
    axes[0].legend(loc="lower left", ncol=2)
    t = list(range(TRAIN_END + 1, 50))
    axes[1].bar(t, [rows[k] for k in t], color=INK2, width=0.7)
    axes[1].set_title("Illicit transactions in each test step")
    axes[1].set_xlabel("time step")
    _save(fig, "per_step_f1.png")


def drift_fig(res):
    dr = res["drift"]
    t = sorted(int(k) for k in dr["feature_drift"])
    ks = [dr["feature_drift"][str(k)]["ks_reject_frac"] for k in t]
    prior = [dr["illicit_prior"][str(k)] for k in t]
    em = [dr["em_estimated_prior"][str(k)] for k in t]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True, gridspec_kw={"hspace": 0.38})
    axes[0].bar(t, ks, color=C[0], width=0.7)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Share of features whose distribution differs from training (KS test, p < 0.01)")
    axes[1].plot(t, prior, color=C[1], marker="o", ms=4, label="true illicit rate (labelled)")
    axes[1].plot(t, em, color=C[2], marker="s", ms=4, label="EM estimate from scores only")
    axes[1].axhline(dr["train_prior"], color=INK2, ls="--", lw=1)
    axes[1].text(44, dr["train_prior"] + 0.004, "training-period rate", va="bottom", color=INK2, fontsize=8)
    axes[1].set_title("Illicit base rate per step")
    axes[1].legend(loc="center left", bbox_to_anchor=(0.56, 0.42), fontsize=8.5)  # empty region over steps 43-47
    names = {"static": "static model", "expanding_retrain": "expanding-window retrain",
             "sliding_retrain_w10": "sliding-window retrain (10)", "em_prior_adjust": "EM prior adjustment",
             "expanding_retrain_retuned_thr": "expanding retrain + re-tuned threshold"}
    for i, (k, lab) in enumerate(names.items()):
        f = dr["strategies"][k]
        axes[2].plot(t, [f[str(s)] for s in t], color=C[i], marker="o", ms=3.5, label=lab)
    for s in dr.get("adwin_alert_steps", []):
        axes[2].axvline(s, color=INK2, lw=0.8, ls=":")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Per-step F1 by adaptation strategy (dotted: label-free ADWIN alarms on alert rate)")
    axes[2].set_xlabel("time step")
    axes[2].legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=3, fontsize=8)
    _save(fig, "drift.png")


def shap_fig(res):
    imp = res["explain"]["shap_top20"]
    items = list(imp.items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(range(len(items)), [v for _, v in items], color=C[0], height=0.65)
    ax.set_yticks(range(len(items)), [k for k, _ in items])
    ax.set_xlabel("mean |SHAP value| (log-odds), XGBoost on AF features")
    ax.set_title("Most influential features")
    ax.grid(axis="y", visible=False)
    _save(fig, "shap_importance.png")


def main():
    res = load_results()
    for fn, needs in ((labels_per_step, "eda"), (model_comparison, "tabular"), (gnn_ablation, "gnn"),
                      (per_step, "tabular"), (drift_fig, "drift"), (shap_fig, "explain")):
        if needs in res:
            fn(res)
        else:
            print(f"skip {fn.__name__}: no {needs} results")


if __name__ == "__main__":
    main()
