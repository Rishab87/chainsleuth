"""ChainSleuth investigator dashboard.

    uv run streamlit run app/dashboard.py

Reads artifacts produced by `python -m chainsleuth.pipeline` (explain + gnn stages).
Never imports torch: GraphSAGE scores are precomputed, SHAP runs on the XGBoost model.
"""

from __future__ import annotations

import json

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chainsleuth import data, explain
from chainsleuth.config import ARTIFACTS_DIR, FIG_DIR, TRAIN_END
from chainsleuth.pipeline import load_results

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
LABEL_NAME = {1: "illicit", 0: "licit", -1: "unknown"}

st.set_page_config(page_title="ChainSleuth", layout="wide")


@st.cache_resource
def load_all():
    d = data.load()
    model = joblib.load(ARTIFACTS_DIR / "xgb_af.joblib")
    meta = json.loads((ARTIFACTS_DIR / "xgb_meta.json").read_text())
    xgb_prob = np.load(ARTIFACTS_DIR / "xgb_prob.npy")
    sage_path = ARTIFACTS_DIR / "sage_prob.npy"
    sage_prob = np.load(sage_path) if sage_path.exists() else None
    sage_meta = json.loads((ARTIFACTS_DIR / "sage_meta.json").read_text()) if sage_path.exists() else None
    und = nx.Graph()
    und.add_nodes_from(range(len(d.nodes)))
    und.add_edges_from(d.edge_index.T.tolist())
    id_to_idx = pd.Series(np.arange(len(d.nodes)), index=d.nodes["tx_id"])
    return d, model, meta, xgb_prob, sage_prob, sage_meta, und, id_to_idx


def neighbourhood_figure(d, und, node, hops, prob):
    sub_nodes = nx.single_source_shortest_path_length(und, node, cutoff=hops)
    if len(sub_nodes) > 300:  # keep the plot readable: closest nodes first
        sub_nodes = dict(sorted(sub_nodes.items(), key=lambda kv: kv[1])[:300])
    sub = und.subgraph(sub_nodes)
    pos = nx.spring_layout(sub, seed=0, k=1 / np.sqrt(max(len(sub), 1)))
    ex, ey = [], []
    for a, b in sub.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#d6d5d0", width=1), hoverinfo="skip", showlegend=False))
    lab = d.label
    for code, name, color, symbol in ((1, "illicit (known)", ORANGE, "diamond"), (0, "licit (known)", BLUE, "circle"),
                                      (-1, "unknown", GREY, "circle-open")):
        ns = [n for n in sub.nodes if lab[n] == code and n != node]
        if not ns:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers", name=name,
            marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1.5, color=color)),
            text=[f"tx {d.nodes.tx_id.iloc[n]}<br>{LABEL_NAME[code]}<br>risk {prob[n]:.2f}<br>hops {sub_nodes[n]}" for n in ns],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[pos[node][0]], y=[pos[node][1]], mode="markers", name="selected",
                             marker=dict(size=18, color="#0b0b0b", symbol="star"),
                             text=[f"tx {d.nodes.tx_id.iloc[node]} (selected)"], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fcfcfb",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=-0.02))
    return fig, len(sub), sub.number_of_edges()


def page_investigate(d, model, meta, xgb_prob, sage_prob, sage_meta, und, id_to_idx):
    st.subheader("Investigate a transaction")
    thr = st.slider("Alert threshold (risk score at or above this is flagged)", 0.05, 0.95, 0.5, 0.05,
                    help="0.5 is the default used in the results tables. Lower it to catch more illicit "
                         "transactions at the cost of more false alarms.")
    test_rows = np.flatnonzero(d.time > TRAIN_END)
    flagged = test_rows[np.argsort(-xgb_prob[test_rows])][:200]
    c1, c2 = st.columns([1, 1])
    with c1:
        pick = st.selectbox("Top-scored test-period transactions", [int(d.nodes.tx_id.iloc[i]) for i in flagged])
    with c2:
        typed = st.text_input("…or enter any transaction ID", "")
    tx = int(typed) if typed.strip().isdigit() else pick
    if tx not in id_to_idx.index:
        st.error(f"Transaction {tx} is not in the Elliptic dataset.")
        return
    node = int(id_to_idx[tx])
    row = d.nodes.iloc[node]

    m = st.columns(5)
    m[0].metric("Time step", int(row.time), "test period" if row.time > TRAIN_END else "training period", delta_color="off")
    m[1].metric("Ground truth", LABEL_NAME[int(row.label)])
    m[2].metric("XGBoost risk", f"{xgb_prob[node]:.3f}", "FLAGGED" if xgb_prob[node] >= thr else "below threshold",
                delta_color="inverse" if xgb_prob[node] >= thr else "off")
    if sage_prob is not None:
        m[3].metric("GraphSAGE risk", f"{sage_prob[node]:.3f}",
                    "FLAGGED" if sage_prob[node] >= thr else "below threshold",
                    delta_color="inverse" if sage_prob[node] >= thr else "off")
    m[4].metric("Degree", und.degree(node))

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Why the model scored it this way** (SHAP, XGBoost)")
        x = d.nodes.loc[[node], d.af_cols].to_numpy(np.float32)
        sv = explain.shap_values(model, x)[0]
        top = np.argsort(-np.abs(sv))[:12][::-1]
        fig = go.Figure(go.Bar(
            x=sv[top], y=[d.af_cols[i] for i in top], orientation="h",
            marker_color=[ORANGE if v > 0 else BLUE for v in sv[top]],
            customdata=x[0, top], hovertemplate="%{y}<br>SHAP %{x:.3f}<br>feature value %{customdata:.3f}<extra></extra>",
        ))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fcfcfb",
                          xaxis_title="contribution to log-odds of illicit (orange pushes towards illicit)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        hops = st.radio("Neighbourhood depth", [1, 2, 3], index=1, horizontal=True)
        fig, n_nodes, n_edges = neighbourhood_figure(d, und, node, hops, xgb_prob)
        st.markdown(f"**{hops}-hop payment neighbourhood**: {n_nodes} transactions, {n_edges} flows (same time step only)")
        st.plotly_chart(fig, use_container_width=True)


def page_overview(res):
    st.subheader("Results on the held-out test period (time steps 35–49)")
    rows = []
    for sec in ("tabular", "hybrid", "gnn"):
        for k, v in res.get(sec, {}).items():
            mm = v["metrics"]
            rows.append({"model": k, "family": sec, "F1 @0.5": mm["f1_at_0.5"]["mean"], "F1 (val threshold)": mm["f1"]["mean"],
                         "precision": mm["precision"]["mean"], "recall": mm["recall"]["mean"], "PR-AUC": mm["pr_auc"]["mean"],
                         "MCC": mm["mcc"]["mean"], "seeds": v["n_seeds"]})
    if rows:
        df = pd.DataFrame(rows).sort_values("F1 @0.5", ascending=False)
        st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns if df[c].dtype == float}), use_container_width=True, hide_index=True)
    for name in ("model_comparison.png", "gnn_edge_ablation.png", "per_step_f1.png"):
        if (FIG_DIR / name).exists():
            st.image(str(FIG_DIR / name))


def page_drift(res):
    st.subheader("Concept drift")
    if "drift" not in res:
        st.info("Run the drift stage first.")
        return
    dr = res["drift"]
    st.dataframe(pd.DataFrame(dr["strategy_summary"]).T.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"ADWIN alarms on the static model's error stream at steps: {dr['adwin_steps']}")
    if (FIG_DIR / "drift.png").exists():
        st.image(str(FIG_DIR / "drift.png"))
    if "patterns" in res:
        st.subheader("Laundering patterns mined with Apriori (training steps), checked on test steps")
        st.dataframe(pd.DataFrame(res["patterns"]["rules"]).head(15), use_container_width=True, hide_index=True)


def main():
    st.title("ChainSleuth")
    st.caption("Illicit Bitcoin transaction detection on the Elliptic graph: graph mining, GNNs, drift-aware learning")
    if not (ARTIFACTS_DIR / "xgb_af.joblib").exists():
        st.error("No trained model found. Run `uv run python -m chainsleuth.pipeline` first.")
        return
    loaded = load_all()
    res = load_results()
    tab1, tab2, tab3 = st.tabs(["Investigate", "Model results", "Drift & patterns"])
    with tab1:
        page_investigate(*loaded)
    with tab2:
        page_overview(res)
    with tab3:
        page_drift(res)


main()
