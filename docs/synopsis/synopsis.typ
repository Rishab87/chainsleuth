#set document(title: "ChainSleuth - Project Synopsis", author: ("Rishab Kumar Jha", "Arush Dwivedi", "Aditya Gupta"))
#set page(paper: "a4", margin: (x: 2cm, top: 1.7cm, bottom: 1.6cm))
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.9em)
#show heading.where(level: 1): it => block(above: 0.95em, below: 0.55em)[
  #set text(size: 13pt, weight: "bold", fill: rgb("#1f3a68"))
  #it.body
  #v(-0.45em)
  #line(length: 100%, stroke: 0.6pt + rgb("#1f3a68"))
]
#show heading.where(level: 2): it => block(above: 0.8em, below: 0.45em)[
  #set text(size: 11pt, weight: "bold")
  #it.body
]

#let accent = rgb("#1f3a68")
#show table: set par(justify: false)
#show table: set text(size: 9pt)

// ───────────────────────── COVER PAGE ─────────────────────────
#page(numbering: none)[
  #set align(center)
  #set par(justify: false)
  #set text(hyphenate: false)
  #show table: set text(size: 11.5pt)
  #v(0.6cm)
  #image("jiit_logo.png", width: 4.2cm)
  #v(0.3cm)
  #text(size: 15pt, weight: "bold", fill: accent)[JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY]
  #v(-0.2cm)
  #text(size: 12pt)[Noida, Uttar Pradesh]

  #v(1.3cm)
  #text(size: 12pt, tracking: 1.5pt)[PROJECT SYNOPSIS]
  #v(-0.1cm)
  #text(size: 12pt)[Machine Learning and Data Mining]

  #v(1.1cm)
  #line(length: 70%, stroke: 0.8pt + accent)
  #v(0.2cm)
  #text(size: 26pt, weight: "bold", fill: accent)[ChainSleuth]
  #v(0.05cm)
  #block(width: 85%)[
    #text(size: 14.5pt)[Detecting Illicit Bitcoin Transactions with Temporal Graph Mining and Graph Neural Networks under Concept Drift]
  ]
  #v(0.2cm)
  #line(length: 70%, stroke: 0.8pt + accent)

  #v(1fr)

  #grid(
    columns: (1fr, 1fr),
    align: (left + top, right + top),
    [
      #text(weight: "bold", fill: accent)[Submitted to:] \
      #v(0.15cm)
      #text(size: 11.5pt)[Dr. Ashish Mishra]
    ],
    [
      #text(weight: "bold", fill: accent)[Submitted by:]
      #v(0.15cm)
      #table(
        columns: 2,
        stroke: none,
        inset: (x: 0pt, y: 3pt),
        column-gutter: 14pt,
        align: (right, right),
        [Rishab Kumar Jha], [23803021],
        [Arush Dwivedi], [23803016],
        [Aditya Gupta], [23803018],
      )
    ],
  )
  #v(1.2cm)
  #text(size: 10.5pt)[November 2026]
]

#set page(numbering: "1", number-align: center)
#counter(page).update(1)

// ───────────────────────── CONTENT ─────────────────────────

= Abstract

Bitcoin is pseudonymous, not anonymous: every payment is written to a public ledger, which turns anti-money-laundering (AML) into a graph-mining problem. *ChainSleuth* is an end-to-end system that flags illicit Bitcoin transactions (ransomware, scams, Ponzi schemes and dark-market payments) on the Elliptic transaction graph of 203,769 transactions and 234,355 payment flows spread over 49 time steps. Only about 2% of transactions are labelled illicit and about 77% carry no label at all, so the task combines extreme class imbalance, semi-supervised learning and non-stationary data.

The project will (i) mine structural features from the transaction graph (centrality, k-core, community and neighbourhood statistics); (ii) benchmark unsupervised anomaly detectors, gradient-boosted trees and graph neural networks (GCN, GraphSAGE, GAT) under a strict, leakage-safe temporal split; (iii) test whether graph structure really adds signal, using edge-shuffling and inductive-versus-transductive ablations; (iv) detect and adapt to concept drift in later time steps; and (v) explain every alert with SHAP and GNNExplainer inside an interactive investigator dashboard. The deliverables are a reproducible pipeline, a comparative study with honest temporal evaluation, and a working forensic tool.

= Problem Statement and Motivation

Analysts trace illicit funds by following payment flows, but manual tracing does not scale. The original Elliptic study @weber2019 found that a Random Forest on hand-crafted features outperformed a Graph Convolutional Network, and it called for methods that combine the two. A 2026 re-evaluation @maganti2026 goes further. It argues that much of the reported GNN advantage comes from evaluation-time leakage through the graph's adjacency. It also reports that every model, feature-only ones included, collapses in the final time steps as the illicit base rate shifts. Two questions are therefore still open:

+ *When does graph structure actually help* illicit-transaction detection, once leakage is removed?
+ *How can a deployed detector notice and survive distribution shift* in a criminal ecosystem that keeps changing?

ChainSleuth addresses both. The aim is a classifier that stays dependable over time and explains its decisions, not a single headline accuracy number.

= Objectives

+ Build a reproducible data pipeline and exploratory analysis of the Elliptic graph: degree distributions, label dynamics over time, and the connected structure within each time step.
+ Engineer graph-mining features (PageRank, k-core number, clustering coefficient, Louvain community statistics, and the label composition of each node's ego-network) computed without crossing the temporal split.
+ Compare four model families under one temporal protocol: unsupervised (Isolation Forest, LOF), classical (Logistic Regression, Random Forest, XGBoost/LightGBM), neural (MLP) and graph (GCN, GraphSAGE, GAT, and a GNN-embedding + XGBoost hybrid).
+ Use the 77% unlabelled transactions through semi-supervised learning (label propagation and self-training) and measure the gain.
+ Quantify concept drift per time step (KS test and PSI on features, ADWIN on the model's error stream) and evaluate adaptation by sliding-window retraining and prior-shift threshold correction.
+ Make every alert explainable (SHAP for tree models, GNNExplainer for GNNs) and expose it through a dashboard where an analyst can inspect a flagged transaction and its k-hop neighbourhood.

= Dataset

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(170),
  inset: 5pt,
  fill: (x, y) => if y == 0 { rgb("#e8edf5") },
  [*Property*], [*Elliptic Bitcoin Dataset* @weber2019],
  [Nodes / edges], [203,769 transactions / 234,355 directed payment flows],
  [Features], [166 per node: time step, 93 local transaction features, 72 one-hop aggregated features],
  [Labels], [4,545 illicit (≈2%), 42,019 licit (≈21%), ≈157k unknown (≈77%)],
  [Time], [49 time steps about two weeks apart; each step is a connected component and no edges cross time steps],
  [Access], [Kaggle, and built into PyTorch Geometric (`EllipticBitcoinDataset`, `EllipticBitcoinTemporalDataset`)],
  [Extension], [Elliptic++ @elmougy2023: 822,942 wallet addresses with 56 features, used for optional wallet-level detection],
)

The data is public and small enough to train every model on a laptop CPU or a free Colab GPU.

= Proposed Methodology

#figure(
  {
    let box_(t) = rect(fill: rgb("#e8edf5"), stroke: 0.6pt + accent, radius: 3pt, inset: 5pt, width: 100%, height: 1.55cm)[#align(center + horizon)[#text(size: 8pt)[#t]]]
    let arr = align(center + horizon)[#text(size: 12pt, fill: accent)[→]]
    grid(
      columns: (1fr, 10pt, 1fr, 10pt, 1.15fr, 10pt, 1fr, 10pt, 1.2fr),
      align: horizon,
      box_[*Ingest & EDA* \ Elliptic / PyG \ temporal split],
      arr,
      box_[*Graph Mining* \ centrality, k-core, \ communities],
      arr,
      box_[*Models* \ IF/LOF, XGBoost \ GCN, SAGE, GAT],
      arr,
      box_[*Drift Layer* \ KS/PSI, ADWIN \ retraining],
      arr,
      box_[*Explain & Serve* \ SHAP, GNNExplainer \ dashboard],
    )
  },
  caption: [ChainSleuth pipeline],
)

== 1. Data Mining and Feature Engineering
The transaction graph is loaded with NetworkX/igraph and PyG. For each node we compute degree, PageRank, betweenness on sampled pivots, k-core number, local clustering coefficient, and Louvain community size and illicit density. Each feature is calculated only from edges and labels visible at training time, because a label leaking across the split would inflate the scores. Association-rule mining (Apriori) over discretised features of known illicit transactions will surface interpretable laundering patterns.

== 2. Unsupervised and Semi-Supervised Learning
Isolation Forest and Local Outlier Factor give a label-free baseline that matches how a new illicit scheme first appears. Label propagation and self-training then use the 157k unlabelled nodes, and the gain over purely supervised training is measured.

== 3. Supervised and Graph Models
Classical models (Logistic Regression, Random Forest, XGBoost/LightGBM) are trained with class weighting and focal loss to handle the 1:9 illicit-to-licit ratio among labelled nodes. GCN @kipf2017, GraphSAGE @hamilton2017 and GAT @velickovic2018 are trained in both transductive and strictly inductive settings. A hybrid model feeds GNN node embeddings into XGBoost. Two ablations test the role of structure: *edge shuffling*, which randomly rewires the edges, and *edge removal*, which reduces each GNN to an MLP.

== 4. Temporal Evaluation and Concept Drift
We follow the standard protocol @weber2019: train on time steps 1–34 and test on 35–49, with no shuffling. Metrics are reported *per time step*. Drift is tracked with the Kolmogorov–Smirnov test and the Population Stability Index on input features, and with ADWIN @bifet2007 on the error stream. Three responses are compared: a static model, sliding-window retraining, and threshold re-calibration for prior shift.

== 5. Explainability and Investigator Dashboard
SHAP @lundberg2017 values explain the tree models and GNNExplainer @ying2019 extracts the subgraph behind each GNN prediction. A Streamlit dashboard lets an analyst enter a transaction ID and see its risk score, its top contributing features, and an interactive view of its k-hop neighbourhood with known illicit nodes highlighted.

= Evaluation Metrics

Illicit-class *precision, recall and F1* are the primary metrics, as in prior work. *PR-AUC* and the *Matthews Correlation Coefficient* are also reported, because accuracy is misleading at a 2% base rate. Results include per-time-step F1 curves, mean ± standard deviation over 5 random seeds, and McNemar's test for pairwise model comparison.

= Tools and Technology Stack

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(170),
  inset: 5pt,
  fill: (x, y) => if y == 0 { rgb("#e8edf5") },
  [*Layer*], [*Tools*],
  [Language & env], [Python 3.12, `uv` (dependency management), Jupyter, Git/GitHub],
  [Data processing], [pandas, NumPy, Polars],
  [Graph mining], [NetworkX, python-igraph, python-louvain, mlxtend (Apriori)],
  [Machine learning], [scikit-learn, XGBoost, LightGBM, imbalanced-learn],
  [Deep learning], [PyTorch, PyTorch Geometric (GCN, GraphSAGE, GAT, GNNExplainer)],
  [Drift detection], [river (ADWIN), SciPy (KS test), custom PSI],
  [Explainability], [SHAP, PyG `Explainer`],
  [Tracking & viz], [MLflow, Matplotlib, Plotly, PyVis],
  [Application], [Streamlit dashboard, Docker for packaging],
  [Hardware], [Laptop CPU for classical models; Google Colab GPU for GNNs],
)

= Work Plan

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170),
  inset: 5pt,
  fill: (x, y) => if y == 0 { rgb("#e8edf5") },
  [*Phase*], [*Activities*], [*Duration*],
  [1], [Literature review, data ingestion, EDA, temporal split, baseline LR/RF], [Weeks 1–2],
  [2], [Graph-mining features, Apriori patterns, Isolation Forest/LOF, XGBoost/LightGBM], [Weeks 3–4],
  [3], [GCN/GraphSAGE/GAT, hybrid model, semi-supervised learning, ablations], [Weeks 5–7],
  [4], [Per-time-step evaluation, drift detection and adaptation strategies], [Weeks 8–9],
  [5], [SHAP/GNNExplainer, Streamlit dashboard, report and presentation], [Weeks 10–11],
)

*Responsibilities.* _Rishab Kumar Jha:_ graph neural networks, hybrid model and overall pipeline architecture. _Arush Dwivedi:_ EDA, graph-mining feature engineering, association rules and unsupervised detection. _Aditya Gupta:_ classical baselines, drift detection and adaptation, explainability and dashboard. All three members share the evaluation and the final report.

= Expected Outcomes

- A reproducible, open-source pipeline that goes from raw Elliptic data to a deployable illicit-transaction detector.
- A leakage-safe comparison of feature-based and graph-based models that states when graph structure helps and when it hurts.
- An empirical study of concept drift on a real financial-crime dataset, with a measured comparison of adaptation strategies.
- An explainable investigator dashboard showing how the detector would be used in a compliance workflow.

#set text(size: 9pt)
#bibliography("refs.yml", title: "References", style: "ieee")
