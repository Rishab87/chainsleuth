#set document(title: "ChainSleuth - Project Report", author: ("Rishab Kumar Jha", "Arush Dwivedi", "Aditya Gupta"))
#set page(paper: "a4", margin: (x: 2cm, top: 1.7cm, bottom: 1.6cm))
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.85em)
#set heading(numbering: "1.1")
#let accent = rgb("#1f3a68")
#show heading.where(level: 1): it => block(above: 1.0em, below: 0.55em)[
  #set text(size: 13pt, weight: "bold", fill: accent)
  #if it.numbering != none [#counter(heading).display() #h(0.3em)]#it.body
  #v(-0.45em)
  #line(length: 100%, stroke: 0.6pt + accent)
]
#show heading.where(level: 2): it => block(above: 0.85em, below: 0.45em)[
  #set text(size: 11pt, weight: "bold")
  #counter(heading).display() #h(0.25em) #it.body
]
#show table: set par(justify: false)
#show table: set text(size: 8.8pt)
#show figure.caption: set text(size: 9pt)
#show figure: set block(above: 0.9em, below: 0.9em)
#set figure(gap: 0.5em, placement: auto)
#set place(clearance: 1.1em)
#let fig(path) = "/reports/figures/" + path
#let hdr = (x, y) => if y == 0 { rgb("#e8edf5") }

// ───────────────────────── COVER PAGE ─────────────────────────
#page(numbering: none)[
  #set align(center)
  #set par(justify: false)
  #set text(hyphenate: false)
  #show table: set text(size: 11.5pt)
  #v(0.6cm)
  #image("img/jiit_logo.png", width: 4.2cm)
  #v(0.3cm)
  #text(size: 15pt, weight: "bold", fill: accent)[JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY]
  #v(-0.2cm)
  #text(size: 12pt)[Noida, Uttar Pradesh]

  #v(1.3cm)
  #text(size: 12pt, tracking: 1.5pt)[PROJECT REPORT]
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
      #text(weight: "bold", fill: accent)[Presented to:] \
      #v(0.15cm)
      #text(size: 11.5pt)[Dr. Ashish Mishra]
    ],
    [
      #text(weight: "bold", fill: accent)[Presented by:]
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

// ───────────────────────── ABSTRACT + CONTENTS ─────────────────────────
#heading(numbering: none, outlined: false)[Abstract]

Graph neural networks (GNNs) are widely reported as the state of the art for detecting illicit transactions on the Elliptic Bitcoin graph. We built *ChainSleuth*, an end-to-end, reproducible pipeline that tests this claim under a strict temporal protocol and studies what happens after deployment. The pipeline trains on time steps 1–34 and evaluates on the unseen steps 35–49. It covers eighteen feature-based configurations (logistic regression, random forest, XGBoost, LightGBM, MLP), three GNN architectures (GCN, GraphSAGE, GAT) with a real / shuffled / no-edge ablation, graph-mining features, a hybrid of trees and GNN embeddings, unsupervised anomaly detection, self-training on 157,205 unlabelled transactions, Apriori rule mining, and concept-drift detection and adaptation.

The main results are as follows. LightGBM on the raw 165 features gives the best illicit-class F1, *0.827 ± 0.001*. The best GNN, GraphSAGE, reaches *0.714 ± 0.014*, and the gap is significant (exact McNemar test, $p = 1.5 times 10^(-42)$). The edge ablation shows that the real transaction graph helps GraphSAGE (0.714 against 0.604 with no edges) but *hurts* GCN (0.562 against 0.598). Every model collapses after time step 42: a static XGBoost model falls from a mean per-step F1 of 0.897 to 0.041. Retraining at every step with a re-tuned threshold recovers it to 0.452. ADWIN on the model's alert rate, which needs no labels, raises an alarm exactly at step 43. ADWIN on the error stream does not. The system ships with an explainable Streamlit dashboard for investigators and 27 automated tests.

#v(0.4em)
#block(inset: (left: 0pt))[
  #set text(size: 10pt)
  #outline(title: [Contents], depth: 2, indent: 1.2em)
]

// ───────────────────────── 1 INTRODUCTION ─────────────────────────
= Introduction

Cryptocurrencies make cross-border value transfer cheap, and that same property makes them useful for ransomware payouts, darknet markets and money laundering. Bitcoin is pseudonymous rather than anonymous: every transaction is public, so the flow of funds forms a graph that can be mined. Anti-money-laundering (AML) teams need a detector that ranks transactions by risk, explains its decisions, and keeps working as criminal behaviour changes.

The Elliptic dataset @weber2019 is the standard public benchmark for this task. Since its release, many papers have reported that GNNs outperform classical models on it. Recent work questions this. Maganti @maganti2026 finds that, under a leakage-free inductive evaluation, feature-only models beat GNNs. Šafář et al. @safar2026 argue that the standard splits themselves are not properly isolated. A final-year project that reruns "GNN beats random forest" would not be adding anything.

*Objectives.* ChainSleuth asks four questions and answers each one with a controlled experiment:

+ Do GNNs beat strong feature-only models when evaluation is strictly temporal and every model gets the same protocol?
+ Is the graph structure itself useful, or would random edges do as well?
+ How badly do models degrade after deployment, and can drift be detected *without* labels?
+ Can classical data-mining tools (graph features, unsupervised detection, association rules, self-training) add value on top of supervised models?

*Contributions.* (i) A reproducible pipeline with one command per stage, fixed seeds and auto-generated results. (ii) An edge ablation showing that only GraphSAGE benefits from the real graph. (iii) A per-step drift study comparing five adaptation strategies, together with a label-free monitor that detects the collapse when it happens. (iv) An investigator dashboard with SHAP explanations and neighbourhood views. (v) A list of methods that did *not* help, with the evidence for each.

// ───────────────────────── 2 RELATED WORK ─────────────────────────
= Related Work

Weber et al. @weber2019 released the Elliptic dataset and reported that a random forest on all features (F1 0.788) beat a GCN (F1 0.628), and that every method failed after a dark-market shutdown around step 43. Our random forest reaches F1 0.809 under the same split. The standard GNN families are the spectral-style GCN @kipf2017, inductive GraphSAGE @hamilton2017, which keeps a separate weight for the node's own features, and attention-based GAT @velickovic2018. Elliptic++ @elmougy2023 adds wallet-level data and again finds tree ensembles strongest. Maganti @maganti2026 reports that in a strict inductive setting random edges can outperform real ones. Šafář et al. @safar2026 reverse-engineer the anonymised features and report leakage between the commonly used splits.

For tabular baselines we use XGBoost @chen2016 and LightGBM @ke2017. For data mining we use Louvain communities @blondel2008, Apriori @agrawal1994, Isolation Forest @liu2008 and LOF @breunig2000. For drift we use ADWIN @bifet2007 and EM prior re-estimation @saerens2002, and for explanations SHAP @lundberg2017 and GNNExplainer @ying2019.

// ───────────────────────── 3 DATA ─────────────────────────
= Dataset and Exploratory Analysis

The Elliptic dataset @weber2019 (PyTorch Geometric mirror) is a directed graph of Bitcoin transactions. Table 1 summarises it. Each node has 166 anonymised features: a time step, 93 _local_ features of the transaction itself (fees, input/output counts, amounts) and 72 _aggregated_ features that summarise its one-hop neighbourhood. Time steps are about two weeks apart.

#figure(
  table(
    columns: (1fr, auto, 1fr, auto),
    stroke: 0.4pt + luma(170), inset: 4.5pt, fill: hdr,
    [*Property*], [*Value*], [*Property*], [*Value*],
    [Transactions (nodes)], [203,769], [Illicit / licit / unknown], [4,545 / 42,019 / 157,205],
    [Directed payment edges], [234,355], [Train steps 1–34 (illicit / licit)], [3,462 / 26,432],
    [Features], [166], [Test steps 35–49 (illicit / licit)], [1,083 / 15,587],
    [Time steps], [49], [Edges crossing time steps], [*0*],
  ),
  caption: [Elliptic dataset statistics, computed by the pipeline and checked against the published values in the test suite.],
)

Three EDA findings shaped the design:

- *77% of nodes are unlabelled.* This motivates the semi-supervised experiment. Unlabelled nodes still take part in message passing.
- *Classes are imbalanced, and the imbalance drifts.* Illicit transactions are 11.6% of labelled training transactions but 6.5% of the test period. Within steps 43–47 the illicit share is only 0.3–2.6% (Figure 1). Accuracy is therefore meaningless; we report illicit-class precision, recall, F1, PR-AUC and MCC.
- *No edge crosses a time step.* Each step is a separate subgraph. With a temporal split, training nodes cannot exchange messages with test nodes, so transductive and inductive GNN training coincide. This rules out the graph leakage criticised in @maganti2026, and a test asserts it.

#figure(image(fig("labels_per_step.png"), width: 88%), caption: [Labelled transactions per time step. The dashed line separates training (1–34) from test (35–49) steps.])

// ───────────────────────── 4 METHODOLOGY ─────────────────────────
= Methodology

== Evaluation protocol

Following @weber2019, models train on labelled transactions in steps 1–34 and are tested on steps 35–49; nothing is shuffled across time. Steps 30–34 serve as a validation block: a model is first fitted on steps 1–29, the GNN epoch count and an alternative decision threshold are chosen on 30–34, and the model is then refitted on all of 1–34. Headline numbers use the default threshold 0.5. We also report F1 at the validation-tuned threshold, which turned out to transfer poorly (itself a sign of drift). Tree models and the MLP are averaged over 5 seeds and GNNs over 3; logistic regression is deterministic. Hyperparameters are fixed and untuned for every model, so the comparison is about model families and not about tuning effort. Paired comparisons use the exact McNemar test @mcnemar1947 on the same test transactions.

== Feature-based models

We evaluate logistic regression, random forest (50 trees and 50 features per split, the settings of @weber2019), XGBoost (400 trees, depth 6, learning rate 0.05), class-weighted XGBoost, LightGBM (400 trees, 63 leaves) and a one-hidden-layer MLP. Each runs on three feature sets: *LF* (93 local features), *AF* (all 165) and *AF+G* (AF plus the graph features below).

== Graph-mining features

For each time step we build the step's subgraph with NetworkX and compute ten structural features per node: in- and out-degree, PageRank, clustering coefficient, k-core number, sampled betweenness ($k = 256$), Louvain @blondel2008 community size and density, mean neighbour degree, and two-hop ego-network size. All are computed per step, so no information flows from the future.

== Graph neural networks and edge ablation

We train two-layer GCN @kipf2017, GraphSAGE @hamilton2017 and GAT @velickovic2018 models (64 hidden units, dropout 0.5, 4 attention heads for GAT) in PyTorch Geometric @fey2019. Training is full-batch on the undirected graph, with Adam (learning rate 0.01, weight decay $5 times 10^(-4)$), class weights 0.3 / 0.7, up to 300 epochs and the epoch count selected on validation PR-AUC. Features are standardised with training-period statistics. To isolate the contribution of structure, each architecture is trained on three graphs:

- *real*: the true transaction graph;
- *shuffled*: edge destinations permuted within each time step, which keeps the number of edges per step but destroys the structure;
- *none*: no edges, so the GNN reduces to a per-node network.

If real edges do not beat both controls, the graph is not adding information.

== Hybrid, unsupervised and semi-supervised models

*Hybrid.* The 64-dimensional GraphSAGE embeddings are appended to the AF features and fed to the tree models.

*Unsupervised.* Isolation Forest @liu2008 (300 trees) and LOF @breunig2000 ($k = 35$, novelty mode, fitted on a 20,000-row sample) use no labels. Each flags the top 11.6% most anomalous test transactions, matching the training illicit rate.

*Self-training.* XGBoost is trained on labelled steps 1–34 and then labels unlabelled training-period nodes with confidence $>= 0.95$ (illicit) or $<= 0.02$ (licit). Each of three rounds adds at most 25% of the pool.

== Association-rule mining

The eight most important XGBoost features are discretised into tertiles (low / mid / high, with bin edges set on training data). Apriori @agrawal1994 then mines rules with the consequent _illicit_ (minimum support 0.01, minimum confidence 0.5, at most 4 items). Every rule is re-scored on the test steps to see whether the pattern survives over time.

== Drift detection and adaptation

Using a static XGBoost model on AF, we measure: (a) input drift per test step, as the share of all 165 features for which a two-sample KS test against the training distribution rejects at $p < 0.01$, plus the mean PSI of the model's top-10 features; (b) the illicit base rate per step, and its estimate from model scores alone via EM @saerens2002; and (c) ADWIN @bifet2007 alarms on two streams ordered by time: the model's *error* indicator, which needs labels, and its *alert* indicator (was the transaction flagged?), which does not.

We compare five strategies, scored step by step:

- *static*: train once on steps 1–34;
- *expanding retrain*: before step $t$, retrain on all labels up to $t-1$, assuming labels arrive one step late;
- *sliding retrain*: as above, using only the last 10 steps;
- *EM prior adjustment*: re-weight the static scores to the estimated prior;
- *expanding retrain with re-tuned threshold*: also choose the threshold on step $t-1$.

== Explainability

TreeSHAP @lundberg2017 gives global and per-transaction feature attributions for XGBoost. GNNExplainer @ying2019, run through the PyG `Explainer` API, gives feature and edge masks for GraphSAGE on the top-scored illicit test transactions.

// ───────────────────────── 5 SYSTEM ─────────────────────────
= System Design and Implementation

#let stage(body, c: rgb("#e8edf5")) = box(fill: c, stroke: 0.6pt + accent, inset: (x: 5pt, y: 4pt), radius: 2pt)[#set text(size: 8.3pt); #body]
#let arr = text(fill: accent, size: 11pt)[$arrow.r$]
#figure(
  align(center, grid(
    columns: 7, column-gutter: 3pt, align: horizon,
    stage[*Ingest* \ download, parquet \ cache, masks],
    arr,
    stage[*Features* \ LF / AF / AF+G \ (NetworkX, Louvain)],
    arr,
    stack(spacing: 3pt,
      stage[*Tabular* LR, RF, XGB, LGBM, MLP],
      stage[*GNN* GCN, SAGE, GAT × 3 graphs],
      stage[*Mining* IF, LOF, Apriori, self-training],
    ),
    arr,
    stack(spacing: 3pt,
      stage[*Evaluate* F1, PR-AUC, MCC, McNemar],
      stage[*Drift* KS, PSI, ADWIN, EM, retraining],
      stage[*Explain* SHAP, GNNExplainer],
    ),
  )) + v(4pt) + align(center)[#arr #h(2pt) #stage(c: rgb("#fdf1e8"))[*Outputs:* `reports/results/*.json` #sym.arrow figures, `RESULTS.md`, Streamlit dashboard, 27 pytest tests]],
  kind: image,
  caption: [ChainSleuth pipeline. Each stage runs in its own process and writes its own results file.],
)

The project is a Python 3.12 package (`src/chainsleuth`, about 1,900 lines including the app and tests) managed with `uv`. `python -m chainsleuth.pipeline --stages …` runs any subset of the eleven stages. Results are written per stage to JSON, and the figures and the results document are regenerated from those files, so every number in this report can be traced to a file. The GNN stage caches each finished (architecture, graph, seed) run, so an interrupted run resumes where it stopped. Re-running a stage reproduces identical numbers.

#figure(
  table(
    columns: (auto, 1fr),
    stroke: 0.4pt + luma(170), inset: 4.5pt, fill: hdr, align: (left, left),
    [*Layer*], [*Tools (versions used)*],
    [Data and graph mining], [pandas 3.1, NumPy 2.5, PyArrow, NetworkX 3.7, python-louvain],
    [Classical ML], [scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, mlxtend 0.25 (Apriori)],
    [Graph learning], [PyTorch 2.15, PyTorch Geometric 2.8 (GCNConv, SAGEConv, GATConv, Explainer)],
    [Drift and statistics], [river 0.27 (ADWIN), SciPy 1.18 (KS, exact binomial McNemar), custom PSI and EM],
    [Explainability], [SHAP 0.54 (TreeSHAP), GNNExplainer],
    [Visualisation and app], [Matplotlib, Plotly, Streamlit 1.67],
    [Engineering], [uv, pytest (27 tests), Apple M5 CPU only],
  ),
  caption: [Technology stack.],
)

*Engineering challenge: OpenMP runtime clash.* On macOS, XGBoost and LightGBM link Homebrew's `libomp` while PyTorch ships its own copy. We reproduced both failure modes in isolation: importing torch first makes XGBoost segfault, and importing the tree libraries first makes multithreaded torch operations deadlock. The fix was architectural. Every stage runs in a separate process, `chainsleuth.tabular` refuses to load if torch is already imported, and `chainsleuth.gnn` drops torch to one thread when a tree library is present (as in the test suite). We also measured Apple's MPS GPU backend for full-batch GNN training and found it slower than the CPU (320–425 ms against 190–260 ms per epoch), so all training runs on the CPU.

*Testing.* 27 pytest tests cover the metrics, edge-shuffle invariants, the step-locality of graph features, every model family on a synthetic graph, PSI/KS, ADWIN, the EM prior, self-training and Apriori. They also check the real data against its published statistics (node, edge and label counts, zero cross-step edges) and run a headless smoke test of the dashboard.

*Investigator dashboard.* The Streamlit app (Figure 3) lets an analyst choose a transaction from the top-scored test-period list or type any ID. It shows XGBoost and GraphSAGE risk scores against an adjustable threshold, a per-transaction SHAP explanation, and an interactive 1–3 hop neighbourhood graph coloured by known label. Further tabs show the model results, the drift study and the mined rules. The app never imports torch, because GNN scores are precomputed.

#figure(image("img/dash_investigate.png", width: 74%), caption: [Investigate tab of the dashboard: an illicit test-period transaction flagged by both models, its SHAP explanation, and its two-hop payment neighbourhood (orange diamonds are known illicit transactions).])

// ───────────────────────── 6 RESULTS ─────────────────────────
= Results and Discussion

All numbers are on labelled transactions in test steps 35–49 (1,083 illicit and 15,587 licit), reported as mean ± standard deviation over seeds.

== Feature-based models against GNNs

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    stroke: 0.4pt + luma(170), inset: 4pt, fill: hdr, align: (left, left, center, center, center, center, center),
    [*Model*], [*Input*], [*F1 \@0.5*], [*Precision*], [*Recall*], [*PR-AUC*], [*MCC*],
    [*LightGBM*], [AF], [*0.827 ± 0.001*], [0.940], [0.730], [*0.806*], [*0.819*],
    [LightGBM], [AF+G], [0.826 ± 0.001], [0.928], [0.729], [0.806], [0.812],
    [XGBoost], [AF], [0.815 ± 0.005], [0.692], [0.744], [0.803], [0.697],
    [Random forest], [AF], [0.809 ± 0.003], [0.679], [0.737], [0.782], [0.681],
    [XGBoost (class-weighted)], [AF], [0.792 ± 0.004], [0.838], [0.737], [0.801], [0.772],
    [LightGBM], [LF], [0.776 ± 0.003], [0.805], [0.716], [0.788], [0.743],
    [LightGBM + SAGE emb. (hybrid)], [AF+emb], [0.746 ± 0.003], [0.921], [0.634], [0.764], [0.751],
    [*GraphSAGE*], [real graph], [*0.714 ± 0.014*], [0.931], [0.559], [0.731], [0.708],
    [MLP], [LF], [0.669 ± 0.044], [0.591], [0.726], [0.596], [0.624],
    [GAT], [real graph], [0.578 ± 0.028], [0.796], [0.467], [0.625], [0.590],
    [GCN], [real graph], [0.562 ± 0.015], [0.878], [0.324], [0.596], [0.516],
    [Logistic regression], [LF], [0.494], [0.435], [0.698], [0.277], [0.513],
  ),
  caption: [Selected models on the test period. F1 uses threshold 0.5; precision, recall and MCC use the validation-tuned threshold; PR-AUC is threshold-free. The full 30-row table is in `reports/RESULTS.md`.],
)

Gradient-boosted trees on all 165 features win clearly (Table 3, Figure 4). LightGBM beats the best GNN by 0.11 F1. On the same test transactions, seed 0 LightGBM is right and GraphSAGE wrong in 253 cases, against 34 the other way (exact McNemar $p = 1.5 times 10^(-42)$); against GCN and GAT the counts are 453 / 33 and 566 / 33. Our random forest (0.809) is close to the 0.788 reported by @weber2019, and our GCN (0.562) is in line with their 0.628, which suggests the protocol is comparable. The aggregated features matter: going from LF to AF lifts LightGBM from 0.776 to 0.827. These 72 features already summarise each node's neighbourhood, which leaves little for message passing to add. The same explanation fits the flat result for hand-crafted graph features (AF+G 0.826 against AF 0.827). The MLP, in contrast, gets *worse* when the aggregated features are added (0.669 on LF against 0.553 on AF), probably because of heavy-tailed features and our untuned settings.

#figure(image(fig("model_comparison.png"), width: 80%), caption: [Illicit F1 at threshold 0.5 for feature-only models, GNNs on the real graph, and the hybrid.])

*The hybrid underperforms (0.746).* The GraphSAGE embeddings for training nodes come from a network fitted on those same labels, so on training data they are over-confident and the trees learn to over-trust them. This is the classic stacking leak; out-of-fold embeddings would be needed to fix it. Weber et al. @weber2019 report a small gain from the same idea (random forest 0.788 to 0.796), so the leak, not the idea, is the likely culprit.

== Does the graph structure help?

#figure(image(fig("gnn_edge_ablation.png"), width: 90%), caption: [Edge ablation: each architecture trained on the real graph, a within-step shuffled graph, and no edges (mean ± sd over 3 seeds).])

The ablation (Figure 5) gives a nuanced answer. For *GraphSAGE*, real edges (0.714) beat no edges (0.604), which beat shuffled edges (0.544). For *GCN*, removing the edges *improves* F1 from 0.562 to 0.598. *GAT* is flat (0.578 real against 0.570 none). Shuffled edges hurt all three, so the real graph carries more information than random structure, but only GraphSAGE can use it. GraphSAGE keeps a separate weight for the node's own features, whereas GCN averages a node's features with its neighbours' and so dilutes a strong per-node signal on a sparse graph. Unlike @maganti2026, we do not find random edges beating real ones, and our setting is equivalent to inductive training because Elliptic has no cross-step edges.

== Concept drift: collapse, detection and recovery

#figure(image(fig("per_step_f1.png"), width: 86%), caption: [Per-step illicit F1 on the test period (threshold 0.5) and the number of illicit transactions per step.])

All models perform well up to step 42 and then collapse (Figure 6). Static XGBoost averages a per-step F1 of 0.897 on steps 35–42 and 0.041 on steps 43–49 (Table 4). This matches the dark-market shutdown at step 43 reported in @weber2019, where even a random forest retrained after every step failed to recover: the illicit behaviour seen after step 42 looks different from anything in training.

#figure(
  table(
    columns: (1fr, auto, auto, auto),
    stroke: 0.4pt + luma(170), inset: 4pt, fill: hdr, align: (left, center, center, center),
    [*Strategy (XGBoost, AF)*], [*Mean F1, 35–49*], [*Steps 35–42*], [*Steps 43–49*],
    [Static (train once on 1–34)], [0.498], [0.897], [0.041],
    [EM prior adjustment @saerens2002], [0.482], [0.904], [0.000],
    [Sliding-window retrain (last 10 steps)], [0.621], [0.894], [0.310],
    [Expanding-window retrain], [0.644], [0.906], [0.345],
    [*Expanding retrain + re-tuned threshold*], [*0.695*], [*0.909*], [*0.452*],
  ),
  caption: [Adaptation strategies, mean per-step illicit F1. Retraining assumes labels arrive one step (about two weeks) late.],
)


Four findings follow from Table 4 and Figure 7.

- *Feature-level tests do not help.* The KS test rejects for 75–96% of all features in every test step, before and after the collapse, so it cannot separate harmless drift from harmful drift.
- *Retraining is what recovers performance.* Expanding-window retraining lifts steps 43–49 from 0.041 to 0.345, and re-tuning the threshold on the latest labelled step lifts them to 0.452. A 10-step sliding window is slightly worse than keeping all history.
- *Prior adjustment does not help.* EM re-weighting gets 0.000 on steps 43–49. When the model no longer recognises the new illicit behaviour, re-scaling its scores cannot recover it.
- *Label-free monitoring works if it watches the right signal.* ADWIN on the error stream alarms at steps 40 and 48, which misses the onset: in steps 43–47 illicit transactions are only 0.3–2.6% of the labelled transactions, so missing all of them barely changes the error rate. ADWIN on the *alert rate*, which needs no labels, alarms at step 43, exactly when the collapse begins; the flagged share falls from 9.1% at step 42 to 0.2% at step 43. Its other alarms (steps 36 and 38) follow real swings in the illicit base rate.

#figure(image(fig("drift.png"), width: 84%), caption: [Top: share of the 165 features failing the KS test ($p < 0.01$) per step. Middle: true and EM-estimated illicit rate. Bottom: per-step F1 for each strategy, with dotted lines at the label-free ADWIN alarms.])

== Data-mining components

#figure(
  table(
    columns: (1fr, auto, auto, auto, auto),
    stroke: 0.4pt + luma(170), inset: 4pt, fill: hdr, align: (left, center, center, center, center),
    [*Method*], [*Precision*], [*Recall*], [*F1*], [*PR-AUC*],
    [Isolation Forest (no labels)], [0.002], [0.003], [0.002], [0.037],
    [LOF (no labels)], [0.038], [0.067], [0.048], [0.066],
    [XGBoost, supervised (threshold 0.5)], [0.928], [0.727], [0.815 ± 0.005], [0.803],
    [XGBoost + self-training], [0.938], [0.722], [0.816 ± 0.002], [0.806],
  ),
  caption: [Unsupervised detectors (test base rate 0.065) and semi-supervised self-training.],
)

*Unsupervised detection fails* (Table 5). Isolation Forest has a PR-AUC of 0.037, *below* the 0.065 random baseline, and LOF is roughly at chance. Illicit transactions are not statistical outliers. If anything they look more ordinary than licit ones, which include unusual high-fan-out exchange activity. *Self-training adds nothing:* three rounds added 7,473 pseudo-labels each (only 51, 27 and 76 of them illicit), and F1 moved from 0.815 to 0.816, within noise. The confident pseudo-labels are almost all easy licit cases that the model already gets right.

*Association rules decay over time.* Apriori found 49 rules with confidence of at least 0.5. The strongest, such as _local_55 = low ∧ local_5 = low ∧ local_90 = low → illicit_, have 73–78% confidence in training (lift ≈ 6.3–6.7) but only 13–37% on the test steps. That is still above the 6.5% base rate, but these are not stable laundering signatures. Rules come in mirrored pairs (for example local_5 / local_14) because some anonymised features are near-duplicates (r = 0.99).

== Explainability

#figure(image(fig("shap_importance.png"), width: 62%), caption: [Global feature importance, mean |SHAP| for XGBoost on AF features.])

TreeSHAP (Figure 8) shows XGBoost relies most on `local_53` (mean |SHAP| 1.02 log-odds), followed by `local_59`, `local_90` and the aggregated feature `agg_32`. GNNExplainer on the top-scored true-illicit transactions picks out a mix of local and aggregated features (for example `agg_57`, `local_90`, `local_55`) and their small two-hop subgraphs (3 to 50 nodes). Because Elliptic's features are anonymised, explanations name feature indices rather than business concepts. This limits their use for real investigations, and @safar2026 tries to address it.

// ───────────────────────── 7 LIMITATIONS ─────────────────────────
= Limitations

- *No hyperparameter tuning* for any model. Tuning could narrow the gap between trees and GNNs, but the edge-ablation and drift conclusions are internal comparisons that do not depend on it.
- *Label delay.* Retraining assumes labels arrive one step (about two weeks) late; in real AML work they can lag by months.
- *Single seed for drift.* The drift study uses one seed; the main tables use 5 (trees) or 3 (GNNs).
- *Dataset validity.* @safar2026 argues that Elliptic's standard splits are not properly isolated. We did not verify this, and any label or feature errors carry through to every result here.
- *Not implemented.* We dropped MLflow tracking and Docker packaging from the synopsis: per-stage JSON results and `uv` lockfiles gave the same reproducibility with less machinery.

// ───────────────────────── 8 CONCLUSION ─────────────────────────
= Conclusion and Future Work

ChainSleuth shows that on Elliptic, under a strict temporal protocol, *gradient-boosted trees beat graph neural networks* by a large and statistically significant margin. The transaction graph is useful only to an architecture that keeps a node's own features separate (GraphSAGE), and it actively hurts GCN. The bigger practical problem is not model choice but *drift*: every model collapses after step 42, standard feature-drift tests raise alarms in every step, and only retraining on fresh labels restores performance. A monitor that watches the model's own alert rate detects the collapse at the right step without any labels. This is the most deployable result of the project, since an AML team can run it from day one.

*Future work:* out-of-fold embeddings to fix the hybrid; temporal GNNs (EvolveGCN, TGN); active learning, so that scarce analyst labels go to the transactions that matter most after drift; and re-running the study on the transparent dataset released by @safar2026.

#v(0.3em)
#set text(size: 9pt)
#bibliography("refs.yml", title: "References", style: "ieee")
