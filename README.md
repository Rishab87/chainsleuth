# ChainSleuth

Illicit Bitcoin transaction detection on the Elliptic transaction graph, using graph mining, graph neural networks and drift-aware learning.

**College project.** Machine Learning and Data Mining, Jaypee Institute of Information Technology (JIIT), Noida, 2026.
Presented to Dr. Ashish Mishra.

| Team member | Enrolment no. |
|---|---|
| Rishab Kumar Jha | 23803021 |
| Arush Dwivedi | 23803016 |
| Aditya Gupta | 23803018 |

**Documents:** [Project report (PDF, 11 pages)](docs/report/ChainSleuth_Report.pdf) · [Synopsis (PDF)](docs/synopsis/ChainSleuth_Synopsis.pdf) · [All results](reports/RESULTS.md)

**In one paragraph.** We take the Elliptic dataset (203,769 Bitcoin transactions, 49 time steps, 2% labelled illicit)
and ask whether graph neural networks really beat classical models when evaluation is strictly temporal (train on
the past, test on the future). They do not. LightGBM on the raw features beats every GNN. The transaction graph helps
only GraphSAGE, and it hurts GCN. Every model collapses after a dark-market shutdown at time step 43. Retraining
partly recovers performance, and a label-free monitor on the model's alert rate detects the collapse at the right
step. A Streamlit dashboard lets an investigator inspect any transaction's risk score, SHAP explanation and payment
neighbourhood.

Every number below comes from `reports/RESULTS.md`, which `python -m chainsleuth.report` generates from the saved results.

## What we found

Test period = labelled transactions in time steps 35–49 (1,083 illicit, 15,587 licit). F1 = illicit-class F1 at threshold 0.5.

| Result | Evidence |
|---|---|
| **Gradient-boosted trees on the raw features win.** LightGBM (all 165 features) gets F1 **0.827 ± 0.001** and PR-AUC 0.806. | `reports/RESULTS.md`, model table |
| **No GNN beats the trees.** Best GNN: GraphSAGE, F1 **0.714 ± 0.014**. GAT 0.578, GCN 0.562. LightGBM vs GraphSAGE on the same test transactions: 253 cases where only LightGBM is right vs 34 where only GraphSAGE is (exact McNemar p = 1.5e-42). | GNN table, significance table |
| **Graph structure helps GraphSAGE but not GCN or GAT.** GraphSAGE: real edges 0.714 > no edges 0.604 > shuffled 0.544. GCN: **no edges 0.598 beats real edges 0.562**. GAT is flat (0.578 / 0.570). Shuffled edges hurt every model, so it is the *specific* structure that matters, and only an architecture that keeps a node's own features separate from its neighbours' (SAGE's root weight) can use it. | `figures/gnn_edge_ablation.png` |
| **Hand-crafted graph-mining features add nothing on top of the 165 features** (LightGBM 0.827 → 0.826). Elliptic's 72 aggregated features already summarise each node's 1-hop neighbourhood. | tabular table, AF vs AF+G |
| **Every model collapses after time step 42.** Static XGBoost: mean per-step F1 0.897 on steps 35–42, **0.041** on 43–49. This matches the collapse reported in the literature. | `figures/per_step_f1.png`, `figures/drift.png` |
| **Retraining is what recovers it.** Retrain each step on all labels available so far, and re-tune the threshold on the latest labelled step: steps 43–49 recover to F1 **0.452** (plain retraining 0.345, 10-step sliding window 0.310). EM prior adjustment (Saerens 2002) does *not* help (0.000): when the model stops recognising the new illicit behaviour, re-weighting its scores cannot bring it back. | drift table |
| **Label-free drift detection works if you monitor the right signal.** ADWIN on the model's *error* stream misses the collapse (alarms at steps 40, 48), because illicit is only 0.3–2.6% of labelled transactions in steps 43–47, so missing all of them barely moves the error rate. ADWIN on the *alert rate* needs no labels and fires **at step 43**. Its other alarms (36, 38) track real swings in the illicit base rate. | drift section |
| **Feature-level drift tests are useless here.** The KS test rejects for 75–96% of features in *every* test step, before and after the collapse. | `figures/drift.png`, top panel |
| **Unsupervised anomaly detection fails, and Isolation Forest is worse than random.** IF PR-AUC 0.037 and LOF 0.066, against a 0.065 base rate. Illicit transactions look *more* ordinary than licit ones, which include high-fan-out exchange activity. | unsupervised table |
| **Self-training on the 157k unlabelled transactions gives nothing** (F1 0.815 → 0.816, within noise). | semi-supervised table |
| **Apriori rules decay out of time.** The top rules → illicit have 73–77% confidence in training (lift ≈ 6.3–6.7) but only 13–37% on the test steps (test base rate 6.5%). | association-rules table |

## Run it

Needs [uv](https://docs.astral.sh/uv/) and about 1 GB of disk for the data. Everything runs on a laptop CPU.

```bash
git clone https://github.com/Rishab87/chainsleuth.git && cd chainsleuth
uv sync                                         # Python 3.12 + all dependencies
```

**Fastest demo (about 5 minutes).** The results and figures are already committed, so you only need the dashboard's model:

```bash
uv run python -m chainsleuth.pipeline --stages explain    # downloads the data (~150 MB), trains the XGBoost model
uv run streamlit run app/dashboard.py                     # opens http://localhost:8501
```

The GraphSAGE score column appears in the dashboard only after the full `gnn` stage has run.

**Full reproduction (about 1.5 hours on an M-series laptop):**

```bash
uv run python -m chainsleuth.pipeline           # every stage, each in its own process
uv run python -m chainsleuth.figures            # reports/figures/*.png
uv run python -m chainsleuth.report             # reports/RESULTS.md
uv run pytest                                   # 27 tests (unit + real-data + dashboard smoke test)
```

Run a subset with `--stages`, e.g. `uv run python -m chainsleuth.pipeline --stages tabular drift`.
Stages: `eda tabular unsupervised gnn hybrid semi drift patterns explain gnn_explain significance`.
`hybrid`, `gnn_explain` and `significance` need `gnn` to have run first.
Passing `--seeds 1` overwrites the committed multi-seed results with single-seed ones.

**Rebuild the report PDF** (needs only uv):

```bash
cd docs/report && uv run --no-project --with typst python -c \
  "import typst; typst.compile('report.typ', output='ChainSleuth_Report.pdf', root='../..')"
```

**Runtime:** `tabular` ≈ 10 min, `drift` ≈ 4 min, `gnn` ≈ 75 min for 9 configurations × 3 seeds
(≈ 45 min if you split it: `--gnn-kinds gcn`, `--gnn-kinds sage` and `--gnn-kinds gat` in three terminals).
The GNN stage caches each finished run in `artifacts/gnn_runs/`, so a killed run resumes where it stopped.
Apple's MPS GPU was *slower* than CPU for these full-batch GNNs (measured: 320–425 ms vs 190–260 ms per epoch).

## Project layout

```
src/chainsleuth/
  data.py            download + load Elliptic (parquet cache), temporal masks
  graph_features.py  per-time-step graph mining: degree, PageRank, clustering, k-core, betweenness, Louvain, 2-hop ego size
  tabular.py         LogReg / RF / XGBoost / LightGBM / MLP; Isolation Forest and LOF
  gnn.py             GCN, GraphSAGE, GAT (PyTorch Geometric), edge-shuffle / no-edge ablations
  semi.py            self-training with pseudo-labels from the 157k unlabelled transactions
  drift.py           KS test, PSI, ADWIN, Saerens EM prior adjustment
  patterns.py        Apriori association rules -> illicit
  explain.py         SHAP (trees), GNNExplainer (GNNs)
  metrics.py         illicit-class P/R/F1, PR-AUC, MCC, per-step F1, McNemar
  pipeline.py        experiment runner (one process per stage)
  figures.py         report figures
  report.py          reports/RESULTS.md
app/dashboard.py     Streamlit investigator dashboard
tests/               unit tests (synthetic graph), real-data invariants, dashboard smoke test
reports/             results JSON, figures, RESULTS.md
docs/                project report and synopsis (Typst sources + PDFs)
```

## Protocol

- **Temporal split** (Weber et al. 2019): train on time steps 1–34, test on 35–49. Nothing is shuffled across time.
  Validation for epoch and threshold selection uses steps 30–34, after which the model is refit on 1–34.
- **Metrics**: illicit-class precision, recall and F1, plus PR-AUC and MCC (accuracy is meaningless at a 6.5% illicit test rate).
  Headline F1 uses the default 0.5 threshold. F1 at the validation-tuned threshold is also reported.
- **Seeds**: 5 for tree models, 3 for GNNs (mean ± sd). Logistic regression is deterministic, so it gets one seed.
- **No leakage through the graph**: the Elliptic graph has **zero edges between time steps** (asserted in
  `tests/test_real_data.py`). Graph features and GNN message passing for training nodes therefore never touch a test
  node. On this dataset transductive and inductive training are the same computation.

## Engineering notes worth knowing

- **macOS OpenMP clash.** XGBoost/LightGBM use Homebrew's `libomp`, and torch ships its own copy. If torch is imported first, XGBoost
  segfaults. If the tree libraries are imported first, multithreaded torch ops deadlock. Both were reproduced in isolation.
  The fix: every pipeline stage runs in its own process. `chainsleuth.tabular` raises a clear error if torch is
  already loaded, and `chainsleuth.gnn` drops torch to one thread if a tree library is loaded (which is what tests do).
  The dashboard never imports torch.
- Elliptic features are anonymised (`local_1..93`, `agg_1..72`), so rules and SHAP explanations name feature
  indices, not business concepts. Some features are near-duplicates (`local_5`/`local_14` r = 0.99,
  `local_53`/`local_55` r = 0.94). That is why the Apriori rules come in mirrored pairs.

## Limitations

- **GNN hyperparameters were not tuned** (2 layers, 64 hidden units, dropout 0.5, lr 0.01, at most 300 epochs, epochs chosen on validation). Tree models were not tuned either. Tuning could narrow the gap, but the edge ablation and the drift collapse do not depend on it.
- **The hybrid model (trees + GraphSAGE embeddings) scores worse than trees alone (0.746 vs 0.827).** The embeddings for training nodes come from a GNN fitted on those same labels, so the trees learn to over-trust them. This is the classic stacking leak. The fix is out-of-fold embeddings, which we have not implemented.
- **Retraining strategies assume labels arrive one time step (~2 weeks) late.** In real AML work, labels can lag by months.
- EM prior adjustment runs on the set of *labelled* test transactions (it never sees their labels), so it is evaluated on the same population as every other strategy.
- The drift stage uses a single seed. Main model tables use 5 seeds for trees and 3 for GNNs.
- Elliptic features are anonymised, so explanations cannot be mapped to real-world transaction properties.
- A 2026 DFRWS paper ("The Enemy of Reproducibility is Opacity: What's Inside the Elliptic Bitcoin Dataset (and Why It Is Wrong)") argues the dataset itself has problems. We have not checked its claims. Any label errors carry through to every result here.

## References

1. M. Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics," KDD Workshop on Anomaly Detection in Finance, 2019. arXiv:1908.02591
2. S. Maganti, "When Graph Structure Becomes a Liability: A Critical Re-Evaluation of Graph Neural Networks for Bitcoin Fraud Detection under Temporal Distribution Shift," arXiv:2604.19514, 2026.
3. Y. Elmougy and L. Liu, "Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics," KDD 2023.
4. M. Saerens, P. Latinne, C. Decaestecker, "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure," Neural Computation, 2002.
5. A. Bifet and R. Gavaldà, "Learning from Time-Changing Data with Adaptive Windowing," SDM 2007.
