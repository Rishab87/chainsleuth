"""Explanations: SHAP for tree models, GNNExplainer for graph models.

Imports are local to each function so a tree-only process never loads torch (see __init__).
"""

from __future__ import annotations

import numpy as np


def shap_values(tree_model, X: np.ndarray) -> np.ndarray:
    """SHAP values (log-odds contributions) for a fitted XGBoost / LightGBM / RF classifier."""
    import shap

    vals = shap.TreeExplainer(tree_model).shap_values(X)
    if isinstance(vals, list):  # older API: one array per class
        vals = vals[1]
    if vals.ndim == 3:  # (n, features, classes)
        vals = vals[..., 1]
    return vals


def gnn_explain_node(model, x, edge_index, node: int, hops: int = 2, epochs: int = 150):
    """Run GNNExplainer on the node's k-hop subgraph. Returns (subset, sub_edge_index, edge_mask, feat_mask)."""
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    subset, sub_ei, mapping, _ = k_hop_subgraph(node, hops, edge_index, relabel_nodes=True, num_nodes=x.shape[0])
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )
    model.eval()
    exp = explainer(x[subset], sub_ei, index=int(mapping[0]))
    feat_mask = exp.node_mask[int(mapping[0])].detach().numpy()
    return subset.numpy(), sub_ei.numpy(), exp.edge_mask.detach().numpy(), feat_mask
