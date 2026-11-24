"""Download and load the Elliptic Bitcoin dataset into plain pandas / numpy structures."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from chainsleuth.config import (
    CACHE_DIR, ELLIPTIC_URL, ILLICIT, LICIT, RAW_DIR, RAW_FILES, TRAIN_END, UNKNOWN,
)

N_LOCAL = 93  # local features after the time-step column (Weber et al.: 94 local incl. time step)
N_AGG = 72


def download(raw_dir=RAW_DIR) -> None:
    """Fetch the three raw CSVs (zipped) from the PyG mirror if they are not already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        target = raw_dir / name
        if target.exists():
            continue
        print(f"downloading {name} ...")
        with urllib.request.urlopen(f"{ELLIPTIC_URL}/{name}.zip") as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member = next(m for m in zf.namelist() if m.endswith(name))
            target.write_bytes(zf.read(member))


@dataclass
class EllipticData:
    """Node table plus directed edge index.

    nodes: one row per transaction, positional index == node id used in edge_index.
        columns: tx_id, time, label (1 illicit / 0 licit / -1 unknown), then feature columns.
    edge_index: int64 array of shape (2, E), source -> destination node ids.
    """

    nodes: pd.DataFrame
    edge_index: np.ndarray
    local_cols: list[str]
    agg_cols: list[str]

    @property
    def af_cols(self) -> list[str]:
        """All 165 features excluding the time step (Weber's 'AF' set minus the time column)."""
        return self.local_cols + self.agg_cols

    @property
    def time(self) -> np.ndarray:
        return self.nodes["time"].to_numpy()

    @property
    def label(self) -> np.ndarray:
        return self.nodes["label"].to_numpy()

    @cached_property
    def labelled(self) -> np.ndarray:
        return self.label != UNKNOWN

    def mask(self, t_from: int, t_to: int, labelled_only: bool = True) -> np.ndarray:
        m = (self.time >= t_from) & (self.time <= t_to)
        return m & self.labelled if labelled_only else m

    @property
    def train_mask(self) -> np.ndarray:
        return self.mask(1, TRAIN_END)

    @property
    def test_mask(self) -> np.ndarray:
        return self.mask(TRAIN_END + 1, 49)


def load(raw_dir=RAW_DIR, cache_dir=CACHE_DIR) -> EllipticData:
    """Load the dataset, using a parquet cache after the first parse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    nodes_pq, edges_npy = cache_dir / "nodes.parquet", cache_dir / "edges.npy"
    local_cols = [f"local_{i}" for i in range(1, N_LOCAL + 1)]
    agg_cols = [f"agg_{i}" for i in range(1, N_AGG + 1)]

    if nodes_pq.exists() and edges_npy.exists():
        return EllipticData(pd.read_parquet(nodes_pq), np.load(edges_npy), local_cols, agg_cols)

    download(raw_dir)
    feats = pd.read_csv(raw_dir / RAW_FILES[0], header=None)
    feats.columns = ["tx_id", "time", *local_cols, *agg_cols]
    classes = pd.read_csv(raw_dir / RAW_FILES[2])
    classes.columns = ["tx_id", "class"]
    label_map = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    classes["label"] = classes["class"].astype(str).map(label_map).astype("int8")

    nodes = feats.merge(classes[["tx_id", "label"]], on="tx_id", how="left", validate="1:1")
    if nodes["label"].isna().any():
        raise ValueError("some transactions have no class row")
    nodes = nodes[["tx_id", "time", "label", *local_cols, *agg_cols]].reset_index(drop=True)
    nodes["time"] = nodes["time"].astype("int16")
    nodes[local_cols + agg_cols] = nodes[local_cols + agg_cols].astype("float32")

    edges = pd.read_csv(raw_dir / RAW_FILES[1])
    edges.columns = ["src", "dst"]
    id_to_idx = pd.Series(np.arange(len(nodes)), index=nodes["tx_id"])
    edge_index = np.stack([id_to_idx.loc[edges["src"]].to_numpy(), id_to_idx.loc[edges["dst"]].to_numpy()])

    nodes.to_parquet(nodes_pq)
    np.save(edges_npy, edge_index.astype(np.int64))
    return EllipticData(nodes, edge_index.astype(np.int64), local_cols, agg_cols)


def summary(d: EllipticData) -> dict:
    lab = d.label
    return {
        "n_nodes": int(len(d.nodes)),
        "n_edges": int(d.edge_index.shape[1]),
        "n_features": len(d.af_cols) + 1,
        "n_illicit": int((lab == ILLICIT).sum()),
        "n_licit": int((lab == LICIT).sum()),
        "n_unknown": int((lab == UNKNOWN).sum()),
        "n_time_steps": int(d.nodes["time"].nunique()),
        "cross_step_edges": int((d.time[d.edge_index[0]] != d.time[d.edge_index[1]]).sum()),
    }
