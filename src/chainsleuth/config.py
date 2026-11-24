"""Paths and experiment constants shared by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

ELLIPTIC_URL = "https://data.pyg.org/datasets/elliptic"
RAW_FILES = ("elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv")

N_TIME_STEPS = 49
TRAIN_END = 34  # standard Weber et al. split: train on steps 1..34, test on 35..49
VAL_START = 30  # steps 30..34 are held out of training for epoch / hyperparameter selection

ILLICIT, LICIT, UNKNOWN = 1, 0, -1
SEEDS = (0, 1, 2, 3, 4)
