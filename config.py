"""Shared paths and constants. Every script imports this module, so the data location is set in one place.

Set MCQ_DATA_DIR to use a data folder outside the repository (default: ./data). See data/README.md.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MCQ_DATA_DIR", ROOT / "data"))

EDM = DATA / "eedi_2020"                                   # NeurIPS 2020 Eedi release (contains data/train_data, data/metadata, data/images)
TRAIN_CSV = EDM / "data" / "train_data" / "train_task_3_4.csv"
META = EDM / "data" / "metadata"
IMAGES = EDM / "data" / "images"
DISTRACTORS = DATA / "distractor_tasks.parquet"            # 899 distractor instances with behavioural features and text_input
MISC_ZIP = DATA / "eedi-mining-misconceptions-in-mathematics.zip"   # Kaggle 2024 bank (misconception_mapping.csv, train.csv)

CACHE = ROOT / "cache"
EMB = CACHE / "embeddings"                                 # one <encoder>.npy per encoder, written by scripts/01_encode.py
RESULTS = ROOT / "results"                                 # small aggregate tables (committed)
FIGURES = ROOT / "figures"                                 # PNG figures (committed)
OUTPUTS = ROOT / "outputs"                                 # item-level outputs that contain Eedi content (git-ignored)
ASSETS = ROOT / "assets"                                   # GUI template, labels, candidate titles, evidence-linked summaries

SEED = 42
GROUP_ENCODER = os.environ.get("GROUP_ENCODER", "Qwen3-Embedding-8B")   # semantic grouping encoder for the candidate audit
GROUP_K = 8

for d in (EMB, RESULTS, FIGURES, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)


def emb(name):
    """Load a cached embedding matrix (899 x d) written by scripts/01_encode.py."""
    import numpy as np
    p = EMB / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/01_encode.py first")
    return np.load(p)
