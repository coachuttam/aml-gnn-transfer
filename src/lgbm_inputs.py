"""Features, labels and the split for LI-Small, built once and kept on disk.

Building the features takes a few minutes, so the result is cached under
`<data root>/lightgbm/` (outside git, like all data). Delete that folder, or pass
`rebuild=True`, to build again.

Example:
    raw = Path("data/Small_LI/LI-Small_Trans.csv")
    inputs = load_inputs(raw, Path("data/lightgbm"))
    inputs.features.shape   # (about 6.9 million, 20)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.lgbm_data import DaySplit, day_split, load_in_multignn_order
from src.lgbm_features import FEATURE_COLUMNS, add_account_codes, build_features

FEATURES_FILE = "features.npy"
TRUTH_FILE = "truth.npy"
SPLIT_FILE = "split.json"


@dataclass(frozen=True)
class LgbmInputs:
    """Everything a LightGBM run needs, in the row order of Multi-GNN's file."""

    features: np.ndarray  # one row per transaction, float32, FEATURE_COLUMNS order
    truth: (
        np.ndarray
    )  # 1 = laundering. Used for training rows, scoring and nothing else
    split: DaySplit
    names: List[str]


def build_inputs(raw_csv: Path) -> LgbmInputs:
    """Read the raw CSV, put it in Multi-GNN's order, split it, compute the features."""
    start = time.time()
    frame = load_in_multignn_order(raw_csv)
    split = day_split(frame["seconds"].to_numpy())
    print("Read %d rows in %.0f s" % (len(frame), time.time() - start))
    truth = frame["is_laundering"].to_numpy().astype(np.int8)
    frame = add_account_codes(frame.drop(columns=["is_laundering"]))
    features = build_features(frame).to_numpy().astype(np.float32)
    print("Built %d features in %.0f s" % (features.shape[1], time.time() - start))
    return LgbmInputs(features, truth, split, list(FEATURE_COLUMNS))


def save_inputs(inputs: LgbmInputs, cache_dir: Path) -> None:
    """Write the three cache files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / FEATURES_FILE, inputs.features)
    np.save(cache_dir / TRUTH_FILE, inputs.truth)
    numbers = {
        "train_end": inputs.split.train_end,
        "validation_end": inputs.split.validation_end,
    }
    (cache_dir / SPLIT_FILE).write_text(
        json.dumps({**numbers, "total": inputs.split.total})
    )


def read_inputs(cache_dir: Path) -> LgbmInputs:
    """Load what `save_inputs` wrote."""
    numbers = json.loads((cache_dir / SPLIT_FILE).read_text())
    return LgbmInputs(
        np.load(cache_dir / FEATURES_FILE),
        np.load(cache_dir / TRUTH_FILE),
        DaySplit(numbers["train_end"], numbers["validation_end"], numbers["total"]),
        list(FEATURE_COLUMNS),
    )


def load_inputs(raw_csv: Path, cache_dir: Path, rebuild: bool = False) -> LgbmInputs:
    """The cached inputs, built from `raw_csv` first if there is no cache yet."""
    if not rebuild and (cache_dir / SPLIT_FILE).exists():
        return read_inputs(cache_dir)
    inputs = build_inputs(raw_csv)
    save_inputs(inputs, cache_dir)
    return inputs
