"""LI-Small for the LightGBM baseline: same row order and same split as Multi-GNN.

The GNN runs read IBM's `formatted_transactions.csv`. This module rebuilds that
file's row order from the raw Kaggle CSV, and ports Multi-GNN's split (whole
days, chosen so train, validation and test hold about 60%, 20% and 20% of the
transactions), so the LightGBM runs train on the same rows and the same label
masks. It is a port of `external/Multi-GNN/data_loading.py` and
`format_kaggle_files.py`, which stay untouched. The kept-label fingerprint of
each run is compared with the Phase C records, which proves the split matches.

In the sorted order the training rows are the first `train_end` rows: Multi-GNN
joins whole days in time order, so position `p` of its training indices is row
`p` here.

Example:
    frame = load_in_multignn_order(Path("data/Small_LI/LI-Small_Trans.csv"))
    split = day_split(frame["seconds"].to_numpy())
    split.train_end   # about 60% of len(frame)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FIRST_TIMESTAMP_OFFSET_SECONDS, SPLIT_SHARES

SECONDS_PER_DAY = 24 * 3600
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
# The raw header repeats "Account", so the columns are named here.
COLUMN_NAMES = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]  # fmt: skip
CATEGORY_COLUMNS = [
    "from_bank", "from_account", "to_bank", "to_account",
    "receiving_currency", "payment_currency", "payment_format",
]  # fmt: skip


@dataclass(frozen=True)
class DaySplit:
    """Row counts of the three periods, in the order of the sorted file."""

    train_end: int
    validation_end: int
    total: int

    @property
    def train(self) -> slice:
        """Rows of the training period."""
        return slice(0, self.train_end)

    @property
    def validation(self) -> slice:
        """Rows of the validation period."""
        return slice(self.train_end, self.validation_end)

    @property
    def test(self) -> slice:
        """Rows of the test period."""
        return slice(self.validation_end, self.total)


def load_in_multignn_order(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """The raw CSV in the row order of IBM's formatted file, with `seconds` added.

    IBM's formatter keeps the raw order, counts seconds from 10 seconds before
    midnight of the first row's day, then sorts by those seconds (stable, so
    equal timestamps keep file order). Multi-GNN subtracts the smallest value.
    """
    frame = pd.read_csv(
        path, header=0, names=COLUMN_NAMES, nrows=max_rows,
        dtype={name: str for name in CATEGORY_COLUMNS},
    )  # fmt: skip
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format=TIMESTAMP_FORMAT)
    midnight = frame["timestamp"].iloc[0].normalize()
    offset = (frame["timestamp"] - midnight).dt.total_seconds()
    seconds = offset.to_numpy().astype(np.int64) + FIRST_TIMESTAMP_OFFSET_SECONDS
    frame["seconds"] = seconds - seconds.min()
    frame = frame.iloc[np.argsort(frame["seconds"].to_numpy(), kind="stable")]
    frame = frame.reset_index(drop=True)
    for column in CATEGORY_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame


def transactions_per_day(seconds: np.ndarray) -> np.ndarray:
    """How many transactions fall in each whole day, as Multi-GNN counts them."""
    day_count = int(seconds.max() / SECONDS_PER_DAY + 1)
    return np.bincount(seconds // SECONDS_PER_DAY, minlength=day_count)


def split_error(totals: np.ndarray, first: int, second: int) -> float:
    """Worst relative miss from 60/20/20 when days are cut at `first` and `second`."""
    parts = [totals[:first].sum(), totals[first:second].sum(), totals[second:].sum()]
    shares = [part / sum(parts) for part in parts]
    return max(abs(share - want) / want for share, want in zip(shares, SPLIT_SHARES))


def day_split(seconds: np.ndarray) -> DaySplit:
    """Multi-GNN's split: the two day cuts closest to 60/20/20 (first best wins)."""
    totals = transactions_per_day(seconds)
    if len(totals) < 3:
        raise ValueError("Need at least 3 days to split, found %d" % len(totals))
    best_cut, best_error = None, None
    for first, second in itertools.combinations(range(len(totals)), 2):
        error = split_error(totals, first, second)
        if best_error is None or error < best_error:
            best_cut, best_error = (first, second), error
    first, second = best_cut
    return DaySplit(
        int(totals[:first].sum()), int(totals[:second].sum()), int(totals.sum())
    )
