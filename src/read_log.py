"""Read the F1 scores Multi-GNN's own training loop wrote to its log.

The stock loop logs one "Train F1", "Validation F1" and "Test F1" line per epoch
(four decimals). For RQ1 we keep that number next to our own scored F1, so
anyone can see how much of a gap comes from the scoring method.

Which epoch does Multi-GNN call "best"? Its loop (training.py) skips epoch 0, then
keeps the last epoch whose validation F1 beat every earlier epoch from 1 onward;
that is also when it saves the checkpoint. We copy that rule exactly. If no later
epoch improves, epoch 0 is the one it reports and no checkpoint exists.

Example:
    epochs = parse_epoch_scores(Path("logs/logs.log").read_text())
    logged = multignn_best_epoch(epochs)
    logged.test_f1
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

START_MARKER = "Retrieving data"  # first line of every Multi-GNN run
SCORE_PATTERN = re.compile(r"(Train|Validation|Test) F1: (\S+)")


@dataclass(frozen=True)
class EpochScores:
    """The three F1 values Multi-GNN logged for one epoch. `epoch` counts from 0."""

    epoch: int
    train_f1: float
    validation_f1: float
    test_f1: float


def last_run_text(log_text: str) -> str:
    """The part of the log from the latest run (logs append across runs)."""
    position = log_text.rfind(START_MARKER)
    return log_text[position:] if position >= 0 else log_text


def parse_epoch_scores(log_text: str) -> List[EpochScores]:
    """One EpochScores per fully logged epoch of the latest run, in order."""
    epochs: List[EpochScores] = []
    pending: dict = {}
    for name, value in SCORE_PATTERN.findall(last_run_text(log_text)):
        pending[name] = float(value)
        if name == "Test" and len(pending) == 3:
            epochs.append(
                EpochScores(
                    len(epochs),
                    pending["Train"],
                    pending["Validation"],
                    pending["Test"],
                )
            )
            pending = {}
    return epochs


def multignn_best_epoch(epochs: List[EpochScores]) -> EpochScores:
    """The epoch Multi-GNN reports as best (and saves a checkpoint for)."""
    if not epochs:
        raise ValueError("No completed epochs found in the log")
    best_validation_f1 = 0.0  # its loop starts here and skips epoch 0
    chosen = epochs[0]
    for scores in epochs[1:]:
        if scores.validation_f1 > best_validation_f1:
            best_validation_f1 = scores.validation_f1
            chosen = scores
    return chosen
