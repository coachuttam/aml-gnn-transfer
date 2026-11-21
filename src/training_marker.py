"""A small file that says "this run finished training", so scoring can be resumed.

Multi-GNN saves its best checkpoint as it goes, so a checkpoint on disk does not
mean training finished: a Colab disconnect leaves one from some earlier epoch.
Scoring that would silently judge an under-trained model. So after training
ends we write this marker (with the label facts and training time the record
needs). A run with a checkpoint *and* a marker only needs scoring; a run with a
checkpoint and no marker was interrupted and must be started again. The marker
also stores the step budget, so a model trained at one budget is never scored
as if it had been trained at another.

Example:
    save_training_marker(model_folder, summary, training_seconds)
    marker = load_training_marker(model_folder)   # None if training never finished
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.train_wrapper import TrainingSummary

MARKER_NAME = "training_complete.json"


@dataclass(frozen=True)
class TrainingMarker:
    """What was learned while training, kept for a scoring-only restart."""

    summary: TrainingSummary
    training_seconds: float
    step_budget_epochs: Optional[int] = None  # None in markers from older code


def marker_file(model_folder: Path) -> Path:
    """Where the marker lives: inside the run's own model folder."""
    return model_folder / MARKER_NAME


def save_training_marker(
    model_folder: Path,
    summary: TrainingSummary,
    training_seconds: float,
    step_budget_epochs: Optional[int] = None,
) -> Path:
    """Write the marker. Atomic, so a crash never leaves half a marker."""
    path = marker_file(model_folder)
    partial = path.with_name(path.name + ".partial")
    content = {
        "summary": asdict(summary),
        "training_seconds": training_seconds,
        "step_budget_epochs": step_budget_epochs,
    }
    partial.write_text(json.dumps(content, indent=2) + "\n")
    os.replace(partial, path)
    return path


def load_training_marker(model_folder: Path) -> Optional[TrainingMarker]:
    """The marker, or None if training did not finish (or never started)."""
    path = marker_file(model_folder)
    if not path.exists():
        return None
    content = json.loads(path.read_text())
    summary = TrainingSummary(**content["summary"])
    return TrainingMarker(
        summary, float(content["training_seconds"]), content.get("step_budget_epochs")
    )
