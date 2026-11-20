"""The pilot's list of runs, and safe restarts after a Colab disconnect.

Pilot = one model (GIN), one HI-Small source model, then scratch and finetune at
three label fractions for two seeds, with a 3-epoch step budget (see config.py).
Runs are listed so that a scratch/finetune pair at one fraction and seed sit side
by side, and seed 0 is finished before seed 1: if time runs out, what exists is
still a fair comparison.

Example:
    for run in pilot_runs():
        print(run.run_id)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import (
    PILOT_FRACTIONS,
    PILOT_MODEL,
    PILOT_SEEDS,
    PILOT_SOURCE_EPOCHS,
    PILOT_SOURCE_SEED,
    PILOT_STEP_BUDGET_EPOCHS,
)
from src.records import make_run_id, model_folder_for_run, record_path


@dataclass(frozen=True)
class PilotRun:
    """One run of the pilot, with the budget it must be started with."""

    model: str
    mode: str
    fraction: float
    seed: int
    step_budget_epochs: int

    @property
    def run_id(self) -> str:
        """The id its record and model folder are named by."""
        return make_run_id(self.model, self.mode, self.fraction, self.seed)

    def command_words(self) -> List[str]:
        """Arguments for `python -m src.run_experiment` (without --environment)."""
        words = [
            "--model", self.model, "--mode", self.mode,
            "--fraction", str(self.fraction), "--seed", str(self.seed),
            "--step-budget-epochs", str(self.step_budget_epochs),
        ]  # fmt: skip
        if self.mode == "finetune":
            words += ["--source-seed", str(PILOT_SOURCE_SEED)]
        return words


def pilot_runs() -> List[PilotRun]:
    """Every pilot run in the order they should be done."""
    runs = [
        PilotRun(PILOT_MODEL, "source", 1.0, PILOT_SOURCE_SEED, PILOT_SOURCE_EPOCHS)
    ]
    for seed in PILOT_SEEDS:
        for fraction in sorted(PILOT_FRACTIONS, reverse=True):
            for mode in ("scratch", "finetune"):
                runs.append(
                    PilotRun(
                        PILOT_MODEL, mode, fraction, seed, PILOT_STEP_BUDGET_EPOCHS
                    )
                )
    return runs


def record_exists(run: PilotRun, records_dir: Path) -> bool:
    """True if this run already has its record: it is finished and is skipped."""
    return record_path(run.run_id, records_dir).exists()


def set_aside_partial_run(run: PilotRun, model_root: Path) -> Optional[Path]:
    """Move a dead attempt's model folder out of the way; never delete it.

    A run with no record but a model folder was interrupted (a Colab
    disconnect). Starting again needs a clean folder, because a run refuses to
    start over an existing checkpoint. Returns where it went, or None.
    """
    folder = model_folder_for_run(run.run_id, model_root)
    if not folder.exists():
        return None
    number = 1
    while folder.with_name("%s_interrupted_%d" % (folder.name, number)).exists():
        number += 1
    destination = folder.with_name("%s_interrupted_%d" % (folder.name, number))
    folder.rename(destination)
    return destination
