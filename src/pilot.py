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
from typing import List, Optional, Sequence

from src.config import (
    PILOT_FRACTIONS,
    PILOT_MODEL,
    PILOT_SEEDS,
    PILOT_SOURCE_EPOCHS,
    PILOT_SOURCE_SEED,
    PILOT_STEP_BUDGET_EPOCHS,
)
from src.records import make_run_id, model_folder_for_run, read_record, record_path
from src.training_marker import load_training_marker

SUPERSEDED_FOLDER = "superseded"


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


def pilot_runs(
    step_budget: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    include_source: bool = True,
) -> List[PilotRun]:
    """Every pilot run in the order they should be done.

    The arguments exist so an estimate can price other budgets or seed counts;
    the defaults are the pilot as configured.
    """
    budget = PILOT_STEP_BUDGET_EPOCHS if step_budget is None else step_budget
    runs = []
    if include_source:
        runs.append(
            PilotRun(PILOT_MODEL, "source", 1.0, PILOT_SOURCE_SEED, PILOT_SOURCE_EPOCHS)
        )
    for seed in PILOT_SEEDS if seeds is None else seeds:
        for fraction in sorted(PILOT_FRACTIONS, reverse=True):
            for mode in ("scratch", "finetune"):
                runs.append(PilotRun(PILOT_MODEL, mode, fraction, seed, budget))
    return runs


def record_exists(run: PilotRun, records_dir: Path) -> bool:
    """True if this run already has its record: it is finished and is skipped."""
    return record_path(run.run_id, records_dir).exists()


def finished_at_this_budget(run: PilotRun, model_root: Path) -> bool:
    """Did this run's model folder finish training at the run's current budget?"""
    marker = load_training_marker(model_folder_for_run(run.run_id, model_root))
    return marker is not None and marker.step_budget_epochs == run.step_budget_epochs


def set_aside_partial_run(run: PilotRun, model_root: Path) -> Optional[Path]:
    """Move a model folder that cannot be reused out of the way; never delete it.

    Reusable means training finished at this run's budget: then the run only
    needs scoring (or recording). Anything else is a dead attempt: interrupted
    training (no marker) or a finished run from a different budget. Starting
    again needs a clean folder, because a run refuses to start over an existing
    checkpoint. Returns where the folder went, or None.
    """
    folder = model_folder_for_run(run.run_id, model_root)
    if not folder.exists() or finished_at_this_budget(run, model_root):
        return None
    reason = "interrupted" if load_training_marker(folder) is None else "other_budget"
    number = 1
    while folder.with_name("%s_%s_%d" % (folder.name, reason, number)).exists():
        number += 1
    destination = folder.with_name("%s_%s_%d" % (folder.name, reason, number))
    folder.rename(destination)
    return destination


def supersede_other_budget_records(
    runs: List[PilotRun], records_dir: Path
) -> List[Path]:
    """Move records made at a different step budget into records_dir/superseded/.

    Nothing is deleted or overwritten (AGENTS.md rule 5). Collecting results
    reads only the records_dir itself, so superseded runs drop out of tables
    but stay on disk for the README to cite.
    """
    moved = []
    for run in runs:
        path = record_path(run.run_id, records_dir)
        if not path.exists():
            continue
        budget = read_record(path).config.get("step_budget_epochs")
        if budget == run.step_budget_epochs:
            continue
        target_dir = records_dir / SUPERSEDED_FOLDER
        target_dir.mkdir(exist_ok=True)
        number = 1
        while (target_dir / ("%s_b%s_%d.json" % (run.run_id, budget, number))).exists():
            number += 1
        target = target_dir / ("%s_b%s_%d.json" % (run.run_id, budget, number))
        path.rename(target)
        moved.append(target)
    return moved
