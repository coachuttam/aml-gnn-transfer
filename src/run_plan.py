"""The plan for one run: ids, datasets, epochs and validation passes.

Decided before any compute is spent. See src/run_experiment.py for how a plan
is run, and src/step_budget.py for how epochs follow from the fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import (
    SMOKE_MAX_EPOCHS,
    SOURCE_DATASET,
    STEP_BUDGET_EPOCHS,
    TARGET_DATASET,
)
from src.records import make_run_id, smoke_run_id
from src.step_budget import epochs_for_budget


@dataclass(frozen=True)
class RunPlan:
    """Everything decided before any compute is spent."""

    run_id: str
    model: str
    mode: str
    fraction: float
    seed: int
    dataset_name: str  # trained on and scored on
    checkpoint_run_id: str  # whose model gets scored
    load_from_run_id: Optional[str]  # whose model fine-tuning starts from
    trains: bool
    finetuned: bool
    step_budget_epochs: int  # epochs at fraction 1.0 that set the step budget
    epochs: int  # epochs this run trains for (0 for zero-shot)
    validations: int  # validation passes in all: the same for every fraction


def planned_epochs(mode: str, budget: int, fraction: float, smoke: bool) -> int:
    """Epochs for a run: none for zero-shot, a short cap for smoke tests."""
    if mode == "zero_shot":
        return 0
    epochs = epochs_for_budget(budget, fraction)
    return min(epochs, SMOKE_MAX_EPOCHS) if smoke else epochs


def plan_run(
    model: str,
    mode: str,
    fraction: float,
    seed: int,
    step_budget_epochs: int = STEP_BUDGET_EPOCHS,
    smoke: bool = False,
    source_seed: Optional[int] = None,
) -> RunPlan:
    """Work out datasets, ids and which checkpoint to use for a run.

    `source_seed` picks which source model a zero-shot or finetune run starts
    from (default: the run's own seed). The pilot has one source model, seed 0.
    """
    run_id = make_run_id(model, mode, fraction, seed)  # also checks mode and fraction
    if mode in ("source", "zero_shot") and fraction != 1.0:
        raise ValueError("Mode %s always uses all labels (fraction 1.0)" % mode)
    source_id = make_run_id(
        model, "source", 1.0, seed if source_seed is None else source_seed
    )
    if smoke:
        run_id, source_id = smoke_run_id(run_id), smoke_run_id(source_id)
    epochs = planned_epochs(mode, step_budget_epochs, fraction, smoke)
    return RunPlan(
        run_id=run_id,
        model=model,
        mode=mode,
        fraction=fraction,
        seed=seed,
        dataset_name=SOURCE_DATASET if mode == "source" else TARGET_DATASET,
        checkpoint_run_id=source_id if mode == "zero_shot" else run_id,
        load_from_run_id=source_id if mode == "finetune" else None,
        trains=mode != "zero_shot",
        finetuned=mode == "finetune",
        step_budget_epochs=step_budget_epochs,
        epochs=epochs,
        validations=min(step_budget_epochs, epochs),
    )


def resolve_step_budget(mode: str, requested: Optional[int]) -> int:
    """The step budget to use. Source runs must choose theirs on purpose."""
    if requested is not None:
        return requested
    if mode == "source":
        raise ValueError(
            "Source runs need an explicit --step-budget-epochs "
            "(HI-Small epochs are decided after the smoke-test timing)"
        )
    return STEP_BUDGET_EPOCHS
