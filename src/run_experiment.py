"""Run one experiment: one model, one mode, one label fraction, one seed.

Steps: plan the run, train (unless zero-shot), score the saved checkpoint
(threshold on validation, test scored once), and write one run record.

Modes (ARCHITECTURE.md): `source` trains on HI-Small, `zero_shot` scores that
HI-Small model on LI-Small, `finetune` starts from it and trains on LI-Small,
`scratch` trains on LI-Small from random weights.

Run from the repo root, inside Multi-GNN's environment, on a GPU machine:
    python -m src.run_experiment --model gin --mode source --seed 0 \
        --step-budget-epochs 100 --environment env.yml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import (
    DATA_ROOT,
    MULTI_GNN_DIR,
    PROJECT_ROOT,
    RUN_MODES,
    SEEDS,
    SOURCE_DATASET,
    STEP_BUDGET_EPOCHS,
    TARGET_DATASET,
)
from src.prepare import build_data_config, make_run_folder, run_folder_for
from src.records import (
    RunRecord,
    git_commit,
    make_run_id,
    model_folder_for_run,
    smoke_run_id,
    write_record,
)
from src.score_checkpoint import (
    ScoringRequest,
    ScoringResult,
    checkpoint_file,
    import_multignn_modules,
    require_checkpoint,
    score_checkpoint,
)
from src.step_budget import epochs_for_budget
from src.train_wrapper import TrainingRequest, TrainingSummary, train_model


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


def plan_run(
    model: str,
    mode: str,
    fraction: float,
    seed: int,
    step_budget_epochs: int = STEP_BUDGET_EPOCHS,
    smoke: bool = False,
) -> RunPlan:
    """Work out datasets, ids and which checkpoint to use for a run."""
    run_id = make_run_id(model, mode, fraction, seed)  # also checks mode and fraction
    if mode in ("source", "zero_shot") and fraction != 1.0:
        raise ValueError("Mode %s always uses all labels (fraction 1.0)" % mode)
    source_id = make_run_id(model, "source", 1.0, seed)
    if smoke:
        run_id, source_id = smoke_run_id(run_id), smoke_run_id(source_id)
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
        epochs=0
        if mode == "zero_shot"
        else epochs_for_budget(step_budget_epochs, fraction),
    )


def scoring_request(plan: RunPlan) -> ScoringRequest:
    """The scoring half of a plan."""
    return ScoringRequest(
        plan.run_id, plan.checkpoint_run_id, plan.model,
        plan.dataset_name, plan.seed, plan.finetuned,
    )  # fmt: skip


def training_request(plan: RunPlan) -> TrainingRequest:
    """The training half of a plan."""
    return TrainingRequest(
        plan.model,
        plan.dataset_name,
        plan.seed,
        plan.epochs,
        plan.fraction,
        plan.finetuned,
    )


def gradient_steps(plan: RunPlan, summary: Optional[TrainingSummary]) -> Optional[int]:
    """Total optimiser steps the run was given, for checking budgets match."""
    return summary.steps_per_epoch * plan.epochs if summary else None


def build_experiment_record(
    plan: RunPlan,
    result: ScoringResult,
    summary: Optional[TrainingSummary],
    runtime_seconds: float,
    environment: str,
    smoke_test: bool,
) -> RunRecord:
    """Combine plan, scores and training facts into one record."""
    config = {
        "dataset": plan.dataset_name,
        "step_budget_epochs": plan.step_budget_epochs,
        "epochs_requested": plan.epochs,
        "steps_per_epoch": summary.steps_per_epoch if summary else None,
        "gradient_steps": gradient_steps(plan, summary),
        "environment": environment,
        "smoke_test": smoke_test,
        "stock_training_loop": False,
        "checkpoint_run_id": plan.checkpoint_run_id,
        "fine_tuned_from": plan.load_from_run_id,
        "train_transactions": summary.train_transactions if summary else None,
        "labels_kept": summary.labels_kept if summary else None,
        "laundering_labels_kept": summary.laundering_labels_kept if summary else None,
        "kept_labels_fingerprint": summary.kept_labels_fingerprint if summary else None,
        "label_mask_seed": plan.seed,
    }
    return RunRecord(
        run_id=plan.run_id, model=plan.model, mode=plan.mode,
        fraction=plan.fraction, seed=plan.seed, config=config,
        ours_commit=git_commit(PROJECT_ROOT),
        multignn_commit=git_commit(MULTI_GNN_DIR),
        metrics={"scored": result.metrics.to_dict()},
        epochs_trained=plan.epochs,
        best_epoch=result.best_epoch,
        runtime_seconds=runtime_seconds,
    )  # fmt: skip


def prepare_run_folder(plan: RunPlan) -> Path:
    """Write data_config.json and model_settings.json for this run."""
    save_dir = model_folder_for_run(plan.run_id)
    load_dir = model_folder_for_run(plan.load_from_run_id or plan.run_id)
    config = build_data_config(DATA_ROOT, save_dir, load_dir)
    return make_run_folder(run_folder_for(plan.run_id, DATA_ROOT), config)


def check_starting_point(plan: RunPlan) -> None:
    """Fail before spending GPU time if a model this run needs is missing."""
    needed = plan.load_from_run_id or (None if plan.trains else plan.checkpoint_run_id)
    if needed is not None:
        require_checkpoint(checkpoint_file(model_folder_for_run(needed), False))


def run_experiment(plan: RunPlan, environment: str, smoke_test: bool) -> Path:
    """Train if needed, score, and write the record. Returns the record's path."""
    check_starting_point(plan)
    run_folder = prepare_run_folder(plan)
    os.chdir(run_folder)  # Multi-GNN reads ./model_settings.json from here
    start = time.time()
    summary = None
    if plan.trains:
        modules = import_multignn_modules()
        data_config = json.loads((run_folder / "data_config.json").read_text())
        model_folder = model_folder_for_run(plan.run_id)
        request = training_request(plan)
        summary = train_model(request, modules, data_config, model_folder)
    result = score_checkpoint(scoring_request(plan))
    record = build_experiment_record(
        plan, result, summary, time.time() - start, environment, smoke_test
    )
    return write_record(record)


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


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=["gin", "pna"])
    parser.add_argument("--mode", required=True, choices=list(RUN_MODES))
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument(
        "--step-budget-epochs",
        type=int,
        default=None,
        help="Epochs at fraction 1.0 that set the step budget (default %d; "
        "required for source runs)" % STEP_BUDGET_EPOCHS,
    )
    parser.add_argument("--environment", required=True, help="env.yml or pip-fallback")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Plan, run, and report where the record went."""
    arguments = parse_arguments()
    budget = resolve_step_budget(arguments.mode, arguments.step_budget_epochs)
    plan = plan_run(
        arguments.model, arguments.mode, arguments.fraction,
        arguments.seed, budget, arguments.smoke_test,
    )  # fmt: skip
    path = run_experiment(plan, arguments.environment, arguments.smoke_test)
    print("Wrote record: %s" % path)


if __name__ == "__main__":
    main()
