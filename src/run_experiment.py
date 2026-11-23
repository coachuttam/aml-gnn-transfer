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
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import (
    DATA_ROOT,
    MODEL_ROOT,
    RUN_MODES,
    SEEDS,
    SOURCE_MODEL_ROOT,
    STEP_BUDGET_EPOCHS,
)
from src.experiment_record import (
    build_experiment_record,
    build_no_improvement_record,
    read_training_settings,
)
from src.prepare import build_data_config, make_run_folder, run_folder_for
from src.read_log import read_validation_curve
from src.records import model_folder_for_run, write_record
from src.run_plan import RunPlan, plan_run, resolve_step_budget
from src.score_checkpoint import (
    ScoringRequest,
    checkpoint_file,
    import_multignn_modules,
    require_checkpoint,
    score_checkpoint,
)
from src.train_wrapper import TrainingRequest, TrainingSummary, train_model
from src.training_marker import (
    MARKER_NAME,
    TrainingMarker,
    load_training_marker,
    save_training_marker,
)


def scoring_request(plan: RunPlan) -> ScoringRequest:
    """The scoring half of a plan."""
    return ScoringRequest(
        plan.run_id, plan.checkpoint_run_id, plan.model,
        plan.dataset_name, plan.seed, plan.finetuned,
    )  # fmt: skip


def training_request(plan: RunPlan, smoke_test: bool = False) -> TrainingRequest:
    """The training half of a plan."""
    return TrainingRequest(
        plan.model,
        plan.dataset_name,
        plan.seed,
        plan.epochs,
        plan.fraction,
        plan.finetuned,
        smoke_test,
        plan.validations,
    )


def prepare_run_folder(plan: RunPlan) -> Path:
    """Write data_config.json and model_settings.json for this run."""
    save_dir = model_folder_for_run(plan.run_id)
    load_dir = (
        model_folder_for_run(plan.load_from_run_id, SOURCE_MODEL_ROOT)
        if plan.load_from_run_id
        else save_dir
    )
    config = build_data_config(DATA_ROOT, save_dir, load_dir)
    return make_run_folder(run_folder_for(plan.run_id, DATA_ROOT), config)


def finished_training(
    plan: RunPlan, model_root: Path = MODEL_ROOT
) -> Optional[TrainingMarker]:
    """The training-complete marker, if this run finished at its current budget.

    A marker from a different budget does not count: scoring it would judge a
    model trained for another number of steps.
    """
    marker = load_training_marker(model_folder_for_run(plan.run_id, model_root))
    if marker is None or marker.step_budget_epochs != plan.step_budget_epochs:
        return None
    return marker


def has_checkpoint(plan: RunPlan, model_root: Path = MODEL_ROOT) -> bool:
    """Did training save this run's best checkpoint?"""
    folder = model_folder_for_run(plan.run_id, model_root)
    return checkpoint_file(folder, plan.finetuned).exists()


def resumes_scoring(plan: RunPlan, model_root: Path = MODEL_ROOT) -> bool:
    """True if this run finished training earlier and only needs scoring."""
    if not plan.trains or finished_training(plan, model_root) is None:
        return False
    return has_checkpoint(plan, model_root)


def check_starting_point(
    plan: RunPlan,
    model_root: Path = MODEL_ROOT,
    source_root: Optional[Path] = None,
) -> None:
    """Fail before spending GPU time if a needed model is missing or would be reused."""
    finished = plan.trains and finished_training(plan, model_root) is not None
    if plan.trains and has_checkpoint(plan, model_root) and not finished:
        raise FileExistsError(
            "%s has a checkpoint but training never finished at this budget "
            "(no matching %s): move the folder aside and start again"
            % (plan.run_id, MARKER_NAME)
        )
    needed = plan.load_from_run_id or (None if plan.trains else plan.checkpoint_run_id)
    if needed is not None and not finished:
        source_folder = model_folder_for_run(needed, source_root or model_root)
        require_checkpoint(checkpoint_file(source_folder, False))


def train_or_resume(
    plan: RunPlan, run_folder: Path, smoke_test: bool
) -> Tuple[TrainingSummary, float, bool]:
    """Training summary, training seconds, and whether training was already done."""
    model_folder = model_folder_for_run(plan.run_id)
    marker = finished_training(plan)
    if marker is not None:
        print("Training already finished for %s: not training again" % plan.run_id)
        return marker.summary, marker.training_seconds, True
    modules = import_multignn_modules()
    data_config = json.loads((run_folder / "data_config.json").read_text())
    start = time.time()
    request = training_request(plan, smoke_test)
    summary = train_model(request, modules, data_config, model_folder)
    seconds = time.time() - start
    save_training_marker(model_folder, summary, seconds, plan.step_budget_epochs)
    return summary, seconds, False


def run_experiment(plan: RunPlan, environment: str, smoke_test: bool) -> Path:
    """Train if needed, score, and write the record. Returns the record's path."""
    check_starting_point(plan, MODEL_ROOT, SOURCE_MODEL_ROOT)
    run_folder = prepare_run_folder(plan)
    os.chdir(run_folder)  # Multi-GNN reads ./model_settings.json from here
    settings = read_training_settings(run_folder, plan.model)
    extras: dict = {}
    summary, training_seconds, resumed = None, 0.0, False
    if plan.trains:
        summary, training_seconds, resumed = train_or_resume(
            plan, run_folder, smoke_test
        )
        # Read the log only now: before training it does not exist yet.
        extras["validation_curve"] = read_validation_curve(
            run_folder, plan.epochs, plan.validations
        )
        if not has_checkpoint(plan):  # a result, not a crash: see experiment_record
            record = build_no_improvement_record(
                plan, summary, training_seconds, environment, smoke_test,
                resumed, settings, extras,
            )  # fmt: skip
            return write_record(record)
    start = time.time()
    result = score_checkpoint(scoring_request(plan))
    runtime = training_seconds + time.time() - start
    record = build_experiment_record(
        plan, result, summary, runtime, environment, smoke_test, resumed,
        training_seconds, settings, extras,
    )  # fmt: skip
    return write_record(record)


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
    parser.add_argument(
        "--source-seed",
        type=int,
        default=None,
        help="Seed of the source model to fine-tune from (default: --seed)",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Appended to this run's id (never to its source model's), e.g. _b20",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Plan, run, and report where the record went."""
    arguments = parse_arguments()
    budget = resolve_step_budget(arguments.mode, arguments.step_budget_epochs)
    plan = plan_run(
        arguments.model, arguments.mode, arguments.fraction,
        arguments.seed, budget, arguments.smoke_test, arguments.source_seed,
        arguments.run_suffix,
    )  # fmt: skip
    path = run_experiment(plan, arguments.environment, arguments.smoke_test)
    print("Wrote record: %s" % path)


if __name__ == "__main__":
    main()
