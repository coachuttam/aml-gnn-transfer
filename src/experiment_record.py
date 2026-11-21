"""Build the JSON record for one finished experiment run.

A run ends in one of two ways, shown in `config["status"]`:
  * "completed": a checkpoint was scored (threshold on validation, test once);
  * "no_improvement": training finished but no epoch ever beat validation F1 0
    after the first epoch, so Multi-GNN saved no checkpoint. That is a result
    (the model never learned to flag anything), not a crash. Test is not scored;
    minority F1, precision and recall are recorded as 0.0 and the other metrics
    as null, so no chart can mistake them for measured values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import MULTI_GNN_DIR, PROJECT_ROOT
from src.records import RunRecord, git_commit
from src.run_plan import RunPlan
from src.score_checkpoint import ScoringResult
from src.train_wrapper import TrainingSummary

STATUS_COMPLETED = "completed"
STATUS_NO_IMPROVEMENT = "no_improvement"
NO_CHECKPOINT_EPOCH = -1  # best_epoch when nothing was saved

# The Multi-GNN settings that decide how a run trains (model_settings.json).
SETTINGS_TO_RECORD = ("lr", "w_ce1", "w_ce2", "dropout", "final_dropout", "n_hidden")


def gradient_steps(plan: RunPlan, summary: Optional[TrainingSummary]) -> Optional[int]:
    """Total optimiser steps the run was given, for checking budgets match."""
    return summary.steps_per_epoch * plan.epochs if summary else None


def read_training_settings(run_folder: Path, model: str) -> Dict[str, float]:
    """Learning rate and loss weights from the model_settings.json the run used."""
    settings = json.loads((run_folder / "model_settings.json").read_text())
    params = settings[model]["params"]
    return {name: params[name] for name in SETTINGS_TO_RECORD}


def build_config(
    plan: RunPlan,
    summary: Optional[TrainingSummary],
    runtime_seconds: float,
    environment: str,
    smoke_test: bool,
    scoring_resumed: bool,
    training_seconds: float,
    training_settings: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """The free-form settings and facts stored in every record."""
    return {
        "status": STATUS_COMPLETED,
        "dataset": plan.dataset_name,
        "step_budget_epochs": plan.step_budget_epochs,
        "epochs_requested": plan.epochs,
        "validations": plan.validations,
        "steps_per_epoch": summary.steps_per_epoch if summary else None,
        "gradient_steps": gradient_steps(plan, summary),
        "environment": environment,
        "smoke_test": smoke_test,
        "scoring_resumed": scoring_resumed,
        # Phase times, so later estimates can use measured numbers per phase.
        "training_seconds": training_seconds,
        "scoring_seconds": runtime_seconds - training_seconds,
        "multignn_settings": training_settings,
        "stock_training_loop": False,
        "checkpoint_run_id": plan.checkpoint_run_id,
        "fine_tuned_from": plan.load_from_run_id,
        "train_transactions": summary.train_transactions if summary else None,
        "labels_kept": summary.labels_kept if summary else None,
        "laundering_labels_kept": summary.laundering_labels_kept if summary else None,
        "kept_labels_fingerprint": summary.kept_labels_fingerprint if summary else None,
        "label_mask_seed": plan.seed,
        "checkpoint_is_last_epoch": (
            summary.checkpoint_is_last_epoch if summary else None
        ),
    }


def make_record(
    plan: RunPlan,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    best_epoch: int,
    runtime_seconds: float,
) -> RunRecord:
    """Attach the plan's identity and both git commits to a config and metrics."""
    return RunRecord(
        run_id=plan.run_id, model=plan.model, mode=plan.mode,
        fraction=plan.fraction, seed=plan.seed, config=config,
        ours_commit=git_commit(PROJECT_ROOT),
        multignn_commit=git_commit(MULTI_GNN_DIR),
        metrics=metrics, epochs_trained=plan.epochs,
        best_epoch=best_epoch, runtime_seconds=runtime_seconds,
    )  # fmt: skip


def build_experiment_record(
    plan: RunPlan,
    result: ScoringResult,
    summary: Optional[TrainingSummary],
    runtime_seconds: float,
    environment: str,
    smoke_test: bool,
    scoring_resumed: bool = False,
    training_seconds: float = 0.0,
    training_settings: Optional[Dict[str, float]] = None,
) -> RunRecord:
    """Combine plan, scores and training facts into one record.

    `scoring_resumed` is True when training finished in an earlier session and
    this one only scored it; the runtime then adds the saved training time.
    """
    config = build_config(
        plan, summary, runtime_seconds, environment, smoke_test,
        scoring_resumed, training_seconds, training_settings,
    )  # fmt: skip
    metrics = {"scored": result.metrics.to_dict()}
    return make_record(plan, config, metrics, result.best_epoch, runtime_seconds)


def build_no_improvement_record(
    plan: RunPlan,
    summary: TrainingSummary,
    runtime_seconds: float,
    environment: str,
    smoke_test: bool,
    scoring_resumed: bool = False,
    training_settings: Optional[Dict[str, float]] = None,
) -> RunRecord:
    """The record for a run whose training saved no checkpoint (see module note)."""
    config = build_config(
        plan, summary, runtime_seconds, environment, smoke_test,
        scoring_resumed, runtime_seconds, training_settings,
    )  # fmt: skip
    config["status"] = STATUS_NO_IMPROVEMENT
    config["status_reason"] = (
        "No epoch after the first had validation F1 above 0, so no checkpoint was "
        "saved. Test was not scored; minority F1, precision and recall are set to 0.0."
    )
    metrics = {
        "scored": {
            "minority_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pr_auc": None,
            "precision_at_100": None,
            "alerts": None,
            "threshold": None,
            "recall_at_alert_budget": None,
        }
    }
    return make_record(plan, config, metrics, NO_CHECKPOINT_EPOCH, runtime_seconds)
