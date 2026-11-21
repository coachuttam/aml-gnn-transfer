"""Tests for resuming a finished-training run by scoring it (no torch, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

import src.run_experiment as experiment
from src.metrics import score_period
from src.run_experiment import (
    check_starting_point,
    plan_run,
    resumes_scoring,
    run_experiment,
)
from src.score_checkpoint import ScoringResult
from src.train_wrapper import TrainingSummary
from src.training_marker import (
    load_training_marker,
    marker_file,
    save_training_marker,
)

SUMMARY = TrainingSummary(1000, 100, 7, "abc123", 3)
SETTINGS_FILE = json.dumps(
    {
        "gin": {
            "params": {
                "lr": 0.006,
                "w_ce1": 1.0,
                "w_ce2": 6.275,
                "dropout": 0.01,
                "final_dropout": 0.1,
                "n_hidden": 66,
            }
        }
    }
)


def finished_scratch_run(model_root: Path):
    """A scratch run with a checkpoint and a training-complete marker on disk."""
    plan = plan_run("gin", "scratch", 0.1, 0, 3)
    folder = model_root / plan.run_id
    folder.mkdir(parents=True)
    (folder / "checkpoint_True.tar").write_text("weights")
    save_training_marker(folder, SUMMARY, 1800.0, 3)
    return plan, folder


def test_the_marker_round_trips_the_summary_and_the_training_time(
    tmp_path: Path,
) -> None:
    save_training_marker(tmp_path, SUMMARY, 12.5)
    marker = load_training_marker(tmp_path)
    assert marker.summary == SUMMARY and marker.training_seconds == 12.5


def test_no_marker_means_training_never_finished(tmp_path: Path) -> None:
    assert load_training_marker(tmp_path) is None


def test_a_checkpoint_with_a_marker_is_resumed_by_scoring_only(tmp_path: Path) -> None:
    plan, _ = finished_scratch_run(tmp_path)
    assert resumes_scoring(plan, tmp_path)


def test_a_checkpoint_without_a_marker_is_not_resumed(tmp_path: Path) -> None:
    plan, folder = finished_scratch_run(tmp_path)
    marker_file(folder).unlink()
    assert not resumes_scoring(plan, tmp_path)


def test_a_marker_without_a_checkpoint_is_not_resumed(tmp_path: Path) -> None:
    plan, folder = finished_scratch_run(tmp_path)
    (folder / "checkpoint_True.tar").unlink()
    assert not resumes_scoring(plan, tmp_path)


def test_a_finished_run_may_start_even_though_its_checkpoint_exists(
    tmp_path: Path,
) -> None:
    plan, _ = finished_scratch_run(tmp_path)
    check_starting_point(plan, tmp_path)


def test_a_finished_finetune_run_resumes_without_its_source_model(
    tmp_path: Path,
) -> None:
    plan = plan_run("gin", "finetune", 0.1, 0, 3)
    folder = tmp_path / plan.run_id
    folder.mkdir()
    (folder / "checkpoint_True_finetuned.tar").write_text("weights")
    save_training_marker(folder, SUMMARY, 5.0, 3)
    check_starting_point(plan, tmp_path)  # source model folder does not exist


def test_a_checkpoint_from_interrupted_training_is_refused_not_scored(
    tmp_path: Path,
) -> None:
    plan, folder = finished_scratch_run(tmp_path)
    marker_file(folder).unlink()
    with pytest.raises(FileExistsError, match="never finished"):
        check_starting_point(plan, tmp_path)


def run_with_fakes(monkeypatch, tmp_path: Path, trained: List[str]) -> Any:
    """Run `run_experiment` with training, scoring and paths replaced by fakes."""
    plan = plan_run("gin", "scratch", 0.1, 0, 3)
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    written: List[Any] = []

    def fake_train(request, modules, data_config, model_folder):
        trained.append(request.model)
        (model_folder).mkdir(parents=True, exist_ok=True)
        (model_folder / "checkpoint_True.tar").write_text("weights")
        return SUMMARY

    def folder_for(run_id, model_root=None):
        return tmp_path / "models" / run_id

    run_folder = tmp_path / "run"
    run_folder.mkdir(exist_ok=True)
    (run_folder / "data_config.json").write_text("{}")
    (run_folder / "model_settings.json").write_text(SETTINGS_FILE)
    monkeypatch.setattr(experiment, "train_model", fake_train)
    monkeypatch.setattr(experiment, "model_folder_for_run", folder_for)
    monkeypatch.setattr(experiment, "import_multignn_modules", lambda: None)
    monkeypatch.setattr(experiment, "prepare_run_folder", lambda plan: run_folder)
    monkeypatch.setattr(
        experiment, "score_checkpoint", lambda request: ScoringResult(metrics, 4)
    )
    monkeypatch.setattr(
        experiment, "write_record", lambda record: written.append(record)
    )
    monkeypatch.chdir(tmp_path)
    return plan, written


def test_a_run_that_finished_training_is_scored_without_training_again(
    monkeypatch, tmp_path: Path
) -> None:
    trained: List[str] = []
    plan, written = run_with_fakes(monkeypatch, tmp_path, trained)
    folder = tmp_path / "models" / plan.run_id
    folder.mkdir(parents=True)
    (folder / "checkpoint_True.tar").write_text("weights")
    save_training_marker(folder, SUMMARY, 1800.0, 3)
    run_experiment(plan, "env", smoke_test=False)
    record = written[0]
    assert trained == []
    assert record.config["scoring_resumed"] is True
    assert record.config["kept_labels_fingerprint"] == "abc123"
    assert record.runtime_seconds >= 1800.0  # the saved training time is counted
    assert record.config["training_seconds"] == 1800.0
    assert record.config["scoring_seconds"] >= 0.0


def test_a_fresh_run_trains_then_writes_the_marker_so_it_can_be_resumed(
    monkeypatch, tmp_path: Path
) -> None:
    trained: List[str] = []
    plan, written = run_with_fakes(monkeypatch, tmp_path, trained)
    run_experiment(plan, "env", smoke_test=False)
    assert trained == ["gin"]
    assert written[0].config["scoring_resumed"] is False
    assert load_training_marker(tmp_path / "models" / plan.run_id) is not None


def test_a_marker_from_a_different_budget_is_not_treated_as_finished(
    tmp_path: Path,
) -> None:
    plan, folder = finished_scratch_run(tmp_path)  # marker says budget 3
    other = plan_run("gin", "scratch", 0.1, 0, 8)
    assert not resumes_scoring(other, tmp_path)
    with pytest.raises(FileExistsError, match="never finished at this budget"):
        check_starting_point(other, tmp_path)


def test_a_marker_from_older_code_without_a_budget_is_not_trusted(
    tmp_path: Path,
) -> None:
    plan, folder = finished_scratch_run(tmp_path)
    save_training_marker(folder, SUMMARY, 1800.0)  # no budget recorded
    assert not resumes_scoring(plan, tmp_path)


def test_training_that_saved_no_checkpoint_is_recorded_not_a_crash(
    monkeypatch, tmp_path: Path
) -> None:
    trained: List[str] = []
    plan, written = run_with_fakes(monkeypatch, tmp_path, trained)

    def train_without_saving(request, modules, data_config, model_folder):
        trained.append(request.model)
        model_folder.mkdir(parents=True, exist_ok=True)
        return SUMMARY  # validation F1 never beat 0: Multi-GNN saved nothing

    def must_not_score(request):
        raise AssertionError("there is no checkpoint to score")

    monkeypatch.setattr(experiment, "train_model", train_without_saving)
    monkeypatch.setattr(experiment, "score_checkpoint", must_not_score)
    run_experiment(plan, "env", smoke_test=False)
    record = written[0]
    assert record.config["status"] == "no_improvement"
    assert record.metrics["scored"]["minority_f1"] == 0.0
    assert record.metrics["scored"]["pr_auc"] is None  # not measured, not a zero
    assert record.best_epoch == -1
    assert record.config["kept_labels_fingerprint"] == "abc123"


def test_a_no_improvement_run_is_not_trained_again_after_a_restart(
    monkeypatch, tmp_path: Path
) -> None:
    trained: List[str] = []
    plan, written = run_with_fakes(monkeypatch, tmp_path, trained)
    folder = tmp_path / "models" / plan.run_id
    folder.mkdir(parents=True)  # finished training, no checkpoint
    save_training_marker(folder, SUMMARY, 900.0, 3)
    run_experiment(plan, "env", smoke_test=False)
    assert trained == []
    assert written[0].config["status"] == "no_improvement"
    assert written[0].config["scoring_resumed"] is True


def test_a_normal_record_says_completed_and_keeps_the_loss_weights(
    monkeypatch, tmp_path: Path
) -> None:
    plan, written = run_with_fakes(monkeypatch, tmp_path, [])
    run_experiment(plan, "env", smoke_test=False)
    config = written[0].config
    assert config["status"] == "completed"
    assert config["multignn_settings"]["w_ce2"] == 6.275
