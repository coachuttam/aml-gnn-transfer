"""Tests for src/record_stock_run.py (the pure record-building part)."""

from __future__ import annotations

import numpy as np

from src.metrics import score_period
from src.read_log import EpochScores
from src.record_stock_run import RunContext, build_record, parse_arguments
from src.score_checkpoint import ScoringRequest, ScoringResult


def make_record(smoke_test: bool = False, logged_epoch: int = 1):
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    request = ScoringRequest("run", "run", "gin", "HI-Small", 0, False)
    context = RunContext(
        "source", 1.0, 3, 90.0, "env.yml", smoke_test, "a" * 40, "b" * 40
    )
    logged = EpochScores(logged_epoch, 0.5, 0.6, 0.55)
    return build_record(request, ScoringResult(metrics, best_epoch=2), context, logged)


def test_the_record_keeps_our_scored_metrics_under_their_own_key() -> None:
    assert make_record().metrics["scored"]["minority_f1"] == 1.0


def test_the_record_stores_both_commits_and_the_runtime() -> None:
    record = make_record()
    assert (record.ours_commit, record.multignn_commit) == ("a" * 40, "b" * 40)
    assert record.runtime_seconds == 90.0


def test_the_record_says_which_environment_and_epoch_made_the_checkpoint() -> None:
    record = make_record()
    assert record.config["environment"] == "env.yml"
    assert record.best_epoch == 2 and record.epochs_trained == 3


def test_a_smoke_test_record_is_marked_so_tables_can_leave_it_out() -> None:
    assert make_record(smoke_test=True).config["smoke_test"] is True


def test_the_command_line_needs_the_run_facts_the_scorer_cannot_know() -> None:
    arguments = parse_arguments(
        "--run-id r --model gin --dataset HI-Small --epochs 3 "
        "--runtime-seconds 9 --environment env.yml".split()
    )
    assert arguments.mode == "source" and arguments.fraction == 1.0


def test_the_record_keeps_multi_gnns_logged_f1_beside_our_scored_f1() -> None:
    metrics = make_record().metrics
    assert metrics["logged_by_multignn"]["minority_f1"] == 0.55
    assert metrics["scored"]["minority_f1"] == 1.0


def test_the_record_flags_when_the_log_and_the_checkpoint_name_different_epochs() -> (
    None
):
    agree = make_record(logged_epoch=1).metrics["logged_by_multignn"]
    differ = make_record(logged_epoch=0).metrics["logged_by_multignn"]
    assert (
        agree["same_epoch_as_checkpoint"] is True
    )  # log epoch 1 -> checkpoint epoch 2
    assert differ["same_epoch_as_checkpoint"] is False
