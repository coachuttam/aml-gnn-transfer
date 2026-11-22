"""Tests for src/metrics.py. Expected numbers are worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    best_f1_threshold,
    precision_at_top,
    recall_at_budget,
    score_period,
    score_run,
)


def test_precision_at_top_counts_only_the_highest_scored_rows() -> None:
    truth = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_top(truth, scores, top_count=2) == 0.5


def test_precision_at_top_is_zero_for_an_empty_period() -> None:
    assert precision_at_top(np.array([]), np.array([]), top_count=100) == 0.0


def test_recall_at_a_budget_bigger_than_the_data_finds_every_laundering_row() -> None:
    truth = np.array([1, 0, 1])
    scores = np.array([0.1, 0.9, 0.5])
    assert recall_at_budget(truth, scores, budget=10_000) == 1.0


def test_recall_at_budget_is_zero_when_the_period_has_no_laundering() -> None:
    assert recall_at_budget(np.zeros(4, dtype=int), np.arange(4.0), budget=2) == 0.0


def test_best_threshold_separates_a_perfectly_ranked_validation_set() -> None:
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert best_f1_threshold(truth, scores) == 0.8


def test_choosing_a_threshold_without_any_laundering_rows_is_an_error() -> None:
    with pytest.raises(ValueError, match="no laundering rows"):
        best_f1_threshold(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]))


def test_score_period_matches_hand_worked_precision_recall_and_f1() -> None:
    truth = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    metrics = score_period(truth, scores, threshold=0.5)  # flags rows 0 and 2
    assert metrics.alerts == 2
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.minority_f1 == 0.5


def test_score_period_gives_zero_f1_when_nothing_is_flagged() -> None:
    truth = np.array([1, 0])
    metrics = score_period(truth, np.array([0.2, 0.1]), threshold=0.9)
    assert (metrics.minority_f1, metrics.precision, metrics.alerts) == (0.0, 0.0, 0)


def test_pr_auc_is_one_when_every_laundering_row_is_ranked_first() -> None:
    truth = np.array([1, 1, 0, 0])
    assert score_period(truth, np.array([0.9, 0.8, 0.2, 0.1]), 0.5).pr_auc == 1.0


def test_alert_budget_recall_has_one_entry_per_configured_budget() -> None:
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    assert set(metrics.recall_at_alert_budget) == {"10000", "25000", "50000"}


def test_score_run_takes_its_threshold_from_validation_not_from_test() -> None:
    validation_truth = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.1, 0.2, 0.8, 0.9])
    test_truth = np.array([0, 1, 0, 1])
    test_scores = np.array([0.1, 0.3, 0.95, 0.99])  # a test-tuned cut-off would differ
    metrics = score_run(validation_truth, validation_scores, test_truth, test_scores)
    assert metrics.threshold == 0.8


def test_metrics_turn_into_a_plain_dictionary_for_the_json_record() -> None:
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    assert metrics.to_dict()["minority_f1"] == 1.0


def test_validation_metrics_use_a_threshold_chosen_on_validation_itself() -> None:
    from src.metrics import score_validation

    truth, scores = np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.4, 0.1])
    result = score_validation(truth, scores)
    assert result.minority_f1 == 1.0 and 0.4 < result.threshold <= 0.6
