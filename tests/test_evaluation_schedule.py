"""Tests for src/evaluation_schedule.py (no torch needed)."""

from __future__ import annotations

import math

import pytest

from src.evaluation_schedule import (
    remember_loaders,
    scheduled_evaluation,
    validation_epochs,
)


def test_three_validations_in_thirty_epochs_fall_at_the_end_of_each_third() -> None:
    assert validation_epochs(30, 3) == {9, 19, 29}


def test_validating_every_epoch_when_epochs_equal_validations() -> None:
    assert validation_epochs(3, 3) == {0, 1, 2}


def test_the_last_epoch_is_always_validated() -> None:
    for epochs in (5, 12, 30, 50, 7):
        assert epochs - 1 in validation_epochs(epochs, 3)


def test_the_number_of_validations_is_the_same_for_every_epoch_count() -> None:
    assert [len(validation_epochs(n, 3)) for n in (3, 6, 12, 30)] == [3, 3, 3, 3]


def test_asking_for_more_validations_than_epochs_validates_every_epoch() -> None:
    assert validation_epochs(2, 5) == {0, 1}
    assert validation_epochs(4, None) == {0, 1, 2, 3}


def test_zero_epochs_cannot_be_validated() -> None:
    with pytest.raises(ValueError):
        validation_epochs(0, 3)


def make_evaluate(schedule):
    holder = {"validation_loader": "VAL", "test_loader": "TEST"}
    return scheduled_evaluation(lambda loader, *rest: 0.5, holder, schedule)


def test_the_test_loader_is_never_run_and_logs_as_not_scored() -> None:
    skipped = make_evaluate(frozenset({0}))("TEST", None)
    assert math.isnan(skipped)
    assert "Test F1: {:.4f}".format(skipped) == "Test F1: not scored"


def test_validation_runs_only_on_scheduled_epochs() -> None:
    evaluate = make_evaluate(frozenset({1}))
    results = [evaluate("VAL", None) for _ in range(3)]
    assert math.isnan(results[0]) and results[1] == 0.5 and math.isnan(results[2])


def test_a_skipped_validation_never_beats_the_best_score_so_far() -> None:
    skipped = make_evaluate(frozenset())("VAL", None)
    assert not skipped > 0.0


def test_the_loaders_are_noticed_when_they_are_built() -> None:
    holder: dict = {}
    get_loaders = remember_loaders(lambda *arguments: ("tr", "val", "te"), holder)
    get_loaders(1, 2, 3)
    assert (holder["validation_loader"], holder["test_loader"]) == ("val", "te")
