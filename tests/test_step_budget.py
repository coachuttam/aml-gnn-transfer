"""Tests for src/step_budget.py."""

from __future__ import annotations

import pytest

from src.config import LABEL_FRACTIONS
from src.step_budget import epochs_for_budget, steps_per_epoch


def test_the_budget_epochs_are_used_as_they_are_at_all_the_labels() -> None:
    assert epochs_for_budget(5, 1.0) == 5


def test_ten_percent_of_the_labels_gets_ten_times_the_epochs() -> None:
    assert epochs_for_budget(5, 0.10) == 50


def test_every_configured_fraction_gets_epochs_matching_its_share() -> None:
    for fraction in LABEL_FRACTIONS:
        assert epochs_for_budget(5, fraction) * fraction == pytest.approx(5)


def test_total_steps_match_across_fractions_to_within_one_batch_per_epoch() -> None:
    full = steps_per_epoch(4_100_000, 8192) * epochs_for_budget(5, 1.0)
    tenth = steps_per_epoch(410_000, 8192) * epochs_for_budget(5, 0.10)
    assert abs(full - tenth) <= epochs_for_budget(5, 0.10)


def test_a_partly_filled_last_batch_still_costs_a_step() -> None:
    assert steps_per_epoch(8193, 8192) == 2


def test_a_budget_of_zero_epochs_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one epoch"):
        epochs_for_budget(0, 1.0)


def test_a_fraction_above_one_is_rejected_for_the_budget() -> None:
    with pytest.raises(ValueError, match="fraction"):
        epochs_for_budget(5, 2.0)
