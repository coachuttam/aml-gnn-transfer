"""Tests for src/read_log.py, using log lines shaped like Multi-GNN's own."""

from __future__ import annotations

import pytest

from src.read_log import multignn_best_epoch, parse_epoch_scores

PREFIX = "2026-11-20 10:00:00,000 [INFO ] "


def log_lines(rows) -> str:
    """Rows of (train, validation, test) F1 turned into log text."""
    text = PREFIX + "Retrieving data\n" + PREFIX + "Illicit ratio = 1 / 2\n"
    for train, validation, test in rows:
        text += PREFIX + "Train F1: %.4f\n" % train
        text += PREFIX + "Validation F1: %.4f\n" % validation
        text += PREFIX + "Test F1: %.4f\n" % test
    return text


def test_every_logged_epoch_is_read_in_order() -> None:
    epochs = parse_epoch_scores(log_lines([(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]))
    assert [(e.epoch, e.validation_f1, e.test_f1) for e in epochs] == [
        (0, 0.2, 0.3),
        (1, 0.5, 0.6),
    ]


def test_only_the_latest_run_in_an_appended_log_is_read() -> None:
    older = log_lines([(0.9, 0.9, 0.9)])
    newer = log_lines([(0.1, 0.2, 0.3)])
    assert len(parse_epoch_scores(older + newer)) == 1


def test_an_epoch_cut_off_before_its_test_line_is_ignored() -> None:
    text = log_lines([(0.1, 0.2, 0.3)]) + PREFIX + "Train F1: 0.5000\n"
    assert len(parse_epoch_scores(text)) == 1


def test_the_best_epoch_is_the_last_one_that_beat_all_earlier_validation_scores() -> (
    None
):
    epochs = parse_epoch_scores(
        log_lines([(0, 0.2, 0.30), (0, 0.5, 0.40), (0, 0.4, 0.90), (0, 0.6, 0.50)])
    )
    best = multignn_best_epoch(epochs)
    assert (best.epoch, best.test_f1) == (3, 0.5)


def test_the_test_score_never_picks_the_epoch() -> None:
    epochs = parse_epoch_scores(
        log_lines([(0, 0.2, 0.1), (0, 0.5, 0.0), (0, 0.4, 0.99)])
    )
    assert multignn_best_epoch(epochs).epoch == 1


def test_epoch_zero_is_reported_only_when_no_later_epoch_improves() -> None:
    epochs = parse_epoch_scores(log_lines([(0, 0.9, 0.8), (0, 0.0, 0.7)]))
    assert multignn_best_epoch(epochs).epoch == 0


def test_a_later_epoch_beats_a_better_epoch_zero_as_multi_gnn_skips_it() -> None:
    epochs = parse_epoch_scores(log_lines([(0, 0.9, 0.8), (0, 0.3, 0.7)]))
    assert multignn_best_epoch(epochs).epoch == 1


def test_a_log_with_no_finished_epoch_is_an_error() -> None:
    with pytest.raises(ValueError, match="No completed epochs"):
        multignn_best_epoch(parse_epoch_scores(PREFIX + "Retrieving data\n"))


def test_a_wrapper_log_with_unscored_test_epochs_still_parses() -> None:
    text = (
        PREFIX
        + "Retrieving data\n"
        + PREFIX
        + "Train F1: 0.1000\n"
        + PREFIX
        + "Validation F1: 0.2000\n"
        + PREFIX
        + "Test F1: not scored\n"
    )
    epochs = parse_epoch_scores(text)
    assert epochs[0].validation_f1 == 0.2 and epochs[0].test_f1 is None


def test_the_validation_curve_has_one_point_per_validated_epoch_counted_from_one() -> (
    None
):
    from src.read_log import validation_curve

    log = (
        "Retrieving data\n"
        "Train F1: 0.1\nValidation F1: not scored\nTest F1: not scored\n"
        "Train F1: 0.2\nValidation F1: 0.3000\nTest F1: not scored\n"
    )
    assert validation_curve(parse_epoch_scores(log)) == [[2, 0.3]]


def test_a_run_folder_without_a_log_has_an_empty_curve(tmp_path) -> None:
    from src.read_log import read_validation_curve

    assert read_validation_curve(tmp_path) == []


def epoch_text(count: int, first_f1: float = 0.1) -> str:
    """`count` logged epochs, every one validated, F1 rising by 0.01 each epoch."""
    text = ""
    for number in range(count):
        text += "Train F1: 0.1\nValidation F1: %.4f\nTest F1: not scored\n" % (
            first_f1 + 0.01 * number
        )
    return text


def test_an_interrupted_attempt_before_the_finished_one_is_left_out_of_the_curve() -> (
    None
):
    from src.read_log import validation_curve

    # No second "Retrieving data" marker, as in the real Phase C logs.
    log = "Retrieving data\n" + epoch_text(3, 0.9) + epoch_text(5, 0.1)
    curve = validation_curve(parse_epoch_scores(log), epochs_run=5, validations=5)
    assert [epoch for epoch, _ in curve] == [1, 2, 3, 4, 5]
    assert curve[0][1] == 0.1 and curve[-1][1] == pytest.approx(0.14)


def test_the_curve_never_has_more_points_than_the_run_had_validations() -> None:
    from src.read_log import validation_curve

    log = "Retrieving data\n" + epoch_text(10)
    curve = validation_curve(parse_epoch_scores(log), validations=4)
    assert len(curve) == 4 and curve[-1][0] == 10


def test_without_limits_the_whole_log_is_the_curve() -> None:
    from src.read_log import validation_curve

    log = "Retrieving data\n" + epoch_text(3)
    assert len(validation_curve(parse_epoch_scores(log))) == 3
