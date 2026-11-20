"""Tests for src/label_mask.py and the masked training loader hook."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import LABEL_FRACTIONS
from src.label_mask import fingerprint, keep_label_positions
from src.train_wrapper import with_masked_training_loader

TRAIN_SIZE = 10_000


def test_ten_percent_labels_are_inside_the_twenty_five_percent_set() -> None:
    small = set(keep_label_positions(TRAIN_SIZE, 0.10, seed=1))
    larger = set(keep_label_positions(TRAIN_SIZE, 0.25, seed=1))
    assert small < larger


def test_every_smaller_fraction_is_inside_every_larger_one() -> None:
    sets = [set(keep_label_positions(TRAIN_SIZE, f, seed=2)) for f in LABEL_FRACTIONS]
    for smaller, larger in zip(sets, sets[1:]):
        assert smaller <= larger


def test_scratch_and_finetune_get_identical_labels_at_the_same_fraction_and_seed() -> (
    None
):
    scratch = keep_label_positions(TRAIN_SIZE, 0.25, seed=0)
    finetune = keep_label_positions(TRAIN_SIZE, 0.25, seed=0)
    assert np.array_equal(scratch, finetune)


def test_different_seeds_keep_different_labels() -> None:
    first = keep_label_positions(TRAIN_SIZE, 0.10, seed=0)
    second = keep_label_positions(TRAIN_SIZE, 0.10, seed=1)
    assert not np.array_equal(first, second)


def test_the_kept_count_follows_the_fraction() -> None:
    assert len(keep_label_positions(TRAIN_SIZE, 0.25, seed=0)) == 2500


def test_all_the_labels_are_kept_at_fraction_one() -> None:
    kept = keep_label_positions(TRAIN_SIZE, 1.0, seed=0)
    assert list(kept) == list(range(TRAIN_SIZE))


def test_kept_positions_are_sorted_unique_and_inside_the_training_period() -> None:
    kept = keep_label_positions(TRAIN_SIZE, 0.5, seed=0)
    assert list(kept) == sorted(set(kept))
    assert kept.min() >= 0 and kept.max() < TRAIN_SIZE


def test_a_fraction_that_rounds_to_nothing_still_keeps_one_label() -> None:
    assert len(keep_label_positions(10, 0.01, seed=0)) == 1


def test_a_fraction_above_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="fraction"):
        keep_label_positions(100, 1.5, seed=0)


def test_an_empty_training_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="no training transactions"):
        keep_label_positions(0, 0.5, seed=0)


def test_equal_label_sets_share_a_fingerprint_and_different_ones_do_not() -> None:
    first = keep_label_positions(TRAIN_SIZE, 0.10, seed=0)
    assert fingerprint(first) == fingerprint(first.copy())
    assert fingerprint(first) != fingerprint(keep_label_positions(TRAIN_SIZE, 0.10, 1))


def test_masked_training_swaps_only_the_training_loader() -> None:
    stock = lambda *arguments: ("stock_train", "stock_validation", "stock_test")  # noqa: E731
    seen = {}

    def make_loader(train_data, kept, transform, args, uses_reverse):
        seen["kept"] = kept
        return "masked_train"

    kept = np.array([1, 5, 9])
    get_loaders = with_masked_training_loader(stock, kept, False, make_loader)
    loaders = get_loaders("tr", "val", "te", None, None, None, None, None)
    assert loaders == ("masked_train", "stock_validation", "stock_test")
    assert list(seen["kept"]) == [1, 5, 9]


def test_a_smoke_run_without_a_stock_checkpoint_gets_its_last_epoch_saved(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from src.train_wrapper import saving_last_epoch_if_none

    saved = []
    modules = SimpleNamespace(
        train_util=SimpleNamespace(
            save_model=lambda *arguments: saved.append(arguments)
        )
    )
    original = lambda *arguments: "final model"  # noqa: E731
    outcome: dict = {}
    train = saving_last_epoch_if_none(
        original, modules, tmp_path / "missing.tar", outcome
    )
    arguments = [None] * 15
    arguments[7], arguments[9], arguments[14] = (
        "optimizer",
        SimpleNamespace(n_epochs=10),
        "config",
    )
    assert train(*arguments) == "final model"
    assert saved == [("final model", "optimizer", 9, arguments[9], "config")]
    assert outcome["last_epoch_saved"] is True


def test_a_checkpoint_saved_during_training_is_left_alone(tmp_path) -> None:
    from types import SimpleNamespace

    from src.train_wrapper import saving_last_epoch_if_none

    checkpoint = tmp_path / "checkpoint_True.tar"
    checkpoint.write_text("best epoch")
    saved: list = []
    modules = SimpleNamespace(train_util=SimpleNamespace(save_model=saved.append))
    outcome: dict = {}
    train = saving_last_epoch_if_none(lambda *a: "model", modules, checkpoint, outcome)
    train(*([None] * 15))
    assert saved == [] and "last_epoch_saved" not in outcome
