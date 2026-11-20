"""Tests for the parts of src/score_checkpoint.py that need no torch.

Importing the module must work on a laptop without torch; the model loading and
scoring itself is only exercised on a machine with Multi-GNN's environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.score_checkpoint import (
    ScoringRequest,
    checkpoint_file,
    multignn_argv,
    require_checkpoint,
)


def make_request() -> ScoringRequest:
    return ScoringRequest("run", "run", "gin", "HI-Small", 0, False)


def test_multignn_gets_the_dataset_folder_name_not_the_kaggle_name() -> None:
    words = multignn_argv(make_request())
    assert words[words.index("--data") + 1] == "Small_HI"


def test_multi_gin_eu_flags_are_all_passed() -> None:
    words = multignn_argv(make_request())
    for flag in ("--emlps", "--reverse_mp", "--ego", "--ports"):
        assert flag in words


def test_unique_name_is_switched_on_so_the_checkpoint_name_can_be_found() -> None:
    assert "--unique_name" in multignn_argv(make_request())


def test_scoring_never_turns_on_training_or_wandb_logging() -> None:
    words = multignn_argv(make_request())
    assert "--testing" in words
    assert "--save_model" not in words and "--finetune" not in words


def test_a_plain_checkpoint_has_multignns_default_name(tmp_path: Path) -> None:
    assert (
        checkpoint_file(tmp_path, finetuned=False) == tmp_path / "checkpoint_True.tar"
    )


def test_a_finetuned_checkpoint_gets_the_finetuned_suffix(tmp_path: Path) -> None:
    expected = tmp_path / "checkpoint_True_finetuned.tar"
    assert checkpoint_file(tmp_path, finetuned=True) == expected


def test_a_missing_checkpoint_gets_an_explanation_not_a_bare_file_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="improves"):
        require_checkpoint(tmp_path / "checkpoint_True.tar")
