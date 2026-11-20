"""Tests for src/download_data.py and src/check_smoke.py (no Kaggle, no GPU)."""

from __future__ import annotations

from pathlib import Path

from src.check_smoke import load_smoke_pair, mismatches
from src.config import DATASETS, KAGGLE_DATASET
from src.download_data import download_command, needs_download
from src.records import RunRecord, make_run_id, smoke_run_id, write_record

HI_SMALL = DATASETS["HI-Small"]


def test_a_dataset_with_no_files_needs_downloading(tmp_path: Path) -> None:
    assert needs_download(HI_SMALL, tmp_path)


def test_a_cached_raw_file_is_not_downloaded_again(tmp_path: Path) -> None:
    folder = tmp_path / "Small_HI"
    folder.mkdir()
    (folder / "HI-Small_Trans.csv").write_text("x")
    assert not needs_download(HI_SMALL, tmp_path)


def test_a_formatted_file_alone_is_enough_to_skip_the_download(tmp_path: Path) -> None:
    folder = tmp_path / "Small_HI"
    folder.mkdir()
    (folder / "formatted_transactions.csv").write_text("x")
    assert not needs_download(HI_SMALL, tmp_path)


def test_the_download_goes_straight_into_the_folder_the_formatter_reads(
    tmp_path: Path,
) -> None:
    command = download_command(HI_SMALL, tmp_path)
    assert command[command.index("-d") + 1] == KAGGLE_DATASET
    assert command[command.index("-f") + 1] == "HI-Small_Trans.csv"
    assert command[command.index("-p") + 1] == str(tmp_path / "Small_HI")
    assert "--unzip" in command


def make_smoke_record(mode: str, **changes) -> RunRecord:
    config = {
        "kept_labels_fingerprint": "abc",
        "gradient_steps": 100,
        "labels_kept": 5000,
        "epochs_requested": 2,
    }
    config.update(changes)
    run_id = smoke_run_id(make_run_id("gin", mode, 0.1, 0))
    return RunRecord(run_id, "gin", mode, 0.1, 0, config, "c", "m", {}, 2, 1, 1.0)


def test_matching_labels_and_steps_give_no_problems() -> None:
    scratch, finetune = make_smoke_record("scratch"), make_smoke_record("finetune")
    assert mismatches(scratch, finetune) == []


def test_a_different_label_set_is_reported() -> None:
    scratch = make_smoke_record("scratch")
    finetune = make_smoke_record("finetune", kept_labels_fingerprint="zzz")
    assert "kept_labels_fingerprint differs" in mismatches(scratch, finetune)[0]


def test_a_different_step_count_is_reported() -> None:
    scratch = make_smoke_record("scratch")
    finetune = make_smoke_record("finetune", gradient_steps=99)
    assert "gradient_steps differs" in mismatches(scratch, finetune)[0]


def test_a_missing_field_is_reported_not_treated_as_a_match() -> None:
    scratch = make_smoke_record("scratch", kept_labels_fingerprint=None)
    finetune = make_smoke_record("finetune", kept_labels_fingerprint=None)
    assert "missing" in mismatches(scratch, finetune)[0]


def test_the_pair_is_loaded_from_the_smoke_records_in_a_folder(tmp_path: Path) -> None:
    write_record(make_smoke_record("scratch"), tmp_path)
    write_record(make_smoke_record("finetune"), tmp_path)
    scratch, finetune = load_smoke_pair(tmp_path, 0.1, 0)
    assert (scratch.mode, finetune.mode) == ("scratch", "finetune")
