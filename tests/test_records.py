"""Tests for src/records.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.records import (
    RunRecord,
    git_commit,
    make_run_id,
    model_folder_for_run,
    read_all_records,
    read_record,
    smoke_run_id,
    write_record,
)


def make_record(run_id: str = "gin_source_f1.0_s0", f1: float = 0.5) -> RunRecord:
    return RunRecord(
        run_id=run_id, model="gin", mode="source", fraction=1.0, seed=0,
        config={"epochs": 2}, ours_commit="a" * 40, multignn_commit="b" * 40,
        metrics={"minority_f1": f1}, epochs_trained=2, best_epoch=1,
        runtime_seconds=12.5,
    )  # fmt: skip


def run_git(folder: Path, *words: str) -> None:
    subprocess.run(
        ["git", "-C", str(folder)] + list(words), check=True, capture_output=True
    )


def test_run_id_reads_as_a_sentence_of_its_settings() -> None:
    assert make_run_id("pna", "finetune", 0.25, 1) == "pna_finetune_f0.25_s1"


def test_run_id_for_all_the_labels_ends_the_fraction_with_one_point_zero() -> None:
    assert make_run_id("gin", "scratch", 1.0, 0) == "gin_scratch_f1.0_s0"


def test_run_id_rejects_a_mode_that_does_not_exist() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        make_run_id("gin", "pretrain", 1.0, 0)


def test_run_id_rejects_a_label_fraction_above_one() -> None:
    with pytest.raises(ValueError, match="fraction"):
        make_run_id("gin", "scratch", 1.5, 0)


def test_each_run_gets_its_own_model_folder_named_by_run_id(tmp_path: Path) -> None:
    folder = model_folder_for_run("pna_finetune_f0.25_s1", tmp_path)
    assert folder == tmp_path / "pna_finetune_f0.25_s1"


def test_a_written_record_reads_back_unchanged(tmp_path: Path) -> None:
    path = write_record(make_record(), tmp_path)
    assert path == tmp_path / "gin_source_f1.0_s0.json"
    assert read_record(path) == make_record()


def test_writing_the_same_run_twice_fails_and_keeps_the_first_result(
    tmp_path: Path,
) -> None:
    write_record(make_record(f1=0.5), tmp_path)
    with pytest.raises(FileExistsError):
        write_record(make_record(f1=0.9), tmp_path)
    assert read_all_records(tmp_path)[0].metrics["minority_f1"] == 0.5


def test_all_records_come_back_sorted_by_run_id(tmp_path: Path) -> None:
    write_record(make_record("b_run"), tmp_path)
    write_record(make_record("a_run"), tmp_path)
    assert [r.run_id for r in read_all_records(tmp_path)] == ["a_run", "b_run"]


def test_reading_records_from_a_folder_that_does_not_exist_gives_an_empty_list(
    tmp_path: Path,
) -> None:
    assert read_all_records(tmp_path / "missing") == []


def test_commit_hash_gains_a_dirty_marker_after_a_tracked_file_changes(
    tmp_path: Path,
) -> None:
    run_git(tmp_path, "init", "-q")
    (tmp_path / "file.txt").write_text("one\n")
    run_git(tmp_path, "add", ".")
    run_git(
        tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"
    )
    clean = git_commit(tmp_path)
    assert len(clean) == 40
    (tmp_path / "file.txt").write_text("two\n")
    assert git_commit(tmp_path) == clean + "+dirty"


def test_untracked_files_do_not_mark_a_commit_as_dirty(tmp_path: Path) -> None:
    run_git(tmp_path, "init", "-q")
    (tmp_path / "file.txt").write_text("one\n")
    run_git(tmp_path, "add", ".")
    run_git(
        tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"
    )
    (tmp_path / "stray.pyc").write_text("junk")
    assert not git_commit(tmp_path).endswith("+dirty")


def test_a_smoke_test_run_id_never_equals_the_real_run_id() -> None:
    assert smoke_run_id("gin_source_f1.0_s0") == "gin_source_f1.0_s0_smoke"


def test_a_suffix_goes_at_the_end_of_the_run_id() -> None:
    assert make_run_id("gin", "scratch", 1.0, 0, "_b20") == "gin_scratch_f1.0_s0_b20"
    assert make_run_id("gin", "zero_shot", 1.0, 0) == "gin_zero_shot_f1.0_s0"
