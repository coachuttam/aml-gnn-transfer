"""Tests for src/prepare.py that need no Kaggle data and no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import DATASETS, MULTI_GNN_DIR
from src.prepare import (
    build_data_config,
    link_raw_file,
    make_run_folder,
    raw_csv_path,
    summarize_formatted_file,
)

HI_SMALL = DATASETS["HI-Small"]


def test_raw_file_is_placed_in_the_folder_multi_gnn_will_read_from(
    tmp_path: Path,
) -> None:
    assert (
        raw_csv_path(HI_SMALL, tmp_path) == tmp_path / "Small_HI" / "HI-Small_Trans.csv"
    )


def test_linking_a_missing_kaggle_file_fails_with_a_clear_message(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Kaggle file not found"):
        link_raw_file(HI_SMALL, tmp_path / "nothing_here", tmp_path / "data")


def test_linked_raw_file_sits_beside_where_the_formatter_will_write(
    tmp_path: Path,
) -> None:
    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "HI-Small_Trans.csv").write_text("header\n")
    link = link_raw_file(HI_SMALL, kaggle_dir, tmp_path / "data")
    assert link.parent == tmp_path / "data" / "Small_HI"
    assert link.read_text() == "header\n"


def test_summary_counts_all_transactions_and_only_the_laundering_ones(
    tmp_path: Path,
) -> None:
    formatted = tmp_path / "formatted_transactions.csv"
    formatted.write_text("EdgeID,Is Laundering\n0,0\n1,1\n2,0\n3,1\n4,0\n")
    summary = summarize_formatted_file(formatted)
    assert summary.transactions == 5
    assert summary.laundering_transactions == 2


def test_run_folder_gets_our_data_config_and_an_unchanged_copy_of_model_settings(
    tmp_path: Path,
) -> None:
    config = build_data_config(tmp_path / "data", tmp_path / "save", tmp_path / "load")
    run_folder = make_run_folder(tmp_path / "run", config)
    written = json.loads((run_folder / "data_config.json").read_text())
    assert written["paths"]["aml_data"] == str(tmp_path / "data")
    original = (MULTI_GNN_DIR / "model_settings.json").read_text()
    assert (run_folder / "model_settings.json").read_text() == original


def test_model_folders_exist_after_making_the_run_folder(tmp_path: Path) -> None:
    config = build_data_config(tmp_path / "data", tmp_path / "save", tmp_path / "load")
    make_run_folder(tmp_path / "run", config)
    assert (tmp_path / "save").is_dir()
    assert (tmp_path / "load").is_dir()
