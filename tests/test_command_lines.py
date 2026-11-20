"""Every command-line entry point must accept the arguments the notebook passes.

A broken option list once reached Kaggle because nothing parsed it. These tests
parse the exact words the notebook uses, and check each module starts with --help.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.prepare import parse_arguments as parse_prepare
from src.record_stock_run import parse_arguments as parse_record
from src.run_experiment import parse_arguments as parse_experiment
from src.score_checkpoint import parse_arguments as parse_score

ENTRY_POINTS = [
    "src.prepare",
    "src.record_stock_run",
    "src.run_experiment",
    "src.score_checkpoint",
    "src.check_environment",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_every_entry_point_starts_and_prints_its_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_prepare_accepts_the_words_the_notebook_passes() -> None:
    arguments = parse_prepare(
        "--raw-dir /kaggle/input/x --run-id gin_source_f1.0_s0_smoke".split()
    )
    assert arguments.run_id == "gin_source_f1.0_s0_smoke"


def test_prepare_can_name_the_run_whose_model_will_be_fine_tuned() -> None:
    arguments = parse_prepare("--load-from-run-id gin_source_f1.0_s0".split())
    assert arguments.load_from_run_id == "gin_source_f1.0_s0"


def test_prepare_defaults_to_the_first_seed_source_run() -> None:
    assert parse_prepare([]).run_id == "gin_source_f1.0_s0"


def test_record_stock_run_accepts_the_words_the_notebook_passes() -> None:
    arguments = parse_record(
        "--run-id r --model gin --dataset HI-Small --seed 0 --epochs 3 "
        "--runtime-seconds 90 --environment env.yml --smoke-test".split()
    )
    assert arguments.smoke_test is True


def test_run_experiment_accepts_the_words_the_notebook_passes() -> None:
    arguments = parse_experiment(
        "--model gin --mode scratch --fraction 0.1 --seed 0 "
        "--step-budget-epochs 1 --environment env.yml --smoke-test".split()
    )
    assert arguments.fraction == 0.1 and arguments.smoke_test is True


def test_score_checkpoint_accepts_a_zero_shot_style_request() -> None:
    arguments = parse_score(
        "--run-id z --checkpoint-run-id s --dataset LI-Small --model pna".split()
    )
    assert arguments.checkpoint_run_id == "s"


def test_prepare_runs_end_to_end_and_writes_the_run_folder_the_notebook_expects(
    tmp_path, monkeypatch
) -> None:
    for folder in ("Small_HI", "Small_LI"):  # pretend the formatter already ran
        (tmp_path / "data" / folder).mkdir(parents=True)
        (tmp_path / "data" / folder / "formatted_transactions.csv").write_text(
            "EdgeID,Is Laundering\n0,0\n1,1\n"
        )
    environment = {
        "PATH": "/usr/bin:/bin",
        "AML_DATA_ROOT": str(tmp_path / "data"),
        "AML_MODEL_ROOT": str(tmp_path / "models"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "src.prepare", "--run-id", "gin_source_f1.0_s0_smoke"],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    run_folder = tmp_path / "data" / "multignn_runs" / "gin_source_f1.0_s0_smoke"
    assert (run_folder / "data_config.json").exists()
    assert (run_folder / "model_settings.json").exists()
    assert (tmp_path / "models" / "gin_source_f1.0_s0_smoke").is_dir()
