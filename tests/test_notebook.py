"""Checks on the Kaggle and Colab notebooks that need no GPU and no network."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from tools.make_notebooks import colab_cells, kaggle_cells, notebook_text

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
KAGGLE = NOTEBOOK_DIR / "kaggle_runner.ipynb"
COLAB = NOTEBOOK_DIR / "colab_runner.ipynb"
BOTH = pytest.mark.parametrize("path", [KAGGLE, COLAB], ids=["kaggle", "colab"])


def code_cells(path: Path) -> List[str]:
    cells = json.loads(path.read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def whole(path: Path) -> str:
    return "\n".join(code_cells(path))


def load_run_helper(path: Path) -> Callable[[str], None]:
    """Execute only the `run` helper from the notebook's first code cell."""
    first = code_cells(path)[0]
    namespace: Dict[str, Any] = {}
    exec(first[: first.index("REPO_URL")], namespace)
    return namespace["run"]


def test_the_committed_notebooks_match_what_the_generator_writes() -> None:
    assert KAGGLE.read_text() == notebook_text(kaggle_cells())
    assert COLAB.read_text() == notebook_text(colab_cells())


@BOTH
def test_every_code_cell_is_plain_python_with_no_shell_or_magic_lines(
    path: Path,
) -> None:
    for source in code_cells(path):
        ast.parse(source)
        for line in source.splitlines():
            assert not line.lstrip().startswith(("!", "%")), line


@BOTH
def test_the_run_helper_shows_output_and_succeeds_on_a_good_command(
    path: Path, capsys
) -> None:
    load_run_helper(path)("echo hello")
    assert "hello" in capsys.readouterr().out


@BOTH
def test_the_run_helper_raises_when_a_command_fails(path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code 3"):
        load_run_helper(path)("exit 3")


@BOTH
def test_the_run_helper_raises_when_a_command_in_a_pipe_fails(path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code"):
        load_run_helper(path)("false | cat")


@BOTH
def test_the_notebook_stops_at_a_missing_gpu_before_doing_any_work(
    path: Path,
) -> None:
    first = code_cells(path)[0]
    assert first.index("nvidia-smi") > first.index("def run")
    assert "GPU_NAME" in first


@BOTH
def test_the_environment_is_checked_before_either_environment_is_trusted(
    path: Path,
) -> None:
    cell = next(c for c in code_cells(path) if "src.pyg_wheels" in c)
    assert cell.count("src.check_environment") == 2
    assert "--only pyg_lib" in cell  # the env.yml path gets the fast sampler too


@BOTH
def test_the_environment_label_records_which_gpu_the_run_used(path: Path) -> None:
    assert 'ENVIRONMENT_LABEL = f"{ENV_LABEL} / {GPU_NAME}"' in whole(path)


@BOTH
def test_smoke_runs_are_followed_by_the_same_labels_and_steps_check(
    path: Path,
) -> None:
    text = whole(path)
    assert "src.check_smoke" in text and "--smoke-test" in text
    assert text.index("smoke_chain(") < text.index("src.check_smoke")


def test_colab_mounts_drive_and_keeps_everything_that_matters_there() -> None:
    text = whole(COLAB)
    for name in ("AML_DATA_ROOT", "AML_GRAPH_CACHE", "AML_MODEL_ROOT"):
        assert name in text
    assert "AML_RUN_RECORDS_DIR" in text and 'drive.mount("/content/drive")' in text


def test_colab_sets_its_folders_before_any_project_code_is_imported() -> None:
    text = whole(COLAB)
    assert text.index("AML_RUN_RECORDS_DIR") < text.index("from src.")


def test_colab_reads_kaggle_json_from_drive_and_never_prints_it() -> None:
    text = whole(COLAB)
    assert 'TOKEN = f"{DRIVE}/kaggle.json"' in text
    assert "print(TOKEN" not in text and "cat {TOKEN}" not in text
    assert "chmod 600" in text


def test_colab_hands_every_run_set_to_the_resumable_runner() -> None:
    text = whole(COLAB)
    assert "src.run_phase" in text and "--set {RUN_SET}" in text
    assert "--gpu-name" in text and "--home {DRIVE}" in text


def test_colab_run_set_defaults_to_the_smoke_test_and_names_every_phase() -> None:
    first = code_cells(COLAB)[0]
    assert 'RUN_SET = "smoke"' in first
    for name in ("pilot", "phaseA", "phaseB", "phaseC", "phaseD"):
        assert '"%s"' % name in first


def test_colab_long_runs_need_an_explicit_switch_that_defaults_to_off() -> None:
    text = whole(COLAB)
    assert "ALLOW_LONG_RUNS = False" in text and "--allow-long" in text


def test_colab_phase_d_seeds_can_be_chosen_in_the_notebook() -> None:
    assert "D_SEEDS = None" in whole(COLAB) and "--seeds" in whole(COLAB)


def test_colab_keeps_no_secrets_in_the_notebook() -> None:
    text = COLAB.read_text().lower()
    assert '"username"' not in text and '"key"' not in text


def test_the_colab_smoke_covers_source_then_scratch_then_finetune() -> None:
    assert 'for mode in ("source", "scratch", "finetune")' in whole(COLAB)


def test_colab_export_curves_set_needs_no_gpu_and_runs_the_exporter() -> None:
    text = whole(COLAB)
    assert '"export_curves"' in code_cells(COLAB)[0]
    assert 'NEEDS_GPU = RUN_SET != "export_curves"' in text
    assert "src.export_curves --home {DRIVE} --sets pilot phaseA" in text
    assert "if NEEDS_GPU:" in code_cells(COLAB)[0]  # the nvidia-smi gate is skipped
