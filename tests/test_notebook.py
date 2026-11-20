"""Checks on notebooks/kaggle_runner.ipynb that need no Kaggle."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_runner.ipynb"


def code_cells() -> List[str]:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def load_run_helper() -> Callable[[str], None]:
    """Execute only the `run` helper from the notebook's first code cell."""
    first = code_cells()[0]
    helper = first[: first.index("REPO_URL")]
    namespace: Dict[str, Any] = {}
    exec(helper, namespace)
    return namespace["run"]


def test_every_code_cell_is_plain_python_with_no_unchecked_shell_or_magic_lines() -> (
    None
):
    for source in code_cells():
        ast.parse(source)
        for line in source.splitlines():
            assert not line.lstrip().startswith(("!", "%")), line


def test_the_run_helper_shows_output_and_succeeds_on_a_good_command(capsys) -> None:
    load_run_helper()("echo hello")
    assert "hello" in capsys.readouterr().out


def test_the_run_helper_raises_when_a_command_fails() -> None:
    with pytest.raises(RuntimeError, match="exit code 3"):
        load_run_helper()("exit 3")


def test_the_run_helper_raises_when_a_command_in_the_middle_of_a_pipe_fails() -> None:
    with pytest.raises(RuntimeError, match="exit code"):
        load_run_helper()("false | cat")


def test_the_notebook_stops_at_a_missing_gpu_before_doing_any_work() -> None:
    first = code_cells()[0]
    assert "nvidia-smi" in first and first.index("nvidia-smi") > first.index("def run")


def test_the_environment_cell_checks_cuda_before_trusting_either_environment() -> None:
    cell = next(c for c in code_cells() if "install_matching_pyg_wheels" in c)
    assert cell.count("src.check_environment") == 2
