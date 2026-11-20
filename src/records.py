"""One JSON record per run, never overwritten (AGENTS.md rule 5).

Tables and charts are rebuilt from these files, never typed by hand.

Example:
    run_id = make_run_id("pna", "finetune", 0.25, 1)   # "pna_finetune_f0.25_s1"
    write_record(record, RUN_RECORDS_DIR)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.config import MODEL_ROOT, RUN_MODES, RUN_RECORDS_DIR


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to understand and rebuild one run's result.

    `config` holds the free-form settings (datasets, flags, epochs asked for,
    which Python environment was used). `ours_commit` and `multignn_commit`
    end in "+dirty" if tracked files had uncommitted changes.
    """

    run_id: str
    model: str
    mode: str
    fraction: float
    seed: int
    config: Dict[str, Any]
    ours_commit: str
    multignn_commit: str
    metrics: Dict[str, Any]
    epochs_trained: int
    best_epoch: int
    runtime_seconds: float


def make_run_id(model: str, mode: str, fraction: float, seed: int) -> str:
    """Run id that reads as its settings, e.g. `pna_finetune_f0.25_s1`."""
    if mode not in RUN_MODES:
        raise ValueError("Unknown mode %r, expected one of %s" % (mode, RUN_MODES))
    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1], got %r" % fraction)
    return "%s_%s_f%s_s%d" % (model, mode, float(fraction), seed)


def model_folder_for_run(run_id: str, model_root: Path = MODEL_ROOT) -> Path:
    """Each run saves its model in its own folder, named by run id."""
    return model_root / run_id


def git_commit(repo_dir: Path) -> str:
    """Current commit hash, with "+dirty" if tracked files have uncommitted edits.

    Untracked files are ignored on purpose: it matches "git diff is empty",
    and stray __pycache__ folders must not mark the submodule as modified.
    """
    commit = _git(repo_dir, "rev-parse", "HEAD")
    changes = _git(repo_dir, "status", "--porcelain", "--untracked-files=no")
    return commit + "+dirty" if changes else commit


def _git(repo_dir: Path, *arguments: str) -> str:
    """Run one git command in `repo_dir` and return its trimmed output."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir)] + list(arguments),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def record_path(run_id: str, runs_dir: Path = RUN_RECORDS_DIR) -> Path:
    """Where the record for `run_id` lives."""
    return runs_dir / (run_id + ".json")


def write_record(record: RunRecord, runs_dir: Path = RUN_RECORDS_DIR) -> Path:
    """Save a record. Raises FileExistsError rather than overwrite an earlier run."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(record.run_id, runs_dir)
    with open(path, "x") as handle:  # "x" = create, fail if it already exists
        json.dump(asdict(record), handle, indent=2)
        handle.write("\n")
    return path


def read_record(path: Path) -> RunRecord:
    """Load one record file."""
    return RunRecord(**json.loads(path.read_text()))


def read_all_records(runs_dir: Path = RUN_RECORDS_DIR) -> List[RunRecord]:
    """Load every record in the folder, sorted by run id."""
    if not runs_dir.exists():
        return []
    return [read_record(path) for path in sorted(runs_dir.glob("*.json"))]
