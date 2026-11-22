"""Run one set of runs (pilot, phaseA to phaseD), safely and resumably.

For every run in the set: skip it if its record already exists (status
completed or no_improvement), otherwise print the estimate, run
`src.run_experiment` as its own process, then print the real runtime, the units
it cost and the running total. After a scratch/finetune pair has both finished,
the pair is checked for identical labels and equal gradient steps.

Records and models go to the set's folders under --home (Google Drive on Colab),
so a disconnect loses nothing: run it again and finished runs are skipped.

Example (Colab, from the repo root):
    python -m src.run_phase --set phaseA --home /content/drive/MyDrive/aml \
        --gpu-name "NVIDIA L4" --environment "env.yml / NVIDIA L4"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.check_smoke import mismatches
from src.config import APPROVAL_GPU_HOURS, FINISHED_STATUSES
from src.phase_estimate import (
    UnitsLedger,
    estimate_minutes,
    estimate_phase,
    gpu_key,
)
from src.phases import (
    PhaseFolders,
    PhaseRun,
    pairs,
    phase_folders,
    phase_names,
    phase_runs,
)
from src.pilot import set_aside_partial_run, supersede_other_budget_records
from src.records import read_record, record_path


def record_status(run: PhaseRun, records_dir: Path) -> Optional[str]:
    """The status in this run's record, or None if it has no record yet."""
    path = record_path(run.run_id, records_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())["config"].get("status", "completed")


def is_finished(run: PhaseRun, records_dir: Path) -> bool:
    """True if the run has a record that ends it: it is skipped, never redone.

    A record with any other status is a surprise, so it stops the notebook
    instead of being silently skipped or overwritten.
    """
    status = record_status(run, records_dir)
    if status is None:
        return False
    if status not in FINISHED_STATUSES:
        raise ValueError(
            "%s has a record with status %r; look at it before running again"
            % (run.run_id, status)
        )
    return True


def check_pair(
    scratch: PhaseRun, finetune: PhaseRun, records_dir: Path, quiet: bool = False
) -> None:
    """Raise if a finished pair differs in kept labels or gradient steps."""
    paths = [record_path(run.run_id, records_dir) for run in (scratch, finetune)]
    if not all(path.exists() for path in paths):
        return
    problems = mismatches(read_record(paths[0]), read_record(paths[1]))
    if problems:
        raise RuntimeError(
            "%s and %s are not a fair pair: %s"
            % (scratch.run_id, finetune.run_id, "; ".join(problems))
        )
    if not quiet:
        print("  pair check ok: same labels and same gradient steps")


def guard_long_run(run: PhaseRun, gpu: str, allow_long: bool) -> None:
    """AGENTS.md rule 6: a run expected to pass the approval hours needs a yes."""
    hours = estimate_minutes(run, gpu) / 60
    if hours > APPROVAL_GPU_HOURS and not allow_long:
        raise RuntimeError(
            "%s is expected to take %.1f h, over the %.0f h that need the owner's "
            "approval. Set ALLOW_LONG_RUNS = True once approved."
            % (run.run_id, hours, APPROVAL_GPU_HOURS)
        )


def child_environment(folders: PhaseFolders) -> Dict[str, str]:
    """This process's environment plus the set's own folders."""
    environment = dict(os.environ)
    environment["AML_RUN_RECORDS_DIR"] = str(folders.records)
    environment["AML_MODEL_ROOT"] = str(folders.models)
    environment["AML_SOURCE_MODEL_ROOT"] = str(folders.source_models)
    environment["PYTHONUNBUFFERED"] = "1"  # so the notebook shows progress as it comes
    return environment


def launch(
    run: PhaseRun, python: str, label: str, environment: Dict[str, str]
) -> float:
    """Run one experiment as its own process; returns the seconds it took."""
    command = shlex.split(python) + ["-m", "src.run_experiment"]
    command += run.command_words() + ["--environment", label]
    start = time.time()
    subprocess.run(command, env=environment, check=True)
    return time.time() - start


def print_plan(name: str, runs: List[PhaseRun], gpu: str, finished: int) -> None:
    """The set's size and estimate, before anything runs."""
    estimate = estimate_phase(runs, gpu)
    print("%s: %d runs, %d already have a record." % (name, len(runs), finished))
    print("Estimate on %s: %.1f h, about %.1f units (longest run %.1f h)." % (
        gpu, estimate.hours, estimate.units, estimate.longest_run_hours))  # fmt: skip


@dataclass(frozen=True)
class RunContext:
    """What every run in a set shares."""

    gpu: str  # 'L4', 'T4' or 'A100'
    ledger: UnitsLedger
    folders: PhaseFolders
    python: str
    label: str
    environment: Dict[str, str]
    allow_long: bool


def prepare_run(run: PhaseRun, folders: PhaseFolders) -> None:
    """Move an interrupted attempt's files aside (never deleted) to start fresh."""
    moved = set_aside_partial_run(run, folders.models)
    if moved:
        print("  earlier attempt of %s moved to %s" % (run.run_id, moved))


def run_one(run: PhaseRun, number: int, total: int, context: RunContext) -> None:
    """Estimate, run, then report real time and units for one run."""
    guard_long_run(run, context.gpu, context.allow_long)
    minutes = estimate_minutes(run, context.gpu)
    print("\n[%d/%d] %s: estimate %.0f min" % (number, total, run.run_id, minutes))
    prepare_run(run, context.folders)
    seconds = launch(run, context.python, context.label, context.environment)
    units = context.ledger.add_run(seconds)
    print("[%d/%d] %s: %.0f min (estimated %.0f min), %.2f units" % (
        number, total, run.run_id, seconds / 60, minutes, units))  # fmt: skip
    print("  running total this session: %.2f units" % context.ledger.total_units)


def make_context(arguments: argparse.Namespace, folders: PhaseFolders) -> RunContext:
    """Collect the shared settings from the command line."""
    return RunContext(
        gpu_key(arguments.gpu_name) or "L4",
        UnitsLedger(arguments.gpu_name),
        folders,
        arguments.python,
        arguments.environment,
        child_environment(folders),
        arguments.allow_long,
    )


def run_set(arguments: argparse.Namespace) -> None:
    """Run every unfinished run of the set, checking each pair as it completes."""
    runs = phase_runs(arguments.set, arguments.seeds, arguments.budget)
    folders = phase_folders(arguments.set, Path(arguments.home))
    for old in supersede_other_budget_records(runs, folders.records):
        print("Superseded (different step budget):", old)
    context = make_context(arguments, folders)
    finished = sum(is_finished(run, folders.records) for run in runs)
    print_plan(arguments.set, runs, context.gpu, finished)
    for number, run in enumerate(runs, start=1):
        if is_finished(run, folders.records):
            print(
                "[%d/%d] %s: record exists, skipping" % (number, len(runs), run.run_id)
            )
            continue
        run_one(run, number, len(runs), context)
        for scratch, finetune in pairs([run] + runs):
            if run in (scratch, finetune):
                check_pair(scratch, finetune, folders.records)
    for scratch, finetune in pairs(runs):  # pairs finished in an earlier session
        check_pair(scratch, finetune, folders.records, quiet=True)
    print("\nDone. Units spent this session: %.2f" % context.ledger.total_units)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=phase_names())
    parser.add_argument("--home", required=True, help="Project folder (Drive on Colab)")
    parser.add_argument("--python", default=sys.executable, help="Python for the runs")
    parser.add_argument("--gpu-name", required=True, help="e.g. 'NVIDIA L4'")
    parser.add_argument("--environment", required=True, help="Label stored in records")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=None, help="Overrides the file")
    parser.add_argument("--allow-long", action="store_true", help="Owner approved >3 h")
    return parser.parse_args(argv)


def main() -> None:
    """Run the chosen set."""
    run_set(parse_arguments())


if __name__ == "__main__":
    main()
