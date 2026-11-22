"""The run sets (pilot, phaseA to phaseD) and the list of runs in each.

`configs/phases.yaml` describes each set; this module reads it and lists the
runs in the order they should be done. A scratch run is always followed by the
finetune run with the same fraction and seed, and one seed is finished before
the next starts, so a stopped set still holds fair pairs.

Example:
    runs = phase_runs("phaseA")
    [run.run_id for run in runs]
    # ['gin_scratch_f1.0_s0_b20', 'gin_finetune_f1.0_s0_b20']
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from src import config
from src.records import make_run_id

REFERENCE_MARK = "$"  # in phases.yaml: "$BUDGET_C" means config.BUDGET_C


@dataclass(frozen=True)
class PhaseRun:
    """One run: what to run and the budget it must be started with."""

    model: str
    mode: str
    fraction: float
    seed: int
    step_budget_epochs: int  # 0 for zero-shot, which trains nothing
    source_seed: int
    run_suffix: str = ""

    @property
    def run_id(self) -> str:
        """The id its record and model folder are named by."""
        return make_run_id(
            self.model, self.mode, self.fraction, self.seed, self.run_suffix
        )

    def command_words(self) -> List[str]:
        """Arguments for `python -m src.run_experiment` (without --environment)."""
        words = [
            "--model", self.model, "--mode", self.mode,
            "--fraction", str(self.fraction), "--seed", str(self.seed),
        ]  # fmt: skip
        if self.mode != "zero_shot":
            words += ["--step-budget-epochs", str(self.step_budget_epochs)]
        if self.mode in ("finetune", "zero_shot"):
            words += ["--source-seed", str(self.source_seed)]
        if self.run_suffix:
            words += ["--run-suffix", self.run_suffix]
        return words


@dataclass(frozen=True)
class PhaseFolders:
    """Where a set keeps its records and models on Drive."""

    records: Path
    models: Path
    source_models: Path
    reuse_records: Optional[Path] = None  # finished runs here count too (same budget)


def resolve(value: Any) -> Any:
    """A "$NAME" string becomes config.NAME; lists are resolved item by item."""
    if isinstance(value, str) and value.startswith(REFERENCE_MARK):
        return getattr(config, value[len(REFERENCE_MARK) :])
    if isinstance(value, list):
        return [resolve(item) for item in value]
    return value


def load_phase(name: str, phases_file: Path = config.PHASES_FILE) -> Dict[str, Any]:
    """One set's settings from phases.yaml, with every "$NAME" resolved."""
    sets = yaml.safe_load(phases_file.read_text())
    if name not in sets:
        raise ValueError("Unknown run set %r, expected one of %s" % (name, list(sets)))
    return {key: resolve(value) for key, value in sets[name].items()}


def phase_names(phases_file: Path = config.PHASES_FILE) -> List[str]:
    """Every run set in phases.yaml."""
    return list(yaml.safe_load(phases_file.read_text()))


def phase_folders(
    name: str, home: Path, phases_file: Path = config.PHASES_FILE
) -> PhaseFolders:
    """The set's records, models and source-model folders inside `home`."""
    settings = load_phase(name, phases_file)
    return PhaseFolders(
        home / settings["records_folder"],
        home / settings["models_folder"],
        home / settings["source_folder"],
        home / load_phase(settings["reuse_from"], phases_file)["records_folder"]
        if settings.get("reuse_from")
        else None,
    )


def require_budget(name: str, budget: Optional[int]) -> int:
    """Stop with a clear message if the budget has not been set yet."""
    if budget is None:
        raise ValueError(
            "%s needs a step budget and none is set. Set BUDGET_C in src/config.py "
            "after reading Phase A, then run again." % name
        )
    return int(budget)


def grid_runs(
    settings: Dict[str, Any], seeds: Sequence[int], budget: int, suffix: str
) -> List[PhaseRun]:
    """Scratch then finetune at each fraction (largest first), seed by seed."""
    runs = []
    for seed in seeds:
        for fraction in sorted(settings["fractions"], reverse=True):
            for mode in settings["arms"]:
                runs.append(
                    PhaseRun(settings["model"], mode, fraction, seed, budget,
                             settings["source_seed"], suffix)
                )  # fmt: skip
    return runs


def single_runs(settings: Dict[str, Any]) -> List[PhaseRun]:
    """The source run and the zero-shot run, if the set has them."""
    model, seed = settings["model"], settings["source_seed"]
    runs = []
    if settings["source_epochs"] is not None:
        runs.append(
            PhaseRun(model, "source", 1.0, seed, settings["source_epochs"], seed)
        )
    if settings["zero_shot"]:
        runs.append(PhaseRun(model, "zero_shot", 1.0, seed, 0, seed))
    return runs


def phase_runs(
    name: str,
    seeds: Optional[Sequence[int]] = None,
    budget: Optional[int] = None,
    phases_file: Path = config.PHASES_FILE,
) -> List[PhaseRun]:
    """Every run of a set, in order. `seeds` and `budget` override the file."""
    settings = load_phase(name, phases_file)
    runs = single_runs(settings)
    if settings["arms"]:
        chosen = require_budget(name, budget or settings["budget"])
        suffix = "_b%d" % chosen if settings["suffix_budget"] else ""
        runs += grid_runs(settings, seeds or settings["seeds"], chosen, suffix)
    return runs


def pairs(runs: List[PhaseRun]) -> List[Tuple[PhaseRun, PhaseRun]]:
    """The (scratch, finetune) runs that share a fraction and seed."""
    scratch = {(r.fraction, r.seed): r for r in runs if r.mode == "scratch"}
    finetune = {(r.fraction, r.seed): r for r in runs if r.mode == "finetune"}
    return [(scratch[key], finetune[key]) for key in scratch if key in finetune]
