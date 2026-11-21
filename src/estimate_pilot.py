"""Estimate GPU hours and Colab compute units for the pilot, before spending any.

With no options it uses the Kaggle T4 timings and stated assumptions in
config.py and compares T4, L4 and A100. With `--gpu` it uses numbers measured on
the GPU you actually have, so the result needs no speed-up guess.

Run (laptop is fine, no torch needed):
    python -m src.estimate_pilot
    python -m src.estimate_pilot --gpu L4 --train-minutes 6 --validation-minutes 1.8 \
        --test-minutes 1.8 --load-minutes 1 --units-per-hour 1.7 --units-remaining 48
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.config import (
    ASSUMED_GRAPH_LOAD_MINUTES,
    ASSUMED_TEST_PASS_MINUTES_LI,
    COLAB_UNITS_AVAILABLE,
    COLAB_UNITS_PER_HOUR,
    GPU_SPEED_CASES,
    HI_TO_LI_SIZE_RATIO,
    MEASURED_TRAIN_MINUTES_PER_FULL_LI_EPOCH,
    MEASURED_VALIDATION_MINUTES_LI,
    PILOT_SOURCE_EPOCHS,
)
from src.pilot import PilotRun, pilot_runs

APPROVAL_HOURS = 3.0  # AGENTS.md rule 6: ask before any run longer than this


@dataclass(frozen=True)
class Timings:
    """Minutes for each phase on LI-Small, on one GPU (HI-Small is scaled by size)."""

    train_per_full_epoch: float = MEASURED_TRAIN_MINUTES_PER_FULL_LI_EPOCH
    validation_pass: float = MEASURED_VALIDATION_MINUTES_LI
    test_pass: float = ASSUMED_TEST_PASS_MINUTES_LI
    graph_load: float = ASSUMED_GRAPH_LOAD_MINUTES


@dataclass(frozen=True)
class RunMinutes:
    """Where one run's minutes go."""

    training: float
    validation: float
    final_scoring: float
    loading: float

    @property
    def total(self) -> float:
        """All minutes for the run."""
        return self.training + self.validation + self.final_scoring + self.loading


def run_minutes(run: PilotRun, timings: Timings = Timings()) -> RunMinutes:
    """Minutes for one run. Every fraction has the same steps (equal-step rule)."""
    size = HI_TO_LI_SIZE_RATIO if run.mode == "source" else 1.0
    budget = run.step_budget_epochs
    training = timings.train_per_full_epoch * size * budget
    validation = timings.validation_pass * size * budget
    # After training: one validation pass (threshold) and one test pass, once.
    final = (timings.validation_pass + timings.test_pass) * size
    return RunMinutes(training, validation, final, 2 * timings.graph_load)


def total_hours(runs: List[PilotRun], timings: Timings = Timings()) -> float:
    """Hours for the whole list on the GPU the timings came from."""
    return sum(run_minutes(run, timings).total for run in runs) / 60


def source_epochs_within(hours: float, timings: Timings = Timings()) -> int:
    """The most source epochs that keep the source run inside `hours`."""
    epochs = 0
    while True:
        trial = PilotRun("gin", "source", 1.0, 0, epochs + 1)
        if run_minutes(trial, timings).total / 60 > hours:
            return epochs
        epochs += 1


def estimate_table(runs: List[PilotRun]) -> Dict[str, Dict[str, float]]:
    """Hours and units per GPU for the realistic and the hopeful speed case."""
    t4_hours = total_hours(runs)
    table: Dict[str, Dict[str, float]] = {}
    for gpu, cases in GPU_SPEED_CASES.items():
        for label, speed in zip(("realistic", "hopeful"), cases):
            if label == "hopeful" and speed == cases[0]:
                continue  # same as the realistic case: show it once
            hours = t4_hours / speed
            table["%s %s" % (gpu, label)] = {
                "speed_vs_t4": speed,
                "hours": hours,
                "units": hours * COLAB_UNITS_PER_HOUR[gpu],
            }
    return table


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options. Timings are minutes measured on the named GPU."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", default=None, help="Use timings measured on this GPU")
    parser.add_argument("--train-minutes", type=float, default=None,
                        help="One full LI-Small epoch of training")  # fmt: skip
    parser.add_argument("--validation-minutes", type=float, default=None)
    parser.add_argument("--test-minutes", type=float, default=None)
    parser.add_argument("--load-minutes", type=float, default=None)
    parser.add_argument("--units-per-hour", type=float, default=None)
    parser.add_argument("--units-remaining", type=float, default=COLAB_UNITS_AVAILABLE)
    return parser.parse_args(argv)


def timings_from(arguments: argparse.Namespace) -> Timings:
    """Defaults, replaced by whatever was measured."""
    base = Timings()
    return Timings(
        arguments.train_minutes or base.train_per_full_epoch,
        arguments.validation_minutes or base.validation_pass,
        arguments.test_minutes or base.test_pass,
        arguments.load_minutes or base.graph_load,
    )


def print_measured(arguments: argparse.Namespace, runs: List[PilotRun]) -> None:
    """The estimate for the GPU you measured, with the 3-hour and units checks."""
    timings = timings_from(arguments)
    rate = arguments.units_per_hour or COLAB_UNITS_PER_HOUR.get(arguments.gpu, 0.0)
    hours = total_hours(runs, timings)
    units = hours * rate
    source_hours = run_minutes(runs[0], timings).total / 60
    comparison_minutes = run_minutes(runs[1], timings).total
    print("On %s: pilot %.1f h, about %.0f units" % (arguments.gpu, hours, units))
    print("Source run alone: %.1f h" % source_hours)
    print("Each comparison run: %.0f min" % comparison_minutes)
    if units > arguments.units_remaining:
        print("OVER: %.0f units needed, %.0f left" % (units, arguments.units_remaining))
    if source_hours > APPROVAL_HOURS:
        fit = source_epochs_within(APPROVAL_HOURS, timings)
        print("Source exceeds %.0f h: %d epochs would fit" % (APPROVAL_HOURS, fit))


def print_comparison(runs: List[PilotRun]) -> None:
    """The default T4 / L4 / A100 table built from the config assumptions."""
    for run in runs[:2] + runs[-1:]:
        print("  %-28s %5.0f min on a T4" % (run.run_id, run_minutes(run).total))
    print("\n%-16s %6s %8s %8s" % ("GPU case", "speed", "hours", "units"))
    for name, row in estimate_table(runs).items():
        flag = "  <-- over budget" if row["units"] > COLAB_UNITS_AVAILABLE else ""
        values = (name, row["speed_vs_t4"], row["hours"], row["units"], flag)
        print("%-16s %5.1fx %8.1f %8.1f%s" % values)
    print("\nUnits available: %d." % COLAB_UNITS_AVAILABLE)
    print("GPU rates and speed-ups are assumptions (see config.py).")


def main() -> None:
    """Print the estimate."""
    arguments = parse_arguments()
    runs = pilot_runs()
    count, epochs = len(runs), PILOT_SOURCE_EPOCHS
    print("Pilot: %d runs, source model for %d epochs" % (count, epochs))
    if arguments.gpu:
        print_measured(arguments, runs)
    else:
        print_comparison(runs)


if __name__ == "__main__":
    main()
