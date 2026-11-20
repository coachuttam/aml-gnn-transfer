"""Estimate GPU hours and Colab compute units for the pilot, before spending any.

The inputs are the Kaggle T4 timings and stated assumptions in config.py. The
estimate is only as good as those inputs: the Colab smoke test gives real
timings, and these numbers should be redone from them.

Run (laptop is fine, no torch needed):
    python -m src.estimate_pilot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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


@dataclass(frozen=True)
class RunMinutes:
    """Where one run's minutes (on a T4) go."""

    training: float
    validation: float
    final_scoring: float
    loading: float

    @property
    def total(self) -> float:
        """All minutes for the run."""
        return self.training + self.validation + self.final_scoring + self.loading


def run_minutes(run: PilotRun) -> RunMinutes:
    """T4 minutes for one run. Every fraction has the same steps (equal-step rule)."""
    size = HI_TO_LI_SIZE_RATIO if run.mode == "source" else 1.0
    training = MEASURED_TRAIN_MINUTES_PER_FULL_LI_EPOCH * size * run.step_budget_epochs
    validation = MEASURED_VALIDATION_MINUTES_LI * size * run.step_budget_epochs
    # After training: one validation pass (threshold) and one test pass, once.
    final = (MEASURED_VALIDATION_MINUTES_LI + ASSUMED_TEST_PASS_MINUTES_LI) * size
    return RunMinutes(training, validation, final, 2 * ASSUMED_GRAPH_LOAD_MINUTES)


def total_t4_hours(runs: List[PilotRun]) -> float:
    """Hours for the whole list on a T4."""
    return sum(run_minutes(run).total for run in runs) / 60


def estimate_table(runs: List[PilotRun]) -> Dict[str, Dict[str, float]]:
    """Hours and units per GPU for the realistic and the hopeful speed case."""
    t4_hours = total_t4_hours(runs)
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


def main() -> None:
    """Print the estimate."""
    runs = pilot_runs()
    print(
        "Pilot: %d runs, source model trained for %d epochs"
        % (len(runs), PILOT_SOURCE_EPOCHS)
    )
    for run in runs[:2] + runs[-1:]:
        minutes = run_minutes(run)
        print("  %-28s %5.0f min on a T4" % (run.run_id, minutes.total))
    print("\n%-16s %6s %8s %8s" % ("GPU case", "speed", "hours", "units"))
    for name, row in estimate_table(runs).items():
        flag = "  <-- over budget" if row["units"] > COLAB_UNITS_AVAILABLE else ""
        values = (name, row["speed_vs_t4"], row["hours"], row["units"], flag)
        print("%-16s %5.1fx %8.1f %8.1f%s" % values)
    print("\nUnits available: %d." % COLAB_UNITS_AVAILABLE)
    print("GPU rates and speed-ups are assumptions (see config.py).")


if __name__ == "__main__":
    main()
