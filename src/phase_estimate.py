"""Time and Colab-unit estimates for a run, and a running total of units spent.

Estimates start from times measured on a Colab L4 in the pilot (config.py) and
are scaled for other GPUs and for PNA. They are rough: the notebook prints the
real runtime and units after each run, so a wrong estimate shows up at once.

Example:
    run = PhaseRun("gin", "scratch", 1.0, 0, 20, 0, "_b20")
    estimate_minutes(run, "L4")        # about 175
    units_for("NVIDIA L4", 3600)       # 1.94
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config import (
    ASSUMED_PNA_TIME_VS_GIN,
    COLAB_UNITS_PER_HOUR,
    GPU_TIME_VS_L4,
    HI_TO_LI_SIZE_RATIO,
    L4_MINUTES_FINAL_SCORING,
    L4_MINUTES_PER_BUDGET_EPOCH,
)
from src.phases import PhaseRun


def gpu_key(gpu_name: str) -> Optional[str]:
    """'NVIDIA L4' -> 'L4', 'Tesla T4' -> 'T4'; None if the GPU has no known rate."""
    for key in COLAB_UNITS_PER_HOUR:
        if key in gpu_name.upper():
            return key
    return None


def units_for(gpu_name: str, seconds: float) -> float:
    """Colab compute units for `seconds` on this GPU (0 if its rate is unknown)."""
    key = gpu_key(gpu_name)
    if key is None:
        return 0.0
    return seconds / 3600 * COLAB_UNITS_PER_HOUR[key]


def estimate_minutes(run: PhaseRun, gpu: str) -> float:
    """Minutes for one run on `gpu` ('L4', 'T4' or 'A100').

    Training and validation both scale with the budget (equal steps at every
    fraction). A source run works on HI-Small, which is a bit smaller.
    """
    size = HI_TO_LI_SIZE_RATIO if run.mode == "source" else 1.0
    model_factor = ASSUMED_PNA_TIME_VS_GIN if run.model == "pna" else 1.0
    training = run.step_budget_epochs * L4_MINUTES_PER_BUDGET_EPOCH * size
    scoring = L4_MINUTES_FINAL_SCORING * size
    return (training + scoring) * model_factor * GPU_TIME_VS_L4[gpu]


def estimate_units(minutes: float, gpu: str) -> float:
    """Units for `minutes` on `gpu`."""
    return minutes / 60 * COLAB_UNITS_PER_HOUR[gpu]


@dataclass(frozen=True)
class PhaseEstimate:
    """Hours and units for a list of runs on one GPU."""

    runs: int
    hours: float
    units: float
    longest_run_hours: float


def estimate_phase(runs: List[PhaseRun], gpu: str) -> PhaseEstimate:
    """Total estimate for a list of runs."""
    minutes = [estimate_minutes(run, gpu) for run in runs]
    total = sum(minutes)
    return PhaseEstimate(
        len(runs), total / 60, estimate_units(total, gpu), max(minutes, default=0) / 60
    )


class UnitsLedger:
    """Adds up the units of the runs finished so far in this notebook session."""

    def __init__(self, gpu_name: str) -> None:
        self.gpu_name = gpu_name
        self.total_units = 0.0

    def add_run(self, seconds: float) -> float:
        """Record one finished run; returns its units."""
        units = units_for(self.gpu_name, seconds)
        self.total_units += units
        return units
