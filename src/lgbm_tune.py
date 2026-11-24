"""Try a few LightGBM settings and feature sets, judged on VALIDATION only.

Nothing here reads a test row or a test label (AGENTS.md rule 2). Each candidate
is trained on the kept labels of three Phase C cells and scored on validation
PR-AUC (the choice) and validation best-F1 (for reading). The table is written
to `results/phases/phaseE/tuning_val.csv`.

Run: python -m src.lgbm_tune
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src import config
from src.lgbm_inputs import LgbmInputs, load_inputs, with_extra_features
from src.lgbm_model import (
    DEFAULT_SETTINGS,
    LightGbmSettings,
    best_iteration,
    fit_best_weight,
    score_both_periods,
)
from src.lgbm_run import kept_training_rows

# (fraction, seed) cells the choice is averaged over: a small, a middle, a full share.
TUNING_CELLS: Tuple[Tuple[float, int], ...] = ((0.10, 0), (0.25, 0), (1.00, 0))
LONG_RUN = replace(DEFAULT_SETTINGS, max_trees=2000, patience=100)


@dataclass(frozen=True)
class Candidate:
    """One thing to try: a name, whether to add the extra features, the settings."""

    name: str
    use_extra: bool
    settings: LightGbmSettings


TUNING_CANDIDATES: Tuple[Candidate, ...] = (
    Candidate("project2_settings", False, DEFAULT_SETTINGS),
    Candidate("more_rounds_patience", False, LONG_RUN),
    Candidate(
        "lower_weights",
        False,
        replace(LONG_RUN, weights=(0.5, 1.0, 2.0, 5.0)),
    ),
    Candidate("extra_features", True, DEFAULT_SETTINGS),
    Candidate("extra_features_long", True, LONG_RUN),
    Candidate(
        "extra_lower_weights",
        True,
        replace(LONG_RUN, weights=(0.5, 1.0, 2.0, 5.0)),
    ),
    Candidate(
        "extra_regularised",
        True,
        replace(
            LONG_RUN, leaves=15, min_child_samples=200, subsample=0.8, column_share=0.8
        ),
    ),
)


def tune_one(
    candidate: Candidate, base: LgbmInputs, extra: Optional[LgbmInputs]
) -> List[Dict[str, object]]:
    """Validation numbers for one candidate in every tuning cell."""
    inputs = extra if candidate.use_extra and extra is not None else base
    results = []
    for fraction, seed in TUNING_CELLS:
        rows = kept_training_rows(inputs.split.train_end, fraction, seed)
        start = time.time()
        fitted = fit_best_weight(inputs, rows, candidate.settings, seed)
        validation = score_both_periods(inputs, fitted, include_test=False)[
            "validation"
        ]
        results.append(
            {
                "candidate": candidate.name,
                "fraction": fraction,
                "seed": seed,
                "features": inputs.features.shape[1],
                "class_weight": fitted.weight,
                "best_iteration": best_iteration(fitted.model),
                "val_pr_auc": validation["pr_auc"],
                "val_best_f1": validation["minority_f1"],
                "seconds": round(time.time() - start, 1),
            }
        )
        print({k: v for k, v in results[-1].items() if k != "candidate"}, flush=True)
    return results


def mean_pr_auc(rows: List[Dict[str, object]], name: str) -> float:
    """A candidate's validation PR-AUC averaged over the tuning cells."""
    values = [float(r["val_pr_auc"]) for r in rows if r["candidate"] == name]  # type: ignore[arg-type]
    if not values:
        return float("nan")
    return float(np.mean(values))


def kept_rows_of_other_candidates(
    path: Path, rerun: List[Candidate]
) -> List[Dict[str, object]]:
    """Rows already in the table for candidates that are not being run again."""
    if not path.exists():
        return []
    names = {candidate.name for candidate in rerun}
    with open(path, newline="") as handle:
        return [
            dict(row) for row in csv.DictReader(handle) if row["candidate"] not in names
        ]


def write_table(rows: List[Dict[str, object]], path: Path) -> None:
    """Save the validation table (no test numbers in it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    """Run every candidate and print which has the best mean validation PR-AUC."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    raw = config.DATA_ROOT / "Small_LI" / "LI-Small_Trans.csv"
    parser.add_argument("--raw-csv", type=Path, default=raw)
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_ROOT / "lightgbm")
    parser.add_argument("--only", nargs="+", help="Run just these candidates")
    arguments = parser.parse_args(argv)
    base = load_inputs(arguments.raw_csv, arguments.cache_dir)
    extra = with_extra_features(base, arguments.raw_csv, arguments.cache_dir)
    folder = config.PHASES_RESULTS_DIR / config.PHASE_E_RUN_SET
    table = folder / "tuning_val.csv"
    chosen = [
        c for c in TUNING_CANDIDATES if not arguments.only or c.name in arguments.only
    ]
    rows = kept_rows_of_other_candidates(table, chosen) if arguments.only else []
    for candidate in chosen:
        print("==", candidate.name, flush=True)
        rows += tune_one(candidate, base, extra)
    write_table(rows, table)
    known = [
        c for c in TUNING_CANDIDATES if any(r["candidate"] == c.name for r in rows)
    ]
    ranked = sorted(known, key=lambda c: -mean_pr_auc(rows, c.name))
    for candidate in ranked:
        print(
            "%-24s mean val PR-AUC %.4f"
            % (candidate.name, mean_pr_auc(rows, candidate.name))
        )


if __name__ == "__main__":
    main()
