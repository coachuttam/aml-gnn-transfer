"""Phase E: train and score the LightGBM baseline on LI-Small, one record per run.

Each run keeps the same training labels as the GNN runs of Phase C (same seed,
same fraction, same `keep_label_positions`), trains only on those labelled
training rows, picks the alert threshold on the validation period and scores the
test period once. Before anything is trained the run's kept-label fingerprint is
compared with the Phase C records; a mismatch stops the run.

Run (laptop, CPU):
    python -m src.lgbm_run --raw-csv data/Small_LI/LI-Small_Trans.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src import config
from src.label_mask import fingerprint, keep_label_positions
from src.lgbm_features import FEATURE_COLUMNS
from src.lgbm_inputs import LgbmInputs, load_inputs, with_extra_features
from src.lgbm_model import (  # noqa: F401  (re-exported for callers and tests)
    DEFAULT_SETTINGS,
    TUNED_SETTINGS,
    FittedModel,
    LightGbmSettings,
    class_weight,
    fit_best_weight,
    fit_one_weight,
    score_both_periods,
)
from src.records import RunRecord, git_commit, make_run_id, record_path, write_record

LGBM_MODELS_FOLDER = "lightgbm_models"  # under the data root, never in git


def kept_training_rows(train_end: int, fraction: float, seed: int) -> np.ndarray:
    """Training rows whose labels this run may use (a prefix of the sorted file)."""
    return keep_label_positions(train_end, fraction, seed)


def phase_c_fingerprints(fraction: float, seed: int, results_dir: Path) -> Set[str]:
    """Kept-label fingerprints of the GIN runs at this fraction and seed."""
    found: Set[str] = set()
    for folder in ("phaseC", "phaseA"):
        records = results_dir / "phases" / folder / "records"
        for path in records.glob("gin_*_f%s_s%d*.json" % (float(fraction), seed)):
            settings = json.loads(path.read_text())["config"]
            found.add(settings["kept_labels_fingerprint"])
    return found


def assert_same_labels_as_phase_c(
    kept: np.ndarray, fraction: float, seed: int, results_dir: Path
) -> str:
    """Stop unless Phase C used exactly these kept labels. Returns the fingerprint."""
    mine = fingerprint(kept)
    expected = phase_c_fingerprints(fraction, seed, results_dir)
    if not expected:
        raise RuntimeError(
            "No Phase C record for fraction %s seed %d in %s: cannot prove the "
            "labels match" % (fraction, seed, results_dir)
        )
    if expected != {mine}:
        raise RuntimeError(
            "Kept labels differ from Phase C (fraction %s, seed %d): %s vs %s. "
            "The split or the label mask does not match the GNN runs."
            % (fraction, seed, mine, sorted(expected))
        )
    return mine


def build_config(
    inputs: LgbmInputs,
    rows: np.ndarray,
    fitted: FittedModel,
    seconds: float,
    settings: LightGbmSettings,
) -> Dict[str, Any]:
    """The settings stored in the record (same keys as GNN records where they fit)."""
    return {
        "status": "completed",
        "dataset": "LI-Small",
        "step_budget_epochs": None,
        "epochs_requested": settings.max_trees,
        "validations": None,
        "steps_per_epoch": None,
        "gradient_steps": None,
        "environment": "local CPU (LightGBM %s)" % lightgbm_version(),
        "smoke_test": False,
        "scoring_resumed": False,
        "training_seconds": seconds,
        "scoring_seconds": 0.0,  # scoring is inside the run time above
        "multignn_settings": None,
        "stock_training_loop": False,
        "checkpoint_run_id": None,
        "fine_tuned_from": None,
        "checkpoint_is_last_epoch": None,
        "validation_curve": [],  # trees are not epochs; no per-epoch curve is kept
        "train_transactions": inputs.split.train_end,
        "labels_kept": len(rows),
        "laundering_labels_kept": int(inputs.truth[rows].sum()),
        "kept_labels_fingerprint": fingerprint(rows),
        "label_mask_seed": None,  # set by the caller: it is the run's seed
        "features": feature_description(inputs),
        "feature_names": inputs.names,
        "class_weight": fitted.weight,
        "validation_pr_auc_by_weight": fitted.validation_pr_auc_by_weight,
        "lightgbm_settings": settings.to_dict(),
    }


def feature_description(inputs: LgbmInputs) -> str:
    """Which feature set the run used, in words."""
    if len(inputs.names) == len(FEATURE_COLUMNS):
        return "Project 2 graph features (past only, no labels)"
    return "Project 2 graph features plus extra fan-in/out and 2-hop features"


def lightgbm_version() -> str:
    """The installed LightGBM version, for the record."""
    import lightgbm

    return lightgbm.__version__


def run_one(
    inputs: LgbmInputs,
    fraction: float,
    seed: int,
    records_dir: Path,
    results_dir: Path,
    settings: LightGbmSettings = DEFAULT_SETTINGS,
) -> Optional[Path]:
    """Train and score one (fraction, seed). Returns the record path, None if done."""
    run_id = make_run_id(config.LIGHTGBM_MODEL, "scratch", fraction, seed)
    if record_path(run_id, records_dir).exists():
        print("%s: record exists, skipping" % run_id)
        return None
    rows = kept_training_rows(inputs.split.train_end, fraction, seed)
    assert_same_labels_as_phase_c(rows, fraction, seed, results_dir)
    start = time.time()
    fitted = fit_best_weight(inputs, rows, settings, seed)
    metrics = score_both_periods(inputs, fitted)
    seconds = time.time() - start
    save_model(fitted, run_id)
    run_config = build_config(inputs, rows, fitted, seconds, settings)
    run_config["label_mask_seed"] = seed
    record = RunRecord(
        run_id, config.LIGHTGBM_MODEL, "scratch", fraction, seed, run_config,
        git_commit(config.PROJECT_ROOT), "n/a (no Multi-GNN code used)", metrics,
        int(fitted.model.n_estimators_),
        int(fitted.model.best_iteration_ or 0), seconds,
    )  # fmt: skip
    print(
        "%s: test F1 %.4f in %.0f s"
        % (run_id, metrics["scored"]["minority_f1"], seconds)
    )
    return write_record(record, records_dir)


def save_model(fitted: FittedModel, run_id: str) -> Path:
    """Keep the trained trees (outside git) so reason codes can be made later."""
    folder = config.DATA_ROOT / LGBM_MODELS_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (run_id + ".txt")
    fitted.model.booster_.save_model(str(path))
    return path


def run_phase_e(
    raw_csv: Path,
    cache_dir: Path,
    records_dir: Path,
    results_dir: Path,
    rebuild: bool,
    simple: bool = False,
) -> List[Path]:
    """Every (fraction, seed) of Phase E; finished runs are skipped.

    `simple` is the first version: Project 2's 16 features and settings. The
    default is the version tuned on validation (extra features, tuned settings).
    """
    inputs = load_inputs(raw_csv, cache_dir, rebuild)
    if not simple:
        inputs = with_extra_features(inputs, raw_csv, cache_dir)
    settings = DEFAULT_SETTINGS if simple else TUNED_SETTINGS
    written = []
    for seed in config.PHASE_C_SEEDS:
        for fraction in sorted(config.PHASE_C_FRACTIONS, reverse=True):
            path = run_one(inputs, fraction, seed, records_dir, results_dir, settings)
            written += [path] if path else []
    return written


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_raw = config.DATA_ROOT / "Small_LI" / "LI-Small_Trans.csv"
    parser.add_argument("--raw-csv", type=Path, default=default_raw)
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_ROOT / "lightgbm")
    phase_e = config.PHASES_RESULTS_DIR / config.PHASE_E_RUN_SET / "records"
    parser.add_argument("--records-dir", type=Path, default=phase_e)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the features")
    parser.add_argument(
        "--simple", action="store_true", help="Project 2's 16 features and settings"
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run Phase E."""
    arguments = parse_arguments()
    written = run_phase_e(
        arguments.raw_csv, arguments.cache_dir, arguments.records_dir,
        config.RESULTS_DIR, arguments.rebuild, arguments.simple,
    )  # fmt: skip
    print("Wrote %d records to %s" % (len(written), arguments.records_dir))


if __name__ == "__main__":
    main()
