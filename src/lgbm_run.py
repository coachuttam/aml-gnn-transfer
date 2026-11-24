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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.metrics import average_precision_score

from src import config
from src.label_mask import fingerprint, keep_label_positions
from src.lgbm_inputs import LgbmInputs, load_inputs
from src.metrics import best_f1_threshold, score_period, score_validation
from src.records import RunRecord, git_commit, make_run_id, record_path, write_record

LGBM_MODELS_FOLDER = "lightgbm_models"  # under the data root, never in git


@dataclass(frozen=True)
class FittedModel:
    """The weight that won on validation PR-AUC and everything needed to score it."""

    model: Any
    weight: float
    validation_pr_auc_by_weight: Dict[str, float]


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


def class_weight(weight: object, truth: np.ndarray) -> float:
    """ "balanced" becomes negatives / positives among the training rows."""
    if weight != "balanced":
        return float(weight)  # type: ignore[arg-type]
    positives = max(int(truth.sum()), 1)
    return (len(truth) - positives) / positives


def fit_one_weight(inputs: LgbmInputs, rows: np.ndarray, weight: float) -> Any:
    """LightGBM on the kept rows; early stopping watches validation PR-AUC."""
    import lightgbm

    model = lightgbm.LGBMClassifier(
        n_estimators=config.LIGHTGBM_MAX_TREES,
        learning_rate=config.LIGHTGBM_LEARNING_RATE,
        num_leaves=config.LIGHTGBM_LEAVES,
        scale_pos_weight=weight,
        random_state=config.LIGHTGBM_SEED,
        verbose=-1,
    )
    validation = inputs.split.validation
    model.fit(
        inputs.features[rows],
        inputs.truth[rows],
        eval_set=[(inputs.features[validation], inputs.truth[validation])],
        eval_metric="average_precision",
        feature_name=inputs.names,
        callbacks=[
            lightgbm.early_stopping(
                config.LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False
            )
        ],
    )
    return model


def fit_best_weight(inputs: LgbmInputs, rows: np.ndarray) -> FittedModel:
    """Try each class weight; keep the one with the best validation PR-AUC."""
    validation = inputs.split.validation
    tried: Dict[str, float] = {}
    best_pr_auc, best_model, best_weight = -1.0, None, 1.0
    for candidate in config.LIGHTGBM_CANDIDATE_WEIGHTS:
        weight = class_weight(candidate, inputs.truth[rows])
        model = fit_one_weight(inputs, rows, weight)
        scores = model.predict_proba(inputs.features[validation])[:, 1]
        pr_auc = float(average_precision_score(inputs.truth[validation], scores))
        tried[str(candidate)] = pr_auc
        if pr_auc > best_pr_auc:
            best_pr_auc, best_model, best_weight = pr_auc, model, weight
    return FittedModel(best_model, best_weight, tried)


def score_both_periods(inputs: LgbmInputs, fitted: FittedModel) -> Dict[str, Any]:
    """Threshold from validation only; the test period is scored once with it."""
    split, truth = inputs.split, inputs.truth
    validation_scores = fitted.model.predict_proba(inputs.features[split.validation])
    test_scores = fitted.model.predict_proba(inputs.features[split.test])
    validation_scores, test_scores = validation_scores[:, 1], test_scores[:, 1]
    threshold = best_f1_threshold(truth[split.validation], validation_scores)
    return {
        "scored": score_period(truth[split.test], test_scores, threshold).to_dict(),
        "validation": score_validation(
            truth[split.validation], validation_scores
        ).to_dict(),
    }


def build_config(
    inputs: LgbmInputs, rows: np.ndarray, fitted: FittedModel, seconds: float
) -> Dict[str, Any]:
    """The settings stored in the record (same keys as GNN records where they fit)."""
    return {
        "status": "completed",
        "dataset": "LI-Small",
        "step_budget_epochs": None,
        "epochs_requested": config.LIGHTGBM_MAX_TREES,
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
        "features": "Project 2 graph features (past only, no labels)",
        "feature_names": inputs.names,
        "class_weight": fitted.weight,
        "validation_pr_auc_by_weight": fitted.validation_pr_auc_by_weight,
        "lightgbm_settings": {
            "max_trees": config.LIGHTGBM_MAX_TREES,
            "learning_rate": config.LIGHTGBM_LEARNING_RATE,
            "num_leaves": config.LIGHTGBM_LEAVES,
            "random_state": config.LIGHTGBM_SEED,
        },
    }


def lightgbm_version() -> str:
    """The installed LightGBM version, for the record."""
    import lightgbm

    return lightgbm.__version__


def run_one(
    inputs: LgbmInputs, fraction: float, seed: int, records_dir: Path, results_dir: Path
) -> Optional[Path]:
    """Train and score one (fraction, seed). Returns the record path, None if done."""
    run_id = make_run_id(config.LIGHTGBM_MODEL, "scratch", fraction, seed)
    if record_path(run_id, records_dir).exists():
        print("%s: record exists, skipping" % run_id)
        return None
    rows = kept_training_rows(inputs.split.train_end, fraction, seed)
    assert_same_labels_as_phase_c(rows, fraction, seed, results_dir)
    start = time.time()
    fitted = fit_best_weight(inputs, rows)
    metrics = score_both_periods(inputs, fitted)
    seconds = time.time() - start
    save_model(fitted, run_id)
    settings = build_config(inputs, rows, fitted, seconds)
    settings["label_mask_seed"] = seed
    record = RunRecord(
        run_id, config.LIGHTGBM_MODEL, "scratch", fraction, seed, settings,
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
    raw_csv: Path, cache_dir: Path, records_dir: Path, results_dir: Path, rebuild: bool
) -> List[Path]:
    """Every (fraction, seed) of Phase E; finished runs are skipped."""
    inputs = load_inputs(raw_csv, cache_dir, rebuild)
    written = []
    for seed in config.PHASE_C_SEEDS:
        for fraction in sorted(config.PHASE_C_FRACTIONS, reverse=True):
            path = run_one(inputs, fraction, seed, records_dir, results_dir)
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
    return parser.parse_args(argv)


def main() -> None:
    """Run Phase E."""
    arguments = parse_arguments()
    written = run_phase_e(
        arguments.raw_csv, arguments.cache_dir, arguments.records_dir,
        config.RESULTS_DIR, arguments.rebuild,
    )  # fmt: skip
    print("Wrote %d records to %s" % (len(written), arguments.records_dir))


if __name__ == "__main__":
    main()
