"""Fit and score one LightGBM model: settings, class-weight choice, scoring.

The default settings are Project 2's. `LightGbmSettings` lets the tuning script
(`src/lgbm_tune.py`) try others, always judged on validation only.

Example:
    fitted = fit_best_weight(inputs, rows)            # Project 2's settings
    scores = score_both_periods(inputs, fitted)       # threshold from validation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from src import config
from src.lgbm_inputs import LgbmInputs
from src.metrics import best_f1_threshold, score_period, score_validation


@dataclass(frozen=True)
class LightGbmSettings:
    """Everything about the model that is not the data. Defaults are Project 2's."""

    max_trees: int = config.LIGHTGBM_MAX_TREES
    patience: int = config.LIGHTGBM_EARLY_STOPPING_ROUNDS
    learning_rate: float = config.LIGHTGBM_LEARNING_RATE
    leaves: int = config.LIGHTGBM_LEAVES
    min_child_samples: int = 20  # LightGBM's own default
    subsample: float = 1.0  # share of rows per tree; below 1 makes the seed matter
    column_share: float = 1.0
    weights: Tuple[object, ...] = config.LIGHTGBM_CANDIDATE_WEIGHTS
    seed: int = config.LIGHTGBM_SEED

    def to_dict(self) -> Dict[str, Any]:
        """Plain dictionary for the run record."""
        return asdict(self)


DEFAULT_SETTINGS = LightGbmSettings()
# Chosen on validation PR-AUC only (src/lgbm_tune.py, results/phases/phaseE/
# tuning_val.csv): more trees and patience, and class weights below 1 as well. Used
# together with the extra features, which gave most of the gain.
TUNED_SETTINGS = LightGbmSettings(
    max_trees=2000, patience=100, weights=(0.5, 1.0, 2.0, 5.0)
)


@dataclass(frozen=True)
class FittedModel:
    """The weight that won on validation PR-AUC and everything needed to score it."""

    model: Any
    weight: float
    validation_pr_auc_by_weight: Dict[str, float]


def class_weight(weight: object, truth: np.ndarray) -> float:
    """ "balanced" becomes negatives / positives among the training rows."""
    if weight != "balanced":
        return float(weight)  # type: ignore[arg-type]
    positives = max(int(truth.sum()), 1)
    return (len(truth) - positives) / positives


def fit_one_weight(
    inputs: LgbmInputs,
    rows: np.ndarray,
    weight: float,
    settings: LightGbmSettings = DEFAULT_SETTINGS,
    run_seed: int = 0,
) -> Any:
    """LightGBM on the kept rows; early stopping watches validation PR-AUC.

    `run_seed` is added to the model seed, so seeds differ when rows are sampled.
    """
    import lightgbm

    model = lightgbm.LGBMClassifier(
        n_estimators=settings.max_trees,
        learning_rate=settings.learning_rate,
        num_leaves=settings.leaves,
        min_child_samples=settings.min_child_samples,
        subsample=settings.subsample,
        subsample_freq=1 if settings.subsample < 1 else 0,
        colsample_bytree=settings.column_share,
        scale_pos_weight=weight,
        random_state=settings.seed + run_seed,
        verbose=-1,
    )
    validation = inputs.split.validation
    model.fit(
        inputs.features[rows],
        inputs.truth[rows],
        eval_set=[(inputs.features[validation], inputs.truth[validation])],
        eval_metric="average_precision",
        feature_name=inputs.names,
        callbacks=[lightgbm.early_stopping(settings.patience, verbose=False)],
    )
    return model


def validation_pr_auc(inputs: LgbmInputs, model: Any) -> float:
    """PR-AUC on the validation period (the only period model choices may use)."""
    validation = inputs.split.validation
    scores = model.predict_proba(inputs.features[validation])[:, 1]
    return float(average_precision_score(inputs.truth[validation], scores))


def fit_best_weight(
    inputs: LgbmInputs,
    rows: np.ndarray,
    settings: LightGbmSettings = DEFAULT_SETTINGS,
    run_seed: int = 0,
) -> FittedModel:
    """Try each class weight; keep the one with the best validation PR-AUC."""
    tried: Dict[str, float] = {}
    best_pr_auc, best_model, best_weight = -1.0, None, 1.0
    for candidate in settings.weights:
        weight = class_weight(candidate, inputs.truth[rows])
        model = fit_one_weight(inputs, rows, weight, settings, run_seed)
        pr_auc = validation_pr_auc(inputs, model)
        tried[str(candidate)] = pr_auc
        if pr_auc > best_pr_auc:
            best_pr_auc, best_model, best_weight = pr_auc, model, weight
    return FittedModel(best_model, best_weight, tried)


def score_both_periods(
    inputs: LgbmInputs, fitted: FittedModel, include_test: bool = True
) -> Dict[str, Any]:
    """Threshold from validation only; the test period is scored once with it.

    With `include_test=False` the test rows are not even read (tuning runs).
    """
    split, truth = inputs.split, inputs.truth
    validation_scores = fitted.model.predict_proba(inputs.features[split.validation])
    validation_scores = validation_scores[:, 1]
    threshold = best_f1_threshold(truth[split.validation], validation_scores)
    metrics: Dict[str, Any] = {
        "validation": score_validation(
            truth[split.validation], validation_scores
        ).to_dict()
    }
    if include_test:
        test_scores = fitted.model.predict_proba(inputs.features[split.test])[:, 1]
        scored = score_period(truth[split.test], test_scores, threshold)
        metrics["scored"] = scored.to_dict()
    return metrics


def best_iteration(model: Any) -> Optional[int]:
    """The tree count early stopping kept (1-based), if it stopped early."""
    return int(model.best_iteration_) if model.best_iteration_ else None
