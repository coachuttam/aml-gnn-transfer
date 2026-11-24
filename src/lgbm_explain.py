"""SHAP reason codes for the highest-scored test alerts of a LightGBM run.

Each alert gets its five features with the largest SHAP value (in absolute
terms), written as "feature = value (+0.84)": a plus sign pushed the alert score
up, a minus sign pushed it down. A beeswarm summary plot of the same alerts sits
beside the table. Uses the trees `src.lgbm_run` saved; nothing is trained here.

Run: python -m src.lgbm_explain [--fraction 0.1 --seed 0]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.lgbm_inputs import LgbmInputs, load_inputs, with_extra_features
from src.lgbm_run import LGBM_MODELS_FOLDER
from src.records import make_run_id


def top_alert_rows(scores: np.ndarray, count: int) -> np.ndarray:
    """Positions of the `count` highest scores, best first (ties keep file order)."""
    return np.argsort(-scores, kind="stable")[:count]


def shap_matrix(booster: Any, features: pd.DataFrame) -> np.ndarray:
    """SHAP values from TreeExplainer, one row per alert, in log-odds."""
    import shap

    values = shap.TreeExplainer(booster).shap_values(features)
    if isinstance(values, list):  # older SHAP returns [negative class, positive class]
        values = values[-1]
    return np.asarray(values)


def reason_codes(
    values: np.ndarray, features: pd.DataFrame, count: int
) -> List[List[str]]:
    """For each alert, its `count` strongest features as readable strings."""
    codes = []
    for alert_values, (_, alert_features) in zip(values, features.iterrows()):
        strongest = np.argsort(-np.abs(alert_values), kind="stable")[:count]
        codes.append(
            [
                "%s = %.4g (%+.2f)"
                % (features.columns[i], alert_features.iloc[i], alert_values[i])
                for i in strongest
            ]
        )
    return codes


def write_reason_codes(
    path: Path, rows: np.ndarray, scores: np.ndarray, truth: np.ndarray, codes: List
) -> None:
    """The CSV: rank, row in the sorted file, score, whether it was laundering, reasons.

    `is_laundering` is the test label, shown to read the alerts, never used to
    train or to pick a threshold.
    """
    head = ["rank", "row", "score", "is_laundering"]
    head += ["reason_%d" % (i + 1) for i in range(config.SHAP_REASONS_PER_ALERT)]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(head)
        for rank, (row, reasons) in enumerate(zip(rows, codes), start=1):
            writer.writerow(
                [rank, row, "%.5f" % scores[rank - 1], truth[row]] + reasons
            )


def draw_summary(values: np.ndarray, features: pd.DataFrame, path: Path) -> None:
    """Beeswarm plot: which features moved the top alerts' scores, and which way."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(values, features, show=False, max_display=15)
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.title("Why the top %d alerts scored high (LightGBM, SHAP)" % len(features))
    plt.figtext(
        0.01, 0.01, "IBM AML (synthetic), LI-Small test period.", fontsize=10,
        color="#6b7280",
    )  # fmt: skip
    figure.savefig(path, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def explain_run(
    inputs: LgbmInputs, booster: Any, output_dir: Path, fraction: float, seed: int
) -> List[Path]:
    """Reason codes and summary plot for the top test alerts of one run."""
    import lightgbm  # noqa: F401  (the booster comes from it)

    split = inputs.split
    scores = booster.predict(inputs.features[split.test])
    best = top_alert_rows(scores, config.SHAP_TOP_ALERTS)
    rows = best + split.test.start
    alerts = pd.DataFrame(inputs.features[rows], columns=inputs.names)
    values = shap_matrix(booster, alerts)
    codes = reason_codes(values, alerts, config.SHAP_REASONS_PER_ALERT)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "f%s_s%d" % (fraction, seed)
    table, plot = (
        output_dir / ("reason_codes_%s.csv" % tag),
        output_dir / ("shap_summary_%s.png" % tag),
    )
    write_reason_codes(table, rows, scores[best], inputs.truth, codes)
    draw_summary(values, alerts, plot)
    return [table, plot]


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options (defaults: the 10% labels, seed 0 run)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=config.DATA_ROOT / "Small_LI" / "LI-Small_Trans.csv",
    )
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_ROOT / "lightgbm")
    return parser.parse_args(argv)


def main() -> None:
    """Explain one saved run."""
    import lightgbm

    arguments = parse_arguments()
    inputs = load_inputs(arguments.raw_csv, arguments.cache_dir)
    run_id = make_run_id(
        config.LIGHTGBM_MODEL, "scratch", arguments.fraction, arguments.seed
    )
    model_file = config.DATA_ROOT / LGBM_MODELS_FOLDER / (run_id + ".txt")
    booster = lightgbm.Booster(model_file=str(model_file))
    if booster.num_feature() > len(inputs.names):  # the tuned model has extra features
        inputs = with_extra_features(inputs, arguments.raw_csv, arguments.cache_dir)
    output_dir = config.PHASES_RESULTS_DIR / config.PHASE_E_RUN_SET
    for path in explain_run(
        inputs, booster, output_dir, arguments.fraction, arguments.seed
    ):
        print("Wrote", path)


if __name__ == "__main__":
    main()
