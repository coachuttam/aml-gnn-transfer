"""Validation F1 per epoch for every run of a set, scratch against fine-tune.

The curves come from Multi-GNN's own log (its argmax F1 on the validation
period), saved in each new record as `validation_curve`. Older records have no
curve in them; `python -m src.export_curves` writes one file per run into
`<set>/curves/` from the logs, and this module reads those files too.
Only epochs that were really validated have a point (the schedule skips others).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.collect import SYSTEM_NAMES, load_records  # noqa: E402
from src.records import RunRecord  # noqa: E402

COLOURS = {"finetune": "#1f5fa8", "scratch": "#6b7280"}
TEXT_COLOUR = "#1f2937"
CURVES_FOLDER = "curves"


def curve_for(record: RunRecord, records_dir: Path) -> Optional[List[List[float]]]:
    """[epoch, F1] points from the record, else from the exported side file."""
    curve = record.config.get("validation_curve")
    if curve:
        return curve
    side_file = records_dir.parent / CURVES_FOLDER / (record.run_id + ".json")
    if side_file.exists():
        return json.loads(side_file.read_text())
    return None


def curves_by_fraction(
    records: List[RunRecord], records_dir: Path
) -> Dict[Tuple[str, float], List[Tuple[RunRecord, List[List[float]]]]]:
    """Runs with a curve, grouped by (model, label share). Scratch/finetune only."""
    groups: Dict[Tuple[str, float], List[Tuple[RunRecord, List[List[float]]]]] = {}
    for record in records:
        curve = curve_for(record, records_dir)
        if curve and record.mode in COLOURS:
            groups.setdefault((record.model, record.fraction), []).append(
                (record, curve)
            )
    return groups


def draw_model(model: str, groups: Dict, path: Path) -> None:
    """One figure per model: a panel per label share, one line per run."""
    fractions = sorted(f for m, f in groups if m == model)
    figure, axes = plt.subplots(
        1,
        len(fractions),
        figsize=(16, 9),
        dpi=100,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    for axis, fraction in zip(axes[0], fractions):
        for record, curve in groups[(model, fraction)]:
            xs, ys = [p[0] for p in curve], [100 * p[1] for p in curve]
            label = SYSTEM_NAMES[record.mode] + " (seed %d)" % record.seed
            axis.plot(
                xs, ys, marker="o", color=COLOURS[record.mode], alpha=0.8, label=label
            )
        axis.set_title(
            "%.0f%% of labels" % (100 * fraction), loc="left", color=TEXT_COLOUR
        )
        axis.set_xlabel("Epoch", color=TEXT_COLOUR)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#e5e7eb")
    axes[0][0].set_ylabel(
        "Validation minority F1 (%), Multi-GNN's own log", color=TEXT_COLOUR
    )
    axes[0][-1].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Validation F1 per validated epoch, every run (%s)" % model,
        x=0.06,
        ha="left",
        fontsize=18,
        color=TEXT_COLOUR,
    )
    figure.text(
        0.06,
        0.02,
        "Only validated epochs have a point. IBM AML (synthetic), LI-Small validation.",
        fontsize=10,
        color="#6b7280",
    )
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def draw_curves(records_dir: Path, output_dir: Path) -> List[Path]:
    """Write val_curves_<model>.png per model that has curves; returns the files."""
    groups = curves_by_fraction(load_records(records_dir), records_dir)
    if not groups:
        print(
            "No validation curves for %s (old records; see src/export_curves.py)"
            % records_dir
        )
        return []
    written = []
    for model in sorted({m for m, _ in groups}):
        path = output_dir / ("val_curves_%s.png" % model)
        draw_model(model, groups, path)
        written.append(path)
    return written
