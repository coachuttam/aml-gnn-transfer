"""Draw the pilot label-scarcity chart (DESIGN_SYSTEM.md): two lines, one per setup.

x: share of LI-Small training labels (evenly spaced). y: minority F1 (%) from 0,
or PR-AUC. With one seed there is no whisker, and the footnote says so.

Run: python -m src.plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402

from src.collect import (  # noqa: E402
    SYSTEM_NAMES,
    ResultRow,
    gaps_by_fraction,
    load_records,
    result_rows,
)
from src.config import RESULTS_DIR  # noqa: E402

FINETUNE_COLOUR = "#1f5fa8"
SCRATCH_COLOUR = "#6b7280"
TEXT_COLOUR = "#1f2937"
IBM_LINE_COLOUR = "#6b7280"
IBM_FINE_TUNED_F1 = 27.38  # IBM paper Table 3, PNA + 5 epochs fine-tuning, LI-Small
FIGURE_SIZE = (16, 9)  # DESIGN_SYSTEM.md: 1600x900 at 100 dpi
DPI = 100
FOOTNOTE = (
    "Pilot: Multi-GIN + EU, one seed (0), equal training steps (8 epochs' worth "
    "at 100% of labels), one shared HI-Small source model.\n"
    "No spread is shown because there is one seed. "
    "IBM AML (synthetic), LI-Small test period."
)


def series(rows: List[ResultRow], mode: str, metric: str) -> Dict[float, float]:
    """Value of `metric` per label share for one setup (percent for F1)."""
    values: Dict[float, float] = {}
    for row in rows:
        value = getattr(row, metric)
        if row.mode == mode and value is not None:
            values[row.fraction] = 100 * value if metric == "minority_f1" else value
    return dict(sorted(values.items()))


def chart_title(rows: List[ResultRow], metric_name: str) -> str:
    """The finding, in words; says so if the pattern is not clean."""
    gaps = gaps_by_fraction(rows)
    if gaps and all(gap > 0 for gap in gaps.values()):
        return (
            "Pre-trained model scores higher at every label share (%s, one seed)"
            % metric_name
        )
    return (
        "Pre-training does not help at every label share (%s, one seed)" % metric_name
    )


def draw_line(
    axis, points: Dict[float, float], position: Dict[float, int], mode: str, colour: str
):
    """One setup's line with value labels."""
    xs = [position[fraction] for fraction in points]
    axis.plot(xs, list(points.values()), marker="o", linewidth=2.5, markersize=8,
              color=colour, label=SYSTEM_NAMES[mode])  # fmt: skip
    for x, value in zip(xs, points.values()):
        text = "%.1f" % value if value > 1 else "%.3f" % value
        axis.annotate(
            text, (x, value), textcoords="offset points", xytext=(0, 10),
            ha="center", color=colour, fontsize=12,
        )  # fmt: skip


def label_gap(
    axis, finetune: Dict[float, float], scratch: Dict[float, float], x: int, unit: str
):
    """Write the gap at the smallest label share directly on the chart."""
    fraction = min(finetune)
    gap = finetune[fraction] - scratch[fraction]
    axis.annotate("%+.1f %s at %.0f%% of labels" % (gap, unit, 100 * fraction),
                  (x, (finetune[fraction] + scratch[fraction]) / 2),
                  textcoords="offset points", xytext=(25, 0), color=TEXT_COLOUR,
                  fontsize=12, va="center")  # fmt: skip


def draw_chart(rows: List[ResultRow], metric: str, path: Path, show_ibm: bool) -> None:
    """Save one chart (minority F1 or PR-AUC) to `path`."""
    finetune, scratch = (
        series(rows, "finetune", metric),
        series(rows, "scratch", metric),
    )
    fractions = sorted(set(finetune) | set(scratch))
    position = {fraction: index for index, fraction in enumerate(fractions)}
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI, facecolor="white")
    draw_line(axis, scratch, position, "scratch", SCRATCH_COLOUR)
    draw_line(axis, finetune, position, "finetune", FINETUNE_COLOUR)
    if finetune and scratch and min(finetune) in scratch:
        unit = "F1 points" if metric == "minority_f1" else "PR-AUC"
        label_gap(axis, finetune, scratch, position[min(finetune)], unit)
    if show_ibm:
        axis.axhline(
            IBM_FINE_TUNED_F1, linestyle="--", color=IBM_LINE_COLOUR, linewidth=1
        )
        axis.annotate(
            "IBM, Table 3 (PNA, fine-tuned)",
            (len(fractions) - 1, IBM_FINE_TUNED_F1),
            textcoords="offset points", xytext=(0, 6), ha="right",
            color=IBM_LINE_COLOUR, fontsize=11,
        )  # fmt: skip
    style_axes(axis, fractions, metric)
    name = "minority F1" if metric == "minority_f1" else "PR-AUC"
    figure.suptitle(
        chart_title(rows, name), x=0.06, ha="left", fontsize=20, color=TEXT_COLOUR
    )
    figure.text(0.06, 0.02, FOOTNOTE, fontsize=11, color=SCRATCH_COLOUR, va="bottom")
    figure.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.95)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def style_axes(axis, fractions: List[float], metric: str) -> None:
    """Plain axes: labels, zero baseline, light grid, no box."""
    axis.set_xticks(range(len(fractions)))
    axis.set_xticklabels(
        ["%.0f%%" % (100 * fraction) for fraction in fractions], fontsize=13
    )
    axis.set_xlim(-0.3, len(fractions) - 0.7)
    axis.set_xlabel(
        "Share of LI-Small training labels used", fontsize=13, color=TEXT_COLOUR
    )
    axis.set_ylabel("Minority F1 (%)" if metric == "minority_f1" else "PR-AUC",
                    fontsize=13, color=TEXT_COLOUR)  # fmt: skip
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e5e7eb")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=13, loc="upper left")


def main(output_dir: Optional[Path] = None) -> None:
    """Write the F1 chart and the PR-AUC chart."""
    folder = output_dir or RESULTS_DIR
    rows = result_rows(load_records())
    folder.mkdir(exist_ok=True)
    draw_chart(rows, "minority_f1", folder / "pilot_label_scarcity.png", True)
    draw_chart(rows, "pr_auc", folder / "pilot_label_scarcity_pr_auc.png", False)
    print("Wrote charts to %s" % folder)


if __name__ == "__main__":
    main()
