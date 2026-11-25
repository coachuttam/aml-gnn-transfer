"""The final charts: test F1 and PR-AUC against label share, every model on one plot.

Lines: Multi-GIN + EU (2 seeds, mean ± std), Multi-PNA + EU (1 seed, hollow
diamonds, no whiskers), and the LightGBM simple baseline (2 seeds), each from
scratch and, for the GNNs, fine-tuned. The F1 chart also carries the published
Egressy et al. (2024) lines. Value labels sit beside the points and are spread
apart where they would touch. The x-axis is categorical, so it says so.

Run: python -m src.plot_final      (writes results/final/*.png)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402

from src.collect import (  # noqa: E402
    Stat,
    SummaryRow,
    record_folders,
    stats_by_fraction,
    summary_rows,  # noqa: E402
)
from src.config import RESULTS_DIR  # noqa: E402
from src.plot import (  # noqa: E402
    DPI,
    FIGURE_SIZE,
    FINETUNE_COLOUR,
    LIGHTGBM_COLOUR,
    SCRATCH_COLOUR,
    TEXT_COLOUR,
    scale,
)  # fmt: skip

OUTPUT_FOLDER = RESULTS_DIR / "final"
# Egressy et al. (2024), minority F1 (%) on LI-Small. Their numbers, not ours.
REFERENCE_F1 = {"gin": 26.88, "pna": 33.07}
PNA_FINETUNE = "#0f766e"
PNA_SCRATCH = "#374151"
# Two points count as "about as good" when this close (F1 points). A reading aid
# for the title only; the table has the exact numbers.
HEADLINE_TOLERANCE_POINTS = 2.0
LABEL_GAP_SHARE = 0.042  # minimum gap between value labels, as a share of the y-range
SPREAD_ROUNDS = 100  # enough for five labels to settle
LABEL_OFFSET_POINTS = 11  # label distance from its marker, in points
X_LIMITS = (-0.25, 2.4)
METRICS = {
    "test_minority_f1": ("Minority F1 (%)", "f1", 44.0),
    "test_pr_auc": ("PR-AUC", "pr_auc", 0.40),
}  # attribute -> (axis label, file part, top of the y-axis)


@dataclass(frozen=True)
class Series:
    """One line on the chart."""

    model: str
    arm: str
    label: str
    colour: str
    marker: str
    hollow: bool = False


SERIES = [
    Series("gin", "scratch", "GIN from scratch", SCRATCH_COLOUR, "o"),
    Series("gin", "finetune", "GIN pre-trained, fine-tuned", FINETUNE_COLOUR, "o"),
    Series("pna", "scratch", "PNA from scratch (1 seed)", PNA_SCRATCH, "D", True),
    Series("pna", "finetune", "PNA fine-tuned (1 seed)", PNA_FINETUNE, "D", True),
    Series("lightgbm", "scratch", "LightGBM, simple baseline", LIGHTGBM_COLOUR, "s"),
]  # fmt: skip
SUBTITLE = (
    "Multi-GIN + EU (2 seeds, mean ± std), Multi-PNA + EU (1 seed, hollow diamonds), "
    "LightGBM simple baseline (2 seeds)"
)


def collect_rows() -> List[SummaryRow]:
    """GIN from Phase C, PNA from Phase D, LightGBM from Phase E (all from records)."""
    rows = summary_rows(record_folders())
    wanted = {"gin": "phaseC", "pna": "phaseD", "lightgbm": "phaseE"}
    return [r for r in rows if wanted.get(r.model) == r.run_set]


def series_stats(
    rows: List[SummaryRow], metric: str
) -> Dict[Tuple[str, str], List[Stat]]:
    """Mean and spread per label share for every (model, arm) on the chart."""
    return {
        (s.model, s.arm): stats_by_fraction(rows, s.model, s.arm, metric)
        for s in SERIES
    }


def spread_labels(values: List[float], gap: float) -> List[float]:
    """Move label heights apart so neighbours are at least `gap` apart.

    Neighbours (in height order) that are too close each move half the missing
    distance, away from each other, and this repeats until nobody overlaps.
    Labels that already have room do not move.
    """
    placed = list(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    for _ in range(SPREAD_ROUNDS):
        moved = False
        for low, high in zip(order, order[1:]):
            missing = gap - (placed[high] - placed[low])
            if missing > 1e-9:
                placed[low] -= missing / 2
                placed[high] += missing / 2
                moved = True
        if not moved:
            break
    return placed


def draw_series(axis, series: Series, points: List[Stat], metric: str) -> None:
    """Markers, a line, and whiskers when there is more than one seed."""
    xs = [fractions_axis()[p.fraction] for p in points]
    ys = [p.mean * scale(metric) for p in points]
    spreads = [p.std * scale(metric) for p in points]
    face = "white" if series.hollow else series.colour
    axis.errorbar(
        xs, ys, yerr=spreads if any(spreads) else None, marker=series.marker,
        linewidth=2.2, markersize=9, capsize=5, color=series.colour,
        markerfacecolor=face, markeredgewidth=2, label=series.label,
        linestyle="--" if series.hollow else "-",
    )  # fmt: skip


def fractions_axis() -> Dict[float, int]:
    """Label shares to category positions (the axis is not to scale)."""
    return {0.1: 0, 0.25: 1, 1.0: 2}


def value_text(value: float, metric: str) -> str:
    """'32.7' for F1 (percent points), '0.266' for PR-AUC."""
    return "%.1f" % value if metric == "test_minority_f1" else "%.3f" % value


def draw_value_labels(axis, stats, metric: str, top: float) -> None:
    """Value labels to the right of each marker, spread apart at each label share."""
    gap = top * LABEL_GAP_SHARE
    for fraction, x in fractions_axis().items():
        entries = []
        for series in SERIES:
            point = next((p for p in stats[(series.model, series.arm)]
                          if p.fraction == fraction), None)  # fmt: skip
            if point is not None:
                entries.append((series, point.mean * scale(metric)))
        heights = spread_labels([y for _, y in entries], gap)
        for (series, y), height in zip(entries, heights):
            axis.annotate(
                value_text(y, metric), (x, height), textcoords="offset points",
                xytext=(LABEL_OFFSET_POINTS, 0), va="center", ha="left",
                fontsize=12, color=series.colour, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            )  # fmt: skip


def draw_references(axis) -> None:
    """The two published F1 numbers as dashed lines, named in the legend."""
    names = {"gin": "GIN", "pna": "PNA"}
    colours = {"gin": FINETUNE_COLOUR, "pna": PNA_FINETUNE}
    for model, value in REFERENCE_F1.items():
        axis.axhline(
            value, linestyle=":", linewidth=1.6, color=colours[model], alpha=0.7,
            label="Egressy et al. 2024, %s: %.2f" % (names[model], value),
        )  # fmt: skip


def lookup(stats, model: str, arm: str, fraction: float) -> Optional[Stat]:
    """The point for one series and label share, or None."""
    return next((p for p in stats[(model, arm)] if p.fraction == fraction), None)


def headline(stats, metric: str) -> str:
    """The title. It states the PNA finding only when the numbers back it."""
    tuned = lookup(stats, "pna", "finetune", 0.1)
    scratch = lookup(stats, "pna", "scratch", 1.0)
    if metric == "test_minority_f1" and tuned and scratch:
        close = abs(tuned.mean - scratch.mean) * 100 <= HEADLINE_TOLERANCE_POINTS
        if close:
            return (
                "PNA fine-tuned on 10%% of labels (%.1f) is close to PNA from scratch "
                "on 100%% (%.1f): one seed" % (100 * tuned.mean, 100 * scratch.mean)
            )
    pairs = [
        (lookup(stats, m, "finetune", f), lookup(stats, m, "scratch", f))
        for m in ("gin", "pna")
        for f in fractions_axis()
    ]
    pairs = [(tuned, plain) for tuned, plain in pairs if tuned and plain]
    leads = bool(pairs) and all(tuned.mean > plain.mean for tuned, plain in pairs)
    name = "minority F1" if metric == "test_minority_f1" else "PR-AUC"
    if leads:
        return "Fine-tuned GNNs lead scratch at every label share (%s)" % name
    return "Fine-tuned does not lead at every label share (%s)" % name


def footnote() -> str:
    """What the chart is, and what its kept labels are."""
    return (
        "IBM AML (synthetic), LI-Small test period. GIN, PNA and LightGBM share the "
        "same kept-label fingerprints at each share and seed.\n"
        "Whiskers: ±1 standard deviation (GIN, LightGBM; PNA has one seed). "
        "GIN budget 20, PNA budget 16. Dotted lines are Egressy et al. (2024), "
        "not our numbers."
    )


def draw_chart(rows: List[SummaryRow], metric: str, path: Path) -> None:
    """Save one chart to `path`."""
    label, _, top = METRICS[metric]
    stats = series_stats(rows, metric)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI, facecolor="white")
    for series in SERIES:
        draw_series(axis, series, stats[(series.model, series.arm)], metric)
    if metric == "test_minority_f1":
        draw_references(axis)
    draw_value_labels(axis, stats, metric, top)
    axis.set_xticks(list(fractions_axis().values()))
    axis.set_xticklabels(["%.0f%%" % (100 * f) for f in fractions_axis()], fontsize=13)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(0, top)
    axis.set_xlabel(
        "Share of LI-Small training labels used (not to scale)", fontsize=13
    )
    axis.set_ylabel(label, fontsize=13, color=TEXT_COLOUR)
    axis.grid(axis="y", color="#e5e7eb")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.suptitle(headline(stats, metric), x=0.06, y=0.97, ha="left", fontsize=19,
                    color=TEXT_COLOUR)  # fmt: skip
    figure.text(0.06, 0.885, SUBTITLE, fontsize=13, color=SCRATCH_COLOUR)
    figure.text(0.06, 0.015, footnote(), fontsize=10.5, color=SCRATCH_COLOUR)
    figure.subplots_adjust(top=0.85, bottom=0.17, left=0.06, right=0.74)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def main() -> None:
    """Write the final F1 and PR-AUC charts."""
    rows = collect_rows()
    for metric, (_, part, _) in METRICS.items():
        draw_chart(rows, metric, OUTPUT_FOLDER / ("label_scarcity_%s.png" % part))
    print("Wrote charts to %s" % OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
