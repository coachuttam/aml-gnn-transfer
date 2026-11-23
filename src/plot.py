"""Draw the label-scarcity charts (DESIGN_SYSTEM.md) for one run set.

Two charts, minority F1 and PR-AUC against the share of LI-Small training
labels, each with one panel per model (GIN and PNA when both were run). Two
lines per panel: trained from scratch (grey) and pre-trained then fine-tuned
(blue). With more than one seed a point is the mean and the whisker is one
standard deviation; with one seed there is no whisker, and the chart says so.
The validation curves of each run are drawn by src/plot_curves.py.

Run: python -m src.plot [--run-set pilot]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402

from src.collect import (  # noqa: E402
    SYSTEM_NAMES,
    Stat,
    SummaryRow,
    record_folders,
    stats_by_fraction,
    summary_rows,
)
from src.config import RESULTS_DIR  # noqa: E402
from src.plot_curves import draw_curves  # noqa: E402

FINETUNE_COLOUR = "#1f5fa8"
SCRATCH_COLOUR = "#6b7280"
TEXT_COLOUR = "#1f2937"
FIGURE_SIZE = (16, 9)  # DESIGN_SYSTEM.md: 1600x900 at 100 dpi
DPI = 100
ARMS = ("scratch", "finetune")
MODEL_NAMES = {"gin": "Multi-GIN + EU", "pna": "Multi-PNA + EU"}
# Published minority F1 (%) on LI-Small, drawn as a dashed reference line.
# Egressy et al. 2024 report 26.88 for Multi-GIN + EU; IBM (Altman et al. 2023,
# Table 3) report 27.38 for PNA fine-tuned. Neither is our number.
REFERENCE_F1: Dict[str, Tuple[float, str]] = {
    "gin": (26.88, "Egressy et al. 2024, Multi-GIN + EU, LI-Small"),
    "pna": (27.38, "IBM, Table 3 (PNA, fine-tuned)"),
}
METRICS = {  # attribute on SummaryRow -> (axis label, file name part)
    "test_minority_f1": ("Minority F1 (%)", "f1"),
    "test_pr_auc": ("PR-AUC", "pr_auc"),
}
StatsByArm = Dict[str, List[Stat]]


def scale(metric: str) -> float:
    """F1 is drawn in percent, PR-AUC as it is."""
    return 100.0 if metric == "test_minority_f1" else 1.0


def gap_is_clear(finetune: Stat, scratch: Stat) -> bool:
    """A gap counts only with several seeds and ranges (mean ± 1 std) apart."""
    if min(finetune.count, scratch.count) < 2:
        return False
    return finetune.mean - finetune.std > scratch.mean + scratch.std


def matched_pairs(by_model: Dict[str, StatsByArm]) -> List[Tuple[Stat, Stat]]:
    """(finetune, scratch) points at the same model and label share."""
    pairs = []
    for stats in by_model.values():
        scratch = {point.fraction: point for point in stats["scratch"]}
        for point in stats["finetune"]:
            if point.fraction in scratch:
                pairs.append((point, scratch[point.fraction]))
    return pairs


def largest_gap_at_fewest_labels(by_model: Dict[str, StatsByArm]) -> bool:
    """True if, in every model, the fine-tune lead is biggest at the smallest share."""
    for name, stats in by_model.items():
        model_pairs = matched_pairs({name: stats})
        ordered = sorted(model_pairs, key=lambda pair: pair[0].fraction)
        gaps = [a.mean - b.mean for a, b in ordered]
        if len(gaps) < 2 or gaps[0] <= max(gaps[1:]):
            return False
    return True


def chart_title(by_model: Dict[str, StatsByArm], metric_name: str) -> str:
    """The finding in words. Never calls a gap inside the seed spread a win."""
    pairs = matched_pairs(by_model)
    if not pairs:
        return "No scratch and fine-tune pair to compare yet"
    seeds = min(min(a.count, b.count) for a, b in pairs)
    if any(a.mean <= b.mean for a, b in pairs):
        return "Pre-training does not help at every label share (%s)" % metric_name
    if all(gap_is_clear(a, b) for a, b in pairs) and largest_gap_at_fewest_labels(
        by_model
    ):
        return "Pre-training helps most when labels are scarce"
    if all(gap_is_clear(a, b) for a, b in pairs):
        return "Pre-trained model beats scratch at every label share (%s, %d seeds)" % (
            metric_name, seeds)  # fmt: skip
    if seeds < 2:
        return "Pre-trained model leads at every share (%s, one seed, no spread)" % (
            metric_name
        )
    return (
        "Pre-trained leads, but not beyond the seed spread everywhere (%s)"
        % metric_name
    )


def draw_line(
    axis, stats: List[Stat], positions: Dict[float, int], arm: str, metric: str
):
    """One setup's line: markers, whiskers when there are several seeds, labels."""
    colour = FINETUNE_COLOUR if arm == "finetune" else SCRATCH_COLOUR
    xs = [positions[point.fraction] for point in stats]
    ys = [point.mean * scale(metric) for point in stats]
    spreads = [point.std * scale(metric) for point in stats]
    axis.errorbar(xs, ys, yerr=spreads if any(spreads) else None, marker="o",
                  linewidth=2.5, markersize=8, capsize=5, color=colour,
                  label=SYSTEM_NAMES[arm])  # fmt: skip
    for x, y in zip(xs, ys):
        text = "%.1f" % y if metric == "test_minority_f1" else "%.3f" % y
        axis.annotate(text, (x, y), textcoords="offset points", xytext=(0, 12),
                      ha="center", color=colour, fontsize=12)  # fmt: skip


def draw_reference(axis, model: str, last_x: int) -> None:
    """The published F1 as a dashed line, labelled with its source."""
    value, label = REFERENCE_F1[model]
    axis.axhline(value, linestyle="--", color=SCRATCH_COLOUR, linewidth=1)
    axis.annotate("%s (%.2f)" % (label, value), (last_x, value),
                  textcoords="offset points", xytext=(0, 6), ha="right",
                  color=SCRATCH_COLOUR, fontsize=10)  # fmt: skip


def style_panel(axis, model: str, fractions: List[float]) -> None:
    """Plain axes: labels, zero baseline, light grid, no box."""
    axis.set_xticks(range(len(fractions)))
    axis.set_xticklabels(["%.0f%%" % (100 * f) for f in fractions], fontsize=13)
    axis.set_xlim(-0.3, len(fractions) - 0.7)
    axis.set_title(
        MODEL_NAMES.get(model, model), loc="left", fontsize=15, color=TEXT_COLOUR
    )
    axis.set_xlabel(
        "Share of LI-Small training labels used", fontsize=13, color=TEXT_COLOUR
    )
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e5e7eb")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def draw_panel(axis, rows: List[SummaryRow], model: str, metric: str) -> StatsByArm:
    """One model's panel. Returns its numbers for the title."""
    stats = {arm: stats_by_fraction(rows, model, arm, metric) for arm in ARMS}
    fractions = sorted({point.fraction for arm in ARMS for point in stats[arm]})
    positions = {fraction: index for index, fraction in enumerate(fractions)}
    for arm in ARMS:
        if stats[arm]:
            draw_line(axis, stats[arm], positions, arm, metric)
    if metric == "test_minority_f1" and model in REFERENCE_F1:
        draw_reference(axis, model, len(fractions) - 1)
    style_panel(axis, model, fractions)
    return stats


def footnote(rows: List[SummaryRow], run_set: str) -> str:
    """What the chart is, in plain words, including how many seeds back each point."""
    counts: Dict[Tuple[str, str, float], int] = {}
    for row in rows:
        key = (row.model, row.arm, row.fraction)
        counts[key] = counts.get(key, 0) + 1
    most, fewest = max(counts.values()), min(counts.values())
    seeds = "%d seed%s" % (most, "" if most == 1 else "s")
    if fewest != most:
        seeds = "%d to %d seeds" % (fewest, most)
    whiskers = (
        "Whiskers: ±1 standard deviation." if most > 1 else "No whiskers: one seed."
    )
    return (
        "Run set: %s. %s per point. %s Equal training steps at every share.\n"
        "IBM AML (synthetic), LI-Small test period." % (run_set, seeds, whiskers)
    )


def draw_chart(rows: List[SummaryRow], metric: str, path: Path, run_set: str) -> None:
    """Save one chart (all models side by side) to `path`."""
    models = [name for name in MODEL_NAMES if any(r.model == name for r in rows)]
    figure, axes = plt.subplots(
        1, len(models), figsize=FIGURE_SIZE, dpi=DPI, sharey=True,
        facecolor="white", squeeze=False,
    )  # fmt: skip
    by_model = {
        m: draw_panel(axes[0][i], rows, m, metric) for i, m in enumerate(models)
    }
    axes[0][0].set_ylabel(METRICS[metric][0], fontsize=13, color=TEXT_COLOUR)
    axes[0][0].legend(frameon=False, fontsize=12, loc="upper left")
    name = "minority F1" if metric == "test_minority_f1" else "PR-AUC"
    figure.suptitle(
        chart_title(by_model, name), x=0.06, ha="left", fontsize=20, color=TEXT_COLOUR
    )
    figure.text(0.06, 0.02, footnote(rows, run_set), fontsize=11, color=SCRATCH_COLOUR)
    figure.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.97, wspace=0.08)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def output_folder(run_set: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Charts sit beside the set's records: results/pilot or results/phases/<set>."""
    return results_dir / ("pilot" if run_set == "pilot" else "phases/%s" % run_set)


def main(argv: Optional[List[str]] = None) -> None:
    """Write the F1 chart, the PR-AUC chart and the validation curves for a run set."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-set", default="pilot")
    arguments = parser.parse_args(argv)
    folders = dict(record_folders())
    rows = summary_rows(list(folders.items()))
    rows = [r for r in rows if r.run_set == arguments.run_set and r.arm in ARMS]
    if not rows:
        raise SystemExit("No scratch/finetune records for %s" % arguments.run_set)
    folder = output_folder(arguments.run_set)
    for metric, (_, part) in METRICS.items():
        path = folder / ("label_scarcity_%s.png" % part)
        draw_chart(rows, metric, path, arguments.run_set)
    draw_curves(folders[arguments.run_set], folder)
    print("Wrote charts to %s" % folder)


if __name__ == "__main__":
    main()
