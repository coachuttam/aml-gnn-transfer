"""The Phase D table: PNA from scratch against PNA fine-tuned, with time and units.

One row per run (PNA has one seed, so nothing is averaged). The source model,
trained on HI-Small, comes first. Hours and Colab units are computed from each
record's runtime and the GPU rate in `src/config.py`; nothing is typed by hand.

Run: python -m src.final_table     (writes results/final_table.md)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.collect import ARM_ORDER, SYSTEM_NAMES, SummaryRow, load_records, summary_row
from src.config import RESULTS_DIR
from src.phase_estimate import units_for
from src.records import RunRecord

PNA_RUN_SET = "phaseD"
TABLE_NAME = "final_table.md"
HEADINGS = [
    "System", "Labels", "Minority F1", "PR-AUC", "Precision@100", "Recall@10k",
    "Best epoch (of trained)", "GPU hours", "Colab units",
]  # fmt: skip


@dataclass(frozen=True)
class TableLine:
    """One run's numbers plus what it cost."""

    row: SummaryRow
    hours: float
    units: float


def gpu_name_of(record: RunRecord) -> str:
    """'env.yml / NVIDIA L4' -> 'NVIDIA L4'."""
    return str(record.config.get("environment", "")).split("/")[-1].strip()


def table_line(record: RunRecord) -> TableLine:
    """A run's headline numbers, hours and units (0 units for an unknown GPU)."""
    units = units_for(gpu_name_of(record), record.runtime_seconds)
    return TableLine(
        summary_row(record, PNA_RUN_SET), record.runtime_seconds / 3600, units
    )


def pna_lines(records_dir: Path) -> List[TableLine]:
    """PNA runs ordered: source first, then by label share, scratch before fine-tune."""
    lines = [table_line(r) for r in load_records(records_dir) if r.model == "pna"]
    return sorted(
        lines,
        key=lambda line: (
            line.row.arm != "source",
            line.row.fraction,
            ARM_ORDER[line.row.arm],
        ),
    )


def format_line(line: TableLine) -> str:
    """One markdown row. Percent for F1, precision and recall; PR-AUC as is."""
    row = line.row
    labels = "HI-Small" if row.arm == "source" else "%.0f%%" % (100 * row.fraction)
    cells = [
        SYSTEM_NAMES[row.arm], labels, "%.1f%%" % (100 * row.test_minority_f1),
        "%.3f" % row.test_pr_auc, "%.2f" % row.test_precision_at_100,
        "%.1f%%" % (100 * row.test_recall_at_10k),
        "%d of %d" % (row.best_epoch, row.epochs), "%.1f" % line.hours,
        "%.1f" % line.units,
    ]  # fmt: skip
    return "| " + " | ".join(cells) + " |"


def markdown_table(lines: List[TableLine]) -> str:
    """The table plus a total row for hours and units."""
    out = ["| " + " | ".join(HEADINGS) + " |"]
    out.append("| " + " | ".join("---" for _ in HEADINGS) + " |")
    out += [format_line(line) for line in lines]
    total_hours = sum(line.hours for line in lines)
    total_units = sum(line.units for line in lines)
    blanks = [""] * (len(HEADINGS) - 3)
    total = ["**Total**"] + blanks + ["%.1f" % total_hours, "%.1f" % total_units]
    out.append("| " + " | ".join(total) + " |")
    return "\n".join(out)


def main() -> None:
    """Write results/final_table.md from the Phase D records."""
    lines = pna_lines(RESULTS_DIR / "phases" / PNA_RUN_SET / "records")
    path = RESULTS_DIR / TABLE_NAME
    path.write_text(markdown_table(lines) + "\n")
    print("Wrote %d rows to %s" % (len(lines), path))


if __name__ == "__main__":
    main()
