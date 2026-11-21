"""Turn pilot run records into one results table (markdown and CSV).

Reads every `results/pilot/*.json` record (never `superseded/`, never smoke
runs) and prints one row per system and label share. Numbers come only from
the records. A `no_improvement` run is shown as 0.0 F1 with a note, never dropped.

Run: python -m src.collect
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import PILOT_RECORDS_DIR, RESULTS_DIR
from src.records import RunRecord, read_all_records

SYSTEM_NAMES = {
    "scratch": "Trained from scratch",
    "finetune": "Pre-trained on HI-Small, then fine-tuned",
    "source": "Trained on HI-Small (source model, HI-Small test)",
}
COMPARISON_MODES = ("scratch", "finetune")
TABLE_COLUMNS = (
    "System", "Labels", "Seeds", "Minority F1", "Precision", "Recall",
    "PR-AUC", "Precision@100", "Alerts", "Note",
)  # fmt: skip
EMPTY = "n/a"  # a metric that was not scored (no_improvement runs)


@dataclass(frozen=True)
class ResultRow:
    """One run's headline numbers, in the units the tables show."""

    mode: str
    fraction: float
    seed: int
    status: str
    minority_f1: float
    precision: float
    recall: float
    pr_auc: Optional[float]
    precision_at_100: Optional[float]
    alerts: Optional[int]


def load_records(records_dir: Path = PILOT_RECORDS_DIR) -> List[RunRecord]:
    """Real runs only: smoke tests are left out (superseded ones are in a subfolder)."""
    records = read_all_records(records_dir)
    return [record for record in records if not record.config.get("smoke_test")]


def to_row(record: RunRecord) -> ResultRow:
    """Pull the scored numbers out of one record."""
    scored = record.metrics["scored"]
    return ResultRow(
        record.mode, record.fraction, record.seed,
        record.config.get("status", "completed"),
        scored["minority_f1"], scored["precision"], scored["recall"],
        scored["pr_auc"], scored["precision_at_100"], scored["alerts"],
    )  # fmt: skip


def result_rows(records: List[RunRecord], model: str = "gin") -> List[ResultRow]:
    """Rows for one model: source first, then each fraction ascending, scratch first."""
    rows = [to_row(record) for record in records if record.model == model]
    order = {"source": 0, "scratch": 1, "finetune": 2}
    rows = [row for row in rows if row.mode in order]
    return sorted(rows, key=lambda row: (row.mode != "source", row.fraction,
                                         order[row.mode], row.seed))  # fmt: skip


def percent(value: float) -> str:
    """0.2284 -> '22.8%'."""
    return "%.1f%%" % (100 * value)


def optional(value: Optional[float], text: str) -> str:
    """Format a metric that may be missing."""
    return EMPTY if value is None else text % value


def table_line(row: ResultRow) -> Tuple[str, ...]:
    """The cells of one table row."""
    note = "no checkpoint: training never beat F1 0" if (
        row.status == "no_improvement") else ""  # fmt: skip
    return (
        SYSTEM_NAMES[row.mode], "%.0f%%" % (100 * row.fraction), str(row.seed),
        percent(row.minority_f1), percent(row.precision), percent(row.recall),
        optional(row.pr_auc, "%.3f"), optional(row.precision_at_100, "%.2f"),
        optional(row.alerts, "%d"), note,
    )  # fmt: skip


def markdown_table(rows: List[ResultRow]) -> str:
    """GitHub markdown table in the DESIGN_SYSTEM column order."""
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |"]
    lines.append("| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(table_line(row)) + " |")
    return "\n".join(lines)


def gaps_by_fraction(rows: List[ResultRow]) -> Dict[float, float]:
    """Finetune minus scratch minority F1 (fraction of 1) per label share."""
    by_key = {(row.mode, row.fraction, row.seed): row for row in rows}
    gaps: Dict[float, float] = {}
    for (mode, fraction, seed), row in by_key.items():
        other = by_key.get(("scratch", fraction, seed))
        if mode == "finetune" and other is not None:
            gaps[fraction] = row.minority_f1 - other.minority_f1
    return dict(sorted(gaps.items()))


def write_csv(rows: List[ResultRow], path: Path) -> None:
    """The same table as CSV, for the chart and for readers."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE_COLUMNS)
        for row in rows:
            writer.writerow(table_line(row))


def main() -> None:
    """Write results/pilot_table.md and .csv from the pilot records."""
    rows = result_rows(load_records())
    if not rows:
        raise SystemExit("No pilot records in %s" % PILOT_RECORDS_DIR)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "pilot_table.md").write_text(markdown_table(rows) + "\n")
    write_csv(rows, RESULTS_DIR / "pilot_table.csv")
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
