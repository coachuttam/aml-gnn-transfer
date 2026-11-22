"""Turn run records into results/summary.csv and a README table.

Reads every record under `results/pilot/records/` and
`results/phases/<set>/records/` (never `superseded/` folders, never smoke runs)
and writes one row per run. Numbers come only from the records. A
`no_improvement` run keeps its row, with F1 0.0 and the other metrics empty.

Run: python -m src.collect
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import (
    ALERT_BUDGETS,
    PILOT_RECORDS_DIR,
    RESULTS_DIR,
)
from src.records import RunRecord, read_all_records

SYSTEM_NAMES = {
    "scratch": "Trained from scratch",
    "finetune": "Pre-trained on HI-Small, then fine-tuned",
    "zero_shot": "Trained on HI-Small only (zero-shot)",
    "source": "Trained on HI-Small (scored on HI-Small test)",
}
ARM_ORDER = {"source": 0, "zero_shot": 1, "scratch": 2, "finetune": 3}
SUMMARY_NAME = "summary.csv"
PILOT_TABLE_NAME = "pilot_table.md"


@dataclass(frozen=True)
class SummaryRow:
    """One run's headline numbers. F1, precision and recall are fractions of 1."""

    run_id: str
    run_set: str
    model: str
    arm: str  # scratch, finetune, zero_shot or source
    fraction: float
    seed: int
    budget: Optional[int]  # step budget in epochs at 100% labels; None for zero-shot
    steps: Optional[int]  # gradient steps actually taken
    epochs: Optional[int]  # epochs trained (more at small fractions: equal steps)
    best_epoch: Optional[int]  # 1-based; -1 when no checkpoint was saved
    status: str
    val_minority_f1: Optional[float]
    test_minority_f1: float
    test_pr_auc: Optional[float]
    test_precision_at_100: Optional[float]
    test_recall_at_10k: Optional[float]
    test_recall_at_25k: Optional[float]
    test_recall_at_50k: Optional[float]


def record_folders(
    results_dir: Path = RESULTS_DIR,
) -> List[Tuple[str, Path]]:
    """(run set, records folder) for the pilot and for each phase folder present."""
    folders = [("pilot", results_dir / "pilot" / "records")]
    phases = results_dir / "phases"
    if phases.exists():
        folders += [(path.name, path / "records") for path in sorted(phases.iterdir())]
    return [(name, path) for name, path in folders if path.exists()]


def load_records(records_dir: Path = PILOT_RECORDS_DIR) -> List[RunRecord]:
    """Real runs only. `read_all_records` ignores subfolders, so superseded stay out."""
    records = read_all_records(records_dir)
    return [record for record in records if not record.config.get("smoke_test")]


def optional_metric(scored: Dict, name: str) -> Optional[float]:
    """A scored metric, or None when the run was not scored."""
    return scored.get(name)


def summary_row(record: RunRecord, run_set: str) -> SummaryRow:
    """Pull one record's numbers into a flat row."""
    scored = record.metrics["scored"]
    validation = record.metrics.get("validation") or {}
    budgets = scored.get("recall_at_alert_budget") or {}
    recalls = [budgets.get(str(budget)) for budget in ALERT_BUDGETS]
    zero_shot = record.mode == "zero_shot"
    return SummaryRow(
        record.run_id, run_set, record.model, record.mode, record.fraction,
        record.seed,
        None if zero_shot else record.config.get("step_budget_epochs"),
        record.config.get("gradient_steps"),
        None if zero_shot else record.epochs_trained,
        None if zero_shot else record.best_epoch,
        record.config.get("status", "completed"),
        validation.get("minority_f1"), scored["minority_f1"],
        optional_metric(scored, "pr_auc"),
        optional_metric(scored, "precision_at_100"), *recalls,
    )  # fmt: skip


def summary_rows(folders: List[Tuple[str, Path]]) -> List[SummaryRow]:
    """All rows, ordered by set, model, arm, fraction and seed."""
    rows = [
        summary_row(record, run_set)
        for run_set, folder in folders
        for record in load_records(folder)
    ]
    return sorted(
        rows,
        key=lambda r: (r.run_set, r.model, ARM_ORDER[r.arm], r.fraction, r.seed),
    )


def write_summary(rows: List[SummaryRow], path: Path) -> None:
    """Write the CSV. Missing values are left empty, not written as 'None'."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([field.name for field in fields(SummaryRow)])
        for row in rows:
            writer.writerow(["" if value is None else value for value in astuple(row)])


@dataclass(frozen=True)
class Stat:
    """One point on a chart: mean over seeds, spread, and how many seeds."""

    fraction: float
    mean: float
    std: float  # sample standard deviation; 0.0 with one seed (no spread exists)
    count: int


def spread(values: List[float]) -> float:
    """Sample standard deviation, or 0.0 when there is only one value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def stats_by_fraction(
    rows: List[SummaryRow], model: str, arm: str, metric: str
) -> List[Stat]:
    """Mean and spread over seeds at each label share, for one model and arm."""
    groups: Dict[float, List[float]] = {}
    for row in rows:
        value = getattr(row, metric)
        if row.model == model and row.arm == arm and value is not None:
            groups.setdefault(row.fraction, []).append(value)
    return [
        Stat(fraction, statistics.mean(values), spread(values), len(values))
        for fraction, values in sorted(groups.items())
    ]


def cell(values: List[Optional[float]], percent: bool) -> str:
    """'22.8%' for one value, '22.8 ± 1.1' (percent points) for several."""
    known = [value for value in values if value is not None]
    if not known:
        return "n/a"
    scale = 100 if percent else 1
    pattern = "%.1f" if percent else "%.3f"
    text = pattern % (scale * statistics.mean(known))
    if len(known) > 1:
        text += " ± " + pattern % (scale * spread(known))
    return text + ("%" if percent and len(known) == 1 else "")


def table_columns() -> List[Tuple[str, str, bool]]:
    """(heading, row attribute, is a percentage) in DESIGN_SYSTEM.md order."""
    return [
        ("Minority F1", "test_minority_f1", True),
        ("PR-AUC", "test_pr_auc", False),
        ("Precision@100", "test_precision_at_100", True),
        ("Recall@25k", "test_recall_at_25k", True),
    ]


def markdown_table(rows: List[SummaryRow], model: str) -> str:
    """Test results of one model: a row per setup and label share, seeds averaged."""
    columns = table_columns()
    head = ["System", "Labels", "Seeds", "Epochs (best/run)"] + [c[0] for c in columns]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join("---" for _ in head) + " |",
    ]
    groups: Dict[Tuple[str, float], List[SummaryRow]] = {}
    for row in rows:
        if row.model == model:
            groups.setdefault((row.arm, row.fraction), []).append(row)
    for (arm, fraction), group in sorted(
        groups.items(), key=lambda g: (g[0][1], ARM_ORDER[g[0][0]])
    ):
        epochs = ", ".join("%s/%s" % (r.best_epoch, r.epochs) for r in group)
        cells = [
            SYSTEM_NAMES[arm],
            "%.0f%%" % (100 * fraction),
            str(len(group)),
            epochs,
        ]
        cells += [
            cell([getattr(r, attr) for r in group], pct) for _, attr, pct in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Write results/summary.csv, and the pilot table for the README."""
    rows = summary_rows(record_folders())
    if not rows:
        raise SystemExit("No run records found under %s" % RESULTS_DIR)
    write_summary(rows, RESULTS_DIR / SUMMARY_NAME)
    pilot = [row for row in rows if row.run_set == "pilot"]
    if pilot:
        table = markdown_table(pilot, "gin") + "\n"
        (RESULTS_DIR / "pilot" / PILOT_TABLE_NAME).write_text(table)
    print("Wrote %d rows to %s" % (len(rows), RESULTS_DIR / SUMMARY_NAME))


if __name__ == "__main__":
    main()
