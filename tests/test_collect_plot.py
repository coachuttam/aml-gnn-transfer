"""Tests for src/collect.py, src/plot.py and src/plot_curves.py (made-up records)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.collect import (
    cell,
    load_records,
    markdown_table,
    record_folders,
    stats_by_fraction,
    summary_row,
    summary_rows,
    write_summary,
)
from src.plot import chart_title, draw_chart, output_folder
from src.plot_curves import curve_for, draw_curves
from src.records import RunRecord, write_record

REQUIRED_COLUMNS = [
    "run_id", "model", "arm", "fraction", "seed", "budget", "steps", "best_epoch",
    "status", "val_minority_f1", "test_minority_f1", "test_pr_auc",
    "test_precision_at_100", "test_recall_at_10k", "test_recall_at_25k",
    "test_recall_at_50k",
]  # fmt: skip


def make_record(
    mode: str, fraction: float, f1: float, seed: int = 0, **config: Any
) -> RunRecord:
    scored: Dict[str, Any] = {
        "minority_f1": f1, "precision": 0.5, "recall": 0.2, "pr_auc": 0.1,
        "precision_at_100": 0.3, "alerts": 40,
        "recall_at_alert_budget": {"10000": 0.5, "25000": 0.6, "50000": 0.7},
    }  # fmt: skip
    settings = {"step_budget_epochs": 8, "gradient_steps": 4000}
    settings.update(config)
    return RunRecord(
        run_id="gin_%s_f%s_s%d" % (mode, fraction, seed), model="gin", mode=mode,
        fraction=fraction, seed=seed, config=settings, ours_commit="a",
        multignn_commit="b", metrics={"scored": scored}, epochs_trained=8,
        best_epoch=3, runtime_seconds=1.0,
    )  # fmt: skip


def pilot_records() -> List[RunRecord]:
    return [
        make_record("scratch", 0.1, 0.157), make_record("finetune", 0.1, 0.228),
        make_record("scratch", 1.0, 0.205), make_record("finetune", 1.0, 0.331),
        make_record("source", 1.0, 0.605),
    ]  # fmt: skip


def rows_of(records: List[RunRecord]):
    return [summary_row(record, "pilot") for record in records]


def three_seed_records(gap: float) -> List[RunRecord]:
    records = []
    for seed, wobble in enumerate((0.00, 0.01, -0.01)):
        records.append(make_record("scratch", 0.1, 0.20 + wobble, seed))
        records.append(make_record("finetune", 0.1, 0.20 + gap + wobble, seed))
    return records


# --- summary.csv ---


def test_superseded_runs_and_smoke_runs_are_left_out(tmp_path: Path) -> None:
    write_record(make_record("scratch", 0.1, 0.1), tmp_path)
    write_record(make_record("scratch", 1.0, 0.9), tmp_path / "superseded")
    write_record(make_record("finetune", 0.1, 0.8, smoke_test=True), tmp_path)
    assert [r.fraction for r in load_records(tmp_path)] == [0.1]


def test_the_summary_file_has_every_column_the_write_up_needs(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    write_summary(rows_of(pilot_records()), path)
    with open(path) as handle:
        header = next(csv.reader(handle))
    for name in REQUIRED_COLUMNS:
        assert name in header


def test_a_summary_row_reads_recall_at_each_alert_budget_from_the_record() -> None:
    row = summary_row(make_record("scratch", 0.1, 0.2), "pilot")
    assert (row.test_recall_at_10k, row.test_recall_at_25k, row.test_recall_at_50k) == (
        0.5,
        0.6,
        0.7,
    )
    assert (row.budget, row.steps, row.best_epoch) == (8, 4000, 3)


def test_validation_f1_is_read_when_the_record_has_it_and_blank_when_not(
    tmp_path: Path,
) -> None:
    with_validation = make_record("scratch", 0.1, 0.2)
    with_validation.metrics["validation"] = {"minority_f1": 0.31}
    assert summary_row(with_validation, "pilot").val_minority_f1 == 0.31
    path = tmp_path / "summary.csv"
    write_summary(rows_of([make_record("scratch", 0.1, 0.2)]), path)
    assert list(csv.DictReader(open(path)))[0]["val_minority_f1"] == ""


def test_a_run_that_never_learned_keeps_its_row_with_empty_metrics(
    tmp_path: Path,
) -> None:
    record = make_record("scratch", 0.25, 0.0, status="no_improvement")
    record.metrics["scored"].update(
        pr_auc=None, precision_at_100=None, recall_at_alert_budget=None
    )
    path = tmp_path / "summary.csv"
    write_summary(rows_of([record]), path)
    row = list(csv.DictReader(open(path)))[0]
    assert row["status"] == "no_improvement" and row["test_minority_f1"] == "0.0"
    assert row["test_pr_auc"] == "" and row["test_recall_at_25k"] == ""


def test_a_zero_shot_row_has_no_budget_and_no_best_epoch() -> None:
    row = summary_row(
        make_record("zero_shot", 1.0, 0.0, step_budget_epochs=0), "phaseB"
    )
    assert row.budget is None and row.best_epoch is None and row.arm == "zero_shot"


def test_records_are_found_in_the_pilot_folder_and_in_each_phase_folder(
    tmp_path: Path,
) -> None:
    write_record(make_record("scratch", 0.1, 0.1), tmp_path / "pilot" / "records")
    write_record(
        make_record("scratch", 0.1, 0.2), tmp_path / "phases" / "phaseC" / "records"
    )
    rows = summary_rows(record_folders(tmp_path))
    assert sorted(row.run_set for row in rows) == ["phaseC", "pilot"]


# --- tables ---


def test_the_table_shows_percentages_and_plain_system_names() -> None:
    table = markdown_table(rows_of(pilot_records()), "gin")
    assert "22.8%" in table and "Trained from scratch" in table
    assert "Pre-trained on HI-Small, then fine-tuned" in table


def test_several_seeds_are_shown_as_mean_plus_or_minus_spread() -> None:
    assert cell([0.20, 0.21, 0.19], percent=True) == "20.0 ± 1.0"
    assert cell([0.25], percent=True) == "25.0%"
    assert cell([None], percent=True) == "n/a"


def test_stats_average_over_seeds_and_report_how_many_seeds() -> None:
    points = stats_by_fraction(
        rows_of(three_seed_records(0.1)), "gin", "scratch", "test_minority_f1"
    )
    assert (
        points[0].count == 3 and round(points[0].mean, 3) == 0.2 and points[0].std > 0
    )


def test_one_seed_has_zero_spread_not_a_made_up_one() -> None:
    points = stats_by_fraction(
        rows_of(pilot_records()), "gin", "scratch", "test_minority_f1"
    )
    assert all(point.std == 0.0 and point.count == 1 for point in points)


# --- charts ---


def title_for(records: List[RunRecord]) -> str:
    rows = rows_of(records)
    by_model = {
        "gin": {
            arm: stats_by_fraction(rows, "gin", arm, "test_minority_f1")
            for arm in ("scratch", "finetune")
        }
    }
    return chart_title(by_model, "minority F1")


def test_with_one_seed_the_title_says_there_is_no_spread() -> None:
    title = title_for(pilot_records())
    assert "leads" in title and "one seed" in title and "beats" not in title


def test_a_gap_inside_the_seed_spread_is_not_called_a_win() -> None:
    assert "beats" not in title_for(three_seed_records(0.005))


def test_a_gap_well_outside_the_seed_spread_is_called_a_win() -> None:
    assert "beats scratch" in title_for(three_seed_records(0.10))


def test_the_title_changes_if_scratch_wins_somewhere() -> None:
    records = [make_record("scratch", 0.1, 0.3), make_record("finetune", 0.1, 0.2)]
    assert "does not help" in title_for(records)


def test_a_chart_file_is_written_for_f1_and_for_pr_auc(tmp_path: Path) -> None:
    rows = [r for r in rows_of(pilot_records()) if r.arm != "source"]
    for metric in ("test_minority_f1", "test_pr_auc"):
        path = tmp_path / (metric + ".png")
        draw_chart(rows, metric, path, "pilot")
        assert path.stat().st_size > 1000


def test_charts_go_beside_the_records_of_their_run_set(tmp_path: Path) -> None:
    assert output_folder("pilot", tmp_path) == tmp_path / "pilot"
    assert output_folder("phaseC", tmp_path) == tmp_path / "phases" / "phaseC"


def test_validation_curves_are_drawn_from_the_record_or_from_an_exported_file(
    tmp_path: Path,
) -> None:
    records_dir = tmp_path / "records"
    in_record = make_record("scratch", 0.1, 0.2, validation_curve=[[4, 0.1], [8, 0.2]])
    write_record(in_record, records_dir)
    side_only = make_record("finetune", 0.1, 0.3)
    write_record(side_only, records_dir)
    (tmp_path / "curves").mkdir()
    (tmp_path / "curves" / (side_only.run_id + ".json")).write_text(
        json.dumps([[4, 0.3]])
    )
    assert curve_for(in_record, records_dir) == [[4, 0.1], [8, 0.2]]
    assert curve_for(side_only, records_dir) == [[4, 0.3]]
    written = draw_curves(records_dir, tmp_path)
    assert [path.name for path in written] == ["val_curves_gin.png"]
    assert written[0].stat().st_size > 1000


def test_no_curves_means_no_file_and_no_crash(tmp_path: Path) -> None:
    write_record(make_record("scratch", 0.1, 0.2), tmp_path / "records")
    assert draw_curves(tmp_path / "records", tmp_path) == []


# --- Phase C reuses finished runs from the pilot and Phase A ---


def test_phase_c_tables_include_phase_a_runs_when_the_budget_matches(
    tmp_path: Path, monkeypatch
) -> None:
    from src import config

    monkeypatch.setattr(config, "BUDGET_C", 20)
    write_record(
        make_record("scratch", 1.0, 0.3, step_budget_epochs=20, run_suffix="_b20"),
        tmp_path / "phases" / "phaseA" / "records",
    )
    write_record(make_record("scratch", 0.1, 0.1), tmp_path / "pilot" / "records")
    rows = summary_rows(record_folders(tmp_path))
    assert [(r.run_set, r.fraction) for r in rows if r.run_set == "phaseC"] == []
    write_record(
        make_record("scratch", 0.1, 0.2), tmp_path / "phases" / "phaseC" / "records"
    )
    rows = summary_rows(record_folders(tmp_path))
    reused = [r for r in rows if r.run_set == "phaseC"]
    assert sorted(r.fraction for r in reused) == [0.1, 1.0]  # own 0.1, reused 1.0


def test_the_phase_c_table_has_nothing_reused_while_the_budget_is_not_set(
    tmp_path: Path, monkeypatch
) -> None:
    from src import config

    monkeypatch.setattr(config, "BUDGET_C", None)
    write_record(make_record("scratch", 0.1, 0.1), tmp_path / "pilot" / "records")
    write_record(
        make_record("scratch", 0.25, 0.2), tmp_path / "phases" / "phaseC" / "records"
    )
    rows = summary_rows(record_folders(tmp_path))
    assert [r.run_set for r in rows].count("phaseC") == 1


# --- export_curves run set ---


def test_export_sets_writes_curves_next_to_each_sets_records(tmp_path: Path) -> None:
    from src.export_curves import export_sets

    record = make_record("scratch", 1.0, 0.3, run_suffix="_b20")
    record = RunRecord(**{**record.__dict__, "run_id": "gin_scratch_f1.0_s0_b20"})
    write_record(record, tmp_path / "phaseA" / "records")
    log = tmp_path / "data" / "multignn_runs" / record.run_id / "logs"
    log.mkdir(parents=True)
    (log / "logs.log").write_text(
        "Retrieving data\nTrain F1: 0.1\nValidation F1: 0.2\nTest F1: not scored\n"
    )
    written = export_sets(tmp_path, ["phaseA"])
    assert written == [tmp_path / "phaseA" / "curves" / (record.run_id + ".json")]
    assert json.loads(written[0].read_text()) == [[1, 0.2]]
