"""Tests for src/collect.py and src/plot.py, using small made-up records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.collect import (
    gaps_by_fraction,
    load_records,
    markdown_table,
    result_rows,
)
from src.plot import chart_title, draw_chart
from src.records import RunRecord, write_record


def make_record(mode: str, fraction: float, f1: float, **config: Any) -> RunRecord:
    scored: Dict[str, Any] = {
        "minority_f1": f1, "precision": 0.5, "recall": 0.2, "pr_auc": 0.1,
        "precision_at_100": 0.3, "alerts": 40,
    }  # fmt: skip
    return RunRecord(
        run_id="gin_%s_f%s_s0" % (mode, fraction), model="gin", mode=mode,
        fraction=fraction, seed=0, config=config, ours_commit="a", multignn_commit="b",
        metrics={"scored": scored}, epochs_trained=8, best_epoch=3, runtime_seconds=1.0,
    )  # fmt: skip


def write_all(folder: Path, records) -> None:
    for record in records:
        write_record(record, folder)


def pilot_records():
    return [
        make_record("scratch", 0.1, 0.157), make_record("finetune", 0.1, 0.228),
        make_record("scratch", 1.0, 0.205), make_record("finetune", 1.0, 0.331),
        make_record("source", 1.0, 0.605),
    ]  # fmt: skip


def test_smoke_runs_and_superseded_runs_are_left_out(tmp_path: Path) -> None:
    write_all(tmp_path, [make_record("scratch", 0.1, 0.1)])
    write_all(tmp_path / "superseded", [make_record("scratch", 1.0, 0.9)])
    smoke = make_record("finetune", 0.1, 0.8, smoke_test=True)
    write_record(smoke, tmp_path)
    assert [r.fraction for r in load_records(tmp_path)] == [0.1]


def test_rows_put_the_source_first_then_each_share_with_scratch_before_finetune() -> (
    None
):
    rows = result_rows(pilot_records())
    order = [(row.mode, row.fraction) for row in rows]
    assert order == [
        ("source", 1.0), ("scratch", 0.1), ("finetune", 0.1),
        ("scratch", 1.0), ("finetune", 1.0),
    ]  # fmt: skip


def test_the_table_shows_percentages_and_plain_system_names() -> None:
    table = markdown_table(result_rows(pilot_records()))
    assert "22.8%" in table and "Trained from scratch" in table
    assert "Pre-trained on HI-Small, then fine-tuned" in table


def test_a_run_that_never_learned_is_shown_with_a_note_not_dropped() -> None:
    record = make_record("scratch", 0.25, 0.0, status="no_improvement")
    scored = dict(record.metrics["scored"], pr_auc=None, precision_at_100=None)
    scored["alerts"] = None
    record = RunRecord(**{**record.__dict__, "metrics": {"scored": scored}})
    table = markdown_table(result_rows([record]))
    assert "0.0%" in table and "n/a" in table and "no checkpoint" in table


def test_the_gap_is_finetune_minus_scratch_at_each_share() -> None:
    gaps = gaps_by_fraction(result_rows(pilot_records()))
    assert round(gaps[0.1], 3) == 0.071 and round(gaps[1.0], 3) == 0.126


def test_the_chart_title_states_the_finding_and_admits_one_seed() -> None:
    title = chart_title(result_rows(pilot_records()), "minority F1")
    assert "higher at every label share" in title and "one seed" in title


def test_the_title_changes_if_scratch_wins_somewhere() -> None:
    records = [make_record("scratch", 0.1, 0.3), make_record("finetune", 0.1, 0.2)]
    assert "does not help" in chart_title(result_rows(records), "minority F1")


def test_a_chart_file_is_written(tmp_path: Path) -> None:
    path = tmp_path / "chart.png"
    draw_chart(result_rows(pilot_records()), "minority_f1", path, True)
    assert path.stat().st_size > 1000
