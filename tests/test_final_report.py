"""Tests for src/final_table.py and src/plot_final.py (made-up numbers)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.collect import Stat, summary_row
from src.final_table import markdown_table, pna_lines
from src.plot_final import draw_chart, headline, spread_labels
from src.records import RunRecord, write_record
from tests.test_collect_plot import make_record


def pna_record(mode: str, fraction: float, f1: float, hours: float) -> RunRecord:
    base = make_record(mode, fraction, f1)
    return RunRecord(
        run_id="pna_%s_f%s_s0" % (mode, fraction), model="pna", mode=mode,
        fraction=fraction, seed=0,
        config=dict(base.config, environment="env.yml / NVIDIA L4"),
        ours_commit="a", multignn_commit="b", metrics=base.metrics,
        epochs_trained=8, best_epoch=3, runtime_seconds=hours * 3600,
    )  # fmt: skip


def test_labels_that_touch_are_pushed_apart_and_others_stay_put() -> None:
    placed = spread_labels([10.0, 10.2, 30.0], gap=1.0)
    assert placed[1] - placed[0] >= 1.0 - 1e-6
    assert placed[2] == 30.0


def test_labels_already_far_apart_do_not_move() -> None:
    assert spread_labels([1.0, 5.0, 9.0], gap=1.0) == [1.0, 5.0, 9.0]


def test_units_are_hours_times_the_l4_rate(tmp_path: Path) -> None:
    write_record(pna_record("scratch", 0.1, 0.2, hours=2.0), tmp_path)
    (line,) = pna_lines(tmp_path)
    assert abs(line.units - 2.0 * 1.94) < 1e-9


def test_the_table_lists_the_source_first_and_ends_with_a_total(tmp_path: Path) -> None:
    write_record(pna_record("scratch", 0.1, 0.2, hours=1.0), tmp_path)
    write_record(pna_record("source", 1.0, 0.6, hours=1.0), tmp_path)
    lines = markdown_table(pna_lines(tmp_path)).splitlines()
    assert "HI-Small" in lines[2]
    assert lines[-1].startswith("| **Total**") and "2.0" in lines[-1]


def stats_with(pna_tuned: float, pna_scratch_full: float) -> Dict:
    def point(fraction: float, mean: float) -> List[Stat]:
        return [Stat(fraction, mean, 0.0, 1)]

    stats = {}
    for model in ("gin", "pna"):
        for arm in ("scratch", "finetune"):
            stats[(model, arm)] = [
                Stat(f, 0.3 if arm == "finetune" else 0.2, 0.0, 1)
                for f in (0.1, 0.25, 1.0)
            ]
    stats[("pna", "finetune")][0] = point(0.1, pna_tuned)[0]
    stats[("pna", "scratch")][2] = point(1.0, pna_scratch_full)[0]
    return stats


def test_the_title_claims_the_pna_match_only_when_the_numbers_are_close() -> None:
    close = headline(stats_with(0.327, 0.343), "test_minority_f1")
    assert "32.7" in close and "34.3" in close and "one seed" in close
    far = headline(stats_with(0.25, 0.343), "test_minority_f1")
    assert "close to" not in far


def test_the_chart_is_written(tmp_path: Path) -> None:
    rows = [
        summary_row(r, "phaseC")
        for r in (make_record("scratch", 0.1, 0.1), make_record("finetune", 0.1, 0.3))
    ]
    path = tmp_path / "chart.png"
    draw_chart(rows, "test_minority_f1", path)
    assert path.stat().st_size > 1000
