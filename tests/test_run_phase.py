"""Tests for src/run_phase.py and src/phase_estimate.py (no GPU, no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src import run_phase
from src.phase_estimate import (
    UnitsLedger,
    estimate_minutes,
    estimate_phase,
    gpu_key,
    units_for,
)
from src.phases import PhaseRun, phase_folders, phase_runs
from src.records import RunRecord, write_record


def fake_record(run: PhaseRun, status: str = "completed", **config: Any) -> RunRecord:
    settings: Dict[str, Any] = {
        "status": status, "gradient_steps": 4336, "labels_kept": 100,
        "kept_labels_fingerprint": "abc", "epochs_requested": 8,
        "step_budget_epochs": run.step_budget_epochs,
    }  # fmt: skip
    settings.update(config)
    scored = {"minority_f1": 0.2, "precision": 0.1, "recall": 0.1, "pr_auc": 0.1,
              "precision_at_100": 0.1, "alerts": 1}  # fmt: skip
    return RunRecord(
        run.run_id, run.model, run.mode, run.fraction, run.seed, settings, "a", "b",
        {"scored": scored}, 8, 3, 1.0,
    )  # fmt: skip


def arguments(home: Path, name: str = "phaseA", **extra: Any) -> argparse.Namespace:
    values = dict(
        set=name, home=str(home), python="python", gpu_name="NVIDIA L4",
        environment="test", seeds=None, budget=None, allow_long=False,
    )  # fmt: skip
    values.update(extra)
    return argparse.Namespace(**values)


def fake_launcher(started: List[str], **config: Any):
    """A stand-in for launch(): writes the run's record instead of training."""

    def launch(
        run: PhaseRun, python: str, label: str, environment: Dict[str, str]
    ) -> float:
        started.append(run.run_id)
        write_record(
            fake_record(run, **config), Path(environment["AML_RUN_RECORDS_DIR"])
        )
        return 3600.0

    return launch


# --- resume ---


def test_a_run_with_a_completed_record_is_skipped(tmp_path: Path) -> None:
    run = phase_runs("phaseB")[0]
    write_record(fake_record(run), tmp_path)
    assert run_phase.is_finished(run, tmp_path)


def test_a_run_with_a_no_improvement_record_is_skipped_too(tmp_path: Path) -> None:
    run = phase_runs("phaseB")[0]
    write_record(fake_record(run, "no_improvement"), tmp_path)
    assert run_phase.is_finished(run, tmp_path)


def test_a_run_with_no_record_is_not_finished(tmp_path: Path) -> None:
    assert not run_phase.is_finished(phase_runs("phaseB")[0], tmp_path)


def test_a_record_with_a_surprising_status_stops_the_notebook(tmp_path: Path) -> None:
    run = phase_runs("phaseB")[0]
    write_record(fake_record(run, "crashed"), tmp_path)
    with pytest.raises(ValueError, match="crashed"):
        run_phase.is_finished(run, tmp_path)


def test_running_a_set_again_skips_finished_runs_and_runs_the_rest(
    tmp_path: Path, monkeypatch
) -> None:
    folders = phase_folders("phaseA", tmp_path)
    first = phase_runs("phaseA")[0]
    write_record(fake_record(first), folders.records)
    started: List[str] = []
    monkeypatch.setattr(run_phase, "launch", fake_launcher(started))
    run_phase.run_set(arguments(tmp_path))
    assert started == ["gin_finetune_f1.0_s0_b20"]


def test_records_go_to_the_sets_own_folder_on_drive(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(run_phase, "launch", fake_launcher([]))
    run_phase.run_set(arguments(tmp_path))
    assert len(list((tmp_path / "phaseA" / "records").glob("*.json"))) == 2


# --- pair checks ---


def pair() -> List[PhaseRun]:
    return phase_runs("phaseA")


def test_a_pair_with_the_same_labels_and_steps_passes(tmp_path: Path) -> None:
    for run in pair():
        write_record(fake_record(run), tmp_path)
    run_phase.check_pair(pair()[0], pair()[1], tmp_path)


def test_a_pair_with_different_gradient_steps_is_rejected(tmp_path: Path) -> None:
    write_record(fake_record(pair()[0]), tmp_path)
    write_record(fake_record(pair()[1], gradient_steps=4000), tmp_path)
    with pytest.raises(RuntimeError, match="gradient_steps"):
        run_phase.check_pair(pair()[0], pair()[1], tmp_path)


def test_a_pair_with_different_kept_labels_is_rejected(tmp_path: Path) -> None:
    write_record(fake_record(pair()[0]), tmp_path)
    write_record(fake_record(pair()[1], kept_labels_fingerprint="zzz"), tmp_path)
    with pytest.raises(RuntimeError, match="kept_labels_fingerprint"):
        run_phase.check_pair(pair()[0], pair()[1], tmp_path)


def test_an_unfair_pair_stops_the_set_right_after_its_second_run(
    tmp_path: Path, monkeypatch
) -> None:
    fair, unfair = fake_launcher([]), fake_launcher([], kept_labels_fingerprint="x")

    def mixed(run, python, label, environment):
        return (fair if run.mode == "scratch" else unfair)(
            run, python, label, environment
        )

    monkeypatch.setattr(run_phase, "launch", mixed)
    with pytest.raises(RuntimeError, match="fair pair"):
        run_phase.run_set(arguments(tmp_path))


# --- estimates and units ---


def test_units_use_the_rate_of_the_gpu_that_was_given() -> None:
    assert units_for("NVIDIA L4", 3600) == pytest.approx(1.94)
    assert units_for("Tesla T4", 3600) == pytest.approx(1.19)
    assert units_for("NVIDIA A100-SXM4-40GB", 3600) == pytest.approx(5.40)


def test_an_unknown_gpu_counts_no_units_and_has_no_key() -> None:
    assert gpu_key("Some New GPU") is None and units_for("Some New GPU", 3600) == 0.0


def test_the_ledger_keeps_a_running_total() -> None:
    ledger = UnitsLedger("NVIDIA L4")
    ledger.add_run(3600)
    ledger.add_run(1800)
    assert ledger.total_units == pytest.approx(1.94 * 1.5)


def test_the_phase_a_estimate_is_about_three_hours_a_run_on_an_l4() -> None:
    estimate = estimate_phase(phase_runs("phaseA"), "L4")
    assert 2.5 < estimate.longest_run_hours < 3.0
    assert estimate.runs == 2 and estimate.units == pytest.approx(estimate.hours * 1.94)


def test_the_zero_shot_estimate_is_minutes_not_hours() -> None:
    assert estimate_minutes(phase_runs("phaseB")[0], "L4") < 10


def test_a_slower_gpu_gets_a_longer_estimate() -> None:
    run = phase_runs("phaseA")[0]
    assert estimate_minutes(run, "T4") > estimate_minutes(run, "L4")


def test_a_run_over_three_hours_is_blocked_until_the_owner_allows_it() -> None:
    long_run = PhaseRun("pna", "scratch", 1.0, 0, 20, 0)
    with pytest.raises(RuntimeError, match="approval"):
        run_phase.guard_long_run(long_run, "L4", allow_long=False)
    run_phase.guard_long_run(long_run, "L4", allow_long=True)


def test_each_run_prints_an_estimate_then_its_real_time_and_units(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(run_phase, "launch", fake_launcher([]))
    run_phase.run_set(arguments(tmp_path))
    out = capsys.readouterr().out
    assert (
        out.count("estimate") >= 3 and "running total this session: 3.88 units" in out
    )


def test_the_child_process_gets_the_sets_folders_in_its_environment(
    tmp_path: Path,
) -> None:
    environment = run_phase.child_environment(phase_folders("phaseC", tmp_path))
    assert environment["AML_RUN_RECORDS_DIR"] == str(tmp_path / "phaseC/records")
    assert environment["AML_SOURCE_MODEL_ROOT"] == str(tmp_path / "pilot/models")


def test_record_status_reads_the_json_without_the_full_record(tmp_path: Path) -> None:
    run = phase_runs("phaseB")[0]
    (tmp_path / (run.run_id + ".json")).write_text(json.dumps({"config": {}}))
    assert run_phase.record_status(run, tmp_path) == "completed"


# --- Phase C reuses the pilot's records when the budget matches ---


def pilot_run(mode: str, fraction: float, seed: int = 0) -> PhaseRun:
    return PhaseRun("gin", mode, fraction, seed, 8, 0)


def test_phase_c_skips_pilot_runs_made_at_the_same_budget(
    tmp_path: Path, monkeypatch
) -> None:
    folders = phase_folders("phaseC", tmp_path)
    for mode in ("scratch", "finetune"):
        write_record(fake_record(pilot_run(mode, 0.1)), folders.reuse_records)
    started: List[str] = []
    monkeypatch.setattr(run_phase, "launch", fake_launcher(started))
    run_phase.run_set(arguments(tmp_path, "phaseC", seeds=[0], budget=8))
    assert "gin_scratch_f0.1_s0" not in started and len(started) == 6


def test_a_pilot_record_at_another_budget_is_not_reused(tmp_path: Path) -> None:
    folders = phase_folders("phaseC", tmp_path)
    write_record(fake_record(pilot_run("scratch", 0.1)), folders.reuse_records)
    other = PhaseRun("gin", "scratch", 0.1, 0, 20, 0)
    assert run_phase.reused_record_dir(other, folders) is None
    assert run_phase.reused_record_dir(pilot_run("scratch", 0.1), folders) is not None


def test_other_sets_never_reuse_anything(tmp_path: Path) -> None:
    assert phase_folders("phaseA", tmp_path).reuse_records is None
