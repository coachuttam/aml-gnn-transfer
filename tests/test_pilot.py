"""Tests for src/pilot.py, src/estimate_pilot.py and the pilot's use of plan_run."""

from __future__ import annotations

from pathlib import Path

from src.config import PILOT_FRACTIONS, PILOT_SEEDS
from src.estimate_pilot import estimate_table, run_minutes, total_hours
from src.pilot import PilotRun, pilot_runs, record_exists, set_aside_partial_run
from src.records import model_folder_for_run
from src.run_experiment import parse_arguments, plan_run


def test_the_pilot_has_one_source_model_then_twelve_comparison_runs() -> None:
    runs = pilot_runs()
    assert [run.mode for run in runs].count("source") == 1
    assert runs[0].mode == "source" and len(runs) == 1 + 2 * 3 * 2


def test_the_pilot_leaves_out_pna_seed_two_and_fifty_percent() -> None:
    runs = pilot_runs()
    assert {run.model for run in runs} == {"gin"}
    assert {run.seed for run in runs} == set(PILOT_SEEDS) and 2 not in PILOT_SEEDS
    assert 0.5 not in PILOT_FRACTIONS and 0.5 not in {r.fraction for r in runs}


def test_scratch_and_finetune_at_one_fraction_and_seed_sit_side_by_side() -> None:
    runs = pilot_runs()[1:]
    for scratch, finetune in zip(runs[0::2], runs[1::2]):
        assert (scratch.mode, finetune.mode) == ("scratch", "finetune")
        assert (scratch.fraction, scratch.seed) == (finetune.fraction, finetune.seed)


def test_seed_zero_is_finished_before_seed_one_starts() -> None:
    seeds = [run.seed for run in pilot_runs()[1:]]
    assert seeds == sorted(seeds)


def test_every_comparison_run_uses_the_three_epoch_budget() -> None:
    budgets = {run.step_budget_epochs for run in pilot_runs()[1:]}
    assert budgets == {3}


def test_every_pilot_run_id_is_unique() -> None:
    ids = [run.run_id for run in pilot_runs()]
    assert len(ids) == len(set(ids))


def test_both_seeds_fine_tune_from_the_one_seed_zero_source_model() -> None:
    finetunes = [r for r in pilot_runs() if r.mode == "finetune" and r.seed == 1]
    arguments = parse_arguments(finetunes[0].command_words() + ["--environment", "x"])
    plan = plan_run(
        "gin", "finetune", arguments.fraction, arguments.seed,
        arguments.step_budget_epochs, source_seed=arguments.source_seed,
    )  # fmt: skip
    assert plan.load_from_run_id == "gin_source_f1.0_s0"
    assert plan.run_id.endswith("_s1")


def test_without_a_source_seed_a_finetune_run_uses_its_own_seed() -> None:
    assert plan_run("gin", "finetune", 0.25, 1).load_from_run_id == "gin_source_f1.0_s1"


def test_a_run_with_a_record_counts_as_finished(tmp_path: Path) -> None:
    run = PilotRun("gin", "scratch", 0.1, 0, 3)
    assert not record_exists(run, tmp_path)
    (tmp_path / (run.run_id + ".json")).write_text("{}")
    assert record_exists(run, tmp_path)


def test_a_dead_attempts_model_folder_is_moved_aside_not_deleted(
    tmp_path: Path,
) -> None:
    run = PilotRun("gin", "scratch", 0.1, 0, 3)
    folder = model_folder_for_run(run.run_id, tmp_path)
    folder.mkdir(parents=True)
    (folder / "checkpoint_True.tar").write_text("weights")
    moved = set_aside_partial_run(run, tmp_path)
    assert not folder.exists()
    assert (moved / "checkpoint_True.tar").read_text() == "weights"


def test_moving_aside_twice_never_overwrites_the_first_set_aside_folder(
    tmp_path: Path,
) -> None:
    run = PilotRun("gin", "scratch", 0.1, 0, 3)
    folder = model_folder_for_run(run.run_id, tmp_path)
    for _ in range(2):
        folder.mkdir(parents=True)
        set_aside_partial_run(run, tmp_path)
    assert len(list(tmp_path.glob("*_interrupted_*"))) == 2


def test_nothing_is_moved_when_there_is_no_earlier_attempt(tmp_path: Path) -> None:
    assert (
        set_aside_partial_run(PilotRun("gin", "scratch", 0.1, 0, 3), tmp_path) is None
    )


def test_a_source_run_costs_more_than_a_comparison_run_on_a_t4() -> None:
    runs = pilot_runs()
    assert run_minutes(runs[0]).total > run_minutes(runs[1]).total


def test_comparison_runs_cost_the_same_at_every_fraction() -> None:
    comparison = [run_minutes(run).total for run in pilot_runs()[1:]]
    assert max(comparison) == min(comparison)


def test_a_faster_gpu_takes_fewer_hours_but_may_use_more_units() -> None:
    table = estimate_table(pilot_runs())
    assert table["A100 hopeful"]["hours"] < table["T4 realistic"]["hours"]
    assert table["A100 realistic"]["units"] > table["T4 realistic"]["units"]
    assert total_hours(pilot_runs()) > 0


def test_every_pilot_command_is_accepted_by_the_run_experiment_parser() -> None:
    for run in pilot_runs():
        arguments = parse_arguments(run.command_words() + ["--environment", "x"])
        assert arguments.mode == run.mode and arguments.seed == run.seed


def test_a_run_that_finished_training_is_not_moved_aside(tmp_path: Path) -> None:
    from src.train_wrapper import TrainingSummary
    from src.training_marker import save_training_marker

    run = PilotRun("gin", "scratch", 0.1, 0, 3)
    folder = model_folder_for_run(run.run_id, tmp_path)
    folder.mkdir(parents=True)
    save_training_marker(folder, TrainingSummary(1, 1, 1, "x", 1), 1.0)
    assert set_aside_partial_run(run, tmp_path) is None
    assert folder.exists()


def test_measured_timings_replace_the_defaults() -> None:
    from src.estimate_pilot import Timings, timings_from
    from src.estimate_pilot import parse_arguments as parse

    arguments = parse(
        ["--gpu", "L4", "--train-minutes", "5", "--validation-minutes", "2"]
    )
    timings = timings_from(arguments)
    assert (timings.train_per_full_epoch, timings.validation_pass) == (5, 2)
    assert timings.graph_load == Timings().graph_load  # not given: default kept


def test_faster_measured_timings_give_fewer_hours() -> None:
    from src.estimate_pilot import Timings, total_hours

    runs = pilot_runs()
    assert total_hours(runs, Timings(5, 1.5, 1.5, 1)) < total_hours(runs)


def test_the_source_epochs_that_fit_in_three_hours_really_fit() -> None:
    from src.estimate_pilot import Timings, run_minutes, source_epochs_within

    epochs = source_epochs_within(3.0, Timings())
    fits = PilotRun("gin", "source", 1.0, 0, epochs)
    too_long = PilotRun("gin", "source", 1.0, 0, epochs + 1)
    assert (
        run_minutes(fits, Timings()).total
        <= 180
        < run_minutes(too_long, Timings()).total
    )
