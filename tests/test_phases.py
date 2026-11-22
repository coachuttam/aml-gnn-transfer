"""Tests for src/phases.py: run ids, run counts and fair pairs for every run set."""

from __future__ import annotations

import pytest

from src import config
from src.phases import load_phase, pairs, phase_folders, phase_names, phase_runs
from src.pilot import pilot_runs
from src.run_plan import plan_run
from src.score_checkpoint import base_argv

EXAMPLE_BUDGET = 8  # BUDGET_C is not set yet, so tests pass a budget themselves


def ids(runs):
    return [run.run_id for run in runs]


def test_phase_a_run_ids_end_in_b20_and_never_match_a_pilot_id() -> None:
    runs = phase_runs("phaseA")
    assert ids(runs) == ["gin_scratch_f1.0_s0_b20", "gin_finetune_f1.0_s0_b20"]
    assert not set(ids(runs)) & set(ids(phase_runs("pilot")))


def test_phase_a_finetune_still_starts_from_the_plain_source_model() -> None:
    finetune = phase_runs("phaseA")[1]
    plan = plan_run(
        finetune.model, finetune.mode, finetune.fraction, finetune.seed,
        finetune.step_budget_epochs, source_seed=finetune.source_seed,
        run_suffix=finetune.run_suffix,
    )  # fmt: skip
    assert plan.run_id == "gin_finetune_f1.0_s0_b20"
    assert plan.load_from_run_id == "gin_source_f1.0_s0"


def test_phase_b_is_one_zero_shot_run_with_its_own_id_and_no_training() -> None:
    (run,) = phase_runs("phaseB")
    assert run.run_id == "gin_zero_shot_f1.0_s0"
    plan = plan_run("gin", "zero_shot", 1.0, 0, run.step_budget_epochs, source_seed=0)
    assert not plan.trains and plan.epochs == 0
    assert plan.checkpoint_run_id == "gin_source_f1.0_s0"
    assert "--step-budget-epochs" not in run.command_words()


def test_phase_a_validates_every_epoch() -> None:
    first = phase_runs("phaseA")[0]
    plan = plan_run("gin", "scratch", 1.0, 0, first.step_budget_epochs)
    assert plan.epochs == 20 and plan.validations == 20


def test_the_pilot_list_is_the_same_as_the_original_pilot_list() -> None:
    assert ids(phase_runs("pilot")) == [run.run_id for run in pilot_runs()]


def test_run_counts_per_phase() -> None:
    assert len(phase_runs("phaseB")) == 1
    assert len(phase_runs("phaseA")) == 2
    assert (
        len(phase_runs("phaseC", budget=EXAMPLE_BUDGET)) == 12
    )  # 3 fractions x 2 arms x 2 seeds
    assert len(phase_runs("phaseD", budget=EXAMPLE_BUDGET)) == 1 + 8  # source + 4 x 2
    assert len(phase_runs("phaseD", seeds=[0, 1, 2], budget=EXAMPLE_BUDGET)) == 25


def test_phase_c_covers_every_fraction_arm_and_seed_once() -> None:
    runs = phase_runs("phaseC", budget=EXAMPLE_BUDGET)
    keys = {(r.fraction, r.mode, r.seed) for r in runs}
    assert len(keys) == 12 and len(set(ids(runs))) == 12
    assert {r.fraction for r in runs} == set(config.PHASE_C_FRACTIONS)
    assert {r.seed for r in runs} == set(config.PHASE_C_SEEDS)


def test_phase_c_and_d_refuse_to_start_while_the_budget_is_not_set(monkeypatch) -> None:
    monkeypatch.setattr(config, "BUDGET_C", None)
    for name in ("phaseC", "phaseD"):
        with pytest.raises(ValueError, match="BUDGET_C"):
            phase_runs(name)


def test_phase_c_uses_the_budget_set_in_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "BUDGET_C", 12)
    assert {r.step_budget_epochs for r in phase_runs("phaseC")} == {12}


def test_phase_d_trains_the_pna_source_model_first_then_the_same_grid() -> None:
    runs = phase_runs("phaseD", budget=EXAMPLE_BUDGET)
    assert runs[0].run_id == "pna_source_f1.0_s0"
    assert {r.model for r in runs} == {"pna"}
    assert {r.fraction for r in runs[1:]} == set(config.LABEL_FRACTIONS)
    assert {r.seed for r in runs} == {0}  # default: one seed


def test_phase_d_models_get_the_adaptation_flags_of_multi_pna_plus_eu() -> None:
    words = base_argv("LI-Small", "pna", 0)
    for flag in ("--emlps", "--reverse_mp", "--ego", "--ports"):
        assert flag in words


def test_seed_zero_is_finished_before_seed_one_starts() -> None:
    seeds = [r.seed for r in phase_runs("phaseC", budget=EXAMPLE_BUDGET)]
    assert seeds == sorted(seeds)


def test_every_scratch_run_is_followed_by_the_finetune_run_of_its_pair() -> None:
    runs = phase_runs("phaseC", budget=EXAMPLE_BUDGET)
    for scratch, finetune in pairs(runs):
        assert runs.index(finetune) == runs.index(scratch) + 1


def test_scratch_and_finetune_in_a_pair_get_equal_epochs_and_validations() -> None:
    for name in ("phaseC", "phaseD", "phaseA"):
        for scratch, finetune in pairs(phase_runs(name, budget=EXAMPLE_BUDGET)):
            a = plan_run(
                "gin", "scratch", scratch.fraction, 0, scratch.step_budget_epochs
            )
            b = plan_run(
                "gin", "finetune", finetune.fraction, 0, finetune.step_budget_epochs
            )
            assert (a.epochs, a.validations) == (b.epochs, b.validations)


def test_finetune_runs_in_every_seed_start_from_the_one_source_model() -> None:
    finetunes = [
        r for r in phase_runs("phaseC", budget=EXAMPLE_BUDGET) if r.mode == "finetune"
    ]
    assert {r.source_seed for r in finetunes} == {0}
    assert all("--source-seed" in r.command_words() for r in finetunes)


def test_every_phase_reference_in_the_yaml_file_exists_in_config() -> None:
    for name in phase_names():
        settings = load_phase(name)
        assert settings["model"] in ("gin", "pna")


def test_phases_that_start_from_the_pilot_source_look_for_it_in_the_pilot_folder(
    tmp_path,
) -> None:
    for name in ("phaseA", "phaseB", "phaseC"):
        assert phase_folders(name, tmp_path).source_models == tmp_path / "pilot/models"
    assert phase_folders("phaseD", tmp_path).source_models == tmp_path / "phaseD/models"


def test_each_phase_keeps_its_own_records_folder() -> None:
    folders = {phase_folders(n, config.PROJECT_ROOT).records for n in phase_names()}
    assert len(folders) == len(phase_names())
