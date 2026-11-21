"""Tests for src/run_experiment.py and src/train_wrapper.py (no torch needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.metrics import score_period
from src.run_experiment import (
    build_experiment_record,
    parse_arguments,
    plan_run,
    resolve_step_budget,
    scoring_request,
    training_request,
)
from src.score_checkpoint import ScoringResult
from src.train_wrapper import (
    TrainingSummary,
    patched_training,
    training_argv,
)


def test_the_source_run_trains_and_scores_on_hi_small() -> None:
    plan = plan_run("gin", "source", 1.0, 0)
    assert (plan.run_id, plan.dataset_name) == ("gin_source_f1.0_s0", "HI-Small")
    assert plan.trains and plan.load_from_run_id is None


def test_the_zero_shot_run_trains_nothing_and_scores_the_source_model_on_li_small() -> (
    None
):
    plan = plan_run("pna", "zero_shot", 1.0, 1)
    assert not plan.trains
    assert plan.dataset_name == "LI-Small"
    assert plan.checkpoint_run_id == "pna_source_f1.0_s1"


def test_the_finetune_run_starts_from_the_source_model_of_the_same_seed() -> None:
    plan = plan_run("pna", "finetune", 0.25, 2)
    assert plan.run_id == "pna_finetune_f0.25_s2"
    assert plan.load_from_run_id == "pna_source_f1.0_s2"
    assert plan.finetuned and plan.dataset_name == "LI-Small"


def test_the_scratch_run_starts_from_nothing() -> None:
    plan = plan_run("gin", "scratch", 0.5, 0)
    assert plan.load_from_run_id is None and not plan.finetuned


def test_source_and_zero_shot_runs_always_use_every_label() -> None:
    for mode in ("source", "zero_shot"):
        with pytest.raises(ValueError, match="all labels"):
            plan_run("gin", mode, 0.5, 0)


def test_a_smoke_plan_renames_the_run_and_the_model_it_starts_from() -> None:
    plan = plan_run("gin", "finetune", 1.0, 0, smoke=True)
    assert plan.run_id == "gin_finetune_f1.0_s0_smoke"
    assert plan.load_from_run_id == "gin_source_f1.0_s0_smoke"


def test_finetune_training_words_ask_multi_gnn_to_load_and_save_models() -> None:
    request = training_request(plan_run("gin", "finetune", 1.0, 0))
    words = training_argv(request)
    assert "--finetune" in words and "--save_model" in words
    assert words[words.index("--n_epochs") + 1] == "5"


def test_training_from_scratch_never_asks_to_load_a_model() -> None:
    words = training_argv(training_request(plan_run("gin", "scratch", 1.0, 0)))
    assert "--finetune" not in words


def test_scoring_a_finetuned_run_looks_for_the_finetuned_checkpoint() -> None:
    assert scoring_request(plan_run("gin", "finetune", 1.0, 0)).finetuned is True


def test_swapped_in_functions_are_put_back_even_if_training_fails() -> None:
    modules = SimpleNamespace(training=SimpleNamespace(get_model="original"))
    with pytest.raises(RuntimeError):
        with patched_training(modules, {"get_model": "replacement"}):
            assert modules.training.get_model == "replacement"
            raise RuntimeError("training crashed")
    assert modules.training.get_model == "original"


def make_record(smoke: bool = False):
    return make_record_for(plan_run("gin", "scratch", 1.0, 0), smoke)


def make_record_for(plan, smoke: bool = False):
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    summary = TrainingSummary(1000, 1000, 7, "abc123", 3)
    return build_experiment_record(
        plan, ScoringResult(metrics, 4), summary, 60.0, "env.yml", smoke
    )


def test_the_record_says_how_many_labels_were_kept_and_how_many_were_laundering() -> (
    None
):
    config = make_record().config
    assert (config["labels_kept"], config["laundering_labels_kept"]) == (1000, 7)


def test_a_wrapper_record_has_scored_metrics_and_no_stock_log_numbers() -> None:
    record = make_record()
    assert "scored" in record.metrics and "logged_by_multignn" not in record.metrics
    assert record.config["stock_training_loop"] is False


def test_a_zero_shot_record_reports_zero_epochs_trained() -> None:
    plan = plan_run("gin", "zero_shot", 1.0, 0)
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    record = build_experiment_record(
        plan, ScoringResult(metrics, 4), None, 60.0, "env.yml", False
    )
    assert record.epochs_trained == 0 and record.config["labels_kept"] is None


def test_smaller_label_fractions_train_proportionally_more_epochs() -> None:
    epochs = [plan_run("gin", "scratch", f, 0).epochs for f in (1.0, 0.5, 0.25, 0.1)]
    assert epochs == [5, 10, 20, 50]


def test_scratch_and_finetune_get_the_same_budget_at_the_same_fraction() -> None:
    scratch = plan_run("gin", "scratch", 0.25, 1)
    finetune = plan_run("gin", "finetune", 0.25, 1)
    assert scratch.epochs == finetune.epochs == 20


def test_the_default_budget_is_five_epochs_at_all_the_labels() -> None:
    assert plan_run("pna", "finetune", 1.0, 0).epochs == 5


def test_a_zero_shot_run_has_no_epochs() -> None:
    assert plan_run("gin", "zero_shot", 1.0, 0).epochs == 0


def test_the_record_stores_the_budget_epochs_and_total_gradient_steps() -> None:
    plan = plan_run("gin", "scratch", 0.25, 0)  # 20 epochs
    config = make_record_for(plan).config
    assert config["step_budget_epochs"] == 5
    assert config["steps_per_epoch"] == 3
    assert config["gradient_steps"] == 60


def test_source_runs_must_choose_their_epochs_explicitly() -> None:
    with pytest.raises(ValueError, match="explicit"):
        resolve_step_budget("source", None)


def test_other_modes_fall_back_to_the_configured_budget() -> None:
    assert resolve_step_budget("finetune", None) == 5
    assert resolve_step_budget("source", 12) == 12


def test_the_command_line_accepts_a_step_budget() -> None:
    arguments = parse_arguments(
        "--model gin --mode source --environment env.yml --step-budget-epochs 7".split()
    )
    assert arguments.step_budget_epochs == 7


def test_a_run_refuses_to_start_over_a_checkpoint_from_unfinished_training(
    tmp_path,
) -> None:
    from src.run_experiment import check_starting_point

    plan = plan_run("gin", "scratch", 1.0, 0)
    folder = tmp_path / plan.run_id
    folder.mkdir()
    (folder / "checkpoint_True.tar").write_text("old")
    with pytest.raises(FileExistsError, match="never finished"):
        check_starting_point(plan, tmp_path)


def test_a_finetune_run_refuses_to_start_without_its_source_model(tmp_path) -> None:
    from src.run_experiment import check_starting_point

    with pytest.raises(FileNotFoundError, match="improves"):
        check_starting_point(plan_run("gin", "finetune", 0.1, 0), tmp_path)


def test_a_finetune_run_may_start_when_its_source_model_exists(tmp_path) -> None:
    from src.run_experiment import check_starting_point

    plan = plan_run("gin", "finetune", 0.1, 0)
    source = tmp_path / plan.load_from_run_id
    source.mkdir()
    (source / "checkpoint_True.tar").write_text("model")
    check_starting_point(plan, tmp_path)


def test_a_smoke_training_request_is_marked_as_smoke() -> None:
    plan = plan_run("gin", "scratch", 0.1, 0)
    assert training_request(plan, smoke_test=True).smoke_test is True
    assert training_request(plan).smoke_test is False


def test_every_label_fraction_gets_the_same_number_of_validation_passes() -> None:
    counts = {
        fraction: plan_run("gin", "scratch", fraction, 0, 3).validations
        for fraction in (0.10, 0.25, 1.0)
    }
    assert counts == {0.10: 3, 0.25: 3, 1.0: 3}


def test_a_smoke_run_is_capped_at_two_epochs_even_at_ten_percent_labels() -> None:
    plan = plan_run("gin", "scratch", 0.10, 0, 1, smoke=True)
    assert plan.epochs == 2
    assert plan_run("gin", "scratch", 0.10, 0, 1).epochs == 10


def test_the_training_request_carries_the_validation_count() -> None:
    plan = plan_run("gin", "scratch", 0.25, 0, 3)
    assert training_request(plan).validations == 3
    assert plan_run("gin", "zero_shot", 1.0, 0).epochs == 0
