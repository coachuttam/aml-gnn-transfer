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
    scoring_request,
    training_request,
)
from src.score_checkpoint import ScoringResult
from src.train_wrapper import (
    TrainingSummary,
    patched_training,
    remember_test_loader,
    training_argv,
    without_test_scoring,
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
    request = training_request(plan_run("gin", "finetune", 1.0, 0), epochs=5)
    words = training_argv(request)
    assert "--finetune" in words and "--save_model" in words
    assert words[words.index("--n_epochs") + 1] == "5"


def test_training_from_scratch_never_asks_to_load_a_model() -> None:
    words = training_argv(training_request(plan_run("gin", "scratch", 1.0, 0), 5))
    assert "--finetune" not in words


def test_scoring_a_finetuned_run_looks_for_the_finetuned_checkpoint() -> None:
    assert scoring_request(plan_run("gin", "finetune", 1.0, 0)).finetuned is True


def test_the_test_loader_is_not_run_during_training() -> None:
    holder = {"test_loader": "TEST"}
    evaluate = without_test_scoring(lambda loader, *rest: 0.99, holder)
    assert evaluate("TEST", None) == 0.0
    assert evaluate("VALIDATION", None) == 0.99


def test_the_test_loader_is_noticed_when_the_loaders_are_built() -> None:
    holder: dict = {}
    get_loaders = remember_test_loader(lambda *arguments: ("tr", "val", "te"), holder)
    get_loaders(1, 2, 3)
    assert holder["test_loader"] == "te"


def test_swapped_in_functions_are_put_back_even_if_training_fails() -> None:
    modules = SimpleNamespace(training=SimpleNamespace(get_model="original"))
    with pytest.raises(RuntimeError):
        with patched_training(modules, {"get_model": "replacement"}):
            assert modules.training.get_model == "replacement"
            raise RuntimeError("training crashed")
    assert modules.training.get_model == "original"


def make_record(smoke: bool = False):
    plan = plan_run("gin", "scratch", 1.0, 0)
    metrics = score_period(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    summary = TrainingSummary(1000, 1000, 7)
    return build_experiment_record(
        plan, ScoringResult(metrics, 4), summary, 5, 60.0, "env.yml", smoke
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
        plan, ScoringResult(metrics, 4), None, 5, 60.0, "env.yml", False
    )
    assert record.epochs_trained == 0 and record.config["labels_kept"] is None


def test_the_command_line_has_no_default_for_epochs() -> None:
    with pytest.raises(SystemExit):
        parse_arguments("--model gin --mode source --environment env.yml".split())
