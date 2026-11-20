"""Tests for src/degree_table.py and src/model_builder.py (no torch needed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.degree_table import (
    degree_histogram,
    load_degree_table,
    save_degree_table,
    table_file,
)
from src.model_builder import build_model, get_model_using_table

# Three accounts: 0 pays 1, 0 pays 2, 1 pays 2.
SOURCES = np.array([0, 0, 1])
TARGETS = np.array([1, 2, 2])


def test_plain_graphs_count_in_degree_only() -> None:
    # in-degrees are 0, 1, 2 for accounts 0, 1, 2
    assert list(degree_histogram(SOURCES, TARGETS, count_both_directions=False)) == [
        1,
        1,
        1,
    ]


def test_reverse_message_passing_graphs_count_in_plus_out_degree() -> None:
    # every account touches two transactions
    table = degree_histogram(SOURCES, TARGETS, count_both_directions=True)
    assert list(table) == [0, 0, 3]


def test_accounts_that_never_appear_in_the_training_graph_are_not_counted() -> None:
    table = degree_histogram(np.array([0]), np.array([5]), count_both_directions=False)
    assert list(table) == [1, 1]  # only accounts 0 and 5, not 1 to 4


def test_table_equals_multi_gnns_formula_when_every_node_appears() -> None:
    sources, targets = np.array([0, 1, 2, 3, 0]), np.array([1, 2, 3, 0, 2])
    multi_gnn_table = np.bincount(np.bincount(targets, minlength=4), minlength=1)
    ours = degree_histogram(sources, targets, count_both_directions=False)
    assert list(ours) == list(multi_gnn_table)


def test_a_graph_with_no_edges_cannot_make_a_table() -> None:
    empty = np.array([], dtype=int)
    with pytest.raises(ValueError, match="no edges"):
        degree_histogram(empty, empty, count_both_directions=False)


def test_a_saved_table_reads_back_unchanged(tmp_path: Path) -> None:
    save_degree_table(tmp_path / "run", np.array([4, 0, 7]))
    assert list(load_degree_table(tmp_path / "run")) == [4, 0, 7]


def test_a_run_without_a_saved_table_gives_none_not_an_error(tmp_path: Path) -> None:
    assert load_degree_table(tmp_path) is None


def test_the_table_lives_inside_the_runs_own_model_folder(tmp_path: Path) -> None:
    assert table_file(tmp_path).parent == tmp_path


def make_modules(calls: list) -> SimpleNamespace:
    def original_get_model(sample_batch, config, args):
        calls.append(args.model)
        return "multi-gnn model"

    return SimpleNamespace(training=SimpleNamespace(get_model=original_get_model))


def test_gin_is_always_built_by_multi_gnns_own_code() -> None:
    calls: list = []
    args = SimpleNamespace(model="gin")
    model = build_model(make_modules(calls), None, None, args, np.array([1, 2]))
    assert (model, calls) == ("multi-gnn model", ["gin"])


def test_pna_without_a_saved_table_falls_back_to_multi_gnns_own_code() -> None:
    calls: list = []
    build_model(make_modules(calls), None, None, SimpleNamespace(model="pna"), None)
    assert calls == ["pna"]


def test_the_swap_in_get_model_passes_calls_through_for_other_models() -> None:
    calls: list = []
    get_model = get_model_using_table(make_modules(calls), np.array([1]))
    assert get_model(None, None, SimpleNamespace(model="gat")) == "multi-gnn model"
