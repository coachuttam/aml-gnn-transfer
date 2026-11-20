"""Tests for src/graph_cache.py (pickle stands in for torch, so no GPU or torch)."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from src.graph_cache import cache_file, load_graph_data


def save_with_pickle(data: Any, path: Path) -> None:
    path.write_bytes(pickle.dumps(data))


def load_with_pickle(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


def make_setup(tmp_path: Path, **changes: Any):
    csv = tmp_path / "data" / "Small_HI" / "formatted_transactions.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text("a,b\n")
    settings = dict(
        data="Small_HI", model="gin", ports=True, tds=False, reverse_mp=True
    )
    settings.update(changes)
    config = {"paths": {"aml_data": str(tmp_path / "data")}}
    return SimpleNamespace(**settings), config


def make_modules(calls: List[str]) -> SimpleNamespace:
    def get_data(args: Any, config: Any) -> Any:
        calls.append("built")
        return ("train", "val", "test")

    return SimpleNamespace(data_loading=SimpleNamespace(get_data=get_data))


def load(modules, args, config, tmp_path):
    return load_graph_data(
        modules, args, config, tmp_path / "cache", save_with_pickle, load_with_pickle
    )


def test_the_second_load_reads_the_cache_instead_of_building_again(
    tmp_path: Path,
) -> None:
    calls: List[str] = []
    args, config = make_setup(tmp_path)
    first = load(make_modules(calls), args, config, tmp_path)
    second = load(make_modules(calls), args, config, tmp_path)
    assert first == second == ("train", "val", "test")
    assert calls == ["built"]


def test_different_graph_settings_get_different_cache_files(tmp_path: Path) -> None:
    with_ports, config = make_setup(tmp_path, ports=True)
    without_ports, _ = make_setup(tmp_path, ports=False)
    cache = tmp_path / "cache"
    assert cache_file(with_ports, config, cache) != cache_file(
        without_ports, config, cache
    )


def test_reformatting_the_csv_makes_the_old_cache_unusable(tmp_path: Path) -> None:
    args, config = make_setup(tmp_path)
    before = cache_file(args, config, tmp_path / "cache")
    csv = tmp_path / "data" / "Small_HI" / "formatted_transactions.csv"
    csv.write_text("a,b\n1,2\n")
    assert cache_file(args, config, tmp_path / "cache") != before


def test_no_half_written_cache_is_left_behind(tmp_path: Path) -> None:
    args, config = make_setup(tmp_path)
    load(make_modules([]), args, config, tmp_path)
    names = [path.name for path in (tmp_path / "cache").iterdir()]
    assert len(names) == 1 and not names[0].endswith(".partial")
