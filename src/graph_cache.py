"""Build each prepared graph once, then reuse it.

Multi-GNN's `get_data` reads the CSV, adds port numbers and builds the graph
objects. That is several minutes of single-core CPU work with the GPU idle, and
training and scoring each did it. Here the result is saved to disk the first
time and loaded the next time, for the same dataset and settings.

The cache holds the graph exactly as `get_data` returns it, before anything
mutates it, so a loaded copy is the same as a freshly built one.

Example:
    data = load_graph_data(modules, args, data_config)   # builds, or loads the cache
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

from src.config import GRAPH_CACHE_ROOT

# The settings that change what `get_data` returns (see Multi-GNN's data_loading.py).
GRAPH_SETTINGS = ("data", "model", "ports", "tds", "reverse_mp")


def graph_settings(args: Any) -> Dict[str, Any]:
    """The parsed arguments that decide the graph's contents."""
    return {name: getattr(args, name) for name in GRAPH_SETTINGS}


def source_fingerprint(csv_path: Path) -> str:
    """A cheap stand-in for the file's contents: its size and modified time."""
    status = csv_path.stat()
    return "%d-%d" % (status.st_size, int(status.st_mtime))


def cache_file(
    args: Any, data_config: Dict[str, Any], cache_root: Path = GRAPH_CACHE_ROOT
) -> Path:
    """Where the graph for these settings is (or will be) saved.

    The name includes the formatted CSV's size and time, so re-formatting the
    data can never leave a stale graph behind.
    """
    csv_path = Path(data_config["paths"]["aml_data"]) / args.data
    csv_path = csv_path / "formatted_transactions.csv"
    key = dict(graph_settings(args), csv=source_fingerprint(csv_path))
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return cache_root / ("%s_%s.pt" % (args.data, digest[:12]))


def default_save(data: Any, path: Path) -> None:
    """Write with torch (imported late: laptops may not have it)."""
    import torch

    torch.save(data, path)


def default_load(path: Path) -> Any:
    """Read with torch. These are our own files, so full unpickling is fine."""
    import torch

    return torch.load(path, weights_only=False)


def load_graph_data(
    modules: Any,
    args: Any,
    data_config: Dict[str, Any],
    cache_root: Path = GRAPH_CACHE_ROOT,
    save: Callable[[Any, Path], None] = default_save,
    load: Callable[[Path], Any] = default_load,
) -> Any:
    """Multi-GNN's `get_data` result, from the cache when it is there."""
    path = cache_file(args, data_config, cache_root)
    if path.exists():
        print("Loading cached graph: %s" % path)
        return load(path)
    data = modules.data_loading.get_data(args, data_config)
    cache_root.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    save(data, partial)
    os.replace(partial, path)  # a crash mid-write never leaves a half-made cache
    print("Saved graph cache: %s" % path)
    return data
