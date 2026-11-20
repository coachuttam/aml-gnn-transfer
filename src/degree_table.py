"""The degree table PNA needs, built from the full training graph.

PNA rescales its messages by the average node degree in the training data. Multi-GNN
works that average out from one neighbour-sampled batch (training.py, get_model),
so it depends on the shuffle and on which edges the training loader seeds from.
Our wrapper builds it once from the whole training graph instead, so it does not
change with the label fraction, and saves it next to the model.

The table is a histogram: entry d is how many nodes have degree d. Like
Multi-GNN, plain graphs count in-degree and reverse-message-passing graphs count
in-degree plus out-degree. Only nodes that appear in the training graph count.

Example:
    table = degree_histogram(source_ids, target_ids, count_both_directions=True)
    save_degree_table(model_folder, table)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

TABLE_FILE_NAME = "degree_table.json"
EDGE_TYPE = ("node", "to", "node")


def degree_histogram(
    source_ids: np.ndarray, target_ids: np.ndarray, count_both_directions: bool
) -> np.ndarray:
    """How many nodes have each degree, counting only nodes that have an edge."""
    if len(source_ids) == 0:
        raise ValueError("Cannot build a degree table from a graph with no edges")
    node_count = int(max(source_ids.max(), target_ids.max())) + 1
    if count_both_directions:
        counted_ids = np.concatenate([target_ids, source_ids])
    else:
        counted_ids = target_ids
    degrees = np.bincount(counted_ids, minlength=node_count)
    in_graph = np.zeros(node_count, dtype=bool)
    in_graph[source_ids] = True
    in_graph[target_ids] = True
    return np.bincount(degrees[in_graph], minlength=1)


def training_degree_histogram(train_data: Any, uses_reverse: bool) -> np.ndarray:
    """Degree table of Multi-GNN's training graph (the `tr_data` from get_data)."""
    graph = train_data[EDGE_TYPE] if uses_reverse else train_data
    edge_index = graph.edge_index.cpu().numpy()
    return degree_histogram(edge_index[0], edge_index[1], uses_reverse)


def table_file(model_folder: Path) -> Path:
    """Where a run keeps its degree table: inside its own model folder."""
    return model_folder / TABLE_FILE_NAME


def save_degree_table(model_folder: Path, table: np.ndarray) -> Path:
    """Write the table as JSON beside the checkpoint."""
    model_folder.mkdir(parents=True, exist_ok=True)
    path = table_file(model_folder)
    path.write_text(json.dumps([int(count) for count in table]) + "\n")
    return path


def load_degree_table(model_folder: Path) -> Optional[np.ndarray]:
    """The saved table, or None if this run never saved one (stock runs)."""
    path = table_file(model_folder)
    if not path.exists():
        return None
    return np.array(json.loads(path.read_text()), dtype=np.int64)
