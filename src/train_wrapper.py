"""Train one Multi-GNN model with Multi-GNN's own `train_gnn`, plus run-time swaps.

Nothing inside external/Multi-GNN is edited. Before calling `train_gnn` we swap
in three functions on its `training` module, and put them back afterwards:

  * `evaluate_homo` / `evaluate_hetero`: skip the test loader, so the test
    period is not scored during training (AGENTS.md rule 2). The stock loop
    scores test every epoch; its logged "Test F1" lines are placeholders here.
  * `get_model`: PNA is built with our full-training-graph degree table.

Everything else (loss, optimiser, epochs, best-validation checkpoint) is stock.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from src.degree_table import save_degree_table, training_degree_histogram
from src.model_builder import get_model_using_table
from src.score_checkpoint import base_argv

EDGE_TYPE = ("node", "to", "node")


@dataclass(frozen=True)
class TrainingRequest:
    """What to train. `finetune` starts from the checkpoint in model_to_load."""

    model: str
    dataset_name: str
    seed: int
    epochs: int
    fraction: float
    finetune: bool


@dataclass(frozen=True)
class TrainingSummary:
    """Facts about the training labels, for the run record."""

    train_transactions: int
    labels_kept: int
    laundering_labels_kept: int


def training_argv(request: TrainingRequest) -> List[str]:
    """Words for Multi-GNN's parser: train, save the best model, no wandb."""
    words = base_argv(request.dataset_name, request.model, request.seed)
    words += ["--n_epochs", str(request.epochs), "--save_model"]
    if request.finetune:
        words.append("--finetune")
    return words


@contextmanager
def patched_training(
    modules: Any, replacements: Dict[str, Callable[..., Any]]
) -> Iterator[None]:
    """Swap functions on Multi-GNN's `training` module, then always restore them."""
    originals = {name: getattr(modules.training, name) for name in replacements}
    for name, function in replacements.items():
        setattr(modules.training, name, function)
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(modules.training, name, function)


def without_test_scoring(
    original: Callable[..., float], holder: Dict[str, Any]
) -> Callable[..., float]:
    """Wrap an evaluate function so it returns 0.0 for the test loader, unrun."""

    def evaluate(loader: Any, *rest: Any) -> float:
        if loader is holder.get("test_loader"):
            return 0.0
        return original(loader, *rest)

    return evaluate


def remember_test_loader(
    original_get_loaders: Callable[..., Any], holder: Dict[str, Any]
) -> Callable[..., Any]:
    """Wrap get_loaders so we know which loader is the test loader."""

    def get_loaders(*arguments: Any) -> Any:
        loaders = original_get_loaders(*arguments)
        holder["test_loader"] = loaders[2]
        return loaders

    return get_loaders


def build_replacements(
    modules: Any, degree_table: Optional[np.ndarray]
) -> Dict[str, Callable[..., Any]]:
    """The functions to swap in for one training run."""
    holder: Dict[str, Any] = {}
    training = modules.training
    return {
        "get_loaders": remember_test_loader(training.get_loaders, holder),
        "evaluate_homo": without_test_scoring(training.evaluate_homo, holder),
        "evaluate_hetero": without_test_scoring(training.evaluate_hetero, holder),
        "get_model": get_model_using_table(modules, degree_table),
    }


def summarize_training_labels(train_data: Any, uses_reverse: bool) -> TrainingSummary:
    """Count training transactions and laundering labels (all are kept, for now)."""
    graph = train_data[EDGE_TYPE] if uses_reverse else train_data
    labels = graph.y.cpu().numpy()
    return TrainingSummary(len(labels), len(labels), int(labels.sum()))


def train_model(
    request: TrainingRequest,
    modules: Any,
    data_config: Dict[str, Any],
    model_folder: Path,
) -> TrainingSummary:
    """Train or fine-tune one model, from inside the run folder."""
    if request.fraction < 1.0:
        raise NotImplementedError("Label masking is added in the next commit")
    args = modules.util.create_parser().parse_args(training_argv(request))
    modules.util.logger_setup()
    modules.util.set_seed(args.seed)
    data = modules.data_loading.get_data(args, data_config)
    uses_reverse = bool(args.reverse_mp)
    table = None
    if args.model == "pna":
        table = training_degree_histogram(data[0], uses_reverse)
        save_degree_table(model_folder, table)
    with patched_training(modules, build_replacements(modules, table)):
        modules.training.train_gnn(*data, args, data_config)
    return summarize_training_labels(data[0], uses_reverse)
