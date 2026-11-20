"""Train one Multi-GNN model with Multi-GNN's own `train_gnn`, plus run-time swaps.

Nothing inside external/Multi-GNN is edited. Before calling `train_gnn` we swap
in functions on its `training` module, and put them back afterwards:

  * `get_loaders`: the training loader only seeds from the kept labels (label
    masking). Every edge stays in the graph; validation and test loaders are
    the stock ones.
  * `evaluate_homo` / `evaluate_hetero`: skip the test loader, so the test
    period is not scored during training (AGENTS.md rule 2). The stock loop
    scores test every epoch; its logged "Test F1" lines are placeholders here.
  * `get_model`: PNA is built with our full-training-graph degree table.

Everything else (loss, optimiser, epochs, best-validation checkpoint) is stock.
Note: with fewer labels an epoch has proportionally fewer training steps.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from src.degree_table import save_degree_table, training_degree_histogram
from src.label_mask import fingerprint, keep_label_positions
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
    kept_labels_fingerprint: str


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


def masked_train_loader(
    train_data: Any, kept: np.ndarray, transform: Any, args: Any, uses_reverse: bool
) -> Any:
    """A training loader whose seed edges are only the kept ones; all edges stay."""
    import torch
    from torch_geometric.loader import LinkNeighborLoader

    kept_tensor = torch.as_tensor(kept)
    graph = train_data[EDGE_TYPE] if uses_reverse else train_data
    seeds = graph.edge_index[:, kept_tensor]
    labels = graph.y[kept_tensor]
    if uses_reverse:
        seeds = (EDGE_TYPE, seeds)
    return LinkNeighborLoader(
        train_data,
        num_neighbors=args.num_neighs,
        edge_label_index=seeds,
        edge_label=labels,
        batch_size=args.batch_size,
        shuffle=True,
        transform=transform,
    )


def with_masked_training_loader(
    original_get_loaders: Callable[..., Any],
    kept: np.ndarray,
    uses_reverse: bool,
    make_loader: Callable[..., Any] = masked_train_loader,
) -> Callable[..., Any]:
    """Wrap get_loaders: stock validation and test loaders, masked training loader."""

    def get_loaders(
        train_data,
        val_data,
        test_data,
        train_inds,
        val_inds,
        test_inds,
        transform,
        args,
    ):
        stock = original_get_loaders(
            train_data, val_data, test_data, train_inds, val_inds, test_inds,
            transform, args,
        )  # fmt: skip
        train_loader = make_loader(train_data, kept, transform, args, uses_reverse)
        return train_loader, stock[1], stock[2]

    return get_loaders


def build_replacements(
    modules: Any,
    degree_table: Optional[np.ndarray],
    kept: np.ndarray,
    uses_reverse: bool,
) -> Dict[str, Callable[..., Any]]:
    """The functions to swap in for one training run."""
    holder: Dict[str, Any] = {}
    training = modules.training
    masked = with_masked_training_loader(training.get_loaders, kept, uses_reverse)
    return {
        "get_loaders": remember_test_loader(masked, holder),
        "evaluate_homo": without_test_scoring(training.evaluate_homo, holder),
        "evaluate_hetero": without_test_scoring(training.evaluate_hetero, holder),
        "get_model": get_model_using_table(modules, degree_table),
    }


def summarize_training_labels(
    train_data: Any, uses_reverse: bool, kept: np.ndarray
) -> TrainingSummary:
    """Count training transactions, kept labels and kept laundering labels."""
    graph = train_data[EDGE_TYPE] if uses_reverse else train_data
    labels = graph.y.cpu().numpy()
    laundering_kept = int(labels[kept].sum())
    if laundering_kept == 0:
        raise ValueError("The kept labels contain no laundering: nothing to learn from")
    return TrainingSummary(len(labels), len(kept), laundering_kept, fingerprint(kept))


def prepare_degree_table(
    args: Any, train_data: Any, model_folder: Path
) -> Optional[np.ndarray]:
    """For PNA: build the table from the full training graph and save it."""
    if args.model != "pna":
        return None
    table = training_degree_histogram(train_data, bool(args.reverse_mp))
    save_degree_table(model_folder, table)
    return table


def train_model(
    request: TrainingRequest,
    modules: Any,
    data_config: Dict[str, Any],
    model_folder: Path,
) -> TrainingSummary:
    """Train or fine-tune one model, from inside the run folder."""
    import torch

    args = modules.util.create_parser().parse_args(training_argv(request))
    modules.util.logger_setup()
    modules.util.set_seed(args.seed)
    train_data, val_data, test_data, train_inds, val_inds, test_inds = (
        modules.data_loading.get_data(args, data_config)
    )
    uses_reverse = bool(args.reverse_mp)
    kept = keep_label_positions(len(train_inds), request.fraction, args.seed)
    table = prepare_degree_table(args, train_data, model_folder)
    summary = summarize_training_labels(train_data, uses_reverse, kept)
    kept_inds = train_inds[torch.as_tensor(kept)]
    replacements = build_replacements(modules, table, kept, uses_reverse)
    with patched_training(modules, replacements):
        modules.training.train_gnn(
            train_data, val_data, test_data, kept_inds, val_inds, test_inds,
            args, data_config,
        )  # fmt: skip
    return summary
