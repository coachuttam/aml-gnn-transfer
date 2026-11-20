"""Train one Multi-GNN model with Multi-GNN's own `train_gnn`, plus run-time swaps.

Nothing inside external/Multi-GNN is edited. Before calling `train_gnn` we swap
in functions on its `training` module, and put them back afterwards:

  * `get_loaders`: the training loader only seeds from the kept labels (label
    masking). Every edge stays in the graph; validation and test loaders are
    the stock ones.
  * `evaluate_homo` / `evaluate_hetero`: skip the test loader (AGENTS.md rule 2)
    and validate only on a fixed schedule (src/evaluation_schedule.py). Skipped
    passes show "not scored" in the log.
  * `get_model`: PNA is built with our full-training-graph degree table.
  * `train_homo` / `train_hetero` (smoke tests only): if the stock loop saved no
    checkpoint, the last epoch's model is saved, so a short run can always be scored.

Everything else (loss, optimiser, epochs, best-validation checkpoint) is stock.
With fewer labels an epoch has fewer steps; src/step_budget.py sets the epochs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Tuple

import numpy as np

from src.degree_table import save_degree_table, training_degree_histogram
from src.evaluation_schedule import (
    remember_loaders,
    scheduled_evaluation,
    validation_epochs,
)
from src.graph_cache import load_graph_data
from src.label_mask import fingerprint, keep_label_positions
from src.model_builder import get_model_using_table
from src.score_checkpoint import base_argv, checkpoint_file
from src.step_budget import steps_per_epoch

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
    smoke_test: bool = False
    validations: Optional[int] = None  # validation passes in all; None = every epoch


@dataclass(frozen=True)
class TrainingSummary:
    """Facts about the training labels, for the run record."""

    train_transactions: int
    labels_kept: int
    laundering_labels_kept: int
    kept_labels_fingerprint: str
    steps_per_epoch: int
    checkpoint_is_last_epoch: bool = False  # True if only the smoke fallback saved


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
    schedule: FrozenSet[int],
) -> Dict[str, Callable[..., Any]]:
    """The functions to swap in for one training run."""
    holder: Dict[str, Any] = {}
    training = modules.training
    masked = with_masked_training_loader(training.get_loaders, kept, uses_reverse)
    return {
        "get_loaders": remember_loaders(masked, holder),
        "evaluate_homo": scheduled_evaluation(training.evaluate_homo, holder, schedule),
        "evaluate_hetero": scheduled_evaluation(
            training.evaluate_hetero, holder, schedule
        ),
        "get_model": get_model_using_table(modules, degree_table),
    }


def saving_last_epoch_if_none(
    original: Callable[..., Any],
    modules: Any,
    checkpoint: Path,
    outcome: Dict[str, Any],
) -> Callable[..., Any]:
    """Wrap a stock train function: if it saved no checkpoint, save the last epoch."""

    def train(*arguments: Any) -> Any:
        model = original(*arguments)
        if not checkpoint.exists():
            # Positions follow Multi-GNN's train_homo / train_hetero signature.
            optimizer, args, data_config = arguments[7], arguments[9], arguments[14]
            last_epoch = args.n_epochs - 1
            modules.train_util.save_model(
                model, optimizer, last_epoch, args, data_config
            )
            outcome["last_epoch_saved"] = True
        return model

    return train


def last_epoch_replacements(
    modules: Any, checkpoint: Path, outcome: Dict[str, Any]
) -> Dict[str, Callable[..., Any]]:
    """The train functions to swap in for smoke tests."""
    training = modules.training
    return {
        "train_homo": saving_last_epoch_if_none(
            training.train_homo, modules, checkpoint, outcome
        ),
        "train_hetero": saving_last_epoch_if_none(
            training.train_hetero, modules, checkpoint, outcome
        ),
    }


def summarize_training_labels(
    train_data: Any, uses_reverse: bool, kept: np.ndarray, batch_size: int
) -> TrainingSummary:
    """Count training transactions, kept labels and kept laundering labels."""
    graph = train_data[EDGE_TYPE] if uses_reverse else train_data
    labels = graph.y.cpu().numpy()
    laundering_kept = int(labels[kept].sum())
    if laundering_kept == 0:
        raise ValueError("The kept labels contain no laundering: nothing to learn from")
    steps = steps_per_epoch(len(kept), batch_size)
    return TrainingSummary(
        len(labels), len(kept), laundering_kept, fingerprint(kept), steps
    )


def prepare_degree_table(
    args: Any, train_data: Any, model_folder: Path
) -> Optional[np.ndarray]:
    """For PNA: build the table from the full training graph and save it."""
    if args.model != "pna":
        return None
    table = training_degree_histogram(train_data, bool(args.reverse_mp))
    save_degree_table(model_folder, table)
    return table


def load_training_data(
    request: TrainingRequest, modules: Any, data_config: Dict[str, Any]
) -> Tuple[Any, Tuple[Any, ...]]:
    """Parse Multi-GNN's arguments, set the seed and load the data."""
    args = modules.util.create_parser().parse_args(training_argv(request))
    modules.util.logger_setup()
    modules.util.set_seed(args.seed)
    return args, load_graph_data(modules, args, data_config)


def train_model(
    request: TrainingRequest,
    modules: Any,
    data_config: Dict[str, Any],
    model_folder: Path,
) -> TrainingSummary:
    """Train or fine-tune one model, from inside the run folder."""
    import torch

    args, data = load_training_data(request, modules, data_config)
    train_data, val_data, test_data, train_inds, val_inds, test_inds = data
    uses_reverse = bool(args.reverse_mp)
    kept = keep_label_positions(len(train_inds), request.fraction, args.seed)
    table = prepare_degree_table(args, train_data, model_folder)
    summary = summarize_training_labels(train_data, uses_reverse, kept, args.batch_size)
    schedule = validation_epochs(request.epochs, request.validations)
    replacements = build_replacements(modules, table, kept, uses_reverse, schedule)
    outcome: Dict[str, Any] = {}
    if request.smoke_test:
        path = checkpoint_file(model_folder, request.finetune)
        replacements.update(last_epoch_replacements(modules, path, outcome))
    kept_inds = train_inds[torch.as_tensor(kept)]
    with patched_training(modules, replacements):
        modules.training.train_gnn(
            train_data, val_data, test_data, kept_inds, val_inds, test_inds,
            args, data_config,
        )  # fmt: skip
    return replace(
        summary, checkpoint_is_last_epoch=outcome.get("last_epoch_saved", False)
    )
