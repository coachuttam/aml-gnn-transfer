"""Score a saved Multi-GNN checkpoint: threshold on validation, test scored once.

Multi-GNN's own scoring only reports F1 from argmax. Here we load the best
checkpoint (saved at the best validation epoch), get the laundering probability
of every validation and test transaction, pick the threshold on validation, and
score the test period exactly once (PRD section 6).

Run from the repo root, inside Multi-GNN's environment:
    python -m src.score_checkpoint --run-id gin_source_f1.0_s0 \
        --dataset HI-Small --model gin --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

from src.config import (
    DATA_ROOT,
    DATASETS,
    MULTI_GNN_ADAPTATION_FLAGS,
    MULTI_GNN_DIR,
    SEEDS,
)
from src.metrics import RunMetrics, score_run
from src.prepare import run_folder_for
from src.records import model_folder_for_run

EDGE_TYPE = ("node", "to", "node")
REVERSE_EDGE_TYPE = ("node", "rev_to", "node")


@dataclass(frozen=True)
class ScoringRequest:
    """What to score. `checkpoint_run_id` differs from `run_id` for zero-shot runs."""

    run_id: str
    checkpoint_run_id: str
    model: str
    dataset_name: str
    seed: int
    finetuned: bool


@dataclass(frozen=True)
class ScoringResult:
    """Test-period metrics, plus the epoch the checkpoint was saved at."""

    metrics: RunMetrics
    best_epoch: int


@dataclass
class LoadedData:
    """Multi-GNN's loaders and index lists, kept together."""

    train_loader: Any
    validation_loader: Any
    test_loader: Any
    test_data: Any
    validation_indices: Any
    test_indices: Any


def multignn_argv(request: ScoringRequest) -> List[str]:
    """Command-line words that make Multi-GNN's own parser build the right arguments."""
    folder = DATASETS[request.dataset_name].folder_name
    words = ["--data", folder, "--model", request.model, "--seed", str(request.seed)]
    # --unique_name is a true/false flag in Multi-GNN, and its checkpoint file
    # names are built from it, so it must be switched on to find them.
    words += ["--unique_name", "--testing"]
    return words + list(MULTI_GNN_ADAPTATION_FLAGS)


def checkpoint_file(model_folder: Path, finetuned: bool) -> Path:
    """Multi-GNN's name for a saved model. Fine-tuned runs get a suffix."""
    suffix = "_finetuned" if finetuned else ""
    return model_folder / ("checkpoint_True%s.tar" % suffix)


def import_multignn_modules() -> SimpleNamespace:
    """Import Multi-GNN's modules late: they need torch, which laptops may lack."""
    sys.dont_write_bytecode = True  # keep __pycache__ out of the submodule folder
    sys.path.insert(0, str(MULTI_GNN_DIR))
    import data_loading
    import train_util
    import training
    import util

    return SimpleNamespace(
        data_loading=data_loading, train_util=train_util, training=training, util=util
    )


def prepare_loaders(
    modules: SimpleNamespace, args: Any, data_config: Any
) -> LoadedData:
    """Build the data and loaders the same way Multi-GNN's train_gnn does."""
    data = modules.data_loading.get_data(args, data_config)
    train_data, validation_data, test_data = data[0], data[1], data[2]
    train_inds, validation_inds, test_inds = data[3], data[4], data[5]
    modules.train_util.add_arange_ids([train_data, validation_data, test_data])
    transform = modules.train_util.AddEgoIds() if args.ego else None
    loaders = modules.train_util.get_loaders(
        train_data, validation_data, test_data,
        train_inds, validation_inds, test_inds, transform, args,
    )  # fmt: skip
    return LoadedData(*loaders, test_data, validation_inds, test_inds)


def model_config(modules: SimpleNamespace, args: Any) -> SimpleNamespace:
    """The settings object get_model expects (Multi-GNN uses a wandb config here)."""
    names = ["lr", "n_hidden", "n_gnn_layers", "dropout", "final_dropout", "n_heads"]
    values = {name: modules.train_util.extract_param(name, args) for name in names}
    return SimpleNamespace(model=args.model, **values)


def load_trained_model(
    modules: SimpleNamespace, args: Any, data: LoadedData, path: Path, device: Any
) -> Tuple[Any, int]:
    """Rebuild the model as train_gnn does, then load the saved weights."""
    import torch
    from torch_geometric.nn import to_hetero

    sample_batch = next(iter(data.train_loader))
    model = modules.training.get_model(sample_batch, model_config(modules, args), args)
    if args.reverse_mp:
        model = to_hetero(model, data.test_data.metadata(), aggr="mean")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Multi-GNN's own evaluation never calls eval(), so dropout stays on there.
    # We switch it off for scoring, which is the correct way.
    return model.to(device).eval(), int(checkpoint["epoch"])


def seed_edge_mask(
    batch_edge_attr: Any, input_id: Any, loader_edge_attr: Any, inds: Any
) -> Any:
    """Which edges in a sampled batch are the seed edges being scored."""
    import torch

    batch_edge_inds = inds.detach().cpu()[input_id.detach().cpu()]
    seed_ids = loader_edge_attr.detach().cpu()[batch_edge_inds, 0]
    return torch.isin(batch_edge_attr[:, 0].detach().cpu(), seed_ids)


def score_batch_plain(
    batch: Any, loader: Any, inds: Any, model: Any, device: Any
) -> Tuple[Any, Any]:
    """Labels and laundering probabilities for one batch of a plain graph."""
    import torch

    mask = seed_edge_mask(batch.edge_attr, batch.input_id, loader.data.edge_attr, inds)
    batch.edge_attr = batch.edge_attr[:, 1:]  # drop the id column, as Multi-GNN does
    batch.to(device)
    out = model(batch.x, batch.edge_index, batch.edge_attr)[mask]
    return batch.y[mask], torch.softmax(out, dim=-1)[:, 1]


def score_batch_reverse(
    batch: Any, loader: Any, inds: Any, model: Any, device: Any
) -> Tuple[Any, Any]:
    """Labels and laundering probabilities for one batch (reverse message passing)."""
    import torch

    forward, reverse = batch[EDGE_TYPE], batch[REVERSE_EDGE_TYPE]
    loader_attr = loader.data[EDGE_TYPE].edge_attr
    mask = seed_edge_mask(forward.edge_attr, forward.input_id, loader_attr, inds)
    forward.edge_attr = forward.edge_attr[:, 1:]
    reverse.edge_attr = reverse.edge_attr[:, 1:]
    batch.to(device)
    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[EDGE_TYPE][
        mask
    ]
    return batch[EDGE_TYPE].y[mask], torch.softmax(out, dim=-1)[:, 1]


def collect_scores(
    loader: Any, inds: Any, model: Any, device: Any, uses_reverse: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Labels and laundering probabilities for every transaction a loader scores."""
    import torch

    score_batch = score_batch_reverse if uses_reverse else score_batch_plain
    truths, scores = [], []
    with torch.no_grad():
        for batch in loader:
            truth, score = score_batch(batch, loader, inds, model, device)
            truths.append(truth.cpu())
            scores.append(score.cpu())
    return torch.cat(truths).numpy().astype(int), torch.cat(scores).numpy()


def collect_both_periods(
    data: LoadedData, model: Any, device: Any, uses_reverse: bool
) -> Tuple[np.ndarray, ...]:
    """Validation and test labels and probabilities. Test is read here once."""
    val = collect_scores(
        data.validation_loader, data.validation_indices, model, device, uses_reverse
    )
    test = collect_scores(
        data.test_loader, data.test_indices, model, device, uses_reverse
    )
    return val + test


def score_checkpoint(request: ScoringRequest) -> ScoringResult:
    """Score the saved model. Run with the working folder set to the run folder."""
    import torch

    modules = import_multignn_modules()
    args = modules.util.create_parser().parse_args(multignn_argv(request))
    config_path = run_folder_for(request.run_id, DATA_ROOT) / "data_config.json"
    modules.util.set_seed(args.seed)
    data = prepare_loaders(modules, args, json.loads(config_path.read_text()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_folder = model_folder_for_run(request.checkpoint_run_id)
    path = checkpoint_file(model_folder, request.finetuned)
    model, best_epoch = load_trained_model(modules, args, data, path, device)
    arrays = collect_both_periods(data, model, device, bool(args.reverse_mp))
    return ScoringResult(score_run(*arrays), best_epoch)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--checkpoint-run-id", default=None, help="Defaults to --run-id"
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--model", required=True, choices=["gin", "pna"])
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--finetuned", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Score one checkpoint and print the result as JSON."""
    arguments = parse_arguments()
    request = ScoringRequest(
        arguments.run_id, arguments.checkpoint_run_id or arguments.run_id,
        arguments.model, arguments.dataset, arguments.seed, arguments.finetuned,
    )  # fmt: skip
    # Multi-GNN reads ./model_settings.json, so start from the run folder.
    os.chdir(run_folder_for(request.run_id, DATA_ROOT))
    result = score_checkpoint(request)
    print(
        json.dumps(
            {"best_epoch": result.best_epoch, "metrics": result.metrics.to_dict()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
