"""Build Multi-GNN models, using our saved degree table for PNA.

Multi-GNN's `get_model` always derives PNA's degree table from the sample batch it
is given. This module builds the same PNA but with a table we supply, and
hands every other model to Multi-GNN's own `get_model` unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np

EDGE_TYPE = ("node", "to", "node")

# Marks our wrapper functions and points back at Multi-GNN's original.
STOCK_ATTRIBUTE = "multignn_original"


def build_model(
    modules: SimpleNamespace,
    sample_batch: Any,
    config: Any,
    args: Any,
    degree_table: Optional[np.ndarray],
    stock_get_model: Optional[Callable[..., Any]] = None,
) -> Any:
    """Multi-GNN's model, with PNA built from `degree_table` when one is given.

    `stock_get_model` must be Multi-GNN's own function. It is looked up on the
    module only when none is passed, and then unwrapped, so a patched module can
    never make this call itself.
    """
    if args.model != "pna" or degree_table is None:
        original = stock_get_model or stock_function(modules.training.get_model)
        return original(sample_batch, config, args)
    return build_pna(modules, sample_batch, config, args, degree_table)


def build_pna(
    modules: SimpleNamespace,
    sample_batch: Any,
    config: Any,
    args: Any,
    degree_table: np.ndarray,
) -> Any:
    """The PNA that Multi-GNN's get_model builds, but with our degree table."""
    import torch
    from torch_geometric.data import HeteroData

    if isinstance(sample_batch, HeteroData):
        feature_count = sample_batch["node"].x.shape[1]
        edge_dimension = sample_batch[EDGE_TYPE].edge_attr.shape[1] - 1
    else:
        feature_count = sample_batch.x.shape[1]
        edge_dimension = sample_batch.edge_attr.shape[1] - 1  # first column is an id
    return modules.models.PNA(
        num_features=feature_count,
        num_gnn_layers=config.n_gnn_layers,
        n_classes=2,
        n_hidden=round(config.n_hidden),
        edge_updates=args.emlps,
        edge_dim=edge_dimension,
        dropout=config.dropout,
        deg=torch.tensor(degree_table, dtype=torch.long),
        final_dropout=config.final_dropout,
    )


def stock_function(function: Callable[..., Any]) -> Callable[..., Any]:
    """Multi-GNN's own function, even if `function` is one of our wrappers."""
    return getattr(function, STOCK_ATTRIBUTE, function)


def get_model_using_table(
    modules: SimpleNamespace, degree_table: Optional[np.ndarray]
) -> Callable[[Any, Any, Any], Any]:
    """A drop-in for `training.get_model`, for the wrapper to swap in at run time.

    Multi-GNN's original is captured here, once, before any patching. Wrapping
    an already wrapped function reuses the same original, so patching twice is safe.
    """
    stock = stock_function(modules.training.get_model)

    def get_model(sample_batch: Any, config: Any, args: Any) -> Any:
        return build_model(modules, sample_batch, config, args, degree_table, stock)

    setattr(get_model, STOCK_ATTRIBUTE, stock)
    return get_model
