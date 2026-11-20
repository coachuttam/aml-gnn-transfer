"""Score a finished stock Multi-GNN run and write its run record.

"Stock" means Multi-GNN's own `main.py` did the training (used for RQ1). This
script then scores the saved checkpoint with `score_checkpoint` and saves one
JSON record under results/runs/. Run from the repo root, inside Multi-GNN's
environment, after the training run has finished:

    python -m src.record_stock_run --run-id gin_source_f1.0_s0 --model gin \
        --dataset HI-Small --seed 0 --epochs 100 --runtime-seconds 5400
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

from src.config import (
    DATA_ROOT,
    DATASETS,
    MULTI_GNN_DIR,
    PROJECT_ROOT,
    RUN_MODES,
    SEEDS,
)
from src.prepare import run_folder_for
from src.records import RunRecord, git_commit, write_record
from src.score_checkpoint import (
    ScoringRequest,
    ScoringResult,
    multignn_argv,
    score_checkpoint,
)


@dataclass(frozen=True)
class RunContext:
    """Facts about how the run was made that the scorer cannot know."""

    mode: str
    fraction: float
    epochs_requested: int
    runtime_seconds: float
    environment: str
    smoke_test: bool
    ours_commit: str
    multignn_commit: str


def build_record(
    request: ScoringRequest, result: ScoringResult, context: RunContext
) -> RunRecord:
    """Combine the scoring result and the run facts into one record."""
    config = {
        "dataset": request.dataset_name,
        "multignn_argv": multignn_argv(request),
        "epochs_requested": context.epochs_requested,
        "environment": context.environment,
        "smoke_test": context.smoke_test,
        "stock_training_loop": True,
    }
    return RunRecord(
        run_id=request.run_id,
        model=request.model,
        mode=context.mode,
        fraction=context.fraction,
        seed=request.seed,
        config=config,
        ours_commit=context.ours_commit,
        multignn_commit=context.multignn_commit,
        metrics={"scored": result.metrics.to_dict()},
        epochs_trained=context.epochs_requested,  # the stock loop never stops early
        best_epoch=result.best_epoch,
        runtime_seconds=context.runtime_seconds,
    )


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True, choices=["gin", "pna"])
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--mode", default="source", choices=list(RUN_MODES))
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--runtime-seconds", type=float, required=True)
    parser.add_argument("--environment", required=True, help="env.yml or pip-fallback")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Score the checkpoint, then write the record."""
    arguments = parse_arguments()
    request = ScoringRequest(
        arguments.run_id, arguments.run_id, arguments.model,
        arguments.dataset, arguments.seed, finetuned=False,
    )  # fmt: skip
    context = RunContext(
        arguments.mode, arguments.fraction, arguments.epochs, arguments.runtime_seconds,
        arguments.environment, arguments.smoke_test,
        git_commit(PROJECT_ROOT), git_commit(MULTI_GNN_DIR),
    )  # fmt: skip
    # Multi-GNN reads ./model_settings.json, so score from the run folder.
    os.chdir(run_folder_for(request.run_id, DATA_ROOT))
    path = write_record(build_record(request, score_checkpoint(request), context))
    print("Wrote record: %s" % path)


if __name__ == "__main__":
    main()
