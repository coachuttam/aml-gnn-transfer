"""Project-wide settings: paths, datasets, seeds, label fractions, model flags.

Every number that shapes an experiment lives here with its reason beside it,
so nobody has to hunt through the code to learn why a value was chosen.

Example:
    from src.config import DATASETS, SEEDS
    DATASETS["HI-Small"].folder_name   # "Small_HI"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_GNN_DIR = PROJECT_ROOT / "external" / "Multi-GNN"
RESULTS_DIR = PROJECT_ROOT / "results"
# Run records can be redirected too (Colab writes them straight to Drive).
RUN_RECORDS_DIR = Path(os.environ.get("AML_RUN_RECORDS_DIR", RESULTS_DIR / "runs"))

# Data and model weights are large and never committed (AGENTS.md rule 8).
# The environment variables let the notebooks point at /kaggle/working or Google Drive.
DATA_ROOT = Path(os.environ.get("AML_DATA_ROOT", PROJECT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("AML_MODEL_ROOT", PROJECT_ROOT / "models"))

# Building the graph (reading the CSV, adding ports) takes several minutes of CPU,
# so it is done once per dataset and settings, then loaded from here by training
# and scoring. On Colab this folder lives on Drive, so it survives a restart.
GRAPH_CACHE_ROOT = Path(os.environ.get("AML_GRAPH_CACHE", DATA_ROOT / "graph_cache"))

# Multi-GNN's main.py reads data_config.json and model_settings.json from the
# folder it is started in, and writes logs/ there. Editing the copies inside the
# submodule would break the "submodule diff is always empty" rule, so every run
# starts from its own folder: <data root>/<this name>/<run id>/.
RUN_FOLDERS_NAME = "multignn_runs"


@dataclass(frozen=True)
class DatasetSpec:
    """One IBM dataset: its name, its Kaggle file, and Multi-GNN's `--data` folder."""

    name: str
    raw_file_name: str
    folder_name: str


# Multi-GNN finds a dataset at <aml_data>/<--data value>/formatted_transactions.csv.
DATASETS: Dict[str, DatasetSpec] = {
    "HI-Small": DatasetSpec("HI-Small", "HI-Small_Trans.csv", "Small_HI"),
    "LI-Small": DatasetSpec("LI-Small", "LI-Small_Trans.csv", "Small_LI"),
}

# Pre-train on the synthetic dataset, then adapt to the target (PRD section 1).
SOURCE_DATASET = "HI-Small"
TARGET_DATASET = "LI-Small"

# Three seeds is the smallest number that gives a spread (PRD section 6).
SEEDS: Tuple[int, ...] = (0, 1, 2)

# Shares of LI-Small training labels kept in the label-scarcity runs (PRD FR5).
# Even steps on a chart: 10, 25, 50, 100.
LABEL_FRACTIONS: Tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)

# Analysts can read about a hundred alerts a day (same value as Project 2).
PRECISION_AT_K = 100

# Alert budgets for "recall at N alerts": same three as Project 2, so results compare.
ALERT_BUDGETS: Tuple[int, ...] = (10_000, 25_000, 50_000)

# The four ways a run can use the data (ARCHITECTURE.md, "Run modes").
RUN_MODES: Tuple[str, ...] = ("source", "zero_shot", "finetune", "scratch")

# Enough epochs to exercise train, save, score and record before any long job.
# Multi-GNN never saves at epoch 0, so one epoch could leave no checkpoint.
SMOKE_TEST_EPOCHS = 3

# Training budget for the label-scarcity runs, counted in gradient steps. It is
# the number of epochs at fraction 1.0; smaller fractions get proportionally more
# epochs, so every run takes about the same number of steps. 5 is IBM's
# fine-tuning setting (Table 3). Scratch and finetune share it at each fraction.
STEP_BUDGET_EPOCHS = 5

# Smoke test of the wrapper path: 10% of labels, and never more than 2 epochs
# (a 1-epoch budget at 10% would be 10 epochs, about 40 minutes on a T4).
SMOKE_STEP_BUDGET_EPOCHS = 1
SMOKE_LABEL_FRACTION = 0.10
SMOKE_MAX_EPOCHS = 2

# Multi-GNN's own default. Not yet checked against IBM's paper settings (RQ1).
REPRODUCTION_EPOCHS = 100

# "Multi-GIN + EU" and "Multi-PNA + EU" in the paper: reverse message passing,
# ego ids, port numbers and edge updates (--emlps). Same flags as Multi-GNN's README.
MULTI_GNN_ADAPTATION_FLAGS: List[str] = ["--emlps", "--reverse_mp", "--ego", "--ports"]

# IBM's published minority F1 on HI-Small (Altman et al. 2023, Table 3), for RQ1.
IBM_REPORTED_MINORITY_F1: Dict[str, float] = {"gin": 64.79, "pna": 68.16}
