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
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_GNN_DIR = PROJECT_ROOT / "external" / "Multi-GNN"
RESULTS_DIR = PROJECT_ROOT / "results"
# Run records can be redirected too (Colab writes them straight to Drive).
RUN_RECORDS_DIR = Path(os.environ.get("AML_RUN_RECORDS_DIR", RESULTS_DIR / "runs"))
# Pilot records are copied here from Drive; kept apart from full-design runs.
PILOT_RECORDS_DIR = RESULTS_DIR / "pilot" / "records"
# Records of Phases A-D are copied from Drive to results/phases/<set>/records/.
PHASES_RESULTS_DIR = RESULTS_DIR / "phases"
# The run sets (pilot, phaseA..D) are described in this file.
PHASES_FILE = PROJECT_ROOT / "configs" / "phases.yaml"

# Data and model weights are large and never committed (AGENTS.md rule 8).
# The environment variables let the notebooks point at /kaggle/working or Google Drive.
DATA_ROOT = Path(os.environ.get("AML_DATA_ROOT", PROJECT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("AML_MODEL_ROOT", PROJECT_ROOT / "models"))
# Fine-tuning and zero-shot runs start from a HI-Small source model that was trained
# in an earlier run set (the pilot), so its folder can differ from MODEL_ROOT.
SOURCE_MODEL_ROOT = Path(os.environ.get("AML_SOURCE_MODEL_ROOT", MODEL_ROOT))

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

# The IBM dataset on Kaggle; the Colab notebook downloads it with the Kaggle API.
KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

# --- The Colab pilot (a smaller first pass; the full design is "v2") ---
# One model, one seed (for now), three fractions, an 8-epoch step budget.
# Deferred to v2: PNA, seed 2 and the 50% fraction (seed 1 pending).
PILOT_MODEL = "gin"
# Seed 0 only for now; seed 1 is added after the owner has seen scratch's 100% curve.
PILOT_SEEDS: Tuple[int, ...] = (0,)
PILOT_FRACTIONS: Tuple[float, ...] = (0.10, 0.25, 1.00)
# Raised from 3: at 3 epochs scratch was still improving, which favoured fine-tuning.
PILOT_STEP_BUDGET_EPOCHS = 8

# Both seeds fine-tune from the same source model (one HI-Small model, seed 0), so
# the finetune spread leaves out pre-training variance. The README says so.
PILOT_SOURCE_SEED = 0

# Epochs for the one HI-Small source model of the pilot. PROPOSED, not yet approved:
# the owner decides after reading the estimate (src/estimate_pilot.py).
PILOT_SOURCE_EPOCHS = 10

# Pilot records and models live apart from the full design's, so v2 runs can
# reuse the same run ids without ever overwriting a pilot record.
PILOT_FOLDER_NAME = "pilot"

# --- Time and cost estimate inputs (src/estimate_pilot.py) ---
# Measured on a Kaggle T4 (GIN + EU, batch size as Multi-GNN's default).
MEASURED_TRAIN_MINUTES_PER_FULL_LI_EPOCH = 10.0  # 1 min per epoch at 10% of labels
MEASURED_VALIDATION_MINUTES_LI = 2.8  # one validation pass over LI-Small
# Assumptions, not measurements: replace them with Colab smoke timings.
ASSUMED_TEST_PASS_MINUTES_LI = 2.8  # taken equal to a validation pass
ASSUMED_GRAPH_LOAD_MINUTES = (
    1.5  # loading a cached graph, once for train and once for score
)
HI_TO_LI_SIZE_RATIO = 5_078_345 / 6_924_049  # transactions in HI-Small / LI-Small
# Colab compute units used per hour (measured by a third party in March 2026,
# https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/ ; they
# change over time, so check the rate shown in your own Colab session).
COLAB_UNITS_PER_HOUR: Dict[str, float] = {"T4": 1.19, "L4": 1.94, "A100": 5.40}
# How much faster than a T4 each GPU might be on this job. Sampling runs on the
# CPU and kept both GPUs idle, so 1.0 is the realistic case and the second is hopeful.
GPU_SPEED_CASES: Dict[str, Tuple[float, float]] = {
    "T4": (1.0, 1.0),
    "L4": (1.0, 1.5),
    "A100": (1.0, 2.5),
}
COLAB_UNITS_AVAILABLE = 59.76  # the owner's balance when the budget was set

# --- Phases A-D (configs/phases.yaml names these constants) ---
# Phase A: a longer budget for GIN at 100% of labels, to see whether scratch and
# finetune are still improving after 8 epochs. Every epoch is validated.
PHASE_A_BUDGET = 20

# Phase C's budget, in epochs at fraction 1.0. None on purpose: the owner sets it
# after reading Phase A, and Phases C and D refuse to start until it is a number.
BUDGET_C: Optional[int] = None

# Phase D: Multi-PNA + EU. One seed by default because PNA is slower than GIN;
# more seeds can be passed to src.run_phase with --seeds.
PHASE_D_SEEDS: Tuple[int, ...] = (0,)
# The PNA source model trains on HI-Small for as many epochs as the GIN one did.
PHASE_D_SOURCE_EPOCHS = PILOT_SOURCE_EPOCHS

# Statuses that mean "this run is finished": a re-run skips it.
FINISHED_STATUSES: Tuple[str, ...] = ("completed", "no_improvement")

# --- Time and units per run (src/phase_estimate.py) ---
# Measured in the pilot on a Colab L4 (results/pilot/records): about 4,100 s to
# train and validate 8 budget epochs, and about 177 s to score at the end.
L4_MINUTES_PER_BUDGET_EPOCH = 8.6  # one epoch's steps at 100%, plus one validation
L4_MINUTES_FINAL_SCORING = 3.0  # validation and test scoring, once per run
# Time on each GPU as a multiple of the L4's. T4: 12.8 min per budget epoch was
# measured on Kaggle against 8.6 on the L4. A100 is a guess (see GPU_SPEED_CASES).
GPU_TIME_VS_L4: Dict[str, float] = {"L4": 1.0, "T4": 1.5, "A100": 0.6}
# Not measured yet: PNA has more parameters per layer than GIN. Replace with the
# first PNA run's real time (the notebook prints it).
ASSUMED_PNA_TIME_VS_GIN = 1.5
# Rule 6 of AGENTS.md: ask the owner before a run longer than this many GPU hours.
APPROVAL_GPU_HOURS = 3.0
