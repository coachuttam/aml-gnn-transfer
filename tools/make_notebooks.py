"""Write notebooks/kaggle_runner.ipynb and notebooks/colab_runner.ipynb.

Run from the repo root after changing any cell:
    python -m tools.make_notebooks
A test checks that the committed notebooks match what this writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tools.notebook_cells import (
    CLONE_REPO,
    ENVIRONMENT,
    ENVIRONMENT_NOTE,
    GPU_GATE,
    RUN_HELPER,
    code,
    markdown,
)

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"

KAGGLE_SETUP = r"""
KAGGLE_DATA = "/kaggle/input/ibm-transactions-for-anti-money-laundering-aml"
RUN_FULL = False  # False = smoke test only; True = full reproduction run
WORK = "/kaggle/working"
REPO = f"{WORK}/repo"

os.environ["AML_DATA_ROOT"] = f"{WORK}/data"
os.environ["AML_MODEL_ROOT"] = f"{WORK}/models"
# Keep __pycache__ out of the Multi-GNN submodule folder.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
"""

KAGGLE_IDS = r"""
from src.config import (
    MULTI_GNN_ADAPTATION_FLAGS, REPRODUCTION_EPOCHS, SEEDS, SMOKE_LABEL_FRACTION,
    SMOKE_STEP_BUDGET_EPOCHS, SMOKE_TEST_EPOCHS,
)
from src.records import make_run_id, smoke_run_id

# The stock run gets its own id in smoke mode, so it never collides with the wrapper's.
REAL_RUN_ID = make_run_id("gin", "source", 1.0, SEEDS[0])
STOCK_RUN_ID = REAL_RUN_ID if RUN_FULL else smoke_run_id(REAL_RUN_ID) + "_stock"
print("Stock run id:", STOCK_RUN_ID)
"""

SMOKE_CHAIN = r'''
def run_timed(label, command):
    """Run one step and print how long it took and on which GPU."""
    start = time.time()
    run(command)
    print(f"[{label}] {(time.time() - start) / 60:.1f} min on {GPU_NAME}", flush=True)


def smoke_chain(python, environment_label, seed, budget, fraction):
    """Source, then scratch and finetune at `fraction`, each as a smoke run."""
    label = shlex.quote(environment_label)
    for mode in ("source", "scratch", "finetune"):
        share = 1.0 if mode == "source" else fraction
        print(f"\n===== wrapper smoke run: {mode}, fraction {share} =====")
        run_timed(
            f"smoke {mode}",
            f"{python} -m src.run_experiment --model gin --mode {mode} --fraction {share} "
            f"--seed {seed} --step-budget-epochs {budget} --environment {label} --smoke-test",
        )
'''


def kaggle_cells() -> List[Dict[str, Any]]:
    """The Kaggle runner: smoke test now, reproduction run when RUN_FULL is on."""
    return [
        markdown("""
# Kaggle runner: smoke test, then (later) the reproduction run

Before running: **Accelerator = GPU (T4)**, **Internet = On**, and add the Kaggle dataset
`ealtman2019/ibm-transactions-for-anti-money-laundering-aml` as an input.

Every step stops the notebook if it fails, so later cells never run on a broken state.
This file is written by `python -m tools.make_notebooks`; change the generator, not this file.

Safe by default: `RUN_FULL = False` runs a smoke test:
1. stock Multi-GNN run, 3 epochs, scored and recorded (`..._smoke_stock`)
2. wrapper runs: source, then 10% scratch, then 10% fine-tune (`..._smoke`), each capped at 2 epochs.
   Smoke runs always keep a last-epoch checkpoint.

Set `RUN_FULL = True` only after the owner has approved the GPU time (AGENTS.md rule 6).
"""),
        code(RUN_HELPER + KAGGLE_SETUP + GPU_GATE),
        code(CLONE_REPO),
        code(KAGGLE_IDS),
        markdown(ENVIRONMENT_NOTE),
        code(ENVIRONMENT),
        markdown(
            "Format HI-Small and LI-Small with IBM's script and write the stock run's `data_config.json` (CPU only, slow)."
        ),
        code(r"""
run(f"{PYTHON} -m src.prepare --raw-dir {KAGGLE_DATA} --run-id {STOCK_RUN_ID}")
"""),
        markdown(
            "Stock Multi-GIN + EU on HI-Small (Multi-GNN's own `main.py`, unmodified)."
        ),
        code(r"""
EPOCHS = REPRODUCTION_EPOCHS if RUN_FULL else SMOKE_TEST_EPOCHS
FLAGS = " ".join(MULTI_GNN_ADAPTATION_FLAGS)
MAIN = f"{REPO}/external/Multi-GNN/main.py"

# Start from the run folder so Multi-GNN reads OUR data_config.json.
os.chdir(f"{WORK}/data/multignn_runs/{STOCK_RUN_ID}")
start = time.time()
run(f"{PYTHON} {MAIN} --data Small_HI --model gin {FLAGS} --seed {SEEDS[0]} --n_epochs {EPOCHS} --testing --save_model --unique_name")
RUNTIME_SECONDS = time.time() - start
os.chdir(REPO)
print(f"{EPOCHS} epochs took {RUNTIME_SECONDS / 60:.1f} min in total (includes data loading)")
"""),
        markdown("""
Score the stock run's checkpoint (threshold on validation, test scored once) and write its record.
Records land in `repo/results/runs/`; download them from the notebook output and commit them.
"""),
        code(r"""
smoke_flag = "" if RUN_FULL else "--smoke-test"
run(f"{PYTHON} -m src.record_stock_run --run-id {STOCK_RUN_ID} --model gin --dataset HI-Small --seed {SEEDS[0]} --epochs {EPOCHS} --runtime-seconds {RUNTIME_SECONDS:.0f} --environment {shlex.quote(ENVIRONMENT_LABEL)} {smoke_flag}")
run(f"cat results/runs/{STOCK_RUN_ID}.json")
"""),
        markdown("""
Wrapper smoke test (skipped when `RUN_FULL` is on): source, then 10% scratch, then 10% fine-tune
from that source. Each run is capped at 2 epochs, validates on the same schedule as real runs, and
writes a `_smoke` record. The chain stops at the first failure.
"""),
        code(
            SMOKE_CHAIN
            + r"""
if RUN_FULL:
    print("RUN_FULL is on: skipping the wrapper smoke runs")
else:
    smoke_chain(PYTHON, ENVIRONMENT_LABEL, SEEDS[0], SMOKE_STEP_BUDGET_EPOCHS, SMOKE_LABEL_FRACTION)
    run(f"{PYTHON} -m src.check_smoke")
"""
        ),
        markdown(
            "Per-epoch scores of the stock run. Pick the best **validation** epoch and read test from that epoch."
        ),
        code(r"""
run(f"grep -E 'Validation F1|Test F1' {WORK}/data/multignn_runs/{STOCK_RUN_ID}/logs/logs.log || true")
"""),
    ]


COLAB_SETUP = r"""
RUN_FULL = False  # False = smoke test only; True = the pilot (about 11 T4-hours: see the estimate cell)
WORK = "/content"
REPO = f"{WORK}/repo"

from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/aml-gnn-transfer"  # kaggle.json lives here; everything we keep is saved here
os.makedirs(DRIVE, exist_ok=True)

# Data, prepared graphs, models and records all live on Drive, so a disconnect loses nothing.
# The pilot's records and models are kept apart from the full design's (and from smoke runs').
SMOKE_STAMP = time.strftime("%Y%m%d-%H%M%S")
RESULTS_HOME = f"{DRIVE}/pilot" if RUN_FULL else f"{DRIVE}/smoke/{SMOKE_STAMP}"
os.environ["AML_DATA_ROOT"] = f"{DRIVE}/data"
os.environ["AML_GRAPH_CACHE"] = f"{DRIVE}/graph_cache"
os.environ["AML_MODEL_ROOT"] = f"{RESULTS_HOME}/models"
os.environ["AML_RUN_RECORDS_DIR"] = f"{RESULTS_HOME}/records"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
print("Records and models go to:", RESULTS_HOME)
"""

COLAB_DOWNLOAD = r"""
# The Kaggle token is read from Drive and copied where the Kaggle CLI looks. It is never printed.
TOKEN = f"{DRIVE}/kaggle.json"
if not os.path.exists(TOKEN):
    raise FileNotFoundError(f"Put your Kaggle API token (kaggle.json) in {DRIVE}/ first")
os.makedirs("/root/.kaggle", exist_ok=True)
run(f"cp {TOKEN} /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json")
run(f"{sys.executable} -m pip install -q kaggle")
run(f"{sys.executable} -m src.download_data")  # skips files already on Drive
run(f"{PYTHON} -m src.prepare --run-id prepare_only")  # IBM's formatter, skipped if already formatted
"""

COLAB_PILOT = r"""
from src.config import MODEL_ROOT, RUN_RECORDS_DIR
from src.pilot import pilot_runs, record_exists, set_aside_partial_run

run(f"{sys.executable} -m src.estimate_pilot")
runs = pilot_runs()
finished = sum(record_exists(item, RUN_RECORDS_DIR) for item in runs)
print(f"\n{finished} of {len(runs)} pilot runs already have a record; those are skipped.")

for number, item in enumerate(runs, start=1):
    if record_exists(item, RUN_RECORDS_DIR):
        print(f"[{number}/{len(runs)}] {item.run_id}: record exists, skipping")
        continue
    moved = set_aside_partial_run(item, MODEL_ROOT)  # a dead attempt's files, kept
    if moved:
        print(f"[{number}/{len(runs)}] earlier attempt of {item.run_id} moved to {moved}")
    words = " ".join(item.command_words())
    run_timed(
        f"{number}/{len(runs)} {item.run_id}",
        f"{PYTHON} -m src.run_experiment {words} --environment {shlex.quote(ENVIRONMENT_LABEL)}",
    )
"""


def colab_cells() -> List[Dict[str, Any]]:
    """The Colab runner: smoke test now, the pilot when RUN_FULL is on."""
    return [
        markdown("""
# Colab runner: smoke test, then the pilot

Before running: **Runtime > Change runtime type > GPU** (L4 or A100 if you have them; T4 works),
and put your Kaggle token as `aml-gnn-transfer/kaggle.json` in your Google Drive.
Colab does not run in the background: keep the tab open. If it disconnects, run all cells again:
finished runs have records on Drive and are skipped, an interrupted run's files are moved aside
(never deleted), and the data, prepared graphs and models are reused from Drive.

Every step stops the notebook if it fails. This file is written by `python -m tools.make_notebooks`.

`RUN_FULL = False` (default) runs the smoke test: source, then 10% scratch, then 10% fine-tune,
each capped at 2 epochs, then checks that scratch and finetune used the same labels and steps.
`RUN_FULL = True` runs the **pilot** (GIN only, one HI-Small source model, fractions 10/25/100%,
seeds 0 and 1, 3-epoch step budget). Turn it on only after the owner has approved the estimate
printed below (AGENTS.md rule 6).
"""),
        code(RUN_HELPER + COLAB_SETUP + GPU_GATE),
        code(CLONE_REPO),
        markdown(ENVIRONMENT_NOTE),
        code(ENVIRONMENT),
        markdown(
            "Download HI-Small and LI-Small from Kaggle (skipped if already on Drive) and format them with IBM's script (CPU only, slow the first time)."
        ),
        code(COLAB_DOWNLOAD),
        markdown(
            "The runs. The first run of each dataset builds its prepared graph (several minutes of CPU, GPU idle) and saves it to Drive; every later train and score step loads it."
        ),
        code(
            SMOKE_CHAIN
            + r"""
if RUN_FULL:
    print("RUN_FULL is on: skipping the smoke runs")
else:
    smoke_chain(PYTHON, ENVIRONMENT_LABEL, 0, 1, 0.10)
    run(f"{sys.executable} -m src.check_smoke")
"""
        ),
        markdown(
            "The pilot (only when `RUN_FULL` is on). Prints the estimate first, then one timed line per run."
        ),
        code(
            r"""
if not RUN_FULL:
    print("RUN_FULL is off: no pilot runs")
else:
"""
            + "\n".join(
                "    " + line if line else line
                for line in COLAB_PILOT.strip().splitlines()
            )
        ),
        markdown(
            "What the records say (our scored values). Copy the records folder into the repo's `results/pilot/` and commit."
        ),
        code(r"""
run(f"grep -H -E '\"(minority_f1|epochs_requested|gradient_steps|labels_kept|kept_labels_fingerprint)\"' {os.environ['AML_RUN_RECORDS_DIR']}/*.json || true")
"""),
    ]


def notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap cells in a notebook document with stable cell ids."""
    for number, cell in enumerate(cells):
        cell["id"] = "cell%02d" % number
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_text(cells: List[Dict[str, Any]]) -> str:
    """The notebook as the JSON text that is saved."""
    return json.dumps(notebook(cells), indent=1) + "\n"


def main() -> None:
    """Write both notebooks."""
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    (NOTEBOOK_DIR / "kaggle_runner.ipynb").write_text(notebook_text(kaggle_cells()))
    (NOTEBOOK_DIR / "colab_runner.ipynb").write_text(notebook_text(colab_cells()))


if __name__ == "__main__":
    main()
