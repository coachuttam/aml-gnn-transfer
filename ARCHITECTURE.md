# Architecture: AML GNN Transfer

How the pieces fit. Read PRD.md for what the project must answer and AGENTS.md for the rules.

## Overview

```
Kaggle: HI-Small_Trans.csv, LI-Small_Trans.csv
        │
        ▼
src/prepare.py ──► external/Multi-GNN/format_kaggle_files.py ──► formatted CSVs + data_config.json
        │
        ▼
src/run_experiment.py  (one run = one model, one mode, one fraction, one seed)
        │   ├─ src/label_mask.py    hides a share of training labels (graph unchanged)
        │   └─ external/Multi-GNN   trains / fine-tunes / scores (code never edited)
        ▼
src/metrics.py ──► results/runs/<run_id>.json   (config, commits, metrics, runtime)
        │
        ▼
src/collect.py ──► results/reproduce.csv, results/label_curve.csv
        │
        ▼
src/plot.py ──► results/label_curve.png

src/lightgbm_baseline.py ──► Project 2's features on LI-Small ──► results/runs/*.json
```

## Folders

```
aml-gnn-transfer/
├─ README.md  PRD.md  ARCHITECTURE.md  DESIGN_SYSTEM.md  AGENTS.md
├─ external/Multi-GNN/        git submodule, pinned commit, never edited
├─ src/
│   ├─ config.py              paths, seeds, fractions, epochs, budgets (named constants)
│   ├─ prepare.py             runs IBM's formatter, writes data_config.json
│   ├─ label_mask.py          picks which training labels to hide, reproducibly
│   ├─ run_experiment.py      command line: --model --mode --fraction --seed
│   ├─ metrics.py             minority F1, PR-AUC, precision@100, recall at budget
│   ├─ records.py             writes and reads one JSON record per run
│   ├─ collect.py             builds the results tables from the records
│   ├─ plot.py                the label-scarcity chart
│   └─ lightgbm_baseline.py   Project 2's pipeline pointed at LI-Small
├─ notebooks/
│   └─ kaggle_runner.ipynb    thin wrapper: install, prepare, call run_experiment
├─ results/
│   ├─ runs/                  one JSON per run (committed; small)
│   ├─ reproduce.csv  label_curve.csv  label_curve.png
├─ tests/
└─ requirements.txt  requirements-train.txt  pytest.ini  .gitignore
```

## Run modes

| Mode | Trained on | Scored on | Answers |
| --- | --- | --- | --- |
| `source` | HI-Small (100% labels) | HI-Small test | RQ1: does our setup match IBM? |
| `zero_shot` | HI-Small (saved model) | LI-Small test | RQ1: IBM's 0.00 row |
| `finetune` | HI-Small, then LI-Small at fraction *f* | LI-Small test | RQ1 (f = 1.0) and RQ2 |
| `scratch` | LI-Small at fraction *f* | LI-Small test | RQ1 (f = 1.0) and RQ2 |

The HI-Small model is trained once per model type and seed, saved, and reused by every `zero_shot` and `finetune` run with that seed.

## Key design decisions

1. **IBM's code is a submodule, never edited.** Anyone can check that our results come from their published code. Our changes live in `src/`.
2. **Mask labels, keep the graph.** A bank with few labels still has all its transactions. So masking removes training labels only; every transaction stays in the graph as context. Validation and test labels are never masked.
3. **Masking is reproducible.** `label_mask.py` picks the kept labels from the seed and the fraction alone, and keeps smaller fractions inside larger ones (the 10% set is part of the 25% set). Curves are then comparable point to point.
4. **Same masks for both setups.** At a given fraction and seed, `scratch` and `finetune` see exactly the same labels. The only difference is the starting weights.
5. **Choices on validation only.** Early stopping and the alert threshold use validation. Test is scored once per run.
6. **One record per run.** Every run writes a JSON file with its config, both git commits (ours and Multi-GNN's), metrics, epochs used and runtime. Tables and charts are rebuilt from these records, never typed by hand.
7. **Equal steps, not equal epochs.** The training budget is the steps in N epochs at fraction 1.0; smaller fractions run proportionally more epochs (`src/step_budget.py`). Scratch and finetune share it at each fraction.
8. **Metrics match Project 2.** The same definitions (`metrics.py` follows Project 2's `train.py`), so GNN and LightGBM results sit in one table.

## Where label masking hooks in (to confirm in week 1)

Multi-GNN selects training transactions inside its own data loading and training code. The plan is to wrap that step from `run_experiment.py` at run time (replace the training index list with the masked one) without changing their files. If that turns out impossible, the fallback is one small patch file in `patches/`, applied by `prepare.py` and described in the README. Either way, the submodule's own files stay as published.

## Compute

GPU runs happen on Kaggle (about 30 free GPU hours a week, 12 hours per session) or Colab. The laptop runs `metrics.py`, `collect.py`, `plot.py`, the tests and the LightGBM baseline. About 31 runs in total, an estimated 15–45 GPU hours (to be re-estimated after the first PNA run).
