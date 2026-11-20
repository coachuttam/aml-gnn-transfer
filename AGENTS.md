# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, Codex) working in this repo. Read PRD.md for what the project must answer and ARCHITECTURE.md for how it fits together.

## Setup and checks

```bash
git submodule update --init                 # brings in external/Multi-GNN at the pinned commit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt             # light: metrics, results, charts, tests, LightGBM
python -m pytest -q                         # must stay green
python -m src.collect                       # rebuild results tables from results/runs/
python -m src.plot                          # rebuild the chart
```

GNN training runs on Kaggle or Colab with a GPU, not the laptop. Use Multi-GNN's own environment there (`external/Multi-GNN/env.yml`). Heavy libraries (`torch`, `torch_geometric`) never go in `requirements.txt`.

## Hard rules

1. **Never edit `external/Multi-GNN`.** It is IBM's published code, pinned to one commit. `git -C external/Multi-GNN diff` must always be empty. Wrap it from `src/`; if a change is truly unavoidable, put it in `patches/` with a reason and stop to tell the owner first.
2. **Never tune on the test period.** Epochs, thresholds and any other choice use validation only. Test is scored once per run.
3. **Mask training labels only.** Label masking removes labels from LI-Small training transactions. Every transaction stays in the graph. Validation and test labels are never masked.
4. **Same labels for both setups.** At a given fraction and seed, `scratch` and `finetune` must use the identical kept-label set. Smaller fractions are subsets of larger ones.
5. **One record per run, never overwritten.** Every run writes `results/runs/<run_id>.json` with config, seed, both git commits, metrics, epochs used and runtime. Tables and charts are built from these files, never typed by hand.
6. **Ask before spending.** Ask the owner before any run expected to exceed 3 GPU hours, and before anything that costs money (Colab units).
7. **Report honestly.** If pre-training does not help, or we cannot reproduce IBM's numbers, the README says so plainly. Show seed spread; never call a gap inside the spread a win.
8. **No data in git.** Kaggle CSVs, formatted files and model weights stay out of the repo. Only small result files (`results/`) are committed.

## Readable code

The owner will read, run and explain every file, so write for a reader, not for cleverness.

- Plain, descriptive names (`keep_label_positions`, not `klp`). No abbreviations a newcomer would have to guess.
- One job per function, about 20 lines or fewer.
- A short docstring on every module and public function, in plain English, with a tiny example where it helps.
- Comments explain **why**, not what.
- No magic numbers. Seeds, fractions, epochs and alert budgets are named constants in `src/config.py`, each with a one-line reason.
- Simple loops and clear `if` statements over dense one-liners.
- Type hints and dataclasses for anything passed between modules.
- `ruff format` and `ruff check` clean.
- Test names read as sentences: `test_ten_percent_labels_are_inside_the_twenty_five_percent_set`.
- Keep each file under about 300 lines.

## Conventions

- Python 3.9+ compatible (the owner's Mac may run 3.9; Kaggle and Colab are newer): start files with `from __future__ import annotations`, use `typing.List` and `Optional`, avoid `match` statements and `X | Y` unions at runtime.
- Metric definitions must match Project 2 (`aml-ibm-experiment/src/ibm/train.py`), so results are comparable.
- Run ids read as sentences of their settings: `pna_finetune_f0.25_s1`.
- Charts and tables follow DESIGN_SYSTEM.md.
- Keep Claude Code prompts and notebook cells short.

## Definition of done

Tests pass, `git -C external/Multi-GNN diff` is empty, every new result has a run record, tables and chart rebuilt from records, README updated with numbers and limits.
