# AML GNN Transfer

Does pre-training a graph neural network on synthetic AML data reduce how many labelled laundering cases a bank needs?

**Status: planned. Nothing has been run yet.** This README will hold the results once the experiments are done.

Docs: [PRD](PRD.md) · [Architecture](ARCHITECTURE.md) · [Design system](DESIGN_SYSTEM.md) · [Agent instructions](AGENTS.md)

## The question

Banks have very few confirmed laundering cases to learn from. IBM's AML paper tested a fix: train a graph network on one synthetic dataset (HI-Small), then fine-tune it on another (LI-Small).

| Model (minority F1, %) | LI-Small |
| --- | --- |
| PNA trained on HI-Small only | 0.00 |
| PNA + 5 epochs fine-tuning on LI-Small | 27.38 |

Source: Altman et al. (2023), Table 3. The paper does not say how much LI-Small data the fine-tune used.

**This project measures that:** at 10%, 25%, 50% and 100% of LI-Small's training labels, how does a pre-trained-and-fine-tuned model compare with one trained from scratch?

## Plan

1. **Reproduce** IBM's results with their own code ([IBM/Multi-GNN](https://github.com/IBM/Multi-GNN)): Multi-GIN + EU and Multi-PNA + EU on HI-Small, then the transfer to LI-Small.
2. **Label scarcity:** hide the labels of 90%, 75%, 50% or 0% of LI-Small's training transactions (every transaction stays in the graph), and compare scratch vs fine-tuned. Three seeds each, 24 runs.
3. **Compare** with the LightGBM pipeline from [aml-ibm-experiment](https://github.com/coachuttam/aml-ibm-experiment) on LI-Small.
4. **Write up** the answer, with the chart and limits.

### How runs are compared: equal training steps

With 10% of the labels, an epoch has about a tenth of the gradient steps. Giving every fraction the same number of epochs would starve the small ones. So the training budget is counted in **steps**, not epochs:

- The budget is the steps in N epochs at 100% of the labels. N is `STEP_BUDGET_EPOCHS` in `src/config.py` (5, IBM's fine-tuning setting), or `--step-budget-epochs` on the command line.
- A smaller fraction runs proportionally more epochs, so every run gets about the same number of steps.

| Share of labels | 100% | 50% | 25% | 10% |
| --- | --- | --- | --- | --- |
| Epochs (N = 5) | 5 | 10 | 20 | 50 |

- Trained from scratch and pre-trained-then-fine-tuned runs at the same fraction get the same budget and the same kept labels. The only difference is the starting weights.
- Multi-GNN runs whole epochs, so step totals match to within one batch per epoch. Each run record stores its actual step count.
- The checkpoint is the epoch with the best validation F1. Test is not scored during training.
- Validation is run **the same number of times for every fraction** (as many passes as the budget has epochs, evenly spread, always ending on the last epoch), because each pass over LI-Small takes about three minutes. Epochs in between log "not scored" and cannot be picked as best. Multi-GNN never saves a checkpoint after epoch 0, so at 100% (where epoch 0 is itself a validation point) there is one fewer candidate epoch than at smaller fractions.
- Building the graph (reading the CSV, adding ports) is done once per dataset and cached (`src/graph_cache.py`); training and scoring reuse it.
- A limit to keep in mind: the same small budget may leave a model trained from scratch underfitted, so it can be a weak baseline.

Every run is scored with minority F1 (to compare with IBM) and with PR-AUC, precision in the top 100 and recall at an alert budget (to compare with Project 2). Thresholds and epochs are chosen on validation; test is scored once.

### The Colab pilot (a first, smaller pass)

Compute is limited, so the first results come from a **pilot**, not the full design. Every number from it is labelled "pilot".

| | Pilot | Deferred to v2 |
| --- | --- | --- |
| Model | GIN + EU | PNA + EU |
| Source model | one HI-Small model, seed 0 | one per seed |
| Label fractions | 10%, 25%, 100% | 50% |
| Seeds | 0, 1 | 2 |
| Step budget | 3 epochs at 100% | 5 (IBM's fine-tuning setting) |
| Runs | 1 source + 12 (scratch and finetune) | 24 |

Limits that come with it:
- If Colab disconnects, a run that had finished training but not scoring is scored from its saved checkpoint (a `training_complete.json` marker proves training finished; its record says `scoring_resumed`). A run interrupted mid-training is set aside and started again.
- Both seeds fine-tune from the **same** source model, so the finetune spread leaves out pre-training variance.
- Two seeds is a thin spread. A gap inside it is not a result.
- A 3-epoch budget may leave scratch under-trained, which would flatter pre-training.
- The source model's epochs (`PILOT_SOURCE_EPOCHS`) are the owner's call, made after reading `python -m src.estimate_pilot`.

Pilot records are kept in `results/pilot/`, apart from `results/runs/`, so v2 runs can never overwrite them. Run it with `notebooks/colab_runner.ipynb` (`RUN_FULL = True` only after approval).

## Results

Not run yet.

## Quick start

GNN training needs a GPU (Kaggle or Colab). The laptop handles results, charts and tests.

```bash
git clone --recurse-submodules https://github.com/coachuttam/aml-gnn-transfer.git
cd aml-gnn-transfer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

On Kaggle: open `notebooks/kaggle_runner.ipynb`, attach the [IBM AML dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), switch the GPU on, and run the cells.

On Colab: open `notebooks/colab_runner.ipynb`, put your Kaggle token at `aml-gnn-transfer/kaggle.json` in Drive, switch the GPU on, and run the cells. It starts as a smoke test; the notebooks are written by `python -m tools.make_notebooks`.

## Related projects

- [aml-alert-investigator](https://github.com/coachuttam/aml-alert-investigator): fine-tuned LLM copilot for AML alert triage.
- [aml-ibm-experiment](https://github.com/coachuttam/aml-ibm-experiment): LightGBM ranking vs rules on IBM HI-Small.

## Sources

- Altman, E. et al. (2023). [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models](https://arxiv.org/abs/2306.16424). NeurIPS Datasets and Benchmarks.
- Egressy, B. et al. (2024). [Provably Powerful Graph Neural Networks for Directed Multigraphs](https://arxiv.org/abs/2306.11586). AAAI.
- [IBM/Multi-GNN](https://github.com/IBM/Multi-GNN), Apache 2.0.

All data is synthetic (IBM). This is a research prototype, not a production system.
