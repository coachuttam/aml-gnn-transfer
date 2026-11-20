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

Every run is scored with minority F1 (to compare with IBM) and with PR-AUC, precision in the top 100 and recall at an alert budget (to compare with Project 2). Thresholds and epochs are chosen on validation; test is scored once.

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

## Related projects

- [aml-alert-investigator](https://github.com/coachuttam/aml-alert-investigator): fine-tuned LLM copilot for AML alert triage.
- [aml-ibm-experiment](https://github.com/coachuttam/aml-ibm-experiment): LightGBM ranking vs rules on IBM HI-Small.

## Sources

- Altman, E. et al. (2023). [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models](https://arxiv.org/abs/2306.16424). NeurIPS Datasets and Benchmarks.
- Egressy, B. et al. (2024). [Provably Powerful Graph Neural Networks for Directed Multigraphs](https://arxiv.org/abs/2306.11586). AAAI.
- [IBM/Multi-GNN](https://github.com/IBM/Multi-GNN), Apache 2.0.

All data is synthetic (IBM). This is a research prototype, not a production system.
