# PRD: AML GNN Transfer

Status: planned (nothing run yet) · Owner: Uttam · Last updated: 20 Nov 2026

## 1. Problem

Banks have very few labelled laundering cases. Most laundering is never detected, and the cases that are confirmed take months. A model trained from scratch on one bank's labels starts with almost nothing to learn from.

IBM's AML paper suggests a way round this: pre-train a model on synthetic data, then fine-tune it on a small amount of real data. Their first test (Altman et al. 2023, Table 3) shows the idea can work. A graph network trained on HI-Small scored 0.00 minority F1 on LI-Small, and 27.38 after 5 epochs of fine-tuning on LI-Small. The paper does not say how much LI-Small data that fine-tune used, and that is the number a bank would need to know.

## 2. Users

| User | Need |
| --- | --- |
| AML data-science lead at a bank or vendor | To know whether starting from a synthetic-data model saves labelling effort, and by how much |
| Reviewer (hiring manager, CTO, technical peer) | Evidence that the owner can reproduce a published result, extend it honestly and explain it |
| Owner | A deep-learning project that completes the set: classic ML (Project 2), generative AI (Projects 1 and 3), graph deep learning (this one) |

## 3. Research questions

- **RQ1 (reproduce):** Do we get close to IBM's published numbers with their code? Targets: Multi-GIN + EU 64.79 and Multi-PNA + EU 68.16 minority F1 on HI-Small; PNA 0.00 → 27.38 on LI-Small after fine-tuning.
- **RQ2 (main):** At 10%, 25%, 50% and 100% of LI-Small's training labels, how does a model fine-tuned from HI-Small compare with one trained from scratch?
- **RQ3 (context):** How do the GNNs compare with the LightGBM pipeline from Project 2 on LI-Small, both trained from scratch and trained on HI-Small?

## 4. Goals and non-goals

Goals
- Reproduce IBM's baseline and transfer results within a few F1 points.
- Measure the label-scarcity curve with three seeds per point.
- Report every result with the same metrics as Project 2, plus minority F1 for comparison with the paper.
- Publish the code, results and one clear chart; write it up as LinkedIn Article 5.

Non-goals (v1)
- Real bank data. Everything is IBM's synthetic data.
- New GNN architectures. We use IBM's Multi-GNN code unchanged.
- Medium or Large datasets. Small only, so the work fits free GPU time.
- A user interface. Results are tables and a chart; an app tab is a possible v2.
- Production claims. This is a research prototype.

## 5. Functional requirements

| ID | Requirement | Acceptance criteria |
| --- | --- | --- |
| FR1 | Data preparation | HI-Small and LI-Small formatted with Multi-GNN's `format_kaggle_files.py`; `data_config.json` written by our script, not by hand |
| FR2 | Untouched upstream code | Multi-GNN is a git submodule pinned to one commit; `git diff` inside it is always empty |
| FR3 | Reproduction runs | Multi-GIN + EU and Multi-PNA + EU on HI-Small; PNA trained on HI-Small, tested on LI-Small (zero-shot), fine-tuned on LI-Small, and trained on LI-Small from scratch |
| FR4 | Label masking | Our wrapper hides the labels of a chosen share of LI-Small training transactions. All transactions stay in the graph. Validation and test labels are never masked |
| FR5 | Label-scarcity runs | 4 fractions (0.10, 0.25, 0.50, 1.00) × 2 setups (scratch, fine-tuned) × 3 seeds (0, 1, 2) = 24 runs |
| FR6 | Metrics | Every run reports minority F1, precision, recall, PR-AUC, precision in the top 100, recall at fixed alert budgets, and the threshold (chosen on validation) |
| FR7 | Run records | One JSON file per run: config, seed, both git commit hashes, metrics, epochs used, runtime. Never overwritten |
| FR8 | Results tables | `results/reproduce.csv` and `results/label_curve.csv` built from the run records by a script |
| FR9 | Chart | Minority F1 (and PR-AUC) against label fraction: two lines (scratch, fine-tuned) with seed spread |
| FR10 | LightGBM comparison | Project 2's feature pipeline run on LI-Small, from scratch and trained on HI-Small, scored the same way |
| FR11 | Write-up | README results section, limitations, and a LinkedIn article draft |

## 6. Evaluation protocol

- **Split:** 60/20/20 by time, as in Multi-GNN. The graph for validation includes training and validation transactions; the graph for test includes all, and only test transactions are scored.
- **Choices on validation only:** epochs (early stopping), threshold (best F1 on validation). Test is scored once per run.
- **Fair comparison:** scratch and fine-tuned runs at the same fraction use the same masked labels and the same seed.
- **Spread:** report mean and standard deviation over 3 seeds. A gap smaller than the spread is not called a win.

## 7. Success criteria

- RQ1: our numbers within about 5 F1 points of IBM's, or a written explanation of the gap.
- RQ2: all 24 runs complete, chart published, and a plain-English answer to "how many labels does pre-training save?", whether the answer is positive or not.
- RQ3: LightGBM and GNN results in one table with identical metrics.

## 8. Milestones

| Week | Milestone |
| --- | --- |
| 1 | Setup done; reproduction runs done (RQ1) |
| 2 | 24 label-scarcity runs done (RQ2) |
| 3 | LightGBM comparison (RQ3), README results, chart, article draft |

## 9. Risks

| Risk | Response |
| --- | --- |
| Can't reproduce IBM's numbers | Stop before RQ2; check flags, split and epochs; document the gap |
| PNA too slow on a free GPU | Run RQ2 with Multi-GIN + EU and say so |
| Label masking needs changes inside Multi-GNN | Patch at run time from our wrapper; if impossible, keep one documented patch file and say so in the README |
| Results swing between seeds at 10% | Report the spread; add seeds only if budget allows |
| Kaggle quota runs out | Colab compute units, within a $50 cap |

## 10. Open questions

- How many LI-Small transactions did IBM fine-tune on? (Not stated in the paper.)
- Why does the paper list 3.6K laundering transactions in HI-Small when the Kaggle file has 5,177? Use what IBM's script produces and note the difference.
- Where exactly does Multi-GNN choose the training transactions? (Find this in week 1; FR4 depends on it.)
