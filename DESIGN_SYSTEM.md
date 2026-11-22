# Design system: AML GNN Transfer results

This project has no app in v1. Its outputs are tables, one main chart and a README, read by AML leads, hiring managers and engineers. They follow the same look as Project 2 (`aml-ibm-experiment`) so the projects read as one family.

## Principles

- **Lead with the answer.** Every chart title states the finding ("Pre-training halves the labels needed"), not the axes ("F1 vs fraction"). If the finding is negative, the title says that.
- **Show the spread.** Every point from several seeds shows its range (±1 standard deviation). A gap inside the range is not called a difference.
- **Plain words for systems.** Readers see "Trained from scratch" and "Pre-trained on HI-Small, then fine-tuned", not `scratch` and `finetune`.
- **Say what the data is.** Every chart and table carries a footnote: "IBM AML (synthetic), LI-Small test period."

## Names

| In code | In tables and charts |
| --- | --- |
| `scratch` | Trained from scratch |
| `finetune` | Pre-trained on HI-Small, then fine-tuned |
| `zero_shot` | Trained on HI-Small only |
| `gin` + flags | Multi-GIN + EU |
| `pna` + flags | Multi-PNA + EU |
| `lightgbm` | LightGBM (Project 2 features) |

## Colours

| Role | Colour |
| --- | --- |
| Pre-trained / fine-tuned (the idea being tested) | Blue `#1f5fa8` |
| Trained from scratch (the comparison) | Grey `#6b7280` |
| LightGBM / rules baseline | Red-orange `#c2410c` |
| Text | `#1f2937`; footnotes `#6b7280` |

## Main chart: label-scarcity curve

- x-axis: share of LI-Small training labels used (10%, 25%, 50%, 100%), evenly spaced.
- y-axis: minority F1 (%) from 0. A second version with PR-AUC goes in `results/`.
- Two lines with markers: blue (fine-tuned) and grey (scratch). Each point is the mean over seeds, with a whisker of ±1 standard deviation (no whisker with one seed, and the footnote says so). With two models run, one panel per model.
- Label the gap at 10% directly on the chart ("+X F1 points").
- A dashed horizontal line for IBM's reported fine-tuned value (27.38) if it falls in range, labelled "IBM, Table 3".
- Sizes: 1600×900 for the README, 1080×1350 for LinkedIn. White background, sans-serif, no chart-junk.

## Tables

- Systems as rows, metrics as columns, in this order: Minority F1, Precision, Recall, PR-AUC, Precision@100, Alerts.
- F1, precision and recall as percentages with one decimal (27.4%). PR-AUC with three decimals (0.312).
- Seed results as mean ± standard deviation (27.4 ± 1.1).
- IBM's published figures, where shown, sit in their own column headed "IBM paper" and are never mixed with ours.

## README results section

1. One-sentence answer to RQ2.
2. The chart.
3. The results table.
4. "What this does and doesn't show": two to four bullets, including limits (synthetic data, one dataset pair, three seeds).
