| System | Labels | Minority F1 | PR-AUC | Precision@100 | Recall@10k | Best epoch (of trained) | GPU hours | Colab units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trained on HI-Small (scored on HI-Small test) | HI-Small | 66.6% | 0.659 | 0.96 | 84.5% | 8 of 10 | 2.8 | 5.5 |
| Trained from scratch | 10% | 21.3% | 0.126 | 0.58 | 48.0% | 40 of 160 | 6.4 | 12.4 |
| Pre-trained on HI-Small, then fine-tuned | 10% | 32.7% | 0.266 | 0.87 | 59.2% | 10 of 160 | 6.4 | 12.4 |
| Trained from scratch | 25% | 20.0% | 0.151 | 0.68 | 48.6% | 44 of 64 | 6.4 | 12.3 |
| Pre-trained on HI-Small, then fine-tuned | 25% | 35.8% | 0.284 | 0.88 | 62.1% | 4 of 64 | 6.3 | 12.3 |
| Trained from scratch | 100% | 34.3% | 0.300 | 0.95 | 57.0% | 16 of 16 | 6.5 | 12.7 |
| Pre-trained on HI-Small, then fine-tuned | 100% | 36.7% | 0.321 | 0.92 | 55.1% | 12 of 16 | 6.4 | 12.3 |
| **Total** |  |  |  |  |  |  | 41.2 | 80.0 |
