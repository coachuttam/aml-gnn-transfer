| System | Labels | Seeds | Epochs (best/run) | Minority F1 | PR-AUC | Precision@100 | Recall@25k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trained from scratch | 10% | 1 | 60/80 | 15.7% | 0.072 | 35.0% | 46.4% |
| Pre-trained on HI-Small, then fine-tuned | 10% | 1 | 40/80 | 22.8% | 0.165 | 76.0% | 59.4% |
| Trained from scratch | 25% | 1 | 24/32 | 16.1% | 0.095 | 44.0% | 56.4% |
| Pre-trained on HI-Small, then fine-tuned | 25% | 1 | 4/32 | 26.6% | 0.276 | 92.0% | 69.0% |
| Trained on HI-Small (scored on HI-Small test) | 100% | 1 | 9/10 | 60.5% | 0.571 | 91.0% | 84.2% |
| Trained from scratch | 100% | 1 | 7/8 | 20.5% | 0.210 | 83.0% | 66.8% |
| Pre-trained on HI-Small, then fine-tuned | 100% | 1 | 8/8 | 33.1% | 0.280 | 90.0% | 74.1% |
