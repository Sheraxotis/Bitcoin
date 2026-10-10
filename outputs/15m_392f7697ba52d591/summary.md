# 15m classification

UTC evaluation: [2025-04-01T00:00:00+00:00, 2026-09-01T00:00:00+00:00); 49727 shared rows.
Shared features: open, high, low, close, volume, return, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume.
Training window: 90 calendar days; refit every 24 hours.
Target: next-bar return > 0. Only labels known at the training cutoff are used.
Every model uses the same features, training splits and prediction rows.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 51.73% | 51.70% | 51.68% | 45.68% | 48.50% | 0.692535 | 0.249638 | 0.526353 |
| knn | 50.27% | 50.27% | 50.02% | 49.20% | 49.61% | 1.894910 | 0.299545 | 0.502946 |
| rf | 50.65% | 50.64% | 50.43% | 47.76% | 49.06% | 0.725745 | 0.264429 | 0.509825 |
| zero | 50.25% | 50.00% | 0.00% | 0.00% | 0.00% | 17.930860 | 0.497476 | 0.500000 |
| majority | 49.81% | 49.70% | 49.24% | 29.04% | 36.54% | 0.693191 | 0.250022 | 0.497671 |

Zero predicts non-up with probability 0; majority uses the current training up-rate.
Log loss clips exact zero probabilities; beating that baseline does not establish useful probability quality.
Accuracy, precision, recall and F1 are higher-is-better; log loss and Brier are lower-is-better.
ROC sweeps probability thresholds; AUC measures ranking, not probability calibration or trading profit.
AUC is undefined for a single actual class. Trading signals remain at the configured fixed threshold 0.5.
Probability quality is not trading profitability. This repeatedly inspected history is exploratory.
