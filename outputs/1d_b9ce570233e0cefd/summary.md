# 1d classification

UTC evaluation: [2025-04-01T00:00:00+00:00, 2026-09-01T00:00:00+00:00); 517 shared rows.
Shared features: open, high, low, close, volume, return, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume.
Training window: 90 calendar days; refit every 24 hours.
Target: next-bar return > 0. Only labels known at the training cutoff are used.
Every model uses the same features, training splits and prediction rows.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 44.29% | 44.28% | 44.78% | 46.15% | 45.45% | 0.752403 | 0.275142 | 0.434511 |
| knn | 50.48% | 50.50% | 50.82% | 47.69% | 49.21% | 1.378251 | 0.293462 | 0.493931 |
| rf | 47.39% | 47.39% | 47.64% | 46.54% | 47.08% | 0.810097 | 0.297888 | 0.470563 |
| zero | 49.71% | 50.00% | 0.00% | 0.00% | 0.00% | 18.126402 | 0.502901 | 0.500000 |
| majority | 46.62% | 46.69% | 45.83% | 33.85% | 38.94% | 0.701624 | 0.254218 | 0.455275 |

Zero predicts non-up with probability 0; majority uses the current training up-rate.
Log loss clips exact zero probabilities; beating that baseline does not establish useful probability quality.
Accuracy, precision, recall and F1 are higher-is-better; log loss and Brier are lower-is-better.
ROC sweeps probability thresholds; AUC measures ranking, not probability calibration or trading profit.
AUC is undefined for a single actual class. Trading signals remain at the configured fixed threshold 0.5.
Probability quality is not trading profitability. This repeatedly inspected history is exploratory.
