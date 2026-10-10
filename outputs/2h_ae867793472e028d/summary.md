# 2h classification

UTC evaluation: [2025-01-01T00:00:00+00:00, 2025-01-03T00:00:00+00:00); 23 shared rows.
Shared features: open, high, low, close, volume, return, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, return_lag1, return_7d, ma_deviation, volatility, price_range, body_ratio, upper_shadow_ratio, lower_shadow_ratio, volume_change, relative_volume, taker_buy_ratio.
Training window: 90 calendar days; refit every 24 hours.
Target: next-bar return > 0. Only labels known at the training cutoff are used.
Every model uses the same features, training splits and prediction rows.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_pca | 52.17% | 56.75% | 71.43% | 35.71% | 47.62% | 0.691568 | 0.249144 | 0.626984 |
| knn_pca | 39.13% | 40.08% | 50.00% | 35.71% | 41.67% | 2.328092 | 0.328696 | 0.436508 |
| zero | 39.13% | 50.00% | 0.00% | 0.00% | 0.00% | 21.939615 | 0.608696 | 0.500000 |
| majority | 60.87% | 50.00% | 60.87% | 100.00% | 75.68% | 0.681925 | 0.244395 | 0.527778 |

Zero predicts non-up with probability 0; majority uses the current training up-rate.
Log loss clips exact zero probabilities; beating that baseline does not establish useful probability quality.
Accuracy, precision, recall and F1 are higher-is-better; log loss and Brier are lower-is-better.
ROC sweeps probability thresholds; AUC measures ranking, not probability calibration or trading profit.
AUC is undefined for a single actual class. Trading signals remain at the configured fixed threshold 0.5.
Probability quality is not trading profitability. This repeatedly inspected history is exploratory.
