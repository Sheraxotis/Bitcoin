# 2h classification

UTC evaluation: [2025-01-01T00:00:00+00:00, 2026-01-01T00:00:00+00:00); 4379 shared rows.
Shared features: open, high, low, close, volume, return, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, return_lag1, return_7d, ma_deviation, volatility, price_range, body_ratio, upper_shadow_ratio, lower_shadow_ratio, volume_change, relative_volume, taker_buy_ratio.
Training window: 90 calendar days; refit every 24 hours.
Target: next-bar return > 0. Only labels known at the training cutoff are used.
Every model uses the same features, training splits and prediction rows.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 52.52% | 52.12% | 52.98% | 67.14% | 59.23% | 0.700107 | 0.253000 | 0.526407 |
| knn | 49.21% | 49.15% | 50.55% | 51.45% | 50.99% | 1.908383 | 0.301256 | 0.494646 |
| rf | 50.88% | 50.84% | 52.17% | 52.29% | 52.23% | 0.719968 | 0.262079 | 0.509406 |
| logistic_pca | 53.23% | 52.81% | 53.49% | 68.43% | 60.05% | 0.696063 | 0.251171 | 0.531794 |
| knn_pca | 49.39% | 49.34% | 50.72% | 51.53% | 51.12% | 1.989044 | 0.303101 | 0.494657 |
| zero | 48.64% | 50.00% | 0.00% | 0.00% | 0.00% | 18.511573 | 0.513588 | 0.500000 |
| majority | 51.15% | 50.28% | 51.53% | 82.48% | 63.43% | 0.693027 | 0.249940 | 0.496508 |

Zero predicts non-up with probability 0; majority uses the current training up-rate.
Log loss clips exact zero probabilities; beating that baseline does not establish useful probability quality.
Accuracy, precision, recall and F1 are higher-is-better; log loss and Brier are lower-is-better.
ROC sweeps probability thresholds; AUC measures ranking, not probability calibration or trading profit.
AUC is undefined for a single actual class. Trading signals remain at the configured fixed threshold 0.5.
Probability quality is not trading profitability. This repeatedly inspected history is exploratory.
