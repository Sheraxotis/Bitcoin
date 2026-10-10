# 2h classification

UTC evaluation: [2025-04-01T00:00:00+00:00, 2025-04-03T00:00:00+00:00); 23 shared rows.
Shared features: open, high, low, close, volume, return, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume.
Training window: 90 calendar days; refit every 24 hours.
Target: next-bar return > 0. Only labels known at the training cutoff are used.
Every model uses the same features, training splits and prediction rows.

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Log loss | Brier | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 52.17% | 50.00% | 52.17% | 100.00% | 68.57% | 0.684618 | 0.245750 | 0.674242 |
| knn | 52.17% | 51.89% | 53.85% | 58.33% | 56.00% | 0.675147 | 0.245217 | 0.583333 |
| rf | 56.52% | 57.20% | 62.50% | 41.67% | 50.00% | 0.668972 | 0.238757 | 0.632576 |
| svm | 52.17% | 52.27% | 54.55% | 50.00% | 52.17% | 0.691805 | 0.249328 | 0.537879 |
| fnn | 52.17% | 51.14% | 52.94% | 75.00% | 62.07% | 0.674627 | 0.240872 | 0.575758 |
| zero | 47.83% | 50.00% | 0.00% | 0.00% | 0.00% | 18.805384 | 0.521739 | 0.500000 |
| majority | 52.17% | 50.00% | 52.17% | 100.00% | 68.57% | 0.692331 | 0.249592 | 0.435606 |

Zero predicts non-up with probability 0; majority uses the current training up-rate.
Log loss clips exact zero probabilities; beating that baseline does not establish useful probability quality.
Accuracy, precision, recall and F1 are higher-is-better; log loss and Brier are lower-is-better.
ROC sweeps probability thresholds; AUC measures ranking, not probability calibration or trading profit.
AUC is undefined for a single actual class. Trading signals remain at the configured fixed threshold 0.5.
Probability quality is not trading profitability. This repeatedly inspected history is exploratory.
