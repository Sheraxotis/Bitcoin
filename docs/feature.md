import numpy as np

O, H, L, C, V, B = (
    df["open"], df["high"], df["low"],
    df["close"], df["volume"],
    df["taker_buy_base_asset_volume"]
)

n = 96  # 示例：均线、波动率、相对成交量使用一天窗口
r = C.pct_change(fill_method=None)

df["return_lag1"] = r.shift(1)
df["return_7d"] = C / C.shift(672) - 1
df["ma_deviation"] = C / C.rolling(n).mean() - 1
df["volatility"] = r.rolling(n).std(ddof=1)
df["price_range"] = (H - L) / C

candle_range = (H - L).replace(0, np.nan)
df["body_ratio"] = (C - O).abs() / candle_range
df["upper_shadow_ratio"] = (
    H - df[["open", "close"]].max(axis=1)
) / candle_range
df["lower_shadow_ratio"] = (
    df[["open", "close"]].min(axis=1) - L
) / candle_range

df["volume_change"] = V / V.shift(1).replace(0, np.nan) - 1
df["relative_volume"] = (
    V / V.shift(1).rolling(n).mean().replace(0, np.nan) - 1
)
df["taker_buy_ratio"] = B / V.replace(0, np.nan)