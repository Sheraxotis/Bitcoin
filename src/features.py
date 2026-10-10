"""Causal feature registry; all models in an experiment share its selected columns."""
import inspect
import numpy as np
import pandas as pd

from src.prepare_data import ALL_FEATURES


def current_return(frame, step):
    return frame.close.pct_change(fill_method=None).where(frame.open_time.diff().eq(step))


def bars_per_day(step):
    step = pd.Timedelta(step)
    day = pd.Timedelta(days=1)
    if step <= pd.Timedelta(0) or day % step:
        raise ValueError('Feature interval must divide one UTC day')
    return int(day / step)


def continuous_history(frame, step, periods):
    """Require every transition in the requested trailing history."""
    return frame.open_time.diff().eq(step).rolling(periods, min_periods=periods).sum().eq(periods)


def return_lag1(frame, step):
    return current_return(frame, step).shift(1).where(continuous_history(frame, step, 2))


def return_7d(frame, step):
    periods = 7 * bars_per_day(step)
    return (frame.close / frame.close.shift(periods) - 1).where(continuous_history(frame, step, periods))


def ma_deviation(frame, step):
    periods = bars_per_day(step)
    values = frame.close / frame.close.rolling(periods).mean() - 1
    return values if periods == 1 else values.where(continuous_history(frame, step, periods - 1))


def volatility(frame, step):
    periods = max(2, bars_per_day(step))
    values = current_return(frame, step).rolling(periods).std(ddof=1)
    return values.where(continuous_history(frame, step, periods))


def price_range(frame, step):
    return (frame.high - frame.low) / frame.close


def body_ratio(frame, step):
    return (frame.close - frame.open).abs() / (frame.high - frame.low).replace(0, np.nan)


def upper_shadow_ratio(frame, step):
    return (frame.high - frame[['open', 'close']].max(axis=1)) / (frame.high - frame.low).replace(0, np.nan)


def lower_shadow_ratio(frame, step):
    return (frame[['open', 'close']].min(axis=1) - frame.low) / (frame.high - frame.low).replace(0, np.nan)


def volume_change(frame, step):
    values = frame.volume / frame.volume.shift(1).replace(0, np.nan) - 1
    return values.where(continuous_history(frame, step, 1))


def relative_volume(frame, step):
    periods = bars_per_day(step)
    denominator = frame.volume.shift(1).rolling(periods).mean().replace(0, np.nan)
    return (frame.volume / denominator - 1).where(continuous_history(frame, step, periods))


def taker_buy_ratio(frame, step):
    return frame.taker_buy_base_asset_volume / frame.volume.replace(0, np.nan)


FEATURE_REGISTRY = {
    'return': current_return,
    'return_lag1': return_lag1,
    'return_7d': return_7d,
    'ma_deviation': ma_deviation,
    'volatility': volatility,
    'price_range': price_range,
    'body_ratio': body_ratio,
    'upper_shadow_ratio': upper_shadow_ratio,
    'lower_shadow_ratio': lower_shadow_ratio,
    'volume_change': volume_change,
    'relative_volume': relative_volume,
    'taker_buy_ratio': taker_buy_ratio,
}
RAW_FEATURES = set(ALL_FEATURES) - {'return'}

# Include helper implementations for derived features without changing the
# identity of experiments selecting only the original return/raw features.
FEATURE_DEPENDENCIES = {
    'return_lag1': (current_return, continuous_history),
    'return_7d': (bars_per_day, continuous_history),
    'ma_deviation': (bars_per_day, continuous_history),
    'volatility': (bars_per_day, current_return, continuous_history),
    'volume_change': (continuous_history,),
    'relative_volume': (bars_per_day, continuous_history),
}


def feature_signature(columns):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Choose a nonempty, unique shared feature list')
    unknown = set(columns) - RAW_FEATURES - FEATURE_REGISTRY.keys()
    if unknown:
        raise ValueError(f'Unknown features: {sorted(unknown)}')
    return {name: (inspect.getsource(FEATURE_REGISTRY[name]) + ''.join(
                inspect.getsource(helper) for helper in FEATURE_DEPENDENCIES.get(name, ())))
            if name in FEATURE_REGISTRY else 'raw-v1'
            for name in columns}


def build_features(frame, columns, step):
    feature_signature(columns)
    frame = frame.copy()
    for name in columns:
        if name in FEATURE_REGISTRY:
            values = FEATURE_REGISTRY[name](frame, step)
            if not values.index.equals(frame.index):
                raise ValueError(f'Feature index mismatch: {name}')
            frame[name] = values
    # Shared mask: every model trains and predicts on exactly these rows.
    frame['valid_features'] = np.isfinite(frame[columns]).all(axis=1)
    frame['valid_sample'] = frame.valid_features & np.isfinite(frame.target) & frame.target_time.notna()
    return frame
