"""Daily fixed-window OLS; all evaluation uses identical paired rows."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .prepare_data import FEATURES, ALL_FEATURES, month_start


def build_pipeline(config):
    name = config.get('model', 'linear')
    params = config.get('model_params', {})
    if name == 'linear':
        estimator = LinearRegression(fit_intercept=True, **params)
    elif name == 'knn':
        estimator = KNeighborsRegressor(**params)
    elif name == 'rf':
        estimator = RandomForestRegressor(**params)
    else:
        raise ValueError(f'Unknown model: {name}')
    return make_pipeline(StandardScaler(), estimator)


def metrics(rows):
    if rows.empty:
        return dict(n=0, mae_model=np.nan, rmse_model=np.nan,
                    mae_baseline=np.nan, rmse_baseline=np.nan, rmse_improvement=np.nan)
    model = rows.predicted_return - rows.actual_return
    baseline = rows.baseline_return - rows.actual_return
    rm, rb = np.sqrt(np.mean(model ** 2)), np.sqrt(np.mean(baseline ** 2))
    return dict(n=len(rows), mae_model=np.mean(abs(model)), rmse_model=rm,
                mae_baseline=np.mean(abs(baseline)), rmse_baseline=rb,
                rmse_improvement=1 - rm / rb if rb != 0 else np.nan)


def bounds(rows, column, prefix):
    return {prefix + '_start': rows[column].min(), prefix + '_end': rows[column].max()}


def rolling_train(df, config, out):
    features = config.get('feature_columns', FEATURES)
    if not features or len(features) != len(set(features)) or not set(features) <= set(ALL_FEATURES):
        raise ValueError('Invalid feature_columns; only market fields and return allowed')
    df = df.copy()
    df['valid_features'] = np.isfinite(df[features]).all(axis=1)
    df['valid_sample'] = df.valid_features & np.isfinite(df.target) & df.target_time.notna()
    is_daily_bar = config.get('interval', '15m') == '1d'
    window_months = config.get('window_months', 3)
    start = month_start(config['months'][window_months if is_daily_bar else 1])
    lookback = pd.DateOffset(months=window_months) if is_daily_bar else pd.Timedelta(days=31)
    bar_step = pd.Timedelta(days=1) if is_daily_bar else pd.Timedelta(minutes=15)
    end = month_start(config['months'][-1]) + pd.offsets.MonthBegin(1)
    predictions, coefficients, days, skipped = [], [], [], []
    update_hours = config.get('update_hours', 24)
    step = pd.Timedelta(hours=update_hours)
    feature_times = pd.DatetimeIndex(df.feature_time)
    assert feature_times.is_monotonic_increasing
    for day in pd.date_range(start, end, freq=step, inclusive='left'):
        window = df.iloc[feature_times.searchsorted(day - lookback):feature_times.searchsorted(day)]
        train = window.loc[window.valid_sample & (window.target_time <= day)]
        if train.empty:
            raise ValueError(f'No valid training samples at {day}')
        assert (train.target_time <= day).all()
        assert (train.feature_time < day).all()
        assert (train.feature_time >= day - lookback).all()
        pipeline = build_pipeline(config)
        pipeline.fit(train[features], train.target)
        candidate = df.iloc[feature_times.searchsorted(day):feature_times.searchsorted(day + step)]
        for row in candidate.loc[~candidate.valid_sample].itertuples():
            reason = 'invalid_input' if not row.valid_features else 'missing_or_noncontiguous_target'
            skipped.append(dict(feature_time=row.feature_time, train_cutoff=day, reason=reason))
        test = candidate.loc[candidate.valid_sample]
        assert (test.feature_time >= day).all()
        result = pd.DataFrame(dict(feature_time=test.feature_time, target_time=test.target_time,
                                   train_cutoff=day, actual_return=test.target,
                                   predicted_return=pipeline.predict(test[features]) if len(test) else [],
                                   baseline_return=0.0))
        eligible = (result.target_time >= start) & (result.target_time < end)
        for row in result.loc[~eligible].itertuples():
            skipped.append(dict(feature_time=row.feature_time, train_cutoff=day,
                                reason='target_outside_evaluation_month'))
        evaluated = result.loc[eligible]
        predictions.append(evaluated)
        days.append(dict(prediction_date=day.date().isoformat(), train_cutoff=day,
                         window_start=day - lookback, n_train=len(train),
                         n_candidate=len(candidate), n_predictions=len(test),
                         n_skipped_invalid=len(candidate)-len(test),
                         n_excluded_target_month=int((~eligible).sum()),
                         **bounds(train, 'feature_time', 'train_feature'),
                         **bounds(train, 'target_time', 'train_target'),
                         **bounds(test, 'feature_time', 'prediction_feature'),
                         **bounds(evaluated, 'target_time', 'evaluation_target'),
                         **metrics(evaluated)))
        if config.get('model', 'linear') != 'linear':
            continue
        model = pipeline.named_steps['linearregression']
        scaler = pipeline.named_steps['standardscaler']
        coefficients.append(dict(train_cutoff=day, intercept=model.intercept_,
                                 coefficient_unit='decimal return per one training standard deviation',
                                 intercept_unit='decimal return at training feature means',
                                 **{f'coef_{f}': c for f, c in zip(features, model.coef_)},
                                 **{f'mean_{f}': v for f, v in zip(features, scaler.mean_)},
                                 **{f'scale_{f}': v for f, v in zip(features, scaler.scale_)}))
    pred = pd.concat(predictions, ignore_index=True).sort_values('feature_time')
    if pred.empty:
        raise ValueError('No eligible out-of-sample predictions')
    assert not pred.feature_time.duplicated().any()
    assert not pred.target_time.duplicated().any()
    assert (pred.feature_time >= pred.train_cutoff).all()
    assert (pred.target_time == pred.feature_time + bar_step).all()
    assert np.isfinite(pred[['actual_return', 'predicted_return', 'baseline_return']]).all().all()
    overall = metrics(pred)
    assert sum(d['n'] for d in days) == overall['n']
    pred.to_csv(out / 'predictions.csv', index=False)
    updates = pd.DataFrame(days)
    updates.to_csv(out / 'training_updates.csv', index=False)
    if update_hours == 24:
        updates.to_csv(out / 'daily_metrics.csv', index=False)
    else:
        pd.DataFrame([dict(prediction_date=date, **metrics(rows)) for date, rows in
                      pred.groupby(pred.feature_time.dt.strftime('%Y-%m-%d'))]).to_csv(out / 'daily_metrics.csv', index=False)
    pd.DataFrame([overall]).to_csv(out / 'overall_metrics.csv', index=False)
    if coefficients:
        pd.DataFrame(coefficients).to_csv(out / ('daily_coefficients.csv' if update_hours == 24 else 'hourly_coefficients.csv'), index=False)
    pd.DataFrame(skipped, columns=['feature_time', 'train_cutoff', 'reason']).to_csv(out / 'skipped_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, column in [(config.get('model', 'linear').upper() + ' model', 'predicted_return'), ('Zero baseline', 'baseline_return')]:
        ax.plot(pred.target_time, ((pred[column] - pred.actual_return) ** 2).cumsum(), label=name)
    ax.set(xlabel='Target time (UTC)', ylabel='Cumulative squared error (decimal return squared)',
           title='BTCUSDT spot: next daily return' if is_daily_bar else 'BTCUSDT spot: next 15-minute return')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d', tz=start.tz))
    ax.tick_params(axis='x', rotation=25)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out / 'cumulative_squared_error.png', dpi=160)
    plt.close(fig)
    return overall
