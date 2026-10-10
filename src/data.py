"""Native verified archives by default; UTC aggregation only when requested."""
import pandas as pd
from src.download_data import download
from src.prepare_data import COLUMNS, INTERVALS, add_features, gap_ranges, prepare


def aggregate_bars(frame, interval):
    step = INTERVALS[interval]
    source_step = INTERVALS['15m']
    count = int(step / source_step)
    raw = frame[COLUMNS].set_index('open_time')
    grouped = raw.resample(step, origin='start_day', closed='left', label='left')
    sums = ['volume', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    aggregated = grouped.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                              'close_time': 'last', 'ignore': 'last', **{name: 'sum' for name in sums}})
    # Missing constituents invalidate the whole bar; never forward-fill or shorten it.
    aggregated = aggregated.loc[grouped.size().eq(count)].reset_index()
    if aggregated.empty:
        raise ValueError(f'No complete {interval} bars')
    return add_features(aggregated, step)


def load_data(config, raw_dir):
    source = config.get('source_interval', config['interval'])
    if source not in INTERVALS or (source != config['interval'] and source != '15m'):
        raise ValueError('Use native archives or explicitly aggregate 15m to 2h/1d')
    paths, manifest = download({**config, 'interval': source}, raw_dir)
    frame, audit = prepare(paths, source)
    if source != config['interval']:
        frame = aggregate_bars(frame, config['interval'])
        step = INTERVALS[config['interval']]
        expected = pd.date_range(audit['requested_start'], audit['requested_end_exclusive'],
                                 freq=step, inclusive='left')
        missing = expected.difference(pd.DatetimeIndex(frame.open_time))
        audit = {**audit, 'source_interval': source, 'source_rows': audit['actual_rows'],
                 'interval': config['interval'], 'actual_rows': len(frame), 'expected_rows': len(expected),
                 'actual_start': frame.open_time.min().isoformat(),
                 'actual_last_open': frame.open_time.max().isoformat(),
                 'missing_bars': len(missing), 'missing_intervals': gap_ranges(missing, step),
                 'valid_samples': int(frame.valid_sample.sum())}
    return frame, audit, manifest
