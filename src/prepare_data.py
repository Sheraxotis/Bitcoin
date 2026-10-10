"""Read CSV members without extraction; preserve raw fields and audit gaps."""
import csv
import io
import zipfile

import numpy as np
import pandas as pd

COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
           'close_time', 'quote_asset_volume', 'number_of_trades',
           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'return']
ALL_FEATURES = FEATURES + ['quote_asset_volume', 'number_of_trades',
                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
STEP = pd.Timedelta(minutes=15)
INTERVALS = {'15m': STEP, '2h': pd.Timedelta(hours=2), '1d': pd.Timedelta(days=1)}


def month_start(month):
    return pd.Timestamp(month + '-01', tz='UTC')


def read_archive(month, path, step=STEP):
    frames = []
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            if member.is_dir() or not member.filename.lower().endswith('.csv'):
                continue
            with z.open(member) as stream:
                rows = list(csv.reader(io.TextIOWrapper(stream, encoding='utf-8-sig')))
            if not rows or any(len(row) != len(COLUMNS) for row in rows):
                raise ValueError(f"Empty CSV or incorrect field count: {member.filename}")
            # Header recognition is explicit; numeric first data rows are never removed.
            normalized = [s.strip().lower().replace(' ', '_') for s in rows[0]]
            if normalized[0] == 'open_time':
                if normalized != COLUMNS:
                    raise ValueError(f"Unrecognized header: {member.filename}")
                rows = rows[1:]
            frame = pd.DataFrame(rows, columns=COLUMNS)
            if frame.empty:
                raise ValueError('CSV contains no data rows')
            for col in COLUMNS:
                frame[col] = pd.to_numeric(frame[col], errors='raise')
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError('Nonfinite raw numeric data')
            unit = 'us' if month >= '2025-01' else 'ms'
            for col in ('open_time', 'close_time'):
                if (frame[col] % 1 != 0).any():
                    raise ValueError('Noninteger timestamp')
                frame[col] = pd.to_datetime(frame[col], unit=unit, utc=True)
            start = month_start(month)
            end = start + pd.offsets.MonthBegin(1)
            if not frame.open_time.between(start, end, inclusive='left').all():
                raise ValueError(f"Timestamp unit/month mismatch: {month}")
            if not ((frame.close_time >= frame.open_time) &
                    (frame.close_time < frame.open_time + step)).all():
                raise ValueError('close_time outside its candle')
            frames.append(frame)
    if not frames:
        raise ValueError('No CSV members')
    return pd.concat(frames, ignore_index=True)


def add_features(df, step=STEP):
    df = df.copy()
    df['bar_end'] = df.open_time + step
    df['feature_time'] = df.bar_end
    previous_contiguous = df.open_time.diff().eq(step)
    next_contiguous = (df.open_time.shift(-1) - df.open_time).eq(step)
    df['return'] = df.close.pct_change(fill_method=None).where(previous_contiguous)
    df['target'] = (df.close.shift(-1) / df.close - 1).where(next_contiguous)
    df['target_time'] = df.bar_end.shift(-1).where(next_contiguous)
    df['valid_features'] = np.isfinite(df[FEATURES]).all(axis=1)
    df['valid_sample'] = df.valid_features & np.isfinite(df.target) & df.target_time.notna()
    return df


def gap_ranges(missing, step=STEP):
    ranges = []
    for time in missing:
        if ranges and time == pd.Timestamp(ranges[-1]['end']) + step:
            ranges[-1]['end'] = time.isoformat()
            ranges[-1]['bars'] += 1
        else:
            ranges.append(dict(start=time.isoformat(), end=time.isoformat(), bars=1))
    return ranges


def prepare(paths, interval='15m'):
    if interval not in INTERVALS:
        raise ValueError('Unsupported interval')
    step = INTERVALS[interval]
    df = pd.concat([read_archive(month, path, step) for month, path in paths], ignore_index=True)
    raw_rows = len(df)
    df = df.drop_duplicates()
    duplicate_rows = raw_rows - len(df)
    if df.open_time.duplicated().any():
        raise ValueError('Conflicting rows with identical open_time')
    df = df.sort_values('open_time').reset_index(drop=True)
    invalid = (~(df[['open', 'high', 'low', 'close']] > 0).all(axis=1) |
               (df.volume < 0) | (df.high < df[['open', 'close']].max(axis=1)) |
               (df.low > df[['open', 'close']].min(axis=1)) | (df.high < df.low))
    if invalid.any():
        raise ValueError(f"Invalid OHLCV rows: {int(invalid.sum())}; training stopped")
    start = month_start(paths[0][0])
    end = month_start(paths[-1][0]) + pd.offsets.MonthBegin(1)
    expected = pd.date_range(start, end, freq=step, inclusive='left')
    if not df.open_time.isin(expected).all():
        raise ValueError(f'open_time not aligned to UTC {interval} grid')
    missing = expected.difference(pd.DatetimeIndex(df.open_time))
    df = add_features(df, step)
    audit = dict(symbol='BTCUSDT', market='spot', interval=interval, price_unit='USDT/BTC',
                 volume_unit='BTC', timezone='UTC', requested_start=start.isoformat(),
                 requested_end_exclusive=end.isoformat(), actual_start=df.open_time.min().isoformat(),
                 actual_last_open=df.open_time.max().isoformat(), raw_rows=raw_rows,
                 actual_rows=len(df), expected_rows=len(expected), duplicate_rows=duplicate_rows,
                 invalid_rows=0, missing_bars=len(missing), missing_intervals=gap_ranges(missing, step),
                 valid_samples=int(df.valid_sample.sum()))
    return df, audit
