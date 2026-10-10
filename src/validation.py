"""Time-split interface. Only fixed-length rolling validation is implemented."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class Split:
    cutoff: object
    train: object
    test: object
    metadata: dict


def rolling_splits(frame, config):
    start = pd.Timestamp(config['evaluation_start'])
    end = pd.Timestamp(config['evaluation_end'])
    update = pd.Timedelta(hours=config['update_hours'])
    window = pd.Timedelta(days=config['window_days'])
    for cutoff in pd.date_range(start, end, freq=update, inclusive='left'):
        train = frame.loc[frame.valid_sample & (frame.feature_time >= cutoff-window) &
                          (frame.feature_time < cutoff) & (frame.target_time <= cutoff)]
        test = frame.loc[frame.valid_sample & (frame.feature_time >= cutoff) &
                         (frame.feature_time < min(cutoff+update, end)) &
                         (frame.target_time >= start) & (frame.target_time < end)]
        if not len(train):
            raise ValueError(f'No valid training rows at {cutoff}')
        assert (train.target_time <= cutoff).all() and (test.feature_time >= cutoff).all()
        metadata = dict(train_cutoff=cutoff, window_start=cutoff-window, n_train=len(train), n_test=len(test),
                        train_feature_start=train.feature_time.min(), train_feature_end=train.feature_time.max(),
                        train_target_end=train.target_time.max(), train_up_rate=float((train.target>0).mean()))
        yield Split(cutoff, train.index, test.index, metadata)


VALIDATION_REGISTRY = {'rolling': rolling_splits}


def make_splits(frame, config):
    method = config.get('validation', 'rolling')
    if method not in VALIDATION_REGISTRY:
        raise ValueError(f'Validation {method!r} is not implemented; available: rolling')
    return list(VALIDATION_REGISTRY[method](frame, config))
