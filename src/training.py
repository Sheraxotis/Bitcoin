"""Train one independent model on shared causal splits."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from src.models import build_model


def predict_split(frame, columns, split, name, params):
    train, test = frame.loc[split.train], frame.loc[split.test]
    if test.empty:
        return np.empty(0)
    y = (train.target > 0).astype(int)
    if y.nunique() == 1:
        return np.full(len(test), float(y.iloc[0]))
    model = build_model(name, params).fit(train[columns], y)
    classes = list(model.classes_)
    return model.predict_proba(test[columns])[:, classes.index(1)] if 1 in classes else np.zeros(len(test))


def prediction_base(frame, splits):
    rows = []
    for split in splits:
        test = frame.loc[split.test]
        if len(test):
            rows.append(pd.DataFrame(dict(feature_time=test.feature_time, target_time=test.target_time,
                        train_cutoff=split.cutoff, actual_return=test.target, actual_class=(test.target>0).astype(int),
                        zero_class=0, zero_probability=0.,
                        majority_class=int(split.metadata['train_up_rate']>.5),
                        majority_probability=split.metadata['train_up_rate'])))
    if not rows:
        raise ValueError('No valid prediction rows in the evaluation range')
    base = pd.concat(rows, ignore_index=True)
    assert not base.feature_time.duplicated().any()
    return base


def train_model(frame, columns, splits, base, name, params, parallel_days=1):
    probabilities = []
    with ThreadPoolExecutor(max_workers=parallel_days) as pool:
        jobs = pool.map(lambda split: predict_split(frame, columns, split, name, params), splits)
        for index, probability in enumerate(jobs):
            probabilities.append(probability)
            if index % 30 == 0:
                print(f'{name}: completed {index+1}/{len(splits)} updates', flush=True)
    result = base[['feature_time', 'target_time', 'train_cutoff', 'actual_return', 'actual_class']].copy()
    result['probability'] = np.concatenate(probabilities)
    result['predicted_class'] = (result.probability > .5).astype(int)
    return result
