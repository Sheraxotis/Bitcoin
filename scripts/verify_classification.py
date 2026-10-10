"""Verify shared rows, known-label cutoffs, metrics and independent first-split fits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import load_data
from src.evaluation import scores
from src.experiment import normalize_config, read_predictions, read_updates
from src.features import build_features
from src.prepare_data import INTERVALS
from src.training import prediction_base, predict_split
from src.validation import make_splits
from src.metadata import read_run

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/1d_b9ce570233e0cefd')
    out = parser.parse_args().output_dir.resolve()
    config = normalize_config(read_run(out)['config'])
    frame = read_predictions(out/'predictions.csv')
    data, _, _ = load_data(config, ROOT/'data/raw')
    data = build_features(data, config['feature_columns'], INTERVALS[config['interval']])
    splits = make_splits(data, config)
    base = prediction_base(data, splits)
    pd.testing.assert_frame_equal(frame[list(base.columns)], base, check_exact=False, rtol=0, atol=1e-14)
    pd.testing.assert_frame_equal(read_updates(out/'training_updates.csv'), pd.DataFrame([s.metadata for s in splits]),
                                  check_exact=False, rtol=0, atol=1e-14)
    assert (frame.target_time-frame.feature_time).eq(INTERVALS[config['interval']]).all()
    result = pd.read_csv(out/'overall_metrics.csv').set_index('model')
    for name in list(config['models'])+['zero','majority']:
        probability = frame[name+'_probability']
        assert probability.between(0,1).all()
        assert (frame[name+'_class'] == (probability>.5).astype(int)).all()
        for key, value in scores(frame.actual_class, frame[name+'_class'], probability).items():
            if key in result.columns:
                np.testing.assert_allclose(value, result.loc[name,key], atol=1e-12)
    first = next(s for s in splits if len(s.test))
    for name, params in config['models'].items():
        probability = predict_split(data, config['feature_columns'], first, name, params)
        np.testing.assert_allclose(probability, frame[name+'_probability'].iloc[:len(first.test)], rtol=0, atol=1e-12)
    print(f'Classification checks passed; {len(frame)} shared rows and independent first-split probabilities verified.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
