"""Config normalization, experiment identity and independently reusable model results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import numpy as np
import pandas as pd
from src.data import load_data
from src.evaluation import write_reports
from src.features import build_features, feature_signature
from src.models import model_signature
from src.prepare_data import ALL_FEATURES, INTERVALS, month_start
from src.training import prediction_base, predict_split, train_model
from src.validation import make_splits
from src.metadata import atomic_json, read_run, update_run


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_config(config):
    config = dict(config)
    interval = config.pop('bar_interval', config.get('interval', '15m'))
    interval = {'15min': '15m'}.get(interval, interval)
    if interval not in INTERVALS:
        raise ValueError('Supported bar intervals: 15m, 2h, 1d')
    if config.get('symbol', 'BTCUSDT') != 'BTCUSDT' or config.get('market', 'spot') != 'spot':
        raise ValueError('This project supports BTCUSDT spot only')
    if config.get('horizon_bars', 1) != 1:
        raise ValueError('Only next-bar labels are implemented')
    config['interval'] = interval
    config['source_interval'] = config.get('source_interval', interval)
    for key, default in dict(symbol='BTCUSDT', market='spot', window_days=90, update_hours=24,
                             timeout_seconds=30, retries=2, parallel_days=1, validation='rolling').items():
        config.setdefault(key, default)
    config.setdefault('feature_columns', list(ALL_FEATURES))
    for key in ('window_days', 'update_hours', 'parallel_days'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if pd.Timedelta(hours=config['update_hours']) % INTERVALS[interval]:
        raise ValueError('Retraining interval must be a multiple of the bar interval')
    feature_signature(config['feature_columns'])
    months = config.get('months', [])
    if not months:
        raise ValueError('Specify consecutive archive months')
    starts = [month_start(m) for m in months]
    if any(b != a+pd.offsets.MonthBegin(1) for a, b in zip(starts, starts[1:])):
        raise ValueError('Archive months must be consecutive and ordered')
    earliest = starts[0]+pd.Timedelta(days=config['window_days'])
    latest = starts[-1]+pd.offsets.MonthBegin(1)
    start = pd.Timestamp(config.get('evaluation_start', earliest))
    end = pd.Timestamp(config.get('evaluation_end', latest))
    start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
    end = end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    if not earliest <= start < end <= latest or start != start.floor(INTERVALS[interval]):
        raise ValueError('Evaluation dates must follow initial history, stay in archive range and align to bars')
    config['evaluation_start'], config['evaluation_end'] = start.isoformat(), end.isoformat()
    if not config.get('models'):
        raise ValueError('Select at least one model')
    for name, params in config['models'].items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not isinstance(params, dict):
            raise ValueError('Model names must be safe identifiers and parameters must be objects')
        model_signature(name, params)
    return config


EXPERIMENT_FIELDS = ('symbol', 'market', 'months', 'interval', 'source_interval', 'window_days',
                     'update_hours', 'validation', 'feature_columns', 'evaluation_start', 'evaluation_end')


def experiment_identity(frame, config, manifest, versions):
    columns = ['feature_time', 'target_time', 'target', 'valid_sample']+config['feature_columns']
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()
    source_dir = Path(__file__).parent
    implementation = {name: file_digest(source_dir/name) for name in
                      ('data.py', 'prepare_data.py', 'validation.py', 'training.py')}
    return digest(dict(settings={key: config[key] for key in EXPERIMENT_FIELDS},
                       archives=[{key: item[key] for key in ('month', 'url', 'sha256')} for item in manifest],
                       features=feature_signature(config['feature_columns']), data=data_hash,
                       implementation=implementation, versions=versions, format_version=1))


def read_predictions(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    for col in ('feature_time', 'target_time', 'train_cutoff'):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def check_predictions(pred, base):
    columns = ['feature_time', 'target_time', 'train_cutoff', 'actual_class']
    pd.testing.assert_frame_equal(pred[columns].reset_index(drop=True), base[columns].reset_index(drop=True))
    np.testing.assert_allclose(pred.actual_return, base.actual_return, rtol=0, atol=1e-14)
    if not pred.probability.between(0, 1).all() or not (pred.predicted_class == (pred.probability>.5)).all():
        raise ValueError('Invalid cached probabilities or direction labels')


def load_cached(folder, identity, base):
    try:
        status = read_run(folder.parent.parent).get('model_cache', {}).get(folder.name)
        if status is None:
            status = json.loads((folder/'run_status.json').read_text(encoding='utf-8'))
        path = folder/'predictions.csv'
        if status['status'] != 'success' or status['identity'] != identity or status['sha256'] != file_digest(path):
            return None
        pred = read_predictions(path)
        check_predictions(pred, base)
        return pred
    except (OSError, ValueError, KeyError, AssertionError):
        return None


def save_cached(folder, identity, pred, origin):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'predictions.csv'
    temporary = folder/'predictions.csv.part'
    pred.to_csv(temporary, index=False)
    temporary.replace(path)
    out = folder.parent.parent
    document = read_run(out)
    cache = document.get('model_cache', {})
    cache[folder.name] = dict(status='success', identity=identity, sha256=file_digest(path), n=len(pred), origin=origin)
    update_run(out, model_cache=cache)


def read_updates(path):
    frame = pd.read_csv(path, float_precision='round_trip')
    for col in ('train_cutoff', 'window_start', 'train_feature_start', 'train_feature_end', 'train_target_end'):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def legacy_predictions(directory, config, manifest, versions, base, splits):
    directory = Path(directory)
    document = read_run(directory)
    old = normalize_config(document['config'])
    if any(old[key] != config[key] for key in EXPERIMENT_FIELDS):
        raise ValueError('Legacy import has different shared experiment settings')
    if document['status']['status'] != 'success':
        raise ValueError('Legacy experiment was not successful')
    old_versions = old.get('dependency_versions')
    if old_versions is None:
        old_versions = json.loads((directory/'dependency_versions.json').read_text())
    if any(old_versions.get(key) != value for key, value in versions.items()):
        raise ValueError('Legacy dependency versions differ; rerun rather than import')
    old_manifest = document['download_manifest']
    if [(r['month'], r['sha256']) for r in old_manifest] != [(r['month'], r['sha256']) for r in manifest]:
        raise ValueError('Legacy raw archive hashes differ')
    pred = read_predictions(directory/'predictions.csv')
    pd.testing.assert_frame_equal(pred[list(base.columns)].reset_index(drop=True), base.reset_index(drop=True),
                                  check_exact=False, rtol=0, atol=1e-14)
    updates = read_updates(directory/'training_updates.csv')
    expected = pd.DataFrame([s.metadata for s in splits])
    pd.testing.assert_frame_equal(updates[list(expected.columns)], expected, check_exact=False, rtol=0, atol=1e-14)
    return old, pred


def run_experiment(config, root, output_dir=None, force=False, import_legacy=None):
    config = normalize_config(config)
    versions = dict(python=platform.python_version(), **{p: importlib.metadata.version(p) for p in
                    ('pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib')})
    frame, audit, manifest = load_data(config, root/'data/raw')
    frame = build_features(frame, config['feature_columns'], INTERVALS[config['interval']])
    audit['valid_samples'] = int(frame.valid_sample.sum())
    splits = make_splits(frame, config)
    base = prediction_base(frame, splits)
    experiment_id = experiment_identity(frame, config, manifest, versions)
    out = Path(output_dir) if output_dir else root/'outputs'/f"{config['interval']}_{experiment_id[:16]}"
    out = out.resolve()
    if (out/'run.json').exists() or (out/'run_config.json').exists():
        previous = read_run(out)['config']
        if previous.get('experiment_id') != experiment_id:
            raise ValueError('Output directory belongs to a different experiment; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    update_run(out, config={**config, 'experiment_id': experiment_id, 'dependency_versions': versions},
               status=dict(status='running', experiment_id=experiment_id))
    try:
        update_run(out, download_manifest=manifest, data_audit=audit)
        print(f'Experiment {experiment_id[:16]}: {len(base)} predictions; first training window '
              f'{splits[0].metadata["n_train"]} samples; output {out}', flush=True)
        legacy = legacy_predictions(import_legacy, config, manifest, versions, base, splits) if import_legacy else None
        result = base.copy()
        actions = {}
        for name, params in config['models'].items():
            identity = digest(dict(experiment=experiment_id, model=model_signature(name, params)))
            folder = out/'models'/f'{name}_{identity[:16]}'
            pred = None if force else load_cached(folder, identity, base)
            action = 'reused' if pred is not None else 'trained'
            if pred is None and not force and legacy and legacy[0]['models'].get(name) == params:
                old = legacy[1]
                if name+'_probability' in old:
                    pred = base[['feature_time', 'target_time', 'train_cutoff', 'actual_return', 'actual_class']].copy()
                    pred['probability'], pred['predicted_class'] = old[name+'_probability'], old[name+'_class']
                    first = next(s for s in splits if len(s.test))
                    check = predict_split(frame, config['feature_columns'], first, name, params)
                    np.testing.assert_allclose(check, pred.probability.iloc[:len(first.test)], rtol=0, atol=1e-12)
                    check_predictions(pred, base)
                    action = 'imported'
            if pred is None:
                pred = train_model(frame, config['feature_columns'], splits, base, name, params, config['parallel_days'])
            check_predictions(pred, base)
            if action != 'reused':
                save_cached(folder, identity, pred, action)
            result[name+'_probability'], result[name+'_class'] = pred.probability, pred.predicted_class
            actions[name] = action
            print(f'{name}: {action}', flush=True)
        result.to_csv(out/'predictions.csv', index=False)
        pd.DataFrame([s.metadata for s in splits]).to_csv(out/'training_updates.csv', index=False)
        metrics = write_reports(result, list(config['models'])+['zero', 'majority'], config, out)
        update_run(out, status=dict(status='success', experiment_id=experiment_id, n=len(result), models=actions))
        print(metrics.to_string(index=False))
        return out
    except Exception as exc:
        update_run(out, status=dict(status='failed', experiment_id=experiment_id, error=str(exc)))
        raise
