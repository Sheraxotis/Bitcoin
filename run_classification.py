"""Config-driven classification with independent, validated model caches."""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from src.experiment import run_experiment

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/config_classification_15m_90days.json')
    parser.add_argument('--output-dir', type=Path, help='default: automatic experiment-ID directory')
    parser.add_argument('--interval', choices=['15m', '2h', '1d'], help='override bar interval')
    parser.add_argument('--models', nargs='+', help='run a subset of models configured in JSON')
    parser.add_argument('--force', action='store_true', help='retrain selected models despite valid caches')
    parser.add_argument('--import-legacy', type=Path, help='validate and import an existing classification result')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.interval:
        config['interval'] = args.interval
        config.pop('bar_interval', None)
        config.pop('source_interval', None)
    if args.models:
        missing = set(args.models)-config['models'].keys()
        if missing:
            parser.error(f'Models not configured: {sorted(missing)}')
        config['models'] = {name: config['models'][name] for name in args.models}
    with threadpool_limits(limits=1):
        run_experiment(config, ROOT, args.output_dir, args.force, args.import_legacy)


if __name__ == '__main__':
    main()
