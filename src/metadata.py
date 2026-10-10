"""One run.json per experiment, including model caches and backtest settings."""
import json
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_run(directory):
    directory = Path(directory)
    path = directory/'run.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    # Read-only compatibility for old result directories not yet migrated.
    result = {'model_cache': {}, 'backtests': {}}
    for key, filename in [('config', 'run_config.json'), ('status', 'run_status.json'),
                          ('download_manifest', 'download_manifest.json'), ('data_audit', 'data_audit.json'),
                          ('execution_settings', 'execution_settings.json')]:
        path = directory/filename
        if path.exists():
            result[key] = json.loads(path.read_text(encoding='utf-8'))
    if 'config' not in result:
        raise FileNotFoundError(f'No experiment metadata at {directory}')
    return result


def update_run(directory, **sections):
    directory = Path(directory)
    result = read_run(directory) if (directory/'run.json').exists() or (directory/'run_config.json').exists() else {}
    result.update(sections)
    atomic_json(directory/'run.json', result)
    return result
