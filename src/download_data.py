"""Official public archives, bounded requests and verified local cache."""
import hashlib
import io
import re
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m"


def archive_url(month, interval='15m'):
    if interval not in ('15m', '2h', '1d'):
        raise ValueError('Unsupported interval')
    return f"{BASE.rsplit('/', 1)[0]}/{interval}/BTCUSDT-{interval}-{month}.zip"


def verify(payload, checksum, name):
    lines = checksum.decode("utf-8-sig").strip().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Invalid CHECKSUM for {name}")
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", lines[0])
    if not match or match[2] != name:
        raise ValueError(f"CHECKSUM digest/filename invalid: {name}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != match[1].lower():
        raise ValueError(f"SHA256 mismatch: {name}; training stopped")
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError(f"Not a ZIP archive: {name}")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
        if not any(n.lower().endswith('.csv') for n in z.namelist()):
            raise ValueError(f"No CSV member: {name}")
    return digest


def download(config, raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    retry = Retry(total=config['retries'], backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'])
    paths, manifest = [], []
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=retry))
        for month in config['months']:
            url = archive_url(month, config['interval'])
            path = raw_dir / url.rsplit('/', 1)[-1]
            checkpath = Path(str(path) + '.CHECKSUM')
            cached = path.exists() and checkpath.exists()
            if cached:
                payload, checksum = path.read_bytes(), checkpath.read_bytes()
            else:
                responses = []
                for endpoint in (url, url + '.CHECKSUM'):
                    print(f"Downloading {endpoint}", flush=True)
                    response = session.get(endpoint, timeout=(10, config['timeout_seconds']))
                    response.raise_for_status()
                    if 'html' in response.headers.get('Content-Type', '').lower():
                        raise ValueError(f"Unexpected HTML: {endpoint}")
                    responses.append(response.content)
                payload, checksum = responses
            digest = verify(payload, checksum, path.name)
            if not cached:
                # Files only enter the cache after complete integrity verification.
                for destination, content in ((path, payload), (checkpath, checksum)):
                    temporary = Path(str(destination) + '.part')
                    temporary.write_bytes(content)
                    temporary.replace(destination)
            print(f"Verified {path.name} ({'cache' if cached else 'download'})", flush=True)
            paths.append((month, path))
            manifest.append(dict(month=month, url=url, sha256=digest, cached=cached))
    return paths, manifest
