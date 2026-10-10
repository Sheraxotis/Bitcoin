"""Small hand-built checks only; never used as experiment data."""
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.download_data import verify
from src.prepare_data import COLUMNS, STEP, add_features, prepare, read_archive
from src.rolling_train import metrics, rolling_train


def archive(path, times, prices, header=False):
    rows = []
    for time, price in zip(times, prices):
        unit = 1000 if time.year >= 2025 else 1000000
        rows.append([time.value // unit, price, price, price, price, 1,
                     (time + STEP).value // unit - 1, price, 1, 0, 0, 0])
    content = pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False, header=header)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('../not_extracted.csv', content)


class PipelineChecks(unittest.TestCase):
    def test_daily_bars_three_calendar_months(self):
        times = pd.date_range('2025-01-01', '2025-07-01', freq='D', inclusive='left', tz='UTC')
        prices = 100 + np.arange(len(times)) * .1
        frame = add_features(pd.DataFrame(dict(open_time=times, open=prices, high=prices,
                             low=prices, close=prices, volume=1.0)), pd.Timedelta(days=1))
        self.assertAlmostEqual(frame.loc[1, 'return'], 100.1/100-1)
        self.assertEqual(frame.loc[1, 'feature_time'], pd.Timestamp('2025-01-03T00:00Z'))
        self.assertEqual(frame.loc[1, 'target_time'], pd.Timestamp('2025-01-04T00:00Z'))
        config = dict(interval='1d', window_months=3, months=['2025-01','2025-02','2025-03','2025-04','2025-05','2025-06'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            out = Path(directory)
            rolling_train(frame, config, out)
            updates = pd.read_csv(out / 'training_updates.csv')
            may31 = updates.loc[updates.train_cutoff.str.startswith('2025-05-31')].iloc[0]
            self.assertEqual(pd.Timestamp(may31.window_start), pd.Timestamp('2025-02-28T00:00Z'))
            self.assertEqual(updates.n_train.iloc[0], 88)
            pred = pd.read_csv(out / 'predictions.csv')
            self.assertEqual(len(pred), 90)
            self.assertTrue(((pd.to_datetime(pred.target_time) - pd.to_datetime(pred.feature_time)) == pd.Timedelta(days=1)).all())

    def test_hand_returns_times_and_gap(self):
        frame = pd.DataFrame(dict(open_time=pd.to_datetime([
            '2026-07-01T00:00Z', '2026-07-01T00:15Z',
            '2026-07-01T00:30Z', '2026-07-01T01:00Z']),
            open=[100, 110, 99, 120], high=[100, 110, 99, 120],
            low=[100, 110, 99, 120], close=[100, 110, 99, 120], volume=[1]*4))
        result = add_features(frame)
        self.assertAlmostEqual(result.loc[1, 'return'], .1)
        self.assertAlmostEqual(result.loc[1, 'target'], -.1)
        self.assertEqual(result.loc[1, 'feature_time'], pd.Timestamp('2026-07-01T00:30Z'))
        self.assertEqual(result.loc[1, 'target_time'], pd.Timestamp('2026-07-01T00:45Z'))
        self.assertTrue(pd.isna(result.loc[0, 'return']))
        self.assertTrue(pd.isna(result.loc[2, 'target']))
        self.assertTrue(pd.isna(result.loc[3, 'return']))
        self.assertEqual(result.valid_sample.sum(), 1)

    def test_archive_units_header_checksum_and_duplicates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for month, header in [('2026-07', False), ('2024-07', True)]:
                path = Path(directory) / 'example.zip'
                times = pd.date_range(month + '-01', periods=3, freq='15min', tz='UTC')
                archive(path, times, [100, 110, 99], header)
                frame = read_archive(month, path)
                self.assertEqual(len(frame), 3)
                self.assertEqual(frame.open_time.iloc[0], times[0])
                payload = path.read_bytes()
                checksum = f'{hashlib.sha256(payload).hexdigest()}  example.zip'.encode()
                verify(payload, checksum, 'example.zip')
                with self.assertRaises(ValueError):
                    verify(payload + b'corrupt', checksum, 'example.zip')
                with self.assertRaises(ValueError):
                    read_archive('2026-08', path)
                duplicate, audit = prepare([(month, path), (month, path)])
                self.assertEqual(audit['duplicate_rows'], 3)
                self.assertEqual(len(duplicate), 3)
                archive(path, times, [100, 110, -1])
                with self.assertRaises(ValueError):
                    prepare([(month, path)])

    def test_daily_cutoff_window_and_paired_metrics(self):
        # Deliberately artificial, isolated in a temporary directory.
        times = pd.date_range('2026-07-01', '2026-09-01', freq='15min', inclusive='left', tz='UTC')
        prices = 100 + np.arange(len(times)) * .001
        frame = pd.DataFrame(dict(open_time=times, open=prices, high=prices,
                                  low=prices, close=prices, volume=1.0))
        frame = add_features(frame)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            out = Path(directory)
            overall = rolling_train(frame, {'months':['2026-07', '2026-08']}, out)
            pred = pd.read_csv(out / 'predictions.csv')
            daily = pd.read_csv(out / 'daily_metrics.csv')
            self.assertEqual(overall['n'], 2975)
            self.assertEqual(len(daily), 31)
            self.assertEqual(daily.n_train.iloc[0], 2974)
            self.assertEqual(daily.n_train.iloc[1], 2976)
            self.assertFalse(pred.feature_time.duplicated().any())
            self.assertEqual(pd.to_datetime(pred.target_time, utc=True).max(), pd.Timestamp('2026-08-31T23:45Z'))
            self.assertEqual(daily.n.sum(), len(pred))
            self.assertAlmostEqual(metrics(pred)['rmse_model'], overall['rmse_model'])
            coefficients = pd.read_csv(out / 'daily_coefficients.csv')
            cutoff = pd.Timestamp('2026-08-01T00:00Z')
            selected = frame.loc[frame.valid_sample & (frame.feature_time < cutoff) & (frame.target_time <= cutoff)]
            self.assertAlmostEqual(coefficients.mean_close.iloc[0], selected.close.mean())
        zeros = pd.DataFrame(dict(actual_return=[0.], predicted_return=[0.], baseline_return=[0.]))
        self.assertTrue(np.isnan(metrics(zeros)['rmse_improvement']))


if __name__ == '__main__':
    unittest.main()
