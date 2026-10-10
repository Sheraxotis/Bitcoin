"""Run the real-data pipeline from the project root, or any working directory."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import traceback

import pandas as pd
from src.download_data import archive_url, download
from src.prepare_data import FEATURES, ALL_FEATURES, month_start, prepare
from src.rolling_train import rolling_train, build_pipeline
from src.metadata import update_run

ROOT = Path(__file__).resolve().parent


def validate_config(config):
    build_pipeline(config)
    features = config.get('feature_columns', FEATURES)
    if features not in (FEATURES, ALL_FEATURES):
        raise ValueError('Supported features: default six or all ten market inputs')
    if config.get('update_hours', 24) not in (1, 24):
        raise ValueError('update_hours must be 1 or 24')
    is_daily_bar = config.get('interval') == '1d'
    if config.get('interval') not in ('15m', '1d'):
        raise ValueError('interval must be 15m or 1d')
    if is_daily_bar and (config.get('window_months') not in (3, 12) or config.get('update_hours', 24) != 24):
        raise ValueError('Daily bars require 3 or 12 calendar months and daily updates')
    fixed = [('symbol', 'BTCUSDT'), ('market', 'spot')]
    if not is_daily_bar:
        fixed.append(('window_days', 31))
    for key, value in fixed:
        if config.get(key) != value:
            raise ValueError(f'This prototype requires {key}={value}')
    months = config['months']
    if is_daily_bar and len(months) <= config['window_months']:
        raise ValueError('Daily experiment requires history longer than its training window')
    if len(months) < 2 or any(len(m) != 7 or month_start(m).strftime('%Y-%m') != m for m in months):
        raise ValueError('At least two YYYY-MM months required')
    if any(month_start(a) + pd.offsets.MonthBegin(1) != month_start(b) for a, b in zip(months, months[1:])):
        raise ValueError('Months must be consecutive and chronological')
    if not 0 <= config['retries'] <= 5 or not 0 < config['timeout_seconds'] <= 120:
        raise ValueError('Retries must be 0..5 and timeout 0..120 seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--processed-file', type=Path, default=ROOT / 'data/processed/btcusdt_15m.csv')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = 'configuration'
    config = {}
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        validate_config(config)
        versions = {'python': platform.python_version(), **{
            package: importlib.metadata.version(package)
            for package in ['pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib']}}
        update_run(out, config={**config, 'dependency_versions': versions}, status={'status': 'running'})
        stage = 'download and checksum validation'
        paths, manifest = download(config, ROOT / 'data' / 'raw')
        update_run(out, download_manifest=manifest)
        stage = 'cleaning'
        df, audit = prepare(paths, config['interval'])
        args.processed_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.processed_file, index=False)
        update_run(out, data_audit=audit)
        stage = 'rolling training and evaluation'
        overall = rolling_train(df, config, out)
        lookback_text = f"{config['window_months']} 个日历月 [T-{config['window_months']}个月,T)" if config['interval'] == '1d' else '31 日 [T-31天,T)'
        horizon_text = '日线（下一天）' if config['interval'] == '1d' else '15 分钟'
        evaluation_start = month_start(config['months'][config['window_months'] if config['interval'] == '1d' else 1])
        comparison = '模型 RMSE 低于零收益基准' if overall['rmse_model'] < overall['rmse_baseline'] else '模型未超过零收益基准（RMSE）'
        improvement = f"{overall['rmse_improvement']:.2%}" if pd.notna(overall['rmse_improvement']) else '不可计算（基准 RMSE 为零）'
        model_text = {'linear': '含截距 LinearRegression', 'knn': 'KNeighborsRegressor', 'rf': 'RandomForestRegressor'}[config.get('model', 'linear')]
        coefficient_text = ('线性模型系数文件的单位为每一训练标准差对应的小数收益率，截距为训练均值处的小数收益率。'
                            if config.get('model', 'linear') == 'linear' else '本模型没有线性回归截距/系数，不输出线性系数文件。')
        text = f'''# 实际运行结果

来源：Binance 官方公开现货月度归档；BTCUSDT；价格 USDT/BTC，成交量 BTC；所有时间 UTC。
请求范围：[{audit['requested_start']}, {audit['requested_end_exclusive']})。
实际 open_time 范围：{audit['actual_start']} 至 {audit['actual_last_open']}。
实际 {audit['actual_rows']} 根，理论 {audit['expected_rows']} 根；缺失 {audit['missing_bars']}、完全重复 {audit['duplicate_rows']}、无效 {audit['invalid_rows']}。
缺失区间见 run.json 的 data_audit；没有填充行情。有效特征标签样本 {audit['valid_samples']}；样本外评价 {overall['n']} 行。

每 {config.get('update_hours', 24)} 小时在 UTC 整点 T 用最近 {lookback_text} 的 feature_time 样本重新训练，要求 target_time<=T。
两次更新之间模型固定，特征在每根 K 线结束后获得，预测下一根 {horizon_text}收益率。
评价 target_time 范围：[{evaluation_start.isoformat()}, {audit['requested_end_exclusive']})。
StandardScaler 仅 fit 当次训练样本，随后为 {model_text}；无调参。
模型参数：{json.dumps(config.get('model_params', {}), ensure_ascii=False)}。RF也沿用训练集标准化流程；树模型不依赖这种尺度处理。
输入 {len(config.get('feature_columns', FEATURES))} 个：{', '.join(config.get('feature_columns', FEATURES))}。

| 指标（小数收益率单位） | 模型 | 零收益基准 |
| --- | ---: | ---: |
| MAE | {overall['mae_model']:.10g} | {overall['mae_baseline']:.10g} |
| RMSE | {overall['rmse_model']:.10g} | {overall['rmse_baseline']:.10g} |

{comparison}；RMSE 改善率 {improvement}。两者使用完全相同的有效预测行，总体误差直接从全部行计算。
daily_metrics.csv 按 feature_time 的 UTC 日期汇总；缺失输入/标签及跨评价边界排除见 skipped_predictions.csv。
{coefficient_text}
每次训练样本数和实际时间范围见 training_updates.csv。
不能把该系数解释为原始价格每变动一单位的效应，也不能根据单个系数大小宣称贡献。

局限：短样本、单币种、市场非平稳、OHLC 原始价格高度共线，OLS 系数可能不稳定。
误差降低不等于交易盈利；没有交易回测。本原型也未完成课程的至少两种课堂方法比较。
运行入口：`python run.py`（扩展实验参数见 README.md）；配置、依赖版本、校验来源与状态统一见 run.json。
'''
        (out / 'summary.md').write_text(text, encoding='utf-8')
        update_run(out, status=dict(status='success', stage='complete', metrics=overall))
        print(text)
        return 0
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        if not (out/'run.json').exists():
            update_run(out, config=config)
        update_run(out, status=dict(status='failed', stage=stage, error=message))
        urls = '\n'.join(f'- {archive_url(m, config.get("interval", "15m"))}\n- {archive_url(m, config.get("interval", "15m"))}.CHECKSUM' for m in config.get('months', []))
        failure = f'''# 本次运行未完成

失败阶段：{stage}。
原因：{message}

本次未产生完整实验结果，不能报告有效预测样本数或误差比较。
若目录有既往结果，不代表本次成功；请以 run.json 的 status 为准。
手动下载下面的 ZIP 和同名 .CHECKSUM 原样放入 data/raw/，保持文件名，然后运行 `python run.py`。
缓存仍须通过 SHA256/ZIP 校验才可清洗训练；不要以假数据替代。

{urls}
'''
        (out / 'summary.md').write_text(failure, encoding='utf-8')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
