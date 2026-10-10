# BTC 分类实验与简易回测

使用 Binance BTCUSDT 现货真实历史数据。分类框架支持配置驱动的周期、共用特征、模型和固定窗口滚动训练；每个模型独立缓存预测。

## 架构

```text
run_classification.py       分类命令入口
backtest_classification.py 回测命令入口
configs/                    实验配置
src/
  download_data.py          官方归档下载、SHA256 校验与缓存
  prepare_data.py           原始数据清洗、时间与价格检查
  data.py                   数据加载与完整 UTC 周期聚合
  features.py               共用特征注册表与统一有效样本
  models.py                 模型注册表、预处理与概率接口
  validation.py             时间划分器，目前仅固定窗口滚动
  training.py               根据共用划分，独立训练一个模型
  experiment.py             配置、实验 ID、模型缓存和旧结果导入
  evaluation.py             通用指标与报告
  metadata.py               单个 run.json 的读取、更新与兼容
outputs/                    实验结果及独立模型预测缓存
scripts/                    分类终端核验
tests/                      隔离的手工数据检查，不进入真实实验
docs/                       用户的 feature.md，保留
```

`run.py`、`config.json`、`src/rolling_train.py` 继续保留原始回归流程；本次重构针对分类。回归历史结果已清理。

## 实验规则

- 所有模型使用相同特征、训练时间划分和有效预测行。新特征造成缺失时，对所有模型统一剔除相应样本。
- `interval`：`15m`、`2h` 或 `1d`，默认统一下载各自周期的官方 K 线并缓存。仅显式设置 `source_interval: "15m"` 时，2h/1d 才从 15m 聚合。官方说明：https://github.com/binance/binance-public-data 。
- 聚合使用 UTC 左闭右开边界，只接受完整组成 K 线。OHLC 取首/最大/最小/末，成交相关字段求和，再计算 return 和标签。
- `window_days`：训练窗口的日历天数；`update_hours`：模型更新间隔。默认 90 天窗口、24 小时更新。
- 标签目前固定为下一根 K 线收益率大于 0。输入周期为 2h 时预测下一根 2h；15m 信号每 2h 调仓仍然是 15m 预测。
- `evaluation_start` / `evaluation_end` 可指定 UTC 评价范围；默认从首月起加训练窗口，至末月之后的月初，不含结束时刻。
- `validation: "rolling"` 是当前唯一实现。单次验证和扩展窗口尚未实现，选择它们会明确报错。
- 特征函数必须只使用当前及过去的已完成 K 线，不能使用未来数据或全样本拟合。标准化在各次训练内部完成。

## 独立模型缓存

输出目录自动命名为 `outputs/<周期>_<实验ID>/`：

```text
15m_<实验ID>/
  run.json                  配置、状态、数据来源、审计、模型缓存及回测配置
  predictions.csv           汇总预测，保留宽表兼容回测
  training_updates.csv      共用窗口边界与训练/预测样本数
  overall_metrics.csv       模型及基准指标
  summary.md
  roc_curve.png
  models/
    logistic_<模型ID>/
      predictions.csv       通用列：时间、真实收益/类别、probability、predicted_class
```

实验 ID 包含数据内容与来源哈希、周期、特征及实现、标签数据、窗口、时间划分、核心训练实现和依赖版本；不包含所选模型列表或并行线程数量。模型 ID 再加入该模型的参数、工厂实现和版本。

| 改动 | 行为 |
| --- | --- |
| 新增模型 | 实验 ID 不变，旧模型复用预测，只训练新模型 |
| 修改某个模型参数或实现 | 只重新训练该模型 |
| 修改共用特征、周期、数据、窗口或验证划分 | 新实验 ID，所有模型重新训练 |
| 修改指标、手续费或交易频率 | 复用预测，重新评估或回测 |
| 缓存不完整或 SHA256 不符 | 重新训练该模型，不使用损坏结果 |

缓存只保存预测，不保存每天的模型对象。重复运行仍会校验原始数据、重新构造特征和时间划分，但不会重新拟合命中缓存的模型。不要同时向同一个实验目录运行多个进程。

## 输出文件的用途

| 文件 | 记录内容 | 使用场景 |
| --- | --- | --- |
| `summary.md` | 指标与实验规则的摘要 | 日常查看 |
| `overall_metrics.csv` | 每个模型的总体分类指标 | 比较模型效果 |
| `predictions.csv` | 所有模型每根 K 线的概率、类别、真实收益和时间 | 回测必需，汇总结果可由模型缓存重建 |
| `training_updates.csv` | 每次训练的窗口、样本数和已知标签边界 | 排查样本量及时间泄漏 |
| `run.json` | 配置、实际依赖、成功状态、下载来源、数据审计、模型缓存和回测设置 | 回测必需，复现、缓存校验与排查 |
| `roc_curve.png` | 上涨概率在不同阈值下的 TPR/FPR，图例标明 AUC | 比较分类排序能力 |

`models/<模型ID>/predictions.csv` 是单个模型的通用预测缓存；其身份、完整性哈希及成功状态统一保存在实验根目录 `run.json` 的 `model_cache` 中。预测缓存与 `run.json` 一起保留，才能避免重新训练。根目录的 CSV 是方便评估、回测的汇总；模型目录中的 CSV 用于独立复用。

回测目录的 `metrics.csv` 是收益、回撤和交易次数；`equity_curve.csv` 是逐根仓位与净值；费率和交易频率放在实验根目录 `run.json` 的 `backtests` 中；`summary.md` 和两张净值图用于查看结果。回测设置不参与训练缓存。输出到实验目录外的独立回测目录时，该目录单独保存一个 `run.json`。

每个实验只保留一个 JSON：`config` 是共用配置及实际依赖版本，`status` 是运行状态，`download_manifest` 是官方来源与 SHA256，`data_audit` 是数据质量，`model_cache` 是各模型缓存记录，`backtests` 是各费率/调仓场景的设置。输入配置仍在 `configs/`，不同实验的记录各自独立。

平时重点看训练 `summary.md` / `overall_metrics.csv`，以及回测 `summary.md` / `net_equity.png`；其他文件用于复用和排查，不需要逐个打开。

不再生成分月指标或独立依赖版本文件。`requirements.txt` 声明安装要求，`run.json` 的 `config.dependency_versions` 记录实际运行环境，两者用途不同。ROC/AUC 衡量排序能力，不衡量概率校准或交易盈利；绘图不会改变回测使用的 0.5 阈值。

`training_updates.csv` 每行是一轮训练。默认相邻 `train_cutoff` 相差一天，因为 `update_hours=24`；`train_cutoff-window_start` 是 90 天，才是训练窗口长度。15m 是输入和预测周期，当天通常有 `n_test=96` 条预测。每日更新、90 天训练和每 15 分钟预测是三个独立概念。

## 运行

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '4'
python run_classification.py
```

默认使用 `configs/config_classification_15m_90days.json`。重复运行会显示 `reused`。新周期：

```powershell
python run_classification.py --interval 2h
python run_classification.py --config configs/config_classification_90days.json
```

2025 全年原生 2h 分类实验：

```powershell
python run_classification.py --config configs/config_classification_2h_2025.json
```

该配置沿用 21 个共用特征及 logistic、knn、rf，并增加 logistic_pca、knn_pca 两个变体，90 个日历日训练窗口，每 24 小时更新，预测下一根 2h 的涨跌。评价范围为 UTC `[2025-01-01, 2026-01-01)`；额外读取 2024-09 至 2024-12 的官方原生 2h 数据作为训练历史与特征预热，不使用 15m 聚合，也不导入旧实验预测。

已完成结果：`outputs/2h_ce10b249d1df7c34/`，4,379 条共用评价预测、365 次每日更新，三个模型均新训练。16 个月共 5,844 根原始 K 线，无缺口、重复或异常 OHLC；预测行、时间边界、指标及首轮独立重算核验通过。年底目标时间等于评价右边界的一条预测被排除。此次仅运行分类实验，未运行回测。

随后加入两个 PCA 变体，复用原三个模型缓存，仅训练新增变体，仍使用相同的 4,379 条评价行。logistic_pca 的 Accuracy 为 53.23%、AUC 为 0.5318、Log Loss 为 0.6961；原 logistic 分别为 52.52%、0.5264、0.7001。knn_pca 的 AUC 为 0.4947，基本无改善，Log Loss 从 1.9084 升至 1.9890。此为已查看历史上的探索性对照，不代表独立验证或交易收益。14 项自动测试通过。

```powershell
python -m scripts.verify_classification --output-dir outputs/2h_ce10b249d1df7c34
```

首次训练新周期会全量拟合该配置中的模型。仅运行配置中的某些模型，或强制重训：

```powershell
python run_classification.py --models logistic rf
python run_classification.py --models rf --force
```

`--models` 限制本次运行的模型；汇总报告也只包含本次选择的模型，其余独立缓存保留。再次不带此参数运行即可汇总所有配置模型。

`--output-dir` 可以指定目录，但已有目录若属于不同实验会拒绝覆盖。更改共用配置时使用自动目录最方便。

## 增加特征或模型

在 `src/features.py` 增加因果特征函数，并注册到 `FEATURE_REGISTRY`；在配置的 `feature_columns` 中统一选择它。函数接口是 `(frame, step) -> 与输入同索引的 Series`。所有模型使用这一个列表。新特征建议增加手工价格、缺口和未来数据不影响历史值的检查。

`docs/feature.md` 中的 11 个特征已实现，两个分类配置现在共用 21 个特征：原有十个字段加 `return_lag1`、`return_7d`、`ma_deviation`、`volatility`、`price_range`、`body_ratio`、`upper_shadow_ratio`、`lower_shadow_ratio`、`volume_change`、`relative_volume`、`taker_buy_ratio`。用户原始特征说明保持不变。

一天窗口按周期换算为 15m 的 96 根、2h 的 12 根、1d 的 1 根；7 日收益对应 672、84、7 根。波动率使用样本标准差（ddof=1），日线至少取两根收益率；日线的一日均线偏离恒为 0。相对成交量只用前一天窗口的成交量均值，不包含当前成交量。历史窗口不跨缺口，预热不足或零分母返回 NaN，由所有模型的共用掩码剔除。

新增特征已启用，下一次运行会创建新实验并重新训练全部配置模型；旧十特征结果保留，不能用 `--import-legacy` 导入到新特征实验。若需要复用旧实验，应使用其 `run.json.config` 中保存的原始特征配置。特征身份同时记录所调用辅助函数的源码，修改历史窗口逻辑也会使对应缓存失效。

实现验证：13 项自动检查通过；已校验并读取缓存中的 2025-01 至 2026-08 真实归档，15m 共 58,368 根、57,695 个有效特征标签样本，日线共 608 根、600 个有效样本。两个周期的首轮 logistic、knn、rf 均成功拟合并输出有效概率；尚未运行新特征的全量滚动训练或回测。这些有效样本数包含训练历史，并非全部是样本外预测数。7 日特征预热会减少首个训练窗口的有效行，但不改变 90 个日历日的窗口边界。

已有模型名：`logistic`、`knn`、`rf`、`svm`、`fnn`。在 JSON 的 `models` 中新增，例如：

```json
"svm": {"C": 1.0, "kernel": "rbf", "calibration_fraction": 0.2},
"fnn": {"hidden_layer_sizes": [64, 32], "max_iter": 300, "random_state": 42}
```

再次运行完整配置，旧模型会命中缓存。SVM 使用训练窗口前段拟合，后段按时间校准概率；不启用 SVC 自带的随机概率校准。FNN 使用 sklearn MLPClassifier，当前禁止随机划分的 early_stopping；关注终端中的未收敛提示。

PCA 变体 `logistic_pca`、`knn_pca` 使用 `StandardScaler → PCA → 分类器`。分类器参数沿用原模型，额外参数 `pca_n_components: 0.95` 固定保留 95% 方差，采用确定性的完整 SVD、不启用 whitening。标准化及 PCA 都只在每次训练窗口内拟合，预测时不更新。各窗口实际主成分数可不同，保留方差不等于保留预测信号。当前只在 2025 年 2h 配置中启用；新增变体不改变共用特征、实验 ID 或原模型缓存。

增加其他模型时，在 `src/models.py` 增加工厂并注册 `ModelDefinition`。返回的对象提供 `fit`、`predict_proba` 和 `classes_`；预处理由工厂放在训练 Pipeline 内。修改工厂依赖的自定义辅助实现时，应提升该模型的 `version`。不用改训练、指标和回测中的模型名称列表。

## 已有结果与本次验证

已有完整分类预测经过原始归档哈希、依赖、共用行、训练窗口及首批模型独立复算后，导入新缓存：

- `outputs/15m_392f7697ba52d591/`：49,727 个评价样本。
- `outputs/1d_b9ce570233e0cefd/`：517 个评价样本。
- 重复的旧分类结果已清理，有效回测迁入当前目录。15m 的 `backtest` 是单边 0.01%，`backtest_fee_0_1pct` 是单边 0.1%，`backtest_2h_fee_0_01pct` 是 2h 调仓、单边 0.01%；日线的 `backtest` 是单边 0.1%。
- `outputs/2h_8a32acd1336aea33/` 是两天真实行情的五模型流程检查，23 个评价样本。FNN 的短迭代设置曾出现未收敛提示，不是完整模型效果比较。

以下导入命令只适用于原十特征配置（需从旧实验的 `run.json.config` 恢复），不能用于当前已启用 21 特征的默认配置：

```powershell
python run_classification.py --import-legacy outputs/15m_392f7697ba52d591
python run_classification.py --config configs/config_classification_90days.json --import-legacy outputs/1d_b9ce570233e0cefd
```

核验与测试仅在终端显示：

```powershell
python -m scripts.verify_classification --output-dir outputs/15m_392f7697ba52d591
python -m unittest discover -s tests -v
```

## 回测

回测自动读取配置中的模型。已有旧预测和新框架汇总预测均可使用：

```powershell
python backtest_classification.py --predictions-dir outputs/15m_392f7697ba52d591 --fee 0.0001 --rebalance-hours 2
python backtest_classification.py --predictions-dir outputs/2h_8a32acd1336aea33 --fee 0.0001
```

回测子目录只输出 `summary.md`、`metrics.csv`、`equity_curve.csv`、`net_equity.png`、`gross_equity.png`，设置并入实验根目录 `run.json`，不生成交易明细、核验或场景对比文件。单边 0.01% 对应小数 0.0001；只在仓位变化时扣费，并计入起始和期末交易。

零收益预测是分类基准；现金是交易中的零收益基准，BTC 自身价格对应买入持有。回测按完成 K 线的收盘参考价即时成交，未模拟延迟、滑点和价差。已反复查看的历史用于探索；分类指标提升不等于交易盈利。

目录清理后，`configs/` 只保留两个分类配置，`scripts/` 只保留分类核验。历史过程文档和可再生成的 `data/processed/` 已删除，原始 ZIP/CHECKSUM 保留。用户的 `docs/feature.md` 保留。回归入口及其根配置、模块仍有测试引用，继续保留。旧目录 `outputs/classification_15m_90days/` 如仍存在，只包含被外部进程占用而暂未删除的旧指标文件。

汇总预测中的 `zero_class=0` 表示基准始终预测非上涨，`zero_probability=0` 表示它给上涨的概率为零。它们不驱动模型交易；该基准的概率损失会受极端零概率影响，不应仅凭超过它判断概率有用。

