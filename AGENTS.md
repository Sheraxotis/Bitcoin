# BTC 分类实验：Agent 接手与扩展指南

## 先读这里

这是已经运行过真实数据的 Python 项目，当前主线是 **BTCUSDT 现货涨跌分类、可配置周期、统一特征、按模型独立缓存、简易回测**。本文件替代最初“仅 15m、31 天、六特征、线性回归”的原型要求；原型代码仍保留，不能把旧约束重新施加到当前分类框架。

开始工作时阅读本文件、README.md 和相关实现，再修改用户指定的部分。优先复用现有接口，避免复制出每周期/每模型一份训练脚本。不要仅为了熟悉项目启动全量训练。

当前范围：BTCUSDT、spot、UTC。可以按用户请求新增模型和特征；不自行增加币种、合约、实盘交易或超参数搜索。不编造下载成功、训练完成或盈利，不用合成数据冒充真实行情。

## 代码地图与执行流程

```text
JSON 输入配置
  -> src/data.py：下载对应周期或显式聚合
  -> src/prepare_data.py：原始清洗、连续性及下一根标签
  -> src/features.py：统一特征与有效样本
  -> src/validation.py：共用时间划分
  -> src/experiment.py：实验身份、各模型缓存/训练、汇总
  -> src/evaluation.py：分类指标、ROC 和总结
  -> backtest_classification.py：读取保存的预测回测
```

| 文件 | 责任 |
| --- | --- |
| `run_classification.py` | 分类 CLI，读取配置、周期/模型覆盖、导入与强制重训 |
| `src/download_data.py` | 官方 ZIP/CHECKSUM 下载、超时重试、SHA256/ZIP 校验和原始缓存 |
| `src/data.py` | 统一数据加载；可选的完整 UTC 聚合 |
| `src/prepare_data.py` | CSV 字段、时间单位、去重/冲突、OHLC 检查、缺口与标签 |
| `src/features.py` | `FEATURE_REGISTRY`、`RAW_FEATURES`、特征身份和共用有效行 |
| `src/models.py` | `MODEL_REGISTRY`、`ModelDefinition`、模型工厂与概率适配 |
| `src/validation.py` | `Split`、`VALIDATION_REGISTRY`、训练与预测行索引 |
| `src/training.py` | 按共用划分训练单个模型、处理单类训练集、保存通用预测列 |
| `src/experiment.py` | 配置规范化、实验/模型身份、结果缓存、旧结果校验导入 |
| `src/metadata.py` | 统一 `run.json` 的读写；旧文件只做兼容读取 |
| `src/evaluation.py` | Accuracy、Balanced accuracy、Precision、Recall、F1、Log Loss、Brier、ROC AUC |
| `backtest_classification.py` | 动态读取模型名单，现货全仓 BTC/现金策略、费用与净值 |
| `tests/test_framework.py` | 聚合、原生周期、特征掩码、概率接口、ROC、缓存复用与失效 |
| `tests/test_pipeline.py` | 原有下载/清洗、标签配对和回归时间边界检查 |

`run.py`、根目录 `config.json`、`src/rolling_train.py` 是保留的回归入口，不是当前分类主线。用户已删除回归实验结果，除非明确要求，不重跑它们。

## 当前配置与时间语义

- 默认分类配置：`configs/config_classification_15m_90days.json`。日线配置：`configs/config_classification_90days.json`。输入配置不等于输出的运行记录。
- `interval` 支持 `15m`、`2h`、`1d`；默认下载 **各自周期的官方原生 K 线**。只在显式指定 `source_interval: "15m"` 时将 15m 聚合为 2h/1d，不把所有数据默认聚合。
- 当前完整数据月份为 2025-01 至 2026-08。日期通过配置设置，不将历史月份写死为新的默认限制。
- `feature_columns` 是所有模型共用的有序特征列表，两个分类配置已启用 21 个特征：原十个市场字段加用户 `docs/feature.md` 的 11 个派生特征。不允许按模型各选一套特征冒充公平比较。
- `window_days=90` 是最近 90 个日历日；`update_hours=24` 是每天重新训练。15m 是输入/预测周期，不是重训频率。
- `evaluation_start` / `evaluation_end` 是 UTC 评价范围，右端不含。默认评价起点为首月起点加训练窗口。
- 当前标签仅支持下一根：`target_t=C_(t+1)/C_t-1`；上涨为 `target>0`，零收益和下跌均为 0。当前已知输入 `return_t=C_t/C_(t-1)-1` 不等于 target。
- `feature_time=bar_end_t`，`target_time=bar_end_(t+1)`；断开的 K 线不能计算有效相邻收益率。
- 每次训练使用 `feature_time` 位于 `[T-window,T)` 且 `target_time<=T` 的有效样本。预测特征在到达后才使用，不能假设零点已知全天特征。
- `training_updates.csv` 每行是一轮 fit；日期按天推进表示每天更新，`train_cutoff-window_start` 才是窗口长度。15m 通常每天 96 条预测。
- `validation="rolling"` 是唯一已实现的方式。**单次验证集和扩展窗口只有扩展接口，用户此前明确要求先不实现；除非后续授权，不添加这些代码。**

## 新增模型：最少改动路径

1. 在 `src/models.py` 定义顶层工厂 `def new_model(params): ...`，使用传入参数创建全新模型对象。需要标准化/填补等操作时放入其 Pipeline，不能在全数据上预先 fit。
2. 在 `MODEL_REGISTRY` 注册 `ModelDefinition(new_model, version="1")`。名称为小写安全标识符，不使用保留名 `zero`、`majority`、`buy_hold`、`cash`。
3. 模型统一提供 `fit(X, y)`、`predict_proba(X)`、`classes_`。概率列必须按 `classes_` 找到类别 1，不能假设所有实现的列顺序一致。概率有限且位于 [0,1]。
4. 在目标输入配置的 `models` 对象中增加模型名和参数；所有模型继续使用共用 `feature_columns`。不要修改训练、评估或回测里的名称循环来硬编码新模型。
5. 固定随机种子。自定义概率校准或早停验证只能使用当次训练窗口内部的数据，按时间划分，不能使用外层评价段。
6. 扩展自定义适配器时检查 `model_signature`：当前一般模型识别工厂源码、参数和显式版本，SVM 额外识别适配器源码。**修改工厂调用的自定义类/辅助代码时必须更新版本，或将其源码加入该模型身份。**新增第三方库也要将实际版本纳入该模型身份，防止错误复用。
7. 增加小样本概率接口和缓存隔离检查；用短真实评价段跑通，再按用户范围运行。不要默认新增模型后把全部旧模型强制重训。

示意（替换为用户要求的具体分类器，不把示意参数当成调参结论）：

```python
def new_model(params):
    return make_pipeline(StandardScaler(), NewClassifier(**params))

MODEL_REGISTRY["new_model"] = ModelDefinition(new_model, version="1")
```

已有模型：`logistic`、`knn`、`rf`、`svm`、`fnn`。SVM 前段拟合、后段按时间校准，禁用 `SVC(probability=True)` 的内置校准；FNN 是 sklearn `MLPClassifier`，目前禁用随机划分的 `early_stopping=True`。训练窗口只有单类时由训练层输出常量概率；SVM 内部分区类数不足会明确失败，不能偷偷删掉该模型的预测行。

## 新增特征：统一重训规则

1. 在 `src/features.py` 定义顶层函数 `(frame, step) -> pandas.Series`，输出与输入完全相同的索引。
2. 注册到 `FEATURE_REGISTRY`，并在输入配置的共用 `feature_columns` 中加入名字。真正的原始市场字段才放到 `RAW_FEATURES`；不要把时间戳、ignore、target、target_time 当成特征。
3. 函数只使用当前/过去已完成 K 线。不能用 `shift(-1)`、居中窗口、未来价格或全样本归一化。特征函数收到的 DataFrame 可能含标签，开发者必须主动避免读取它们。
4. 对需要 N 根历史的特征检查窗口连续性。缺口和预热不足返回 NaN，不前向填充价格造行情，也不跨缺口算动量等历史特征。
5. 需要 fit 的变换放到模型训练 Pipeline 内；不要让无状态特征函数在整个实验上学习参数。若新增可配置特征参数，要同时纳入配置和特征/实验身份，不能只改函数运行时变量。
6. 保持共用有效样本掩码：`build_features` 用全部选定特征统一计算 `valid_sample`，所有模型按同一组行训练和评价。
7. 检查手工价格数值、方向和时间，缺口行为，以及“追加/修改未来数据不改变历史特征”。构造数据只用于测试，不能混入正式实验。
8. **增加/修改使用中的特征后创建新实验，所有模型重新训练。**不能复用旧特征下的模型预测来比较新特征下的新模型。

## 缓存与复现的硬规则

用户特征已注册：return_lag1、return_7d、ma_deviation、volatility、price_range、body_ratio、upper_shadow_ratio、lower_shadow_ratio、volume_change、relative_volume、taker_buy_ratio。一日窗口按周期换算 96/12/1 根，7 日为 672/84/7 根；日线波动率使用至少两根收益率，日线一日均线偏离为零。历史窗口不跨缺口，零分母返回 NaN。`FEATURE_DEPENDENCIES` 记录特征辅助函数源码；新特征调用辅助实现时也必须将其纳入身份。旧十特征实验不能导入当前 21 特征配置，也不代表新特征已经完成全量训练。

- 实验 ID 包含共用数据、来源/哈希、特征及实现、周期、时间划分、窗口、核心训练实现和依赖版本；不包含模型列表。模型 ID 加上该模型实现/版本与参数。
- 新增模型：只训练新模型。修改某个模型参数/实现：只训练该模型。修改共用特征、数据、周期、窗口或划分：全部模型进入新实验。
- 修改指标或手续费/调仓频率：复用已有预测，只重新评估/回测。
- 成功状态、完整文件 SHA256、时间/实际标签配对全部通过，才可复用模型缓存。缓存不完整或损坏时重训对应模型，不手工篡改身份来强行命中。
- 修改 `data.py`、`prepare_data.py`、`validation.py`、`training.py` 等参与身份的核心源码会改变实验 ID，即使配置没变。已有目录名不是当前源码会生成的 ID；不要承诺默认命令必然复用旧目录。
- 配置相同且实现语义保持一致时，可使用 `--import-legacy` 导入已保存的分类预测。导入会检查配置、实际依赖、原始哈希、共用行、训练窗口并独立复算首批预测。新特征/新周期结果不能这样导入。
- 不同实验不能覆盖同一个 `--output-dir`。优先使用自动 ID 目录。当前元数据更新不是多进程锁，不要同时向一个实验目录启动多个运行进程。
- `requirements.txt` 是安装要求；`run.json` 中实际依赖版本用于缓存与复现，不能仅凭 requirements 推断已安装版本。

## 输出约定：继续保持精简

每个实验只保留 **一个 `run.json`**。使用 `src/metadata.py` 的 `read_run` / `update_run` 读写，更新时保留其他章节，不直接覆盖整份记录导致缓存或回测信息丢失。

- `config`：共用配置、模型参数、实验 ID、实际依赖版本。
- `status`：成功/失败、样本数、模型 trained/reused/imported 状态。
- `download_manifest`：官方来源、原始 ZIP SHA256、缓存标记。
- `data_audit`：行数、重复、缺失、异常和有效样本。
- `model_cache`：各独立模型缓存的完整身份、预测文件 SHA256、状态。
- `backtests`：各回测子目录的费率、调仓和执行设置。
- 旧记录可能有 `execution_settings`，迁移/更新时保留。

训练输出：`predictions.csv`（宽表汇总）、`training_updates.csv`、`overall_metrics.csv`、`summary.md`、`roc_curve.png`、`run.json`；每个 `models/<模型ID>/` 只存通用格式 `predictions.csv`，其状态并入根目录 `run.json`。

回测读取实验汇总 `predictions.csv` 和 `run.json`，并核对原始价格。子目录只输出 `metrics.csv`、`equity_curve.csv`、`summary.md`、`net_equity.png`、`gross_equity.png`；设置并入实验 `run.json.backtests`。独立输出到实验目录外时可另存一个 `run.json`。

**不要重新生成用户已明确删除的文件**：`trades.csv`、核验 JSON、场景比较文件、`monthly_metrics.csv`、`dependency_versions.json`、累计分类错误图、独立 `run_config.json` / `run_status.json` / `data_audit.json` / `download_manifest.json`。旧文件名称只用于兼容读取。核验信息显示在终端，不创建额外报告/工作文件。历史过程文档与旧回归配置、对比脚本已清理，不重新创建；`docs/feature.md` 是用户添加的文件，必须保留，不擅自改写或删除。

## 数据和指标注意事项

- 官方归档：https://github.com/binance/binance-public-data 。ZIP 后加 `.CHECKSUM`，下载先检查 HTTP、HTML、SHA256、ZIP；直接读取 CSV 成员，不任意解压路径。
- 2025-01-01 起现货时间戳为微秒，更早为毫秒。检查月份、UTC 网格、唯一性、价格/成交量与 OHLC 合法性。字段/规则有变化时查官方说明。
- 不跨缺口造收益，不把未完成聚合 K 线当有效样本。网络/文件失败时清楚报告，不替代为假数据。
- 概率阈值为严格 `>0.5`；ROC/AUC 用概率排名并扫描阈值，绘 ROC 不自动改交易阈值。单一真实类别的 AUC 不可计算。
- `zero_class=0`、`zero_probability=0` 是始终非上涨的方向/概率基准，不是 BTC 价格走势。其极端零概率造成很大 Log Loss，不单凭超过它声称概率有用。
- 多数基准由各训练窗口的上涨比例决定。交易基准是现金与买入持有 BTC；分类正确率/AUC 不等于盈利。
- 回测现货全仓 BTC/现金，不做空、无杠杆；仅仓位切换及首尾交易扣单边费，默认执行是理想化完成 K 线收盘价。未建模延迟、滑点、价差。比较只用相同有效预测行。
- 已多次查看这段历史，属于探索，不称为全新独立验证。短 2h 检查、未收敛 FNN 不能当成模型效果结论。

## 快速运行与验证

在项目根目录执行 PowerShell 命令。Python 环境已有依赖时无需重复安装：

```powershell
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '4'
python run_classification.py --help
python -m unittest discover -s tests -v
```

模型配置增加后运行完整配置以汇总比较；`--models` 只能选择 JSON 中已配置的模型：

```powershell
python run_classification.py --config configs/config_classification_15m_90days.json
python run_classification.py --interval 2h
python run_classification.py --models rf --force
# --import-legacy 仅适用于特征等配置完全相同的结果；当前默认特征已改变。
python -m scripts.verify_classification --output-dir outputs/15m_392f7697ba52d591
python backtest_classification.py --predictions-dir outputs/15m_392f7697ba52d591 --fee 0.0001 --rebalance-hours 2
```

`--models` 限制本轮汇总报告包含的模型，但其他模型缓存保留；再次运行完整配置会汇总所有模型。正式全量 15m 滚动拟合此前约 25 分钟，新模型尤其 SVM 可能更慢。先用较短 `evaluation_end` 检查，使用独立输入配置，不默默覆盖完整实验范围。

Windows 下 PowerShell profile 可能被禁用，工具命令可用 `login=false`；读写中文明确 UTF-8，避免通过默认编码的 Python stdin 管道写中文。现有测试的 TemporaryDirectory 在沙箱中可能遇到已知权限限制，可以申请执行权限后运行，不能将权限失败误报为测试通过。下载同理，成功与失败都按实际结果报告。

## 已有结果与交付要求

- `outputs/15m_392f7697ba52d591/`、`outputs/1d_b9ce570233e0cefd/`：当前保留的完整分类结果，分别 49,727 / 517 个评价样本。重复旧分类目录已清理，其有效回测已迁入对应目录。
- 15m 的 `backtest` 为单边 0.01%，`backtest_fee_0_1pct` 为单边 0.1%，`backtest_2h_fee_0_01pct` 为 2h 调仓、单边 0.01%；日线的 `backtest` 为单边 0.1%，具体设置见各实验 `run.json.backtests`。
- `outputs/classification_15m_90days/overall_metrics.csv` 清理时被外部进程占用，是待删除的旧文件，不能当成完整可运行实验目录。关闭占用程序后可删除此文件及空目录。
- `outputs/2h_8a32acd1336aea33/`：旧来源为 15m 显式聚合的两天五模型流程检查，23 个评价样本，FNN 出现过未收敛提示；不是新的原生 2h 完整实验。
- 原生 2h 下载已用 2025-01 月份验证（372 根）；其余月份会在请求相应实验时下载。
- 当前已有 13 项自动检查，包括用户特征的公式、跨周期窗口、缺口、零分母、因果性和辅助实现身份。交付时说明实际改动、缓存复用/重训情况、验证结果及未运行部分，更新 README 的扩展接口与命令。低影响文档修改不需要启动训练。
- `configs/` 只保留两个分类输入配置，`scripts/` 只保留分类核验入口与包初始化。`data/processed/` 的可再生成 CSV 已清理，分类不依赖它；`data/raw/` 的 ZIP/CHECKSUM 是训练和回测共用的真实数据缓存，不随意删除。原始回归入口、根配置和模块仍有引用和测试，继续保留。
- 清理/迁移输出仅在用户授权范围内，保留原始缓存与无关结果；Windows 递归删除前检查绝对路径，文件被程序占用时报告，不能随意结束用户进程。
