# 双复合择时策略说明

## 定位

本模块把人工确认的 15 条单因子持仓规则复合成两个正式策略：

1. 主策略 `two_speed_category_strong0p25`：类别等权 + 周频锚 + 日度强事件响应。
2. 挑战者 `entry_frequency_inverse_sqrt_weekly_binary`：按历史开仓频率降权 + 周频锚 + 日度强事件响应。策略 ID 为兼容既有输出而保留。

两个策略集中实现在同一个文件：

```text
skills/factor-timing-advisor/runtime/composite_timing_strategies.py
```

输入来自：

```text
results/selected_single_factor_rules/selected_rule_specs.parquet
results/selected_single_factor_rules/selected_rule_daily_positions.parquet
```

每条单因子规则的日度持仓转换为投票分数：

```text
position=1 -> score=+1
position=0 -> score=-1
```

所有综合分数在阈值判断前统一保留 12 位小数，并将绝对值小于 `1e-12` 的结果归零，避免浮点误差把数学上的零分误判为多头或空头。

## 主策略：类别等权两速机制

### 加权

四个类别各分配 25% 总权重，再在类别内部等权：

| 类别 | 当前规则数 | 类别预算 | 当前单规则权重 |
|---|---:|---:|---:|
| 赔率/筹码 | 3 | 25% | 8.33% |
| 赔率/估值 | 3 | 25% | 8.33% |
| 胜率/资金 | 3 | 25% | 8.33% |
| 胜率/量 | 6 | 25% | 4.17% |

综合分数：

```text
score_t = sum(weight_i * child_score_i,t)
```

类别等权避免某一类因子仅因保留规则数量更多而天然控制组合。

### 两速执行

慢速通道是周频稳定锚：

```text
周末综合分数 > 0  -> 下一交易日持仓
周末综合分数 <= 0 -> 下一交易日空仓
```

快速通道每天检查强共识：

```text
日度综合分数 >= +0.25 -> 下一交易日提前开仓
日度综合分数 <= -0.25 -> 下一交易日提前平仓
-0.25 < 日度分数 < +0.25 -> 保持当前状态
```

`+0.25` 等价于加权看多票约 62.5%，`-0.25` 等价于加权看多票约 37.5%。普通分歧等待周频确认，强共识可以在任意交易日收盘后触发。

## 挑战者：开仓频率平方根倒数

训练期固定截至 `2020-12-31`，只统计每条规则实际持仓从 `0 -> 1` 的转换次数：

```text
f_i = 训练期年均实际开仓次数
raw_weight_i = 1 / sqrt(f_i)
weight_i = raw_weight_i / sum(raw_weight)
```

频繁开仓的规则通常更容易重复表达短周期噪声，因此单次话语权较小；低频规则权重更高。使用平方根而非直接使用 `1/f`，是为了避免对高频规则惩罚过度。

挑战者使用与类别等权策略相同的两速执行框架，但综合分数由频率平方根倒数权重计算。

慢速周频锚：

```text
周末频率加权分数 > 0  -> 下一交易日持仓
周末频率加权分数 <= 0 -> 下一交易日空仓
```

快速日度通道：

```text
日度频率加权分数 >= +0.25 -> 下一交易日提前开仓
日度频率加权分数 <= -0.25 -> 下一交易日提前平仓
-0.25 < 日度分数 < +0.25 -> 保持当前状态
```

频率训练期在后续日更中保持固定，不使用 2021 年以后的验证与评估数据重新拟合权重。若正式规则池发生变化，应重新检查新规则在训练期是否有足够开仓事件。

## 运行方式

单独刷新两个复合策略：

```powershell
D:\anaconda\python.exe .\skills\factor-timing-advisor\scripts\run_pipeline.py composite-strategies --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

刷新单因子规则时会自动继续刷新复合策略：

```powershell
D:\anaconda\python.exe .\skills\factor-timing-advisor\scripts\run_pipeline.py selected-rules --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

日更 `score-update` 和全量 `all` 链路也已接入本模块。

## 输出

输出目录：

```text
workspace/runs/default/results/composite_timing_strategies/
```

| 文件 | 含义 |
|---|---|
| `composite_strategy_specs.csv/parquet` | 两个正式策略的参数与规则 |
| `composite_rule_weights.csv/parquet` | 15 条规则在主策略和挑战者中的权重及频率统计 |
| `composite_strategy_daily.csv/parquet` | 两个策略的日度分数、仓位、收益和净值 |
| `composite_strategy_summary.csv/parquet` | 全样本绩效摘要 |
| `composite_strategy_trades.csv/parquet` | 信号日、执行日、开平仓方向和触发来源 |
| `composite_strategy_latest_status.csv/parquet` | 两个策略最新分数、仓位及最近开平仓日期 |
| `current_composite_signal.json` | 供程序读取的最新主策略与挑战者信号 |
| `current_composite_signal.md` | 供人工查看的最新信号摘要 |

## HTML 报告展示

`results/report/timing_report.html` 的第四个顶层模块为“复合策略模块”。模块内部使用两个同级标签：

- 类别等权两速复合策略。
- 开仓频率平方根倒数复合策略。

每个标签先展示加权方式、两速调仓与触发条件和下一交易日执行规则，再依次展示当前信号状态、价格与开平仓点、复合分数与仓位、策略/基准净值、超额净值和历史绩效。页面不使用“主策略/挑战者”称谓。

## 当前复现快照

基于截至 2026-07-10 的 selected-rule 输出，并扣除单边 5bp 换仓成本：

| 策略 | 年化收益 | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| 两速类别等权主策略 | 17.88% | 1.38 | -11.43% |
| 两速频率平方根倒数策略 | 17.52% | 1.34 | -14.31% |

频率平方根倒数策略全样本共有 6 次周中强事件开仓、2 次周中强事件平仓；其余开平仓由周频锚触发。

正式日更结果以输出目录内的最新表格为准。
