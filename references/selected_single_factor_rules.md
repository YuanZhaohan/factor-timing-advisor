# 单因子最优规则汇总

## 定位

`selected_single_factor_rules.py` 汇总当前已经确认的单因子最优持仓规则，只保留正式规则，不保留中间探索版本。

这个模块用于日度增量更新：

- 输入使用日更后生成的 `data/input_snapshot.parquet` 和 `results/signals/signals.parquet`。
- 不重新遍历全量 rule pair。
- 每次只按当前保留的最优规则刷新最新状态、逐笔交易和日度持仓。

正式脚本：

```text
skills/factor-timing-advisor/runtime/selected_single_factor_rules.py
```

单独运行：

```powershell
D:\anaconda\python.exe .\skills\factor-timing-advisor\scripts\run_pipeline.py selected-rules --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

日更链路 `score-update` 和全量链路 `all` 已自动刷新本模块。

说明：

- `score-update` 日更链路默认不再刷新 rule pair；rule pair 研究代码保留，但不作为日常增量报告的必跑步骤。
- 如果需要临时刷新 best rule-pair 输出，可以显式增加 `--refresh-rule-pair`。
- 单因子模块以后以人工确认后的最优规则为准，更新 `selected_single_factor_rules.py` 后由日更链路刷新最新状态。

## 输出

输出目录：

```text
skills/factor-timing-advisor/workspace/runs/default/results/selected_single_factor_rules/
```

输出表：

| 表 | 含义 |
|---|---|
| `selected_rule_specs.csv/parquet` | 当前保留的规则规格 |
| `selected_rule_latest_status.csv/parquet` | 每条规则最新持仓状态、待开/待平信号、最近开平仓日期 |
| `selected_rule_summary.csv/parquet` | 每条规则的历史表现摘要 |
| `selected_rule_trades.csv/parquet` | 逐笔交易明细 |
| `selected_rule_daily_positions.csv/parquet` | 每条规则的日度持仓 |


## 执行约定

- 所有开仓都在开仓信号日后一个交易日入场。
- 所有闭仓都在闭仓信号日后一个交易日出场。
- 等待确认规则只使用原始信号日之后、等待窗口内的数据。
- 止损只在持仓后动态判断，不提前生成静态信号。
- 辅助类指标只作为标签和过滤，不混入本表的独立持仓规则。
