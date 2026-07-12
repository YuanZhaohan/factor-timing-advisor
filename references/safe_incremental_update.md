# 一键追加式更新

## 唯一命令

在项目根目录执行：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py update
```

不要把内部 `score-update` 当作人工入口。`score-update` 会直接写指定结果目录；`update` 才包含历史冻结、暂存运行、验证和安全切换。

## 历史冻结规则

正式基线是：

```text
workspace/runs/default/data/input_snapshot.parquet
```

每次更新会对比原始 `宽基得分.csv`：

- 旧快照中已经存在的 `代码 + 日期` 行始终保留旧值。
- 只吸收严格晚于该代码旧最大日期的新行。
- 历史字段变化、历史行删除和历史回填不会进入本次计算，只记录到审计清单。
- 字段集合变化会直接失败，避免把新旧 schema 混在一次增量更新中。

## 安全切换

更新先复制正式 run 到同级暂存目录，再在暂存目录计算。完成后检查：

- `input_snapshot`
- `signals`
- `monthly_refresh_daily_score`
- `selected_rule_daily_positions`
- `composite_strategy_daily`
- `monthly_strategy_best_equity_default`
- `timing_report.html`

表格必须更新到同一最新日期；关键表在旧快照日期及以前的历史前缀必须保持不变。全部通过后才替换正式 `default`。失败时正式目录不变，失败暂存目录和错误堆栈会保留用于排查。

## 追溯记录

最新一次状态：

```text
workspace/runs/default/results/report/update_status.json
```

所有运行历史：

```text
workspace/runs/default/results/report/update_history.jsonl
```

记录内容包括输入文件哈希、安全输入指纹、代码哈希、新增日期、被忽略的历史变化、关键输出日期、输出文件哈希、历史前缀指纹、运行结果和错误堆栈。

旧输入快照会保存在：

```text
workspace/runs/default/data/history/
```

## 预检

只查看本次会追加什么、忽略什么，不计算正式结果：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py update --dry-run
```

若没有新增交易日，重复运行 `update` 会返回 `no_new_rows`，不会重算策略，只追加一条审计记录。
