# 基准主线流程

当前正式代码已经收进：

- `skills/factor-timing-advisor/runtime/`

正式入口是：

- `skills/factor-timing-advisor/scripts/run_pipeline.py`

## 运行顺序

### 1. 第一次全量跑

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py all --csv .\skills\factor-timing-advisor\workspace\data\宽基得分.csv --output-dir .\skills\factor-timing-advisor\workspace\runs\default
```

这一步会依次执行：

1. `upstream`
2. `strategy`
3. `report`
4. `plot`

### 2. 日常增量更新

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py update
```

这是唯一的人工日更入口。它冻结旧 `input_snapshot` 的历史行，只追加新日期，在暂存目录运行并验证后才切换正式结果。

当前会更新：

- `signals.parquet`
- `event_forward_returns.parquet`
- `monthly_refresh_daily_score.parquet`
- 基准 score 策略净值
- 正式单因子规则
- 两套复合择时策略
- `advisor_summary.json / md / html`
- 全部图
- `update_status.json / update_history.jsonl`

日更不会刷新重型 `open_close_trades` 和全量 `rule_pair` 扫描。

### 3. 只重跑基准 score 策略

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py strategy --input-dir .\skills\factor-timing-advisor\workspace\runs\default --output-dir .\skills\factor-timing-advisor\workspace\runs\default
```

### 4. 只更新报告

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py report --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

### 5. 只重画图

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py plot --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

## 输出目录

```text
{run_dir}/
  data/
  results/
  plots/
```

- `data/`：源数据快照
- `results/`：表格、回测结果、报告
- `plots/`：图

## 主线结构

```text
原始 CSV
  -> append-only preflight（冻结历史，只取新日期）
  -> staging run
  -> signals
  -> events
  -> rule_pair
  -> factor_signal_utility
  -> monthly_refresh_daily_score
  -> baseline score strategy
  -> selected single-factor rules
  -> composite timing strategies
  -> report
  -> plot
  -> history/date verification
  -> promote staging to default
```
