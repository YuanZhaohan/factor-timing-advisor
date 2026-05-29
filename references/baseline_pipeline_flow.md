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
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py score-update --csv .\skills\factor-timing-advisor\workspace\data\宽基得分.csv --output-dir .\skills\factor-timing-advisor\workspace\runs\default
```

当前会更新：

- `signals.csv`
- `event_forward_returns.csv`
- `open_close_trades.csv`
- `monthly_refresh_daily_score.csv`
- 每个 base 因子的最佳 `rule_pair`
- 基准 score 策略净值
- `advisor_summary.json / md / html`
- 全部图

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
  -> signals
  -> events
  -> rule_pair
  -> factor_signal_utility
  -> monthly_refresh_daily_score
  -> baseline score strategy
  -> report
  -> plot
```
