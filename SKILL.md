---
name: factor-timing-advisor
description: 运行宽基因子择时主流程，生成 signals、事件回测、最佳 rule_pair、score 策略、JSON/HTML 报告和图表。适用于全量重跑和日常增量更新。
metadata:
  openclaw:
    emoji: "📊"
user-invocable: true
---

# Factor Timing Advisor

这是宽基因子择时的正式 skill。优先使用 skill 内置脚本，不再调用项目根目录旧测试脚本。

正式入口：

```text
skills/factor-timing-advisor/scripts/run_pipeline.py
```

默认运行目录：

```text
skills/factor-timing-advisor/workspace/runs/default/
```

## 数据格式约定

- 原始输入保留 CSV：`workspace/data/宽基得分.csv`。你每天只需要替换这个文件。
- 运行结果表统一保存为 Parquet：例如 `signals.parquet`、`event_forward_returns.parquet`、`monthly_refresh_daily_score.parquet`。
- 代码读取结果时会优先找 Parquet，找不到时兼容旧 CSV。
- JSON、MD、HTML 和图片保持原格式。

## 每日更新

每天只需要替换输入 CSV，然后跑 `score-update`。

### 0. 拉取最新数据（每次运行前必做）

运行前先从 GitHub 拉取最新的 `宽基得分.csv`，确保数据目录已同步：

```bash
git pull origin main
```

如果 `workspace/data/宽基得分.csv` 有更新，再继续后续步骤。

### 1. 更新输入数据

把最新宽基数据放到：

```text
skills/factor-timing-advisor/workspace/data/宽基得分.csv
```

要求：

- 文件名保持 `宽基得分.csv`，命令不用改。
- 日期列追加到最新交易日。
- 列名和历史文件保持一致。
- 如果回填或重写了历史数据，不要只跑日更，改跑周末全量流程。

### 2. 跑日更链路

在项目根目录执行：

```bash
python skills/factor-timing-advisor/scripts/run_pipeline.py score-update --csv skills/factor-timing-advisor/workspace/data/宽基得分.csv --output-dir skills/factor-timing-advisor/workspace/runs/default
```

日更会刷新：

- `data/input_snapshot.parquet`
- `results/signals/signals.parquet`
- `results/events/event_forward_returns.parquet`
- `results/score/monthly_refresh_daily_score.parquet`
- `results/rule_pair/rule_pair_best_base_summary.parquet`
- `results/rule_pair/rule_pair_best_base_equity_curves.parquet`
- `results/strategy/monthly_strategy_summary_default.parquet`
- `results/strategy/monthly_strategy_best_equity_default.parquet`
- `results/report/advisor_summary.json`
- `results/report/advisor_summary.md`
- `results/report/current_signal_report.md`
- `results/report/signal_points_state.parquet`
- `results/report/signal_points_summary.parquet`
- `results/report/timing_report.html`
- `plots/strategy/`
- `plots/factor/`
- `plots/rule_pair_best/`

日更为了速度不会重算：

```text
results/events/open_close_trades.parquet
```

这个文件属于重型 open-close 全组合统计，放到周末全量流程里更新。

### 3. 查看网页报告

日更完成后打开：

```text
skills/factor-timing-advisor/workspace/runs/default/results/report/timing_report.html
```

报告包含三块：

- 事件驱动模块：信号分布、看多/看空结构、净开仓量时序。
- 综合打分模块：抄底得分、逃顶得分、score 策略净值。
- 单因子规则模块：每个 base 因子的最优开仓/平仓规则与交互图。

正常日更耗时目标：约 3 到 5 分钟内。

## 每周末全量更新

每周末建议跑一次全量更新，用来刷新日更跳过的重型统计、rule_pair 全量扫描、score cache 和全部图表。

在项目根目录执行：

```bash
python skills/factor-timing-advisor/scripts/run_pipeline.py all --csv skills/factor-timing-advisor/workspace/data/宽基得分.csv --output-dir skills/factor-timing-advisor/workspace/runs/default
```

周末全量会覆盖更新：

- `results/events/open_close_trades.parquet`
- `results/rule_pair/rule_pair_summary.parquet`
- `results/rule_pair/equity_curves.parquet`
- `results/rule_pair/rule_pair_summary_by_year_end.parquet`
- `results/score/factor_signal_utility.parquet`
- `results/score/cache/`
- `results/score/monthly_refresh_daily_score.parquet`
- `results/report/timing_report.html`
- `plots/`

原则：

- 工作日只追加最新交易日时，跑 `score-update`。
- 周末、规则变化、历史数据变化后，跑 `all`。
- 如果全量耗时较长，这是正常的；它负责补齐日更为了速度跳过的全量研究统计。

## 常用命令

只重新生成报告：

```bash
python skills/factor-timing-advisor/scripts/run_pipeline.py report --input-dir skills/factor-timing-advisor/workspace/runs/default
```

只重跑 score 策略：

```bash
python skills/factor-timing-advisor/scripts/run_pipeline.py strategy --input-dir skills/factor-timing-advisor/workspace/runs/default --output-dir skills/factor-timing-advisor/workspace/runs/default
```

只重画图：

```bash
python skills/factor-timing-advisor/scripts/run_pipeline.py plot --input-dir skills/factor-timing-advisor/workspace/runs/default
```

## 什么时候必须全量

以下情况不要只跑 `score-update`，要跑 `all` 或至少 `upstream`：

1. 信号生成规则改了。
2. `rule_pair` 回测逻辑改了。
3. score 筛选或得分计算逻辑改了。
4. 历史 CSV 被回填，不是只追加最新交易日。
5. `workspace/runs/default/results/score/cache/` 被删除。
6. 需要更新 `open_close_trades.parquet`。

## 目录约定

```text
skills/factor-timing-advisor/
  agents/
  references/
  runtime/
  scripts/
  workspace/
    data/
      宽基得分.csv
    runs/
      default/
        data/
        results/
          signals/
          events/
          rule_pair/
          score/
          strategy/
          report/
        plots/
          strategy/
          factor/
          rule_pair_best/
```

含义：

- `runtime/`：正式 Python 代码。
- `scripts/`：命令行入口封装。
- `references/`：因子分类、输出解释、信号规则说明等文档。
- `workspace/data/`：原始输入 CSV。
- `workspace/runs/default/`：默认生产结果。
- `results/`：Parquet 表格、JSON、HTML 等结果。
- `plots/`：图片输出。

## 参考文档

按需读取：

- `references/factor_taxonomy.md`
- `references/output_interpretation.md`
- `references/signal_policy.md`
- `references/开平仓规则说明.md`
- `references/baseline_pipeline_flow.md`
- `references/event_condition_checklist.md`
## 代码仓库

当前 skill 同时维护 GitHub 和 Gitee 两个远端：

- GitHub: `https://github.com/YuanZhaohan/factor-timing-advisor.git`
- Gitee: `https://gitee.com/zhaohanyuan/factor-timing-advisor.git`

同步更新时建议两个远端都推：

```bash
git push origin main
git push gitee main
```
