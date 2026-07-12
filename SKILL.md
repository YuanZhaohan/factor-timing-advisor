---
name: factor-timing-advisor
description: 通过唯一 update 接口执行宽基因子择时的追加式一键更新，冻结历史输入，在暂存目录生成并验证 signals、score、单因子规则、复合策略、JSON/HTML 报告和图表后再安全切换正式结果；也支持显式全量重建和单模块维护。
metadata:
  openclaw:
    emoji: "📊"
---

# Factor Timing Advisor

这是宽基因子择时的正式 skill。优先使用 skill 内置脚本，不再调用项目根目录旧测试脚本。

项目根目录下的正式入口：

```text
skills/factor-timing-advisor/scripts/run_pipeline.py
```

默认只使用 `update` 子命令。不要直接调用内部的 `score-update`。

默认运行目录：

```text
workspace/runs/default/
```

## 数据格式约定

- 原始输入保留 CSV：`workspace/data/宽基得分.csv`。你每天只需要替换这个文件。
- 运行结果表统一保存为 Parquet：例如 `signals.parquet`、`event_forward_returns.parquet`、`monthly_refresh_daily_score.parquet`。
- 代码读取结果时会优先找 Parquet，找不到时兼容旧 CSV。
- JSON、MD、HTML 和图片保持原格式。

## 每日更新

每天只需要替换输入 CSV，然后运行一次 `update`。

### 0. 更新输入数据

把最新宽基数据放到：

```text
skills/factor-timing-advisor/workspace/data/宽基得分.csv
```

要求文件名和列名保持不变。原始 CSV 即使包含历史修订，`update` 也会以正式 `input_snapshot` 为冻结基线，只吸收严格晚于旧快照的新交易日，并记录被忽略的历史差异。

### 1. 运行唯一的一键接口

在项目根目录只执行：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py update
```

`update` 会自动：

1. 比较原始 CSV 与上次正式快照，冻结所有历史行，只追加新日期。
2. 把历史差异、输入哈希、代码哈希和新增日期写入审计记录。
3. 复制现有 run 到暂存目录，在暂存目录完成增量计算。
4. 验证关键表的历史前缀完全不变，并检查所有输出更新到同一最新日期。
5. 全部通过后才替换正式 `default`；失败时正式结果保持不动。

日更会刷新：

- `data/input_snapshot.parquet`
- `results/signals/signals.parquet`
- `results/events/event_forward_returns.parquet`
- `results/score/monthly_refresh_daily_score.parquet`
- `results/selected_single_factor_rules/selected_rule_latest_status.parquet`
- `results/selected_single_factor_rules/selected_rule_trades.parquet`
- `results/composite_timing_strategies/composite_strategy_latest_status.parquet`
- `results/composite_timing_strategies/current_composite_signal.json`
- `results/composite_timing_strategies/current_composite_signal.md`
- `results/strategy/monthly_strategy_summary_default.parquet`
- `results/strategy/monthly_strategy_best_equity_default.parquet`
- `results/report/advisor_summary.json`
- `results/report/advisor_summary.md`
- `results/report/current_signal_report.md`
- `results/report/signal_points_state.parquet`
- `results/report/signal_points_summary.parquet`
- `results/report/timing_report.html`
- `results/report/update_status.json`
- `results/report/update_history.jsonl`
- `plots/strategy/`
- `plots/factor/`
- `plots/rule_pair_best/`

日更为了速度不会重算：

```text
results/events/open_close_trades.parquet
```

这个文件属于重型 open-close 全组合统计，只在用户明确要求的 `all` 全量流程中更新。

### 2. 查看网页报告

日更完成后打开：

```text
skills/factor-timing-advisor/workspace/runs/default/results/report/timing_report.html
```

报告包含四块：

- 事件驱动模块：信号分布、看多/看空结构、净开仓量时序。
- 综合打分模块：抄底得分、逃顶得分、score 策略净值。
- 单因子规则模块：正式保留规则的当前状态、最优开仓/平仓规则与交互图。
- 复合策略模块：用两个同级策略标签分别展示策略逻辑、当前仓位、历史开平仓点、策略/基准净值、超额净值和绩效指标。

正常日更耗时目标：约 3 到 5 分钟内。

## 显式全量更新

不要定期自动全量重建。只有用户明确要求重建历史、规则、缓存或重型统计时，才运行 `all`。

在项目根目录执行：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py all --csv .\skills\factor-timing-advisor\workspace\data\宽基得分.csv --output-dir .\skills\factor-timing-advisor\workspace\runs\default
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

全量会改变历史结果，不属于默认的一键更新路径。执行前必须得到用户明确授权并单独保留修订说明。

## 常用命令

只重新生成报告：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py report --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

只重跑 score 策略：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py strategy --input-dir .\skills\factor-timing-advisor\workspace\runs\default --output-dir .\skills\factor-timing-advisor\workspace\runs\default
```

只重画图：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py plot --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

只刷新正式复合择时主策略与挑战者：

```powershell
& 'D:\anaconda\python.exe' .\skills\factor-timing-advisor\scripts\run_pipeline.py composite-strategies --input-dir .\skills\factor-timing-advisor\workspace\runs\default
```

## 什么时候必须全量

以下情况不要自行继续运行，先向用户说明需要显式全量重建：

1. 信号生成规则改了。
2. `rule_pair` 回测逻辑改了。
3. score 筛选或得分计算逻辑改了。
4. 用户明确要求采用历史 CSV 修订，而不是冻结历史、只追加新交易日。
5. `workspace/runs/default/results/score/cache/` 被删除。
6. 需要更新 `open_close_trades.parquet`。

## 目录约定

```text
factor-timing-advisor/
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
- `references/selected_single_factor_rules.md`
- `references/composite_timing_strategies.md`
- `references/safe_incremental_update.md`

## 代码仓库

当前 skill 同时维护 GitHub 和 Gitee 两个远端：

- GitHub: `https://github.com/YuanZhaohan/factor-timing-advisor.git`
- Gitee: `https://gitee.com/zhaohanyuan/factor-timing-advisor.git`

同步更新时建议两个远端都推：

```bash
git push origin main
git push gitee main
```

## 更新运行原则

- 严格按本 `SKILL.md` 的命令和路径执行，不绕路、不改路径、不自己写替代脚本。
- 不要为了“看起来更快”临时改代码、改输出目录、改文件名或跳过主流程步骤。
- 默认每次更新只运行唯一入口：`update`。
- 周末也不默认跑全量回测；只有用户明确要求“全量重跑”“全量回测”“跑 all”“重建 rule_pair / open_close_trades / score cache”时，才运行全量流程。
- 不要直接调用 `score-update`；它是供 `update` 暂存流程使用的内部命令，没有历史冻结和安全切换保护。
- `update` 自动检查 input snapshot、signals、score、单因子日度持仓、复合策略日度结果、基准策略净值和 HTML 的最新日期，不要绕过该检查单独生成 HTML。
- 程序运行慢时不要 kill 或中断进程。命令没输出不代表卡住，Plotly/HTML 图表生成耗时数分钟是正常的，必须耐心等待自然跑完。
- 判断失败前先读完整日志和完整报错堆栈，不要只看最后一行。
- 出错后先定位原始异常、输入日期、输出日期和缺失文件，再决定是否需要重跑。
- 不要删除中间文件、cache 或旧结果来“修复”增量更新。增量更新依赖已有中间文件，删除文件反而会破坏流程并导致重算或结果不一致。
- 只有用户明确要求清理、全量重建或删除文件时，才允许删除中间产物。
