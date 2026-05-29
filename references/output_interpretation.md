# 回测结果解读

本文件说明如何把 `signals.csv`、事件研究、开平仓交易统计和历史净值结果转成择时建议。

## 读取顺序

先检查 `results` 目录是否已有信号表、回测文件、当前信号点状态文本和最终择时摘要。已有结果应直接读取，不要为了写建议重复跑耗时回测。只有缺少必要文件时，才按 `SKILL.md` 的分步命令补齐。

如果是多指数数据，先进入对应指数目录，例如 `results/000985.CSI_中证全指/`，再读取该目录下的结果文件。不要跨指数混读不同目录下的回测结果。

结构化文本应在回测文件之后生成，保证择时建议可以同时引用当前状态和历史证据。

1. 先读 `advisor_summary.json`，获取脚本化的最终结论、分数和固定判断规则。
2. 再读 `advisor_summary.md`，获取人类可读摘要。
3. 再读 `current_signal_report.md`，获取确定性的当前信号点状态摘要。
4. 再读 `signal_points_summary.csv`，查看当前多/空/观望在因子类别、周期和开仓规则类型上的分布。
5. 再读 `signal_points_state.csv`，必要时查看每个 `指数 + 因子 + 开仓规则` 点位的全量状态。
6. 再读 `signals.csv`，看当前或最近窗口触发了哪些原始事件。
7. 用 `factor_taxonomy.md` 给每个信号补充字段类别。
8. 用 `signal_policy.md` 判断这些信号是赔率、胜率还是辅助。
9. 再读回测结果，验证这些信号历史上是否有效。
10. 最后解释为偏多、中性、降仓或观望；如果 `advisor_summary.json` 已存在，最终观点必须使用其中的 `conclusion`。

## 最终确定性摘要

`advisor_summary.json` 是机器可读的最终事实摘要，包含：

- conclusion：脚本按固定阈值给出的最终观点。
- scores：`total_score`、`core_score`、`interpretable_ratio`。
- state_counts：多/空/观望点位数量。
- evidence_counts：按字段默认方向折算后的看多/看空/风险缓和/待确认数量。
- category_evidence：按因子类别折算后的证据。
- frequency_evidence：按原始、季线、年线折算后的证据。
- top_bullish_points / top_bearish_points：贡献最大的点位。
- rule_backtest_overview：规则组合历史超额表现概览。
- decision_rules：脚本使用的固定判断规则。

大模型输出择时建议时，不能自行覆盖 `conclusion`，只能解释为什么脚本给出这个结论，以及有哪些限制。

## 当前信号点状态

`signal_points_state.csv` 是确定性事实表，每一行对应一个：

```text
instrument + factor + open_pattern
```

重点字段：

- current_state：当前是 `多`、`空` 还是 `观望`。
- factor_category：因子类别，例如 `赔率/估值`、`胜率/资金`。
- frequency：因子周期，取 `原始`、`季线`、`年线`。
- signal_role：`核心开仓` 或 `辅助观察`。
- open_rule_style：核心开仓的抄底、低位修复、趋势确认、追高/动量等；辅助指标的风险过滤、结构确认、分歧观察等。
- open_rule_stage：该规则所代表的具体阶段。
- open_pattern_family：开仓规则类型，例如上穿、下方拐点、低位均值回复。
- state_start_date：当前状态从哪一天开始。
- state_age_days：当前状态已经延续多少个交易日。
- expected_remaining_days：按当前状态的历史平均持续期估算还剩多少交易日。
- last_signal_date：最近一次影响该点位状态的事件日期。
- last_signal_pattern：最近一次影响该点位状态的规则。
- mean_long_days / median_long_days：历史上该点位从开仓到下一次静态闭仓事件的平均/中位持续天数。
- mean_flat_days / median_flat_days：历史上该点位从闭仓/反向事件到下一次开仓事件的平均/中位持续天数。

使用规则：

- 大模型给建议时，必须优先引用 `signal_points_state.csv` 和 `signal_points_summary.csv` 的确定性状态。
- `current_state=多` 表示该点位最近一次有效状态切换来自它自己的开仓规则。
- `current_state=空` 表示该因子最近一次有效状态切换来自任一静态闭仓/反向事件；这里不是做空，只是 long/cash 里的空仓或偏谨慎。
- `current_state=观望` 表示该点位尚未出现过有效开仓或闭仓状态切换。
- `signal_role=核心开仓` 才能作为开仓证据，并进一步区分抄底、趋势确认、追高等风格。
- `signal_role=辅助观察` 只能作为风险过滤、结构确认或分歧观察，不能单独作为开仓依据。
- 没有新事件的日期延续上一状态，因此不要只看最新一天触发数量。
- `expected_remaining_days <= 0` 表示当前状态已超过或接近历史平均持续期；如果当前是 `多`，要提高退出关注度；如果当前是 `空`，要关注是否接近下一次修复窗口。
- 不要把 `open_condition + close_condition` 组合当作当前多空投票；它只用于历史净值回测和交易配对统计。

## 当前信号窗口

建议至少看三个窗口：

- 最近 1 个交易日：最新边际变化。
- 最近 5 个交易日：短期信号集中度。
- 最近 20 个交易日：中期状态变化。

对每个窗口统计：

- 开仓信号数量
- 闭仓信号数量
- 各类别信号数量
- 触发信号的因子数量
- 是否集中在少数几个相似因子
- 是否出现赔率、胜率、辅助风险共振

## 事件收益

事件收益用于判断某个单独信号触发后，未来固定持有期是否有统计优势。

重点看：

- count：事件次数，太少不可靠。
- mean_return：平均收益。
- median_return：中位数收益，优先级高于平均收益。
- win_rate：胜率。
- p25_return / p75_return：收益分布是否稳定。

使用规则：

- `count` 很少的信号只能作为观察，不能作为主要依据。
- 平均收益为正但中位数为负，说明可能依赖少数极端样本。
- 胜率高但收益小，需要结合回撤和持仓期判断。

## 开仓到闭仓交易收益

开平仓交易统计用于判断一个开仓规则和一个闭仓规则配对后的单笔交易质量。

重点看：

- trade_count
- mean_trade_return
- median_trade_return
- mean_annualized_trade_return
- median_annualized_trade_return
- max_drawdown
- win_rate
- mean_holding_days

使用规则：

- `median_trade_return` 比 `mean_trade_return` 更稳健。
- `median_annualized_trade_return` 比 `mean_annualized_trade_return` 更稳健。
- `max_drawdown` 过大时，即使收益高也要降权。
- `trade_count` 太少时，不能作为主要证据。

## 历史净值

历史净值用于判断规则组合是否形成可投资的择时策略。

重点看：

- annual_return
- benchmark_annual_return
- excess_annual_return
- max_drawdown
- benchmark_max_drawdown
- excess_max_drawdown
- sharpe
- holding_ratio
- turnover

使用规则：

- 优先看 `excess_annual_return`，因为目标是相对一直持有指数有超额。
- 如果策略收益高但最大回撤也显著更大，需要降权。
- `holding_ratio` 太低的策略可能只是偶然避开下跌。
- `turnover` 太高时，要提示交易成本和稳定性问题。

## 最低可信度门槛

不要只按收益最高排序。建议使用以下门槛做初筛：

- 交易次数不能太少。
- 中位数收益不能明显为负。
- 最大回撤不能明显失控。
- 超额收益不能只来自少数样本。
- 信号最好来自多个类别，而不是同一类重复信号。
- 辅助类信号不能单独作为开仓依据。

## 多信号综合框架

### 偏多

满足多数条件：

- 赔率/估值或赔率/筹码出现修复信号。
- 胜率/量出现扩散改善或技术改善。
- 胜率/资金出现资金改善。
- 辅助/风险状态没有恶化。
- 辅助/资金分歧没有明显恶化。
- 对应规则历史上中位数收益、超额收益和回撤表现可接受。

### 中性

常见情形：

- 开仓和闭仓信号混杂。
- 赔率信号和胜率信号不共振。
- 只有单一类别信号触发。
- 回测表现一般或分布不稳定。

### 降仓

满足多数条件：

- 胜率/量转弱。
- 胜率/资金高位回落。
- 辅助/风险状态恶化，例如波动率上行。
- 辅助/资金分歧恶化，例如情绪熵高、轮动变快、下行抛压加大。
- 筹码或估值赔率已经不便宜。

### 观望

常见情形：

- 信号数量少。
- 主要信号来自辅助类。
- 关键规则交易次数太少。
- 回测结果冲突明显。
- 字段含义尚未确认。

## 输出模板

输出择时建议时使用：

1. 当前结论：
   偏多 / 中性 / 降仓 / 观望

2. 当前信号概览：
   当前信号点多/空/观望数量，按因子类别、周期、开仓规则类型拆分；再补充最近 1 / 5 / 20 日新增事件。

3. 偏多证据：
   哪些赔率和胜率类因子触发了开仓信号，历史表现如何。

4. 风险证据：
   哪些辅助风险、资金分歧、筹码压力或闭仓信号正在触发。

5. 回测证据：
   引用关键规则组合的收益、回撤、胜率和交易次数。

6. 冲突信号：
   哪些信号互相矛盾，哪些证据需要降权。

7. 建议动作：
   维持、加仓、降仓、等待确认，或仅观察。

8. 注意事项：
   样本不足、信号冲突、回撤过大、交易成本、字段含义待确认等。
