# factor-timing-advisor

宽基因子择时研究 OpenClaw/Codex skill，提供每日报告生成与策略分析。

从 [SKILL.md](SKILL.md) 开始了解：

- 每日数据刷新与 HTML 报告生成
- 周末全量重建流程
- 输出目录约定
- 命令行入口

默认输入：

```text
workspace/data/宽基得分.csv
```

运行结果以 Parquet 格式存储。输入 CSV 保持 CSV 格式以便每日替换。

默认 HTML 报告：

```text
workspace/runs/default/results/report/timing_report.html
```

## 数据来源

`workspace/data/宽基得分.csv` 由 dududu 每日手动上传至仓库更新。如果数据未及时刷新，可催更。