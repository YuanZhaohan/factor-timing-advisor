# factor-timing-advisor

OpenClaw/Codex skill for broad-index factor timing research and daily report generation.

Start with [SKILL.md](SKILL.md) for:

- daily data refresh and HTML report generation
- weekend full rebuild workflow
- output directory conventions
- command-line entry points

Default input:

```text
workspace/data/宽基得分.csv
```

Runtime table outputs are stored as Parquet. The input CSV is intentionally kept as CSV for daily replacement.

Default HTML report:

```text
workspace/runs/default/results/report/timing_report.html
```
