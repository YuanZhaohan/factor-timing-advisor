from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from plotly.io._html import get_plotlyjs
except Exception:
    from plotly.offline import get_plotlyjs

from interactive_report import (
    _fig_html,
    _load_factor_descriptions,
    _make_recent_signal_chart_html,
    _make_rule_pair_html,
    _make_score_z20_html,
    _make_strategy_html,
    _read_csv,
    _signal_counts,
)
from baseline_score_strategy import format_rule_name_cn
from timing_config import CODE_COL, DATE_COL


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _category_colors() -> dict[str, str]:
    return {
        "赔率/资金": "#3498db",
        "赔率/估值": "#2ecc71",
        "赔率/筹码": "#e67e22",
        "胜率/估值": "#9b59b6",
        "辅助/筹码结构": "#1abc9c",
        "辅助/资金分歧": "#e74c3c",
        "辅助/风险状态": "#95a5a6",
    }


def _escape(v: Any) -> str:
    return html_escape("" if v is None else str(v))


def _pick_code_col(df: pd.DataFrame) -> str:
    if CODE_COL in df.columns:
        return CODE_COL
    if df.empty:
        return CODE_COL
    return str(df.columns[0])


def _signal_structure_tables(advisor: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signal_structure = advisor.get("signal_structure", {})
    bullish = signal_structure.get("bullish", {}).get("breakdown", []) or []
    bearish = signal_structure.get("bearish", {}).get("breakdown", []) or []
    return bullish, bearish


def build_view_data(input_dir: str | Path, taxonomy_path: str | Path | None = None, report_title: str = "宽基择时信号报告") -> dict[str, Any]:
    input_dir = Path(input_dir)
    results_dir = input_dir / "results"

    advisor = _read_json(results_dir / "report" / "advisor_summary.json")
    factor_desc_map = _load_factor_descriptions(taxonomy_path)

    input_df = _read_csv(input_dir, ["data/input_snapshot.csv", "input_snapshot.csv"])
    strategy_df = _read_csv(
        input_dir,
        [
            "results/strategy/monthly_strategy_best_equity_default.csv",
            "results/strategy/monthly_strategy_best_equity.csv",
            "monthly_strategy_best_equity_default.csv",
            "monthly_strategy_best_equity.csv",
        ],
    )
    strategy_summary_df = _read_csv(
        input_dir,
        [
            "results/strategy/monthly_strategy_summary_default.csv",
            "results/strategy/monthly_strategy_summary.csv",
            "monthly_strategy_summary_default.csv",
            "monthly_strategy_summary.csv",
        ],
    )
    signals_df = _read_csv(
        input_dir,
        [
            "results/signals/signals.csv",
            "signals/signals.csv",
            "signals.csv",
        ],
    )
    rule_best_summary_df = _read_csv(
        input_dir,
        [
            "results/rule_pair/rule_pair_best_base_summary.csv",
            "rule_pair_best_base_summary.csv",
            "results/rule_pair/rule_pair_summary.csv",
            "rule_pair_summary.csv",
        ],
        optional=True,
    )
    rule_best_equity_df = _read_csv(
        input_dir,
        [
            "results/rule_pair/rule_pair_best_base_equity_curves.csv",
            "rule_pair_best_base_equity_curves.csv",
            "results/rule_pair/equity_curves.csv",
            "equity_curves.csv",
        ],
        optional=True,
    )

    if not strategy_df.empty and DATE_COL in strategy_df.columns:
        strategy_df[DATE_COL] = pd.to_datetime(strategy_df[DATE_COL], errors="coerce")
        strategy_df = strategy_df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if not strategy_summary_df.empty and "excess_annual_return" in strategy_summary_df.columns:
        strategy_summary_df = strategy_summary_df.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)
    if not rule_best_summary_df.empty and "excess_annual_return" in rule_best_summary_df.columns:
        rule_best_summary_df = rule_best_summary_df.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)
    if not rule_best_equity_df.empty and DATE_COL in rule_best_equity_df.columns:
        rule_best_equity_df[DATE_COL] = pd.to_datetime(rule_best_equity_df[DATE_COL], errors="coerce")
        rule_best_equity_df = rule_best_equity_df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    strategy_plot_html = ""
    strategy_z20_html = ""
    recent_signal_chart_html = ""
    if not strategy_df.empty:
        summary_row = strategy_summary_df.iloc[0] if not strategy_summary_df.empty else None
        strategy_plot_html = _make_strategy_html(strategy_df, summary_row=summary_row)
        strategy_z20_html = _make_score_z20_html(strategy_df)
    if not input_df.empty and not signals_df.empty:
        recent_signal_chart_html = _make_recent_signal_chart_html(input_df, signals_df, default_visible_days=756)

    rule_pair_cards: list[dict[str, Any]] = []
    if not rule_best_summary_df.empty:
        code_col = _pick_code_col(rule_best_summary_df)
        equity_code_col = _pick_code_col(rule_best_equity_df) if not rule_best_equity_df.empty else CODE_COL
        for _, row in rule_best_summary_df.iterrows():
            factor = str(row.get("factor", ""))
            base_factor = str(row.get("base_factor", factor))
            desc = factor_desc_map.get(base_factor, factor_desc_map.get(factor, {}))
            row_copy = row.copy()
            if not rule_best_equity_df.empty:
                mask = (
                    rule_best_equity_df[equity_code_col].astype(str).eq(str(row[code_col]))
                    & rule_best_equity_df["factor"].astype(str).eq(str(row["factor"]))
                    & rule_best_equity_df["open_condition"].astype(str).eq(str(row["open_condition"]))
                    & rule_best_equity_df["close_condition"].astype(str).eq(str(row["close_condition"]))
                )
                if "base_factor" in rule_best_equity_df.columns and "base_factor" in row.index:
                    mask &= rule_best_equity_df["base_factor"].astype(str).eq(str(row["base_factor"]))
                row_copy.attrs["_equity_df"] = rule_best_equity_df.loc[mask].copy()
            try:
                chart_html = _make_rule_pair_html(input_df, signals_df, row_copy, desc)
            except Exception as exc:
                chart_html = (
                    "<div class='chart-error'>"
                    f"无法生成交互图：{html_escape(str(factor))} | {html_escape(str(exc))}"
                    "</div>"
                )
            rule_pair_cards.append(
                {
                    "factor": factor,
                    "desc": desc,
                    "chart_html": chart_html,
                    "open_condition": row.get("open_condition", ""),
                    "close_condition": row.get("close_condition", ""),
                }
            )

    sc = advisor.get("state_counts", {})
    bullish_count = int(sc.get("多", 0))
    bearish_count = int(sc.get("空", 0))
    watch_count = int(sc.get("观望", 0))
    total_sig = bullish_count + bearish_count + watch_count
    bullish_pct = round(bullish_count / total_sig * 100, 1) if total_sig else 0.0
    bearish_pct = round(bearish_count / total_sig * 100, 1) if total_sig else 0.0

    conclusion = str(advisor.get("conclusion", "观望"))
    conclusion_color = {"偏多": "#2ecc71", "减仓": "#e74c3c", "中性": "#f39c12", "观望": "#95a5a6"}.get(conclusion, "#95a5a6")
    bullish_structure, bearish_structure = _signal_structure_tables(advisor)

    signal_path = results_dir / "signals" / "signals.csv"
    return {
        "title": report_title,
        "latest": advisor.get("latest_date"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "conclusion": conclusion,
        "conclusion_color": conclusion_color,
        "scores": advisor.get("scores", {}),
        "state_counts": sc,
        "category_evidence": advisor.get("category_evidence", []),
        "category_colors": _category_colors(),
        "total_sig": total_sig,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "signal_info": _signal_counts(signal_path, 20),
        "recent_signal_chart_html": recent_signal_chart_html,
        "strategy_plot_html": strategy_plot_html,
        "strategy_z20_html": strategy_z20_html,
        "rule_pair_cards": rule_pair_cards,
        "bullish_structure": bullish_structure,
        "bearish_structure": bearish_structure,
    }


def _render_structure_rows(rows: list[dict[str, Any]], label_key: str) -> str:
    parts = []
    for row in rows:
        parts.append(
            "<tr>"
            f"<td><b>{_escape(row.get(label_key, ''))}</b></td>"
            f"<td>{int(row.get('count', 0) or 0)}</td>"
            f"<td>{float(row.get('count_share', 0) or 0) * 100:.1f}%</td>"
            f"<td>{float(row.get('score_sum', 0) or 0):.3f}</td>"
            f"<td>{int(row.get('core_count', 0) or 0)}</td>"
            f"<td>{int(row.get('auxiliary_count', 0) or 0)}</td>"
            "</tr>"
        )
    return "".join(parts)


def _render_evidence_rows(v: dict[str, Any]) -> str:
    rows = []
    for cat in v["category_evidence"]:
        cname = str(cat["factor_category"])
        net_score = float(cat["net_score"])
        score_color = "#e74c3c" if net_score < -0.3 else "#f39c12" if net_score < 0 else "#2ecc71"
        ccol = v["category_colors"].get(cname, "#888")
        rows.append(
            "<tr>"
            f"<td style='color:{ccol}'><b>{_escape(cname)}</b></td>"
            f"<td>{int(cat['看多'])}</td>"
            f"<td>{int(cat['看空'])}</td>"
            f"<td>{int(cat['风险缓和'])}</td>"
            f"<td>{int(cat['待确认'])}</td>"
            f"<td>{int(cat.get('中性', 0))}</td>"
            f"<td style='color:{score_color}'><b>{net_score:.3f}</b></td>"
            f"<td><b>{_escape(cat['主证据'])}</b></td>"
            "</tr>"
        )
    return "".join(rows)


def _render_category_bars(v: dict[str, Any]) -> str:
    html = []
    for cat in v["category_evidence"]:
        total = int(cat["total"])
        bullish = round(int(cat["看多"]) / total * 100, 1) if total else 0.0
        bearish = round(int(cat["看空"]) / total * 100, 1) if total else 0.0
        neutral = max(0.0, round(100 - bullish - bearish, 1))
        net_score = float(cat["net_score"])
        score_color = "#e74c3c" if net_score < -0.3 else "#f39c12" if net_score < 0 else "#2ecc71"
        ccol = v["category_colors"].get(str(cat["factor_category"]), "#888")
        html.append(
            "<div class='cat-bar-row'>"
            f"<div class='cat-bar-label' style='color:{ccol}'>{_escape(cat['factor_category'])}</div>"
            "<div class='cat-bar-track'>"
            f"<div class='cat-bar-bearish' style='width:{bearish}%'></div>"
            f"<div class='cat-bar-neutral' style='width:{neutral}%'></div>"
            f"<div class='cat-bar-bullish' style='width:{bullish}%'></div>"
            "</div>"
            f"<div class='cat-bar-score' style='color:{score_color}'>{net_score:.2f}</div>"
            "</div>"
        )
    return "".join(html)


def _category_root(category: str) -> str:
    text = str(category or "").strip()
    if not text:
        return "其他"
    return text.replace("／", "/").split("/")[0].strip() or "其他"


def _render_rule_filter(v: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for rp in v["rule_pair_cards"]:
        root = _category_root((rp.get("desc") or {}).get("category", ""))
        counts[root] = counts.get(root, 0) + 1
    preferred = ["胜率", "赔率", "辅助", "其他"]
    roots = [r for r in preferred if r in counts] + sorted(r for r in counts if r not in preferred)
    options = "".join(
        "<label class='rule-filter-option'>"
        f"<input type='checkbox' class='rule-category-check' value='{_escape(root)}' checked onchange='updateRulePairFilter()'>"
        f"<span>{_escape(root)}</span><em>{counts[root]}</em>"
        "</label>"
        for root in roots
    )
    total = sum(counts.values())
    return (
        "<div class='rule-filter'>"
        "<details class='rule-filter-dropdown'>"
        f"<summary>筛选因子类别 <span id='rule-filter-summary'>全部 {total}</span></summary>"
        "<div class='rule-filter-menu'>"
        "<label class='rule-filter-option rule-filter-all'>"
        "<input type='checkbox' id='rule-filter-all' checked onchange='toggleAllRuleCategories(this)'>"
        f"<span>全部</span><em>{total}</em>"
        "</label>"
        f"{options}"
        "</div>"
        "</details>"
        "</div>"
    )


def _render_rule_pair_cards(v: dict[str, Any]) -> str:
    cards = []
    for rp in v["rule_pair_cards"]:
        desc = rp["desc"] or {}
        category = str(desc.get("category", "") or "")
        category_root = _category_root(category)
        parts = []
        if category:
            parts.append(
                f"<span class='factor-cat' style='background:{v['category_colors'].get(category, '#888')}'>{_escape(category)}</span>"
            )
        if desc.get("meaning"):
            parts.append(f"<span class='factor-meaning'>{_escape(desc['meaning'])}</span>")
        meta = "<div class='factor-meta'>" + "".join(parts) + "</div>" if parts else ""
        extra = []
        if desc.get("direction"):
            extra.append(f"<div class='factor-extra'>方向：{_escape(desc['direction'])}</div>")
        if desc.get("observation"):
            extra.append(f"<div class='factor-extra'>观察：{_escape(desc['observation'])}</div>")
        if desc.get("note"):
            extra.append(f"<div class='factor-extra'>注意：{_escape(desc['note'])}</div>")
        open_rule = format_rule_name_cn(str(rp.get("open_condition", "")))
        close_rule = format_rule_name_cn(str(rp.get("close_condition", "")))
        rule_lines = (
            "<div class='rule-pair-rules'>"
            f"<div><b>开仓规则：</b>{_escape(open_rule)}</div>"
            f"<div><b>平仓规则：</b>{_escape(close_rule)}</div>"
            "</div>"
        )
        cards.append(
            f"<div class='rule-pair-card' data-category-root='{_escape(category_root)}' data-category='{_escape(category)}'>"
            "<div class='rule-pair-header'>"
            f"<h3>{_escape(rp['factor'])}</h3>"
            f"{meta}{rule_lines}{''.join(extra)}"
            "</div>"
            f"{rp['chart_html']}"
            "</div>"
        )
    return "".join(cards)


def render_html(v: dict[str, Any]) -> str:
    plotly_js = get_plotlyjs()
    nw_pct = round(100 - v["bullish_pct"] - v["bearish_pct"], 1)
    scores = v["scores"]
    sc = v["state_counts"]
    bullish_count = int(sc.get("多", 0))
    bearish_count = int(sc.get("空", 0))
    watch_count = int(sc.get("观望", 0))
    core_score = float(scores.get("core_score", 0.0))
    total_score = float(scores.get("total_score", 0.0))
    interpretable_ratio = float(scores.get("interpretable_ratio", 0.0)) * 100
    cat_interp_parts = []
    for cat in v["category_evidence"]:
        net_score = float(cat["net_score"])
        score_cls = "pill-bullish" if net_score > 0.3 else "pill-bearish" if net_score < -0.3 else "pill-neutral"
        evidence = str(cat["主证据"])
        evidence_cls = (
            "pill-bullish" if evidence == "看多"
            else "pill-bearish" if evidence == "看空"
            else "pill-risk" if evidence == "风险缓和"
            else "pill-watch"
        )
        cat_interp_parts.append(
            f"<li><b>{_escape(cat['factor_category'])}</b>："
            f"净得分 <span class='keyword-pill {score_cls}'>{net_score:.3f}</span>，"
            f"主证据 <span class='keyword-pill {evidence_cls}'>{_escape(evidence)}</span> "
            f"<span class='mini-count pill-bullish'>看多 {int(cat['看多'])}</span>"
            f"<span class='mini-count pill-bearish'>看空 {int(cat['看空'])}</span>"
            f"<span class='mini-count pill-risk'>风险缓和 {int(cat['风险缓和'])}</span>"
            f"<span class='mini-count pill-watch'>待确认 {int(cat['待确认'])}</span>"
            f"<span class='mini-count pill-neutral'>中性 {int(cat.get('中性', 0))}</span>"
            "</li>"
        )
    cat_interp = "".join(cat_interp_parts)
    evidence_rows = _render_evidence_rows(v)
    category_bars = _render_category_bars(v)
    rule_filter = _render_rule_filter(v)
    rule_pair_cards = _render_rule_pair_cards(v)
    bullish_rows = _render_structure_rows(v.get("bullish_structure", []), "signal_style_bucket")
    bearish_rows = _render_structure_rows(v.get("bearish_structure", []), "bearish_reason_bucket")

    daily_rows = "".join(
        f"<tr><td>{_escape(r['date'])}</td><td>{int(r['open'])}</td><td>{int(r['close'])}</td><td>{int(r['factors'])}</td></tr>"
        for r in v["signal_info"].get("daily", [])
    )
    disclaimer = """
<div class="disclaimer">
<p><b>免责声明</b></p>
<p>本报告由 AI 自动生成，仅供参考，不构成任何投资建议或投资推荐。报告中的所有信号、评分、回测结果均基于历史数据统计分析，历史表现不代表未来收益，不保证盈利或避免亏损。</p>
<p>本报告涉及的因子择时模型、信号规则及策略回测可能存在模型风险、数据偏差、过拟合等局限性。使用者应独立判断，结合自身风险承受能力和投资目标审慎决策，并承担由此产生的全部风险与责任。</p>
<p>报告生成方及模型开发者不对因使用本报告中的任何信息而导致的任何直接或间接损失承担责任。</p>
</div>
"""

    score_desc = """
<div class="strategy-desc">
<p><b>抄底得分（Entry Score）</b>：基于多因子历史信号聚合得到的开仓倾向得分。值越高，表示做多信号越强。</p>
<p><b>逃顶得分（Exit Score）</b>：基于多因子历史信号聚合得到的平仓倾向得分。值越高，表示离场或风控信号越强。</p>
<p><b>净得分（Net Score）</b>：抄底得分减去逃顶得分，正值偏多，负值偏空。</p>
<p><b>20日 zscore + 3日均线</b>：用于更快观察短期分数变化；正式基准策略净值使用的是更稳的长窗口口径。</p>
</div>
"""

    css = f"""
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;color:#333;line-height:1.6}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:40px 20px;text-align:center}}
.header h1{{font-size:28px;margin-bottom:8px}}
.header .date{{font-size:14px;opacity:.86}}
.container{{max-width:1440px;margin:0 auto;padding:20px}}
.conclusion-banner{{background:#fff;border-radius:12px;padding:30px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.08);text-align:center}}
.conclusion-badge{{display:inline-block;background:{v['conclusion_color']};color:#fff;padding:12px 40px;border-radius:50px;font-size:32px;font-weight:700;letter-spacing:4px}}
.conclusion-sub{{margin-top:16px;font-size:14px;color:#666}}
.score-grid{{display:flex;justify-content:center;gap:40px;margin-top:20px;flex-wrap:wrap}}
.score-item{{text-align:center}}
.score-value{{font-size:24px;font-weight:700;color:{v['conclusion_color']}}}
.score-label{{font-size:12px;color:#999;margin-top:4px}}
.keyword-pill{{display:inline-block;border-radius:999px;padding:2px 8px;margin:0 3px;font-size:12px;font-weight:700;line-height:1.5;border:1px solid transparent;white-space:nowrap}}
.mini-count{{font-size:11px;margin:0 2px;padding:1px 7px}}
.pill-bullish{{background:#eafaf1;color:#169b62;border-color:#bfe8d0}}
.pill-bearish{{background:#fff0f0;color:#d62728;border-color:#f3c7c7}}
.pill-watch{{background:#f2f4f7;color:#667085;border-color:#d0d5dd}}
.pill-neutral{{background:#fff7e6;color:#b54708;border-color:#fedf89}}
.pill-risk{{background:#eef4ff;color:#1d4ed8;border-color:#bfdbfe}}
.pill-core{{background:#f5f7ff;color:{v['conclusion_color']};border-color:#d9dde7}}
.card{{background:#fff;border-radius:12px;padding:24px;margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.card h2{{font-size:18px;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #f0f2f5;color:#1a1a2e}}
.card h2 .badge{{display:inline-block;background:#e74c3c;color:#fff;font-size:12px;padding:2px 10px;border-radius:12px;margin-left:8px;vertical-align:middle}}
.disclaimer{{background:#fff8f0;border:1px solid #f0d8b0;border-radius:10px;padding:16px 20px;margin-bottom:24px;font-size:12.5px;line-height:1.7;color:#8b6914}}
.disclaimer p{{margin-bottom:4px}}
.disclaimer b{{color:#b8860b}}
.signal-stats{{display:flex;align-items:center;gap:40px;flex-wrap:wrap}}
.signal-pie{{width:180px;height:180px;border-radius:50%;background:conic-gradient(#2ecc71 0% {v['bullish_pct']}%,#e74c3c {v['bullish_pct']}% {round(v['bullish_pct']+v['bearish_pct'], 1)}%,#95a5a5 {round(v['bullish_pct']+v['bearish_pct'], 1)}% 100%);flex-shrink:0}}
.signal-legend{{flex:1;min-width:200px}}
.legend-item{{display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:14px}}
.legend-dot{{width:12px;height:12px;border-radius:50%;flex-shrink:0}}
.legend-pct{{margin-left:auto;font-weight:700}}
.cat-bar-row{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}
.cat-bar-label{{width:110px;font-size:12px;text-align:right;flex-shrink:0}}
.cat-bar-track{{flex:1;height:20px;background:#f0f2f5;border-radius:10px;overflow:hidden;display:flex}}
.cat-bar-bullish{{height:100%;background:#2ecc71}}
.cat-bar-bearish{{height:100%;background:#e74c3c}}
.cat-bar-neutral{{height:100%;background:#95a5a5}}
.cat-bar-score{{width:60px;font-size:12px;text-align:right;flex-shrink:0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 12px;text-align:center;border-bottom:1px solid #eee}}
th{{background:#f8f9fa;font-weight:600;color:#555}}
tr:hover{{background:#f8f9fa}}
.rule-pair-grid{{display:grid;grid-template-columns:1fr;gap:20px}}
.rule-pair-card{{background:#fafafa;border-radius:10px;padding:16px;border:1px solid #e8e8e8}}
.rule-pair-card.hidden{{display:none}}
.rule-pair-header{{margin-bottom:12px}}
.rule-pair-header h3{{font-size:15px;color:#1a1a2e;margin-bottom:6px}}
.rule-filter{{margin:12px 0 16px 0}}
.rule-filter-dropdown{{position:relative;display:inline-block;min-width:260px}}
.rule-filter-dropdown summary{{list-style:none;cursor:pointer;border:1px solid #d9dde7;background:#f8fafc;border-radius:8px;padding:9px 14px;font-size:13px;font-weight:700;color:#344054;user-select:none}}
.rule-filter-dropdown summary::-webkit-details-marker{{display:none}}
.rule-filter-dropdown summary span{{font-weight:600;color:#667085;margin-left:8px}}
.rule-filter-dropdown[open] summary{{background:#eef2f7}}
.rule-filter-menu{{position:absolute;z-index:20;top:42px;left:0;min-width:280px;background:#fff;border:1px solid #d9dde7;border-radius:10px;box-shadow:0 10px 24px rgba(15,23,42,.16);padding:10px}}
.rule-filter-option{{display:flex;align-items:center;gap:8px;padding:7px 8px;border-radius:6px;font-size:13px;color:#344054;cursor:pointer}}
.rule-filter-option:hover{{background:#f8fafc}}
.rule-filter-option input{{width:14px;height:14px}}
.rule-filter-option span{{flex:1}}
.rule-filter-option em{{font-style:normal;color:#667085;font-size:12px}}
.rule-filter-all{{border-bottom:1px solid #eef2f7;margin-bottom:4px;padding-bottom:9px;font-weight:700}}
.factor-meta{{font-size:12px;margin-bottom:6px;display:flex;align-items:flex-start;gap:8px;flex-wrap:wrap}}
.factor-cat{{display:inline-block;color:#fff;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}}
.factor-meaning{{color:#333;font-size:12.5px;line-height:1.5}}
.factor-extra{{font-size:11.5px;color:#666;margin-top:3px;padding-left:4px;line-height:1.5}}
.rule-pair-rules{{font-size:12.5px;color:#344054;background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:7px 10px;margin:8px 0;line-height:1.6}}
.strategy-desc{{background:#f0f4ff;border:1px solid #d0d8f0;border-radius:8px;padding:14px 16px;margin-bottom:16px;font-size:13px;line-height:1.7}}
.strategy-desc p{{margin-bottom:4px;color:#2c3e50}}
.tab-bar{{display:flex;gap:0;margin-bottom:16px;border-bottom:2px solid #e0e0e0}}
.tab-btn{{padding:8px 20px;cursor:pointer;border:none;background:transparent;font-size:14px;color:#666;border-bottom:2px solid transparent;margin-bottom:-2px}}
.tab-btn:hover{{color:#333}}
.tab-btn.active{{color:{v['conclusion_color']};border-bottom-color:{v['conclusion_color']};font-weight:600}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}
.interpretation{{background:#f8f9fa;border-left:4px solid {v['conclusion_color']};padding:16px;border-radius:4px;margin-top:16px;font-size:14px;line-height:1.8}}
.interpretation p{{margin-bottom:8px}}
.interpretation ul{{margin-left:20px;margin-bottom:8px}}
.interpretation li{{margin-bottom:4px}}
.strategy-label{{font-size:14px;color:#555;margin-bottom:8px;text-align:center;font-weight:600}}
.event-module{{margin:24px 0}}
.module-heading{{display:none}}
.module-tabs{{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 16px 0;padding:12px;background:#fff;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.06)}}
.module-tab-btn{{border:1px solid #d9dde7;background:#f8fafc;color:#344054;border-radius:8px;padding:10px 18px;font-size:14px;font-weight:600;cursor:pointer}}
.module-tab-btn:hover{{background:#eef2f7}}
.module-tab-btn.active{{background:{v['conclusion_color']};border-color:{v['conclusion_color']};color:#fff}}
.module-panel{{display:none}}
.module-panel.active{{display:block}}
.module-intro{{font-size:13.5px;color:#475467;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;line-height:1.7;margin:0 0 14px 0}}
.plotly-wrap{{width:100%;overflow-x:auto}}
.strategy-figures{{display:flex;flex-direction:column;gap:14px}}
.score-z20-figures{{display:flex;flex-direction:column;gap:14px}}
.recent-signal-figures{{display:flex;flex-direction:column;gap:14px;margin-bottom:14px}}
.rule-pair-figures{{display:flex;flex-direction:column;gap:14px}}
.plot-panel{{display:block;width:100%;background:#fff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;margin:0 0 12px 0}}
.plot-panel-title{{font-size:13px;font-weight:700;color:#334155;background:#f8fafc;border-bottom:1px solid #e5e7eb;padding:8px 12px}}
.plot-panel .plotly-graph-div{{border:0!important}}
.footnote{{font-size:12px;color:#667085;line-height:1.6;margin:10px 0 0 0}}
.chart-error{{padding:16px;border:1px solid #f3c7c7;background:#fff4f4;color:#b42318;border-radius:8px;font-size:13px}}
@media(max-width:900px){{.rule-pair-grid{{grid-template-columns:1fr}}.signal-stats{{flex-direction:column;align-items:center}}}}
"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(v['title'])} - {_escape(v['latest'])}</title>
<style>{css}</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="header">
  <h1>{_escape(v['title'])}</h1>
  <div class="date">中证全指 | 最新数据：{_escape(v['latest'])} | 报告生成：{_escape(v['generated_at'])}</div>
</div>
<div class="container">
  {disclaimer}
  <div class="conclusion-banner">
    <div class="conclusion-badge">{_escape(v['conclusion'])}</div>
    <div class="conclusion-sub">
      基于 <span class="keyword-pill pill-core">{v['total_sig']} 个信号点</span>
      <span class="keyword-pill pill-bullish">多 {bullish_count}</span>
      <span class="keyword-pill pill-bearish">空 {bearish_count}</span>
      <span class="keyword-pill pill-watch">观望 {watch_count}</span>
      <span class="keyword-pill pill-core">核心评分 {core_score:.3f}</span>
    </div>
    <div class="score-grid">
      <div class="score-item"><div class="score-value">{core_score:.2f}</div><div class="score-label">核心评分</div></div>
      <div class="score-item"><div class="score-value">{total_score:.2f}</div><div class="score-label">总评分</div></div>
      <div class="score-item"><div class="score-value">{interpretable_ratio:.0f}%</div><div class="score-label">可解释占比</div></div>
    </div>
  </div>

  <div class="card">
    <h2>结论解释</h2>
    <div class="interpretation">
      <p><b>最新结论</b>：系统判定为 <span class="keyword-pill pill-core">{_escape(v['conclusion'])}</span>，core_score = <span class="keyword-pill pill-core">{core_score:.3f}</span>。</p>
      <p><b>信号结构</b>：<span class="keyword-pill pill-core">{v['total_sig']} 个信号点</span> 中，
        <span class="keyword-pill pill-bullish">看多 {bullish_count}（{v['bullish_pct']:.1f}%）</span>
        <span class="keyword-pill pill-bearish">看空 {bearish_count}（{v['bearish_pct']:.1f}%）</span>
        <span class="keyword-pill pill-watch">观望 {watch_count}（{nw_pct:.1f}%）</span>。
      </p>
      <p><b>分项表现</b>：</p>
      <ul>{cat_interp}</ul>
    </div>
  </div>

  <div class="module-tabs">
    <button class="module-tab-btn active" onclick="switchModule(event,'event-module-panel')">事件驱动模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'score-module-panel')">综合打分模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'rule-module-panel')">单因子规则模块</button>
  </div>

  <div id="event-module-panel" class="module-panel active event-module">
    <h2 class="module-heading">事件驱动模块</h2>
    <p class="module-intro">事件驱动模块从每个因子的开仓、平仓事件出发，统计当前市场中看多、看空和观望信号的分布。它不直接给出最终仓位，而是回答最近哪些类型的因子正在触发交易事件、这些事件偏向抄底还是风险释放、不同证据之间是否一致。适合用来解释当下择时观点的来源和结构。</p>

    <div class="card">
      <h2>分析证据</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>类别</th><th>看多</th><th>看空</th><th>风险缓和</th><th>待确认</th><th>中性</th><th>净得分</th><th>主证据</th></tr></thead>
          <tbody>{evidence_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>信号分布概览</h2>
      <div class="signal-stats">
        <div class="signal-pie"></div>
        <div class="signal-legend">
          <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div>看多 <span class="legend-pct">{v['bullish_pct']:.1f}% ({int(sc.get('多', 0))})</span></div>
          <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>看空 <span class="legend-pct">{v['bearish_pct']:.1f}% ({int(sc.get('空', 0))})</span></div>
          <div class="legend-item"><div class="legend-dot" style="background:#95a5a5"></div>观望 <span class="legend-pct">{nw_pct:.1f}% ({int(sc.get('观望', 0))})</span></div>
        </div>
      </div>
      <h3 style="margin-top:20px;font-size:14px;color:#555;">因子类别净得分分布</h3>
      <div class="cat-bar-container">{category_bars}</div>
    </div>

    <div class="card">
      <h2>看多信号结构</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>类型</th><th>数量</th><th>占比</th><th>分值合计</th><th>核心开仓</th><th>辅助观察</th></tr></thead>
          <tbody>{bullish_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>看空信号结构</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>类型</th><th>数量</th><th>占比</th><th>分值合计</th><th>核心平仓</th><th>辅助观察</th></tr></thead>
          <tbody>{bearish_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>最近 20 日信号触发 <span class="badge">{int(v['signal_info'].get('total_recent', 0))} 条开仓</span></h2>
      <div class="plot-container">
        <div class="plotly-wrap">{v['recent_signal_chart_html']}</div>
      </div>
      <p class="footnote">数量口径：以 signals.csv 为来源，按交易日聚合事件条数；signal = 1 计入开仓数量，signal = -1 计入平仓数量，净开仓量 = 开仓数量 - 平仓数量。同一交易日多个因子或规则同时触发会分别计数。图中传入全历史交易日数据，打开时默认显示最近 3 年约 756 个交易日；未触发信号的交易日数量记为 0。图中展示净开仓量、开仓数量和平仓数量的 5 日均线，hover 中保留当日原始数量。</p>
    </div>

  </div>

  <div id="score-module-panel" class="module-panel">
    <h2 class="module-heading">综合打分模块</h2>
    <p class="module-intro">综合打分模块把历史事件回测后的有效信号汇总成抄底得分和逃顶得分，并根据得分变化生成最终择时策略。这里更关注多个信号合成以后是否形成可执行的仓位规则，因此重点观察抄底、逃顶得分的相对强弱、开平仓点以及策略相对基准的净值表现。</p>
  <div class="card">
    <h2>策略净值曲线</h2>
    {score_desc}
    <div class="plot-container">
      <div class="plotly-wrap">{v['strategy_z20_html']}</div>
    </div>
    <div class="plot-container" style="margin-top:24px;">
      <div class="plotly-wrap">{v['strategy_plot_html']}</div>
    </div>
  </div>
  </div>

  <div id="rule-module-panel" class="module-panel">
    <h2 class="module-heading">单因子规则模块</h2>
    <p class="module-intro">单因子规则模块逐个展示每个 base 因子历史上表现最好的开仓和平仓规则组合。它用于回答某个因子单独使用时，什么事件规则最有效、交易点是否合理、超额收益是否稳定。这些结果主要作为规则解释和因子筛选参考，不等同于最终综合策略。</p>

  <div class="card">
    <h2>各 base 指标最优规则组合回测 <span class="badge">{len(v['rule_pair_cards'])} 个因子</span></h2>
    <p style="font-size:13px;color:#666;margin-bottom:16px;">保持原有内容结构，只把时序图改成可交互图。悬停可以查看每个时间点的具体数值。</p>
    {rule_filter}
    <div class="rule-pair-grid">{rule_pair_cards}</div>
  </div>
  </div>
</div>
<script>
function switchModule(e,t) {{
  document.querySelectorAll('.module-panel').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.querySelectorAll('.module-tab-btn').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.getElementById(t).classList.add('active');
  e.target.classList.add('active');
  setTimeout(function() {{
    document.querySelectorAll('#' + t + ' .plotly-graph-div').forEach(function(el) {{
      if (window.Plotly) {{ Plotly.Plots.resize(el); }}
    }});
  }}, 80);
}}
function switchTab(e,t) {{
  document.querySelectorAll('.tab-content,.tab-btn').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.getElementById(t).classList.add('active');
  e.target.classList.add('active');
}}
function toggleAllRuleCategories(box) {{
  document.querySelectorAll('.rule-category-check').forEach(function(el) {{
    el.checked = box.checked;
  }});
  updateRulePairFilter();
}}
function updateRulePairFilter() {{
  var boxes = Array.from(document.querySelectorAll('.rule-category-check'));
  var selected = boxes.filter(function(el) {{ return el.checked; }}).map(function(el) {{ return el.value; }});
  var allBox = document.getElementById('rule-filter-all');
  if (allBox) {{
    allBox.checked = selected.length === boxes.length;
  }}
  var selectedSet = new Set(selected);
  var visible = 0;
  document.querySelectorAll('.rule-pair-card').forEach(function(card) {{
    var show = selectedSet.has(card.dataset.categoryRoot || '其他');
    card.classList.toggle('hidden', !show);
    if (show) {{ visible += 1; }}
  }});
  var summary = document.getElementById('rule-filter-summary');
  if (summary) {{
    if (selected.length === boxes.length) {{
      summary.textContent = '全部 ' + visible;
    }} else if (selected.length === 0) {{
      summary.textContent = '未选择';
    }} else {{
      summary.textContent = selected.join('、') + ' ' + visible;
    }}
  }}
  setTimeout(function() {{
    document.querySelectorAll('#rule-module-panel .rule-pair-card:not(.hidden) .plotly-graph-div').forEach(function(el) {{
      if (window.Plotly) {{ Plotly.Plots.resize(el); }}
    }});
  }}, 80);
}}
document.addEventListener('DOMContentLoaded', updateRulePairFilter);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成宽基择时信号 HTML 报告")
    parser.add_argument("--input-dir", required=True, help="运行结果目录，包含 data/results/plots")
    parser.add_argument("--output", required=True, help="输出 HTML 路径")
    parser.add_argument("--taxonomy", default=None, help="factor_taxonomy.md 路径")
    parser.add_argument("--title", default="宽基择时信号报告", help="报告标题")
    args = parser.parse_args()

    taxonomy = args.taxonomy
    if not taxonomy:
        auto_path = Path(__file__).resolve().parents[1] / "references" / "factor_taxonomy.md"
        if auto_path.exists():
            taxonomy = str(auto_path)

    data = build_view_data(args.input_dir, taxonomy, args.title)
    html = render_html(data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
