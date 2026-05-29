from __future__ import annotations

from html import escape as html_escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
try:
    from plotly.io._html import get_plotlyjs
except Exception:  # pragma: no cover - fallback for older/newer plotly builds
    from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from baseline_score_strategy import format_rule_name_cn, rolling_zscore
from data_cleaning import get_factor_columns
from io_utils import read_table, resolve_table_file, table_candidates
from plotting import factor_plot_roots
from timing_config import (
    CODE_COL,
    DATE_COL,
    NAME_COL,
    PRICE_COL,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_VALUE_COL,
)


def _resolve_run_file(root: str | Path, candidates: list[str]) -> Path:
    base = Path(root)
    for rel in candidates:
        path = base / rel
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {candidates} under {base}")


def _read_csv(root: str | Path, candidates: list[str], *, optional: bool = False) -> pd.DataFrame:
    try:
        path = resolve_table_file(root, candidates)
    except FileNotFoundError:
        if optional:
            return pd.DataFrame()
        raise
    return read_table(path)


def _read_json(root: str | Path, candidates: list[str]) -> dict[str, Any]:
    path = _resolve_run_file(root, candidates)
    return pd.read_json(path, typ="series").to_dict()


def _parse_factor_taxonomy(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    factors: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("### "):
            if current:
                factors[current] = "\n".join(lines).strip()
            current = line[4:].strip()
            lines = []
        elif line.startswith("## "):
            if current:
                factors[current] = "\n".join(lines).strip()
            current = None
            lines = []
        elif current is not None:
            lines.append(line)
    if current:
        factors[current] = "\n".join(lines).strip()
    return factors


def _extract_factor_desc(desc_text: str) -> dict[str, str]:
    if not desc_text:
        return {"category": "", "meaning": "", "direction": "", "observation": "", "note": ""}
    result = {"category": "", "meaning": "", "direction": "", "observation": "", "note": ""}
    for raw in desc_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line[2:] if line.startswith(("- ", "* ")) else line
        if line.startswith("类别："):
            result["category"] = line[3:].strip()
        elif line.startswith("可能含义："):
            result["meaning"] = line[5:].strip()
        elif line.startswith("默认方向："):
            result["direction"] = line[5:].strip()
        elif line.startswith("观察方式："):
            result["observation"] = line[5:].strip()
        elif line.startswith("注意事项："):
            result["note"] = line[5:].strip()
    return result


def _signal_counts(path: str | Path, top_n_days: int = 20) -> dict[str, Any]:
    existing = next((candidate for candidate in table_candidates(path) if candidate.exists()), None)
    if existing is None:
        return {"daily": [], "top_factors": [], "top_patterns": [], "total_recent": 0}
    df = read_table(existing)
    if df.empty:
        return {"daily": [], "top_factors": [], "top_patterns": [], "total_recent": 0}
    df["date"] = df["date"].astype(str)
    recent_dates = sorted(df["date"].dropna().unique().tolist(), reverse=True)[:top_n_days]
    recent = df[df["date"].isin(recent_dates)].copy()
    daily = []
    if not recent.empty:
        day_counts = (
            recent.assign(is_open=recent["signal"].astype(str).eq("1"), is_close=recent["signal"].astype(str).eq("-1"))
            .groupby("date", sort=False)
            .agg(open=("is_open", "sum"), close=("is_close", "sum"), factors=("factor", "nunique"))
            .reset_index()
        )
        daily = day_counts.to_dict(orient="records")
    factor_totals = (
        recent[recent["signal"].astype(str).eq("1")].groupby("factor").size().sort_values(ascending=False).head(30)
    )
    pattern_totals = (
        recent[recent["signal"].astype(str).eq("1")].groupby("pattern").size().sort_values(ascending=False).head(30)
    )
    return {
        "daily": daily,
        "top_factors": list(factor_totals.items()),
        "top_patterns": list(pattern_totals.items()),
        "total_recent": int(recent[recent["signal"].astype(str).eq("1")].shape[0]),
    }


def _make_recent_signal_chart_html(
    input_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    top_n_days: int | None = None,
    default_visible_days: int = 756,
) -> str:
    """Render recent open / close signal counts with closing price."""
    if input_df.empty or signals_df.empty:
        return "<div class='chart-error'>最近信号或行情数据为空，无法生成图表。</div>"

    sig = signals_df.copy()
    sig[SIGNAL_DATE_COL] = pd.to_datetime(sig[SIGNAL_DATE_COL], errors="coerce")
    sig = sig.dropna(subset=[SIGNAL_DATE_COL])
    if sig.empty:
        return "<div class='chart-error'>最近信号日期为空，无法生成图表。</div>"

    price = input_df.copy()
    price[DATE_COL] = pd.to_datetime(price[DATE_COL], errors="coerce")
    price = price.dropna(subset=[DATE_COL])
    if SIGNAL_INSTRUMENT_COL in sig.columns and CODE_COL in price.columns and not sig[SIGNAL_INSTRUMENT_COL].dropna().empty:
        instrument = sig[SIGNAL_INSTRUMENT_COL].astype(str).mode().iloc[0]
        price = price[price[CODE_COL].astype(str).eq(instrument)]
    price = (
        price.assign(plot_date=price[DATE_COL].dt.normalize())
        .sort_values(DATE_COL)
        .groupby("plot_date", as_index=False)
        .agg(close_price=(PRICE_COL, "last"))
    )
    if top_n_days is not None:
        price = price.tail(top_n_days)
    if price.empty:
        return "<div class='chart-error'>最近行情数据为空，无法生成图表。</div>"

    recent_dates = price["plot_date"]
    recent = sig[sig[SIGNAL_DATE_COL].dt.normalize().isin(set(recent_dates))].copy()
    daily = (
        recent.assign(
            open_count=recent[SIGNAL_VALUE_COL].astype(str).eq("1"),
            close_count=recent[SIGNAL_VALUE_COL].astype(str).eq("-1"),
        )
        .assign(plot_date=lambda x: x[SIGNAL_DATE_COL].dt.normalize())
        .groupby("plot_date", as_index=False)
        .agg(open_count=("open_count", "sum"), close_count=("close_count", "sum"))
    )

    chart_df = price.merge(daily, on="plot_date", how="left")
    chart_df[["open_count", "close_count"]] = chart_df[["open_count", "close_count"]].fillna(0)
    chart_df["net_open_count"] = chart_df["open_count"] - chart_df["close_count"]
    chart_df["net_open_count_ma5"] = chart_df["net_open_count"].rolling(5, min_periods=1).mean()
    chart_df["open_count_ma5"] = chart_df["open_count"].rolling(5, min_periods=1).mean()
    chart_df["close_count_ma5"] = chart_df["close_count"].rolling(5, min_periods=1).mean()
    visible_start_idx = max(0, len(chart_df) - default_visible_days)
    visible_range = [chart_df["plot_date"].iloc[visible_start_idx], chart_df["plot_date"].iloc[-1]]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.56, 0.44],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["plot_date"],
            y=chart_df["net_open_count_ma5"],
            name="净开仓量 5日均线",
            mode="lines",
            line=dict(color="#7B2CBF", width=1.8),
            customdata=np.stack([chart_df["net_open_count"].values, chart_df["open_count"].values, chart_df["close_count"].values], axis=1),
            hovertemplate="%{x|%Y-%m-%d}<br>净开仓量5日均线=%{y:.2f}<br>当日净开仓量=%{customdata[0]:.0f}<br>当日开仓=%{customdata[1]:.0f}<br>当日平仓=%{customdata[2]:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["plot_date"],
            y=chart_df["close_price"],
            name="收盘价",
            mode="lines",
            line=dict(color="#FF8080", width=1.2, dash="dash"),
            opacity=0.82,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["plot_date"],
            y=chart_df["open_count_ma5"],
            name="开仓数量 5日均线",
            mode="lines",
            line=dict(color="#044E7E", width=1.7),
            customdata=np.stack([chart_df["open_count"].values], axis=1),
            hovertemplate="%{x|%Y-%m-%d}<br>开仓数量5日均线=%{y:.2f}<br>当日开仓数量=%{customdata[0]:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["plot_date"],
            y=chart_df["close_count_ma5"],
            name="平仓数量 5日均线",
            mode="lines",
            line=dict(color="#FF3333", width=1.7),
            customdata=np.stack([chart_df["close_count"].values], axis=1),
            hovertemplate="%{x|%Y-%m-%d}<br>平仓数量5日均线=%{y:.2f}<br>当日平仓数量=%{customdata[0]:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=0.8, line_dash="dot", line_color="#777777", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=50, r=45, t=28, b=38),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12),
        hoverlabel=dict(font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="净开仓量 5日均线", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="收盘价", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="开仓 / 平仓数量 5日均线", row=2, col=1)
    fig.update_xaxes(range=visible_range)
    return (
        "<div class='recent-signal-figures'>"
        "<div class='plot-panel'><div class='plot-panel-title'>全历史信号触发：默认显示最近3年</div>"
        f"{_fig_html(fig, height=560)}"
        "</div>"
        "</div>"
    )


def _load_factor_descriptions(taxonomy_path: str | Path | None) -> dict[str, dict[str, str]]:
    return {k: _extract_factor_desc(v) for k, v in _parse_factor_taxonomy(taxonomy_path).items()}


def _select_rule_pair_factor_cols(df: pd.DataFrame, factor: str, open_condition: str, close_condition: str) -> list[str]:
    """Select only the factor series actually implied by the chosen rule."""
    if factor not in df.columns:
        return []
    cond = f"{open_condition} {close_condition}"
    if factor.endswith("_季线"):
        base = factor[: -len("_季线")]
    elif factor.endswith("_年线"):
        base = factor[: -len("_年线")]
    else:
        base = factor

    raw_col = base
    season_col = f"{base}_季线"
    year_col = f"{base}_年线"

    if "原始季线年线" in cond:
        candidates = [raw_col, season_col, year_col]
    elif "季线" in cond or "年线" in cond:
        candidates = [factor]
        if factor == raw_col:
            if "季线" in cond:
                candidates.append(season_col)
            if "年线" in cond:
                candidates.append(year_col)
    else:
        candidates = [factor]

    seen = set()
    cols = []
    for col in candidates:
        if col in df.columns and col not in seen:
            cols.append(col)
            seen.add(col)
    return cols


def _long_spans(position_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if position_df.empty:
        return []
    pos = position_df[[DATE_COL, "position"]].copy()
    pos[DATE_COL] = pd.to_datetime(pos[DATE_COL], errors="coerce")
    pos = pos.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_long = False
    start = None
    last_dt = None
    for dt, value in pos[[DATE_COL, "position"]].itertuples(index=False):
        last_dt = pd.Timestamp(dt)
        curr = float(value or 0)
        if not in_long and curr > 0:
            start = pd.Timestamp(dt)
            in_long = True
        elif in_long and curr <= 0:
            spans.append((start, pd.Timestamp(dt)))
            in_long = False
            start = None
    if in_long and start is not None and last_dt is not None:
        spans.append((start, last_dt))
    return spans


def _fig_html(fig: go.Figure, height: int | None = None) -> str:
    if height is not None:
        fig.update_layout(height=height)
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )


def _metric_badge(label: str, value: str, tone: str = "neutral") -> str:
    colors = {
        "neutral": ("#eef2ff", "#3730a3"),
        "good": ("#ecfdf5", "#047857"),
        "warn": ("#fff7ed", "#c2410c"),
        "bad": ("#fef2f2", "#b91c1c"),
    }
    bg, fg = colors.get(tone, colors["neutral"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'padding:6px 10px;border-radius:999px;background:{bg};color:{fg};'
        f'font-size:12px;font-weight:600;margin:4px 6px 0 0;">'
        f'<span>{html_escape(label)}</span><b>{html_escape(value)}</b></span>'
    )


def _position_change_marks(position_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if position_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    pos = position_df[[DATE_COL, "position"]].copy()
    pos[DATE_COL] = pd.to_datetime(pos[DATE_COL], errors="coerce")
    pos = pos.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    prev = pos["position"].shift(1).fillna(0.0)
    open_df = pos[(prev <= 0) & (pos["position"] > 0)].copy()
    close_df = pos[(prev > 0) & (pos["position"] <= 0)].copy()
    return open_df, close_df


def _make_strategy_figure(strategy_df: pd.DataFrame, summary_row: pd.Series | None = None) -> go.Figure:
    df = strategy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("strategy dataframe is empty")
    title_open = format_rule_name_cn(str(df["open_rule"].dropna().iloc[0])) if "open_rule" in df else "开仓"
    title_close = format_rule_name_cn(str(df["close_rule"].dropna().iloc[0])) if "close_rule" in df else "平仓"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.30, 0.34, 0.36],
    )

    # Row 1: open/close region and price
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#4C78A8", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    if "position" in df.columns:
        for start, end in _long_spans(df):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(246,199,199,0.28)",
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )
    if "open_event" in df.columns:
        open_df = df[df["open_event"].astype(int).eq(1)]
        if not open_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=open_df[DATE_COL],
                    y=open_df[PRICE_COL],
                    name="开仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#169B62", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>开仓价=%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
    if "close_event" in df.columns:
        close_df = df[df["close_event"].astype(int).eq(1)]
        if not close_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=close_df[DATE_COL],
                    y=close_df[PRICE_COL],
                    name="平仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#D62728", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>平仓价=%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

    # Row 2: z-scores and price
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df["entry_z"],
            name="抄底得分",
            mode="lines",
            line=dict(color="#D62728", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>抄底得分=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df["exit_z"],
            name="逃顶得分",
            mode="lines",
            line=dict(color="#2CA02C", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>逃顶得分=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#FF8080", width=1.2, dash="dash"),
            opacity=0.65,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Row 3: equity curves
    fig.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df["strategy_equity"],
            name="策略净值",
            mode="lines",
            line=dict(color="#1D3557", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>策略净值=%{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    if "excess_equity" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[DATE_COL],
                y=df["excess_equity"],
                name="超额曲线",
                mode="lines",
                line=dict(color="#C1121F", width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>超额曲线=%{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    metrics = []
    if summary_row is not None:
        metrics = [
            _metric_badge("超额年化", f"{float(summary_row.get('excess_annual_return', 0.0)):.2%}", "good" if float(summary_row.get("excess_annual_return", 0.0)) >= 0 else "bad"),
            _metric_badge("夏普", f"{float(summary_row.get('sharpe', 0.0)):.2f}"),
            _metric_badge("最大回撤", f"{float(summary_row.get('max_drawdown', 0.0)):.2%}", "bad" if float(summary_row.get("max_drawdown", 0.0)) < 0 else "good"),
            _metric_badge("持仓占比", f"{float(summary_row.get('holding_ratio', 0.0)):.2%}"),
            _metric_badge("交易次数", f"{int(summary_row.get('trade_count', 0))}"),
        ]

    fig.update_layout(
        template="plotly_white",
        height=1180,
        margin=dict(l=45, r=35, t=95, b=35),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12),
        hoverlabel=dict(font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)),
        title=dict(
            text="最优规则组合策略净值 vs 中证全指基准",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="收盘价", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="抄底 / 逃顶得分", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="收盘价", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="净值", row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
    for y in (0,):
        fig.add_hline(y=y, line_width=0.8, line_dash="dot", line_color="#666666", row=2, col=1)
    if metrics:
        fig.add_annotation(
            x=0.5,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=" ".join(metrics),
            align="center",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=12),
        )
    return fig


def _make_strategy_html(strategy_df: pd.DataFrame, summary_row: pd.Series | None = None) -> str:
    """Render the retained score strategy as three independent Plotly panels."""
    df = strategy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("strategy dataframe is empty")

    font = dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)
    hover_font = dict(font=font)
    template = "plotly_white"

    # Panel 1: price with open / close markers and long regions.
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#4C78A8", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        )
    )
    if "position" in df.columns:
        for start, end in _long_spans(df):
            fig_price.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(246,199,199,0.28)",
                line_width=0,
                layer="below",
            )
    if "open_event" in df.columns:
        open_df = df[df["open_event"].astype(int).eq(1)]
        if not open_df.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=open_df[DATE_COL],
                    y=open_df[PRICE_COL],
                    name="开仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#169B62", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>开仓价=%{y:.2f}<extra></extra>",
                )
            )
    if "close_event" in df.columns:
        close_df = df[df["close_event"].astype(int).eq(1)]
        if not close_df.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=close_df[DATE_COL],
                    y=close_df[PRICE_COL],
                    name="平仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#D62728", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>平仓价=%{y:.2f}<extra></extra>",
                )
            )
    fig_price.update_layout(
        template=template,
        height=340,
        margin=dict(l=50, r=45, t=38, b=22),
        font=font,
        hoverlabel=hover_font,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
    )
    fig_price.update_yaxes(title_text="收盘价")

    # Panel 2: entry / exit score and price.
    fig_score = make_subplots(specs=[[{"secondary_y": True}]])
    fig_score.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df["entry_z"],
            name="抄底得分",
            mode="lines",
            line=dict(color="#D62728", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>抄底得分=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_score.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df["exit_z"],
            name="逃顶得分",
            mode="lines",
            line=dict(color="#2CA02C", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>逃顶得分=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_score.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#FF8080", width=1.2, dash="dash"),
            opacity=0.75,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_score.add_hline(y=0.0, line_width=0.8, line_dash="dot", line_color="#666666")
    fig_score.update_layout(
        template=template,
        height=380,
        margin=dict(l=50, r=45, t=24, b=22),
        font=font,
        hoverlabel=hover_font,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig_score.update_yaxes(title_text="抄底 / 逃顶得分", secondary_y=False)
    fig_score.update_yaxes(title_text="收盘价", secondary_y=True)

    # Panel 3: equity curves.
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=df[DATE_COL],
            y=df["strategy_equity"],
            name="策略净值",
            mode="lines",
            line=dict(color="#1D3557", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>策略净值=%{y:.4f}<extra></extra>",
        )
    )
    if "excess_equity" in df.columns:
        fig_equity.add_trace(
            go.Scatter(
                x=df[DATE_COL],
                y=df["excess_equity"],
                name="超额曲线",
                mode="lines",
                line=dict(color="#C1121F", width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>超额曲线=%{y:.4f}<extra></extra>",
            )
        )
    fig_equity.add_hline(y=1.0, line_width=0.9, line_dash="dot", line_color="#777777")
    if summary_row is not None:
        metrics = " ".join(
            [
                _metric_badge("超额年化", f"{float(summary_row.get('excess_annual_return', 0.0)):.2%}", "good" if float(summary_row.get("excess_annual_return", 0.0)) >= 0 else "bad"),
                _metric_badge("夏普", f"{float(summary_row.get('sharpe', 0.0)):.2f}"),
                _metric_badge("最大回撤", f"{float(summary_row.get('max_drawdown', 0.0)):.2%}", "bad" if float(summary_row.get("max_drawdown", 0.0)) < 0 else "good"),
                _metric_badge("持仓占比", f"{float(summary_row.get('holding_ratio', 0.0)):.2%}"),
                _metric_badge("交易次数", f"{int(summary_row.get('trade_count', 0))}"),
            ]
        )
        fig_equity.add_annotation(
            x=0.5,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=metrics,
            align="center",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=12),
        )
    fig_equity.update_layout(
        template=template,
        height=360,
        margin=dict(l=50, r=45, t=54, b=40),
        font=font,
        hoverlabel=hover_font,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_equity.update_yaxes(title_text="净值")

    return (
        "<div class='strategy-figures'>"
        "<div class='plot-panel'><div class='plot-panel-title'>1. 价格与开平仓点</div>"
        f"{_fig_html(fig_price, height=340)}"
        "</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>2. 抄底得分与逃顶得分</div>"
        f"{_fig_html(fig_score, height=380)}"
        "</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>3. 策略净值与超额曲线</div>"
        f"{_fig_html(fig_equity, height=360)}"
        "</div>"
        "</div>"
    )


def _make_score_z20_fig(strategy_df: pd.DataFrame) -> go.Figure:
    df = strategy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("strategy dataframe is empty")

    z20_df = df[[DATE_COL, PRICE_COL]].copy()
    z20_df["entry_z_20_3"] = rolling_zscore(df["entry_score"], 20).rolling(3, min_periods=1).mean()
    z20_df["exit_z_20_3"] = rolling_zscore(df["exit_score"], 20).rolling(3, min_periods=1).mean()
    z20_df = z20_df.tail(750).reset_index(drop=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        row_heights=[0.5, 0.5],
    )
    fig.add_trace(
        go.Scatter(
            x=z20_df[DATE_COL],
            y=z20_df["entry_z_20_3"],
            name="抄底得分 (20Z+3MA)",
            mode="lines",
            line=dict(color="#D62728", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>抄底得分 (20Z+3MA)=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=z20_df[DATE_COL],
            y=z20_df[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#4C78A8", width=1.0),
            opacity=0.75,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=z20_df[DATE_COL],
            y=z20_df["exit_z_20_3"],
            name="逃顶得分 (20Z+3MA)",
            mode="lines",
            line=dict(color="#2CA02C", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>逃顶得分 (20Z+3MA)=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=z20_df[DATE_COL],
            y=z20_df[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#4C78A8", width=1.0),
            opacity=0.75,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_white",
        height=780,
        margin=dict(l=45, r=35, t=80, b=35),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12),
        hoverlabel=dict(font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)),
        title=dict(text="最近750个交易日得分时序", x=0.5, xanchor="center", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="得分", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="收盘价", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="得分", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="收盘价", row=2, col=1, secondary_y=True)
    return fig


def _make_score_z20_html(strategy_df: pd.DataFrame) -> str:
    """Render 20-day z-score / 3-day MA score charts as separate panels."""
    df = strategy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("strategy dataframe is empty")

    z20_df = df[[DATE_COL, PRICE_COL]].copy()
    z20_df["entry_z_20_3"] = rolling_zscore(df["entry_score"], 20).rolling(3, min_periods=1).mean()
    z20_df["exit_z_20_3"] = rolling_zscore(df["exit_score"], 20).rolling(3, min_periods=1).mean()
    z20_df = z20_df.tail(750).reset_index(drop=True)

    font = dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)
    hover_font = dict(font=font)

    def make_one(score_col: str, score_name: str, color: str) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=z20_df[DATE_COL],
                y=z20_df[score_col],
                name=score_name,
                mode="lines",
                line=dict(color=color, width=1.8),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{score_name}=%{{y:.4f}}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=z20_df[DATE_COL],
                y=z20_df[PRICE_COL],
                name="收盘价",
                mode="lines",
                line=dict(color="#4C78A8", width=1.1, dash="dash"),
                opacity=0.78,
                hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_hline(y=0.0, line_width=0.8, line_dash="dot", line_color="#777777")
        fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=50, r=45, t=26, b=38),
            font=font,
            hoverlabel=hover_font,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        fig.update_yaxes(title_text="得分", secondary_y=False)
        fig.update_yaxes(title_text="收盘价", secondary_y=True)
        return fig

    entry_fig = make_one("entry_z_20_3", "抄底得分 (20Z+3MA)", "#D62728")
    exit_fig = make_one("exit_z_20_3", "逃顶得分 (20Z+3MA)", "#2CA02C")
    return (
        "<div class='score-z20-figures'>"
        "<div class='plot-panel'><div class='plot-panel-title'>1. 抄底得分 20日Z值 + 3日均线（最近750个交易日）</div>"
        f"{_fig_html(entry_fig, height=340)}"
        "</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>2. 逃顶得分 20日Z值 + 3日均线（最近750个交易日）</div>"
        f"{_fig_html(exit_fig, height=340)}"
        "</div>"
        "</div>"
    )


def _make_factor_figure(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    factor_root: str,
    instrument: str,
    factor_desc: dict[str, str] | None = None,
) -> go.Figure:
    factor_cols = [c for c in [factor_root, f"{factor_root}_季线", f"{factor_root}_年线"] if c in df.columns]
    if not factor_cols:
        raise ValueError(f"No factor columns found for {factor_root}")
    group = df[df[CODE_COL].astype(str).eq(str(instrument))].copy()
    if group.empty:
        raise ValueError(f"Instrument not found: {instrument}")
    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    signal_rows = signals[
        signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))
        & signals[SIGNAL_FACTOR_COL].astype(str).isin(factor_cols)
    ].copy()
    signal_rows[SIGNAL_DATE_COL] = pd.to_datetime(signal_rows[SIGNAL_DATE_COL], errors="coerce")
    signal_rows = signal_rows.dropna(subset=[SIGNAL_DATE_COL])
    signal_rows = signal_rows.merge(
        group[[DATE_COL, *factor_cols, PRICE_COL]],
        left_on=SIGNAL_DATE_COL,
        right_on=DATE_COL,
        how="left",
    )
    value_col = factor_root if factor_root in group.columns else factor_cols[0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#044E7E", "#FF3333", "#7B2CBF"]
    for idx, col in enumerate(factor_cols):
        fig.add_trace(
            go.Scatter(
                x=group[DATE_COL],
                y=group[col],
                name=col,
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=1.6),
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}=%{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=group[DATE_COL],
            y=group[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#FF8080", width=1.2, dash="dash"),
            opacity=0.85,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    if not signal_rows.empty and value_col in signal_rows.columns:
        open_rows = signal_rows[signal_rows[SIGNAL_VALUE_COL].astype(str).eq("1")].copy()
        close_rows = signal_rows[signal_rows[SIGNAL_VALUE_COL].astype(str).eq("-1")].copy()
        if not open_rows.empty:
            fig.add_trace(
                go.Scatter(
                    x=open_rows[SIGNAL_DATE_COL],
                    y=open_rows[value_col],
                    name="+1 开仓",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=9, color="#169B62", line=dict(color="white", width=0.5)),
                    customdata=np.stack(
                        [
                            open_rows[SIGNAL_FACTOR_COL].astype(str).values,
                            open_rows[SIGNAL_PATTERN_COL].astype(str).values,
                        ],
                        axis=1,
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>因子=%{customdata[0]}<br>规则=%{customdata[1]}<br>因子值=%{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )
        if not close_rows.empty:
            fig.add_trace(
                go.Scatter(
                    x=close_rows[SIGNAL_DATE_COL],
                    y=close_rows[value_col],
                    name="-1 平仓",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=9, color="#D62728", line=dict(color="white", width=0.5)),
                    customdata=np.stack(
                        [
                            close_rows[SIGNAL_FACTOR_COL].astype(str).values,
                            close_rows[SIGNAL_PATTERN_COL].astype(str).values,
                        ],
                        axis=1,
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>因子=%{customdata[0]}<br>规则=%{customdata[1]}<br>因子值=%{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )

    for y in (0, 1, -1):
        fig.add_hline(y=y, line_width=0.8, line_dash="dot", line_color="#999999")

    desc = factor_desc or {}
    title_bits = [factor_root]
    if desc.get("category"):
        title_bits.append(desc["category"])
    if desc.get("meaning"):
        title_bits.append(desc["meaning"])
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=45, r=35, t=30, b=30),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12),
        hoverlabel=dict(font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)),
        title=dict(text="", x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="因子值 / sigma", secondary_y=False)
    fig.update_yaxes(title_text="收盘价", secondary_y=True)
    return fig


def _make_rule_pair_figure(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    best_row: pd.Series,
    factor_desc: dict[str, str] | None = None,
) -> go.Figure:
    code_key = CODE_COL if CODE_COL in best_row.index else ("浠ｇ爜" if "浠ｇ爜" in best_row.index else best_row.index[0])
    code = str(best_row.get(code_key, ""))
    factor = str(best_row["factor"])
    open_condition = str(best_row["open_condition"])
    close_condition = str(best_row["close_condition"])
    factor_cols = [c for c in [factor, f"{factor}_季线", f"{factor}_年线"] if c in df.columns]
    if not factor_cols:
        raise ValueError(f"No factor columns found for {factor}")

    group = df[df[CODE_COL].astype(str).eq(code)].copy()
    if group.empty:
        raise ValueError(f"Instrument not found: {code}")
    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    equity_df = group[[DATE_COL, PRICE_COL]].copy()
    best_equity = None
    # best equity rows are already date aligned; fall back if missing
    if "results" in best_row.index:
        pass

    # infer from the best-base equity file by the same key
    # the caller may filter the correct row before calling
    selected_factor = factor
    key_mask = (
        signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(code)
        & signals[SIGNAL_FACTOR_COL].astype(str).eq(selected_factor)
        & signals[SIGNAL_PATTERN_COL].isin([open_condition, close_condition])
    )
    signal_points = signals[key_mask].copy()
    signal_points[SIGNAL_DATE_COL] = pd.to_datetime(signal_points[SIGNAL_DATE_COL], errors="coerce")
    signal_points = signal_points.dropna(subset=[SIGNAL_DATE_COL])

    # Use the equity curve for this exact best row if it exists in the merged table.
    # The caller passes the full best-base equity table in the view data.
    # Here we filter the same key from the preloaded dataframe via attrs on best_row.
    if "_equity_df" in best_row.attrs:
        best_equity = best_row.attrs["_equity_df"]
    if best_equity is None or best_equity.empty:
        raise ValueError(f"Missing equity curve for {factor} | {open_condition} -> {close_condition}")
    best_equity = best_equity.copy()
    best_equity[DATE_COL] = pd.to_datetime(best_equity[DATE_COL], errors="coerce")
    best_equity = best_equity.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    start_dt = best_equity[DATE_COL].min()
    end_dt = best_equity[DATE_COL].max()
    group = group[group[DATE_COL].between(start_dt, end_dt)].copy()

    signal_points = signal_points[
        signal_points[SIGNAL_DATE_COL].between(start_dt, end_dt)
    ].copy()
    if not signal_points.empty:
        signal_points = signal_points.merge(
            group[[DATE_COL, *factor_cols, PRICE_COL]],
            left_on=SIGNAL_DATE_COL,
            right_on=DATE_COL,
            how="left",
        )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]],
        row_heights=[0.24, 0.26, 0.50],
    )

    # Row 1: price + long zones + trade markers
    fig.add_trace(
        go.Scatter(
            x=group[DATE_COL],
            y=group[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#4C78A8", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    if "position" in best_equity.columns:
        for start, end in _long_spans(best_equity.rename(columns={DATE_COL: DATE_COL})):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(246,199,199,0.28)",
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )
    if "position" in best_equity.columns:
        prev = best_equity["position"].shift(1).fillna(0.0)
        open_dates = best_equity.loc[(prev <= 0) & (best_equity["position"] > 0), DATE_COL]
        close_dates = best_equity.loc[(prev > 0) & (best_equity["position"] <= 0), DATE_COL]
        open_prices = group.set_index(DATE_COL).reindex(open_dates)[PRICE_COL]
        close_prices = group.set_index(DATE_COL).reindex(close_dates)[PRICE_COL]
        if not open_dates.empty:
            fig.add_trace(
                go.Scatter(
                    x=open_dates,
                    y=open_prices.values,
                    name="开仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#169B62", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>开仓价=%{y:.2f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
        if not close_dates.empty:
            fig.add_trace(
                go.Scatter(
                    x=close_dates,
                    y=close_prices.values,
                    name="平仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#D62728", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>平仓价=%{y:.2f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

    # Row 2: strategy equity and excess
    fig.add_trace(
        go.Scatter(
            x=best_equity[DATE_COL],
            y=best_equity["strategy_equity"],
            name="策略净值",
            mode="lines",
            line=dict(color="#1D3557", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>策略净值=%{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    if "excess_equity" in best_equity.columns:
        fig.add_trace(
            go.Scatter(
                x=best_equity[DATE_COL],
                y=best_equity["excess_equity"],
                name="超额曲线",
                mode="lines",
                line=dict(color="#C1121F", width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>超额曲线=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    metrics = (
        f"超额年化: {float(best_row.get('excess_annual_return', 0.0)):.2%}   "
        f"夏普: {float(best_row.get('sharpe', 0.0)):.2f}   "
        f"最大回撤: {float(best_row.get('max_drawdown', 0.0)):.2%}"
    )

    # Row 3: factor + price + signal points
    colors = ["#044E7E", "#FF3333", "#7B2CBF"]
    for idx, col in enumerate(factor_cols):
        fig.add_trace(
            go.Scatter(
                x=group[DATE_COL],
                y=group[col],
                name=col,
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=1.6),
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}=%{y:.4f}<extra></extra>",
                showlegend=(idx == 0),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=group[DATE_COL],
            y=group[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#FF8080", width=1.2, dash="dash"),
            opacity=0.85,
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    if not signal_points.empty and selected_factor in signal_points.columns:
        open_points = signal_points[signal_points[SIGNAL_VALUE_COL].astype(str).eq("1")].copy()
        close_points = signal_points[signal_points[SIGNAL_VALUE_COL].astype(str).eq("-1")].copy()
        if not open_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=open_points[SIGNAL_DATE_COL],
                    y=open_points[selected_factor],
                    name="+1 开仓",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="#169B62", line=dict(color="white", width=0.5)),
                    customdata=np.stack(
                        [
                            open_points[SIGNAL_PATTERN_COL].astype(str).values,
                            open_points[SIGNAL_FACTOR_COL].astype(str).values,
                        ],
                        axis=1,
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>规则=%{customdata[0]}<br>因子=%{customdata[1]}<br>因子值=%{y:.4f}<extra></extra>",
                    showlegend=True,
                ),
                row=3,
                col=1,
                secondary_y=False,
            )
        if not close_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=close_points[SIGNAL_DATE_COL],
                    y=close_points[selected_factor],
                    name="-1 平仓",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="#D62728", line=dict(color="white", width=0.5)),
                    customdata=np.stack(
                        [
                            close_points[SIGNAL_PATTERN_COL].astype(str).values,
                            close_points[SIGNAL_FACTOR_COL].astype(str).values,
                        ],
                        axis=1,
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>规则=%{customdata[0]}<br>因子=%{customdata[1]}<br>因子值=%{y:.4f}<extra></extra>",
                    showlegend=True,
                ),
                row=3,
                col=1,
                secondary_y=False,
            )
    for y in (0, 1, -1):
        fig.add_hline(y=y, line_width=0.8, line_dash="dot", line_color="#999999", row=3, col=1)

    desc = factor_desc or {}
    fig.update_layout(
        template="plotly_white",
        height=1380,
        margin=dict(l=45, r=35, t=90, b=35),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12),
        hoverlabel=dict(font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)),
        title=dict(text="", x=0.5, xanchor="center", font=dict(size=17)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        hovermode="x unified",
    )
    fig.add_annotation(
        x=0.5,
        y=1.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=metrics,
        align="center",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DDDDDD",
        borderwidth=1,
        font=dict(size=12),
    )
    fig.update_yaxes(title_text="收盘价", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="净值", row=2, col=1)
    fig.update_yaxes(title_text="因子值 / sigma", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="收盘价", row=3, col=1, secondary_y=True)
    return fig


def _make_rule_pair_html(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    best_row: pd.Series,
    factor_desc: dict[str, str] | None = None,
) -> str:
    """Render the rule-pair chart as three independent Plotly figures.

    A single multi-row Plotly figure makes the hover line, global legend and
    multiple right axes visually fight each other in long time-series charts.
    Keeping the three panels separate matches the static report layout more
    closely and avoids apparent overlap.
    """
    code_key = CODE_COL if CODE_COL in best_row.index else ("浠ｇ爜" if "浠ｇ爜" in best_row.index else best_row.index[0])
    code = str(best_row.get(code_key, ""))
    factor = str(best_row["factor"])
    open_condition = str(best_row["open_condition"])
    close_condition = str(best_row["close_condition"])
    factor_cols = _select_rule_pair_factor_cols(df, factor, open_condition, close_condition)
    if not factor_cols:
        raise ValueError(f"No factor columns found for {factor}")

    group = df[df[CODE_COL].astype(str).eq(code)].copy()
    if group.empty:
        raise ValueError(f"Instrument not found: {code}")
    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    signal_points = signals[
        signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(code)
        & signals[SIGNAL_FACTOR_COL].astype(str).eq(factor)
        & signals[SIGNAL_PATTERN_COL].isin([open_condition, close_condition])
    ].copy()
    signal_points[SIGNAL_DATE_COL] = pd.to_datetime(signal_points[SIGNAL_DATE_COL], errors="coerce")
    signal_points = signal_points.dropna(subset=[SIGNAL_DATE_COL])

    best_equity = best_row.attrs.get("_equity_df")
    if best_equity is None or best_equity.empty:
        raise ValueError(f"Missing equity curve for {factor} | {open_condition} -> {close_condition}")
    best_equity = best_equity.copy()
    best_equity[DATE_COL] = pd.to_datetime(best_equity[DATE_COL], errors="coerce")
    best_equity = best_equity.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    start_dt = best_equity[DATE_COL].min()
    end_dt = best_equity[DATE_COL].max()
    group = group[group[DATE_COL].between(start_dt, end_dt)].copy()
    signal_points = signal_points[signal_points[SIGNAL_DATE_COL].between(start_dt, end_dt)].copy()
    if not signal_points.empty:
        signal_points = signal_points.merge(
            group[[DATE_COL, *factor_cols, PRICE_COL]],
            left_on=SIGNAL_DATE_COL,
            right_on=DATE_COL,
            how="left",
        )

    font = dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)
    hover_font = dict(font=font)
    template = "plotly_white"

    # Panel 1: price, long zones and trade marks.
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=group[DATE_COL],
            y=group[PRICE_COL],
            name="收盘价",
            mode="lines",
            line=dict(color="#4C78A8", width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
        )
    )
    open_dates = pd.Series(dtype="datetime64[ns]")
    close_dates = pd.Series(dtype="datetime64[ns]")
    trade_source = group[[DATE_COL, PRICE_COL, *factor_cols]].drop_duplicates(subset=[DATE_COL]).set_index(DATE_COL)
    open_trades = pd.DataFrame(columns=[DATE_COL, PRICE_COL, *factor_cols])
    close_trades = pd.DataFrame(columns=[DATE_COL, PRICE_COL, *factor_cols])
    if "position" in best_equity.columns:
        for start, end in _long_spans(best_equity):
            fig_price.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(246,199,199,0.28)",
                line_width=0,
                layer="below",
            )
        prev = best_equity["position"].shift(1).fillna(0.0)
        open_dates = best_equity.loc[(prev <= 0) & (best_equity["position"] > 0), DATE_COL]
        close_dates = best_equity.loc[(prev > 0) & (best_equity["position"] <= 0), DATE_COL]
        open_trades = trade_source.reindex(pd.DatetimeIndex(open_dates)).reset_index().rename(columns={"index": DATE_COL})
        close_trades = trade_source.reindex(pd.DatetimeIndex(close_dates)).reset_index().rename(columns={"index": DATE_COL})
        if not open_trades.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=open_trades[DATE_COL],
                    y=open_trades[PRICE_COL],
                    name="开仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#169B62", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>开仓价=%{y:.2f}<extra></extra>",
                )
            )
        if not close_trades.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=close_trades[DATE_COL],
                    y=close_trades[PRICE_COL],
                    name="平仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#D62728", line=dict(color="white", width=0.6)),
                    hovertemplate="%{x|%Y-%m-%d}<br>平仓价=%{y:.2f}<extra></extra>",
                )
            )
    fig_price.update_layout(
        template=template,
        height=320,
        margin=dict(l=50, r=45, t=24, b=22),
        font=font,
        hoverlabel=hover_font,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig_price.update_yaxes(title_text="收盘价")

    # Panel 2: strategy and excess equity.
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=best_equity[DATE_COL],
            y=best_equity["strategy_equity"],
            name="策略净值",
            mode="lines",
            line=dict(color="#1D3557", width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>策略净值=%{y:.4f}<extra></extra>",
        )
    )
    if "excess_equity" in best_equity.columns:
        fig_equity.add_trace(
            go.Scatter(
                x=best_equity[DATE_COL],
                y=best_equity["excess_equity"],
                name="超额曲线",
                mode="lines",
                line=dict(color="#C1121F", width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>超额曲线=%{y:.4f}<extra></extra>",
            )
        )
    fig_equity.add_hline(y=1.0, line_width=0.9, line_dash="dot", line_color="#777777")
    metrics = (
        f"超额年化: {float(best_row.get('excess_annual_return', 0.0)):.2%}   "
        f"夏普: {float(best_row.get('sharpe', 0.0)):.2f}   "
        f"最大回撤: {float(best_row.get('max_drawdown', 0.0)):.2%}"
    )
    fig_equity.add_annotation(
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=metrics,
        align="center",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DDDDDD",
        borderwidth=1,
        font=dict(size=12),
    )
    fig_equity.update_layout(
        template=template,
        height=320,
        margin=dict(l=50, r=45, t=48, b=22),
        font=font,
        hoverlabel=hover_font,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_equity.update_yaxes(title_text="净值")

    # Panel 3: factor, price and actual trade marks. If the rule uses
    # raw/season/year lines, split them into stacked rows for cleaner reading.
    row_count = len(factor_cols)
    factor_height = 260 * row_count + 80
    fig_factor = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.18 / max(row_count, 1)),
        specs=[[{"secondary_y": True}] for _ in factor_cols],
    )
    colors = ["#044E7E", "#1F77B4", "#7B2CBF"]
    open_rule_cn = format_rule_name_cn(open_condition)
    close_rule_cn = format_rule_name_cn(close_condition)
    for idx, col in enumerate(factor_cols, start=1):
        fig_factor.add_trace(
            go.Scatter(
                x=group[DATE_COL],
                y=group[col],
                name=col,
                mode="lines",
                line=dict(color=colors[(idx - 1) % len(colors)], width=1.6),
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}=%{y:.4f}<extra></extra>",
            ),
            row=idx,
            col=1,
            secondary_y=False,
        )
        fig_factor.add_trace(
            go.Scatter(
                x=group[DATE_COL],
                y=group[PRICE_COL],
                name="收盘价",
                mode="lines",
                line=dict(color="#FF8080", width=1.1, dash="dash"),
                opacity=0.78,
                hovertemplate="%{x|%Y-%m-%d}<br>收盘价=%{y:.2f}<extra></extra>",
                showlegend=(idx == 1),
            ),
            row=idx,
            col=1,
            secondary_y=True,
        )

        open_points = open_trades[[DATE_COL, col]].rename(columns={DATE_COL: SIGNAL_DATE_COL})
        close_points = close_trades[[DATE_COL, col]].rename(columns={DATE_COL: SIGNAL_DATE_COL})
        open_points = open_points.dropna(subset=[col])
        close_points = close_points.dropna(subset=[col])
        if not open_points.empty:
            fig_factor.add_trace(
                go.Scatter(
                    x=open_points[SIGNAL_DATE_COL],
                    y=open_points[col],
                    name="实际开仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="#169B62", line=dict(color="white", width=0.5)),
                    customdata=np.repeat(open_rule_cn, len(open_points)),
                    hovertemplate="%{x|%Y-%m-%d}<br>开仓规则=%{customdata}<br>因子值=%{y:.4f}<extra></extra>",
                    showlegend=(idx == 1),
                ),
                row=idx,
                col=1,
                secondary_y=False,
            )
        if not close_points.empty:
            fig_factor.add_trace(
                go.Scatter(
                    x=close_points[SIGNAL_DATE_COL],
                    y=close_points[col],
                    name="实际平仓点",
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="#D62728", line=dict(color="white", width=0.5)),
                    customdata=np.repeat(close_rule_cn, len(close_points)),
                    hovertemplate="%{x|%Y-%m-%d}<br>平仓规则=%{customdata}<br>因子值=%{y:.4f}<extra></extra>",
                    showlegend=(idx == 1),
                ),
                row=idx,
                col=1,
                secondary_y=False,
            )
        for y in (0, 1, -1):
            fig_factor.add_hline(y=y, line_width=0.8, line_dash="dot", line_color="#999999", row=idx, col=1)
        fig_factor.update_yaxes(title_text=col, row=idx, col=1, secondary_y=False)
        fig_factor.update_yaxes(title_text="收盘价", row=idx, col=1, secondary_y=True)
    fig_factor.update_layout(
        template=template,
        height=factor_height,
        margin=dict(l=50, r=45, t=24, b=40),
        font=font,
        hoverlabel=hover_font,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig_factor.update_xaxes(title_text="", row=row_count, col=1)

    return (
        "<div class='rule-pair-figures'>"
        "<div class='plot-panel'><div class='plot-panel-title'>1. 价格与持仓区间</div>"
        f"{_fig_html(fig_price, height=320)}"
        "</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>2. 策略净值与超额曲线</div>"
        f"{_fig_html(fig_equity, height=320)}"
        "</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>3. 因子值、收盘价与实际交易点</div>"
        f"{_fig_html(fig_factor, height=factor_height)}"
        "</div>"
        "</div>"
    )
