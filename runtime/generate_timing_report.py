from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from baseline_score_strategy import format_rule_name_cn, rolling_zscore
from composite_timing_strategies import (
    CHALLENGER_STRATEGY_ID,
    PRIMARY_STRATEGY_ID,
)
from reporting import _state_direction_score, score_signal_points_for_advisor
from timing_config import (
    CODE_COL,
    CORE_CATEGORIES,
    DATE_COL,
    NAME_COL,
    PRICE_COL,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_VALUE_COL,
    STATE_FLAT,
    STATE_LONG,
)


COMPOSITE_STRATEGY_META: dict[str, dict[str, Any]] = {
    PRIMARY_STRATEGY_ID: {
        "name": "类别等权两速复合策略",
        "weighting": "四类单因子各分配 25% 权重，再在类别内部等权分配。",
        "rebalance": "以周频综合信号作为稳定锚；日度综合分数达到 +0.25 或 -0.25 时，允许周中提前开仓或平仓。",
        "execution": "所有信号在收盘后确认，并于下一交易日执行；日度分数处于 (-0.25, +0.25) 时保持现有仓位。",
        "strong_threshold": 0.25,
    },
    CHALLENGER_STRATEGY_ID: {
        "name": "开仓频率平方根倒数复合策略",
        "weighting": "单规则权重与截至 2020-12-31 训练期年均开仓频率的平方根倒数成正比，并归一化为 100%。",
        "rebalance": "以周频频率加权信号作为稳定锚；日度频率加权分数达到 +0.25 或 -0.25 时，允许周中提前开仓或平仓。",
        "execution": "所有信号在收盘后确认，并于下一交易日执行；训练期权重保持固定，不使用后续评估期数据重新拟合。",
        "strong_threshold": 0.25,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _category_colors() -> dict[str, str]:
    return {
        "赔率/资金": "#3498db",
        "赔率/估值": "#2ecc71",
        "赔率/筹码": "#e67e22",
        "胜率/估值": "#9b59b6",
        "胜率/量": "#7B2CBF",
        "胜率/资金": "#2563eb",
        "辅助/筹码结构": "#1abc9c",
        "辅助/资金分歧": "#e74c3c",
        "辅助/风险状态": "#95a5a6",
        "辅助/技术状态": "#64748b",
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


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        num = float(value)
    except Exception:
        return "--"
    if pd.isna(num):
        return "--"
    return f"{num:.{digits}f}"


def _format_pct_value(value: Any, digits: int = 1) -> str:
    try:
        num = float(value)
    except Exception:
        return "--"
    if pd.isna(num):
        return "--"
    return f"{num * 100:.{digits}f}%"


def _strategy_latest_status(strategy_df: pd.DataFrame) -> dict[str, Any]:
    if strategy_df.empty:
        return {}
    df = strategy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        return {}

    latest = df.iloc[-1]
    position = pd.to_numeric(pd.Series([latest.get("position", 0.0)]), errors="coerce").iloc[0]
    if pd.isna(position):
        position = 0.0
    if position >= 0.99:
        position_label = "多头满仓"
        position_class = "pill-bullish"
    elif position >= 0.49:
        position_label = "多头半仓"
        position_class = "pill-neutral"
    else:
        position_label = "空仓"
        position_class = "pill-bearish"

    return {
        "date": str(pd.Timestamp(latest[DATE_COL]).date()),
        "position": float(position),
        "position_label": position_label,
        "position_class": position_class,
        "entry_z": pd.to_numeric(pd.Series([latest.get("entry_z")]), errors="coerce").iloc[0],
        "exit_z": pd.to_numeric(pd.Series([latest.get("exit_z")]), errors="coerce").iloc[0],
        "open_event": int(pd.to_numeric(pd.Series([latest.get("open_event", 0)]), errors="coerce").fillna(0).iloc[0]),
        "close_event": int(pd.to_numeric(pd.Series([latest.get("close_event", 0)]), errors="coerce").fillna(0).iloc[0]),
        "open_rule": format_rule_name_cn(str(latest.get("open_rule", ""))),
        "close_rule": format_rule_name_cn(str(latest.get("close_rule", ""))),
    }


def _z_bucket(value: float) -> tuple[float, float, str]:
    edges = [-float("inf"), *[step / 4 for step in range(-8, 9)], float("inf")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if value >= lo and value < hi:
            if lo == -float("inf"):
                return lo, hi, f"< {hi:g}"
            if hi == float("inf"):
                return lo, hi, f">= {lo:g}"
            return lo, hi, f"[{lo:g}, {hi:g})"
    return edges[-2], edges[-1], f">= {edges[-2]:g}"


def _score_z20_forward_stats(strategy_df: pd.DataFrame, horizon: int = 5) -> list[dict[str, Any]]:
    if strategy_df.empty or PRICE_COL not in strategy_df.columns:
        return []
    df = strategy_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    required = {"entry_score", "exit_score", PRICE_COL}
    if df.empty or not required.issubset(df.columns):
        return []

    close = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df["entry_z_20_3"] = rolling_zscore(df["entry_score"], 20).rolling(3, min_periods=1).mean()
    df["exit_z_20_3"] = rolling_zscore(df["exit_score"], 20).rolling(3, min_periods=1).mean()
    df["future_5d_return"] = close.shift(-horizon) / close - 1.0

    rows: list[dict[str, Any]] = []
    specs = [
        ("抄底得分20Z", "entry_z_20_3", 1.0, "未来5日上涨胜率"),
        ("逃顶得分20Z", "exit_z_20_3", -1.0, "未来5日下跌胜率"),
    ]
    for name, col, direction, win_label in specs:
        latest_valid = df[[DATE_COL, col]].dropna()
        if latest_valid.empty:
            continue
        latest = latest_valid.iloc[-1]
        current_value = float(latest[col])
        lo, hi, bucket_label = _z_bucket(current_value)
        sample = df[df[col].ge(lo) & df[col].lt(hi) & df["future_5d_return"].notna()].copy()
        directional_ret = sample["future_5d_return"] * direction
        wins = directional_ret[directional_ret > 0]
        losses = directional_ret[directional_ret < 0]
        avg_win = float(wins.mean()) if not wins.empty else pd.NA
        avg_loss = float((-losses).mean()) if not losses.empty else pd.NA
        payoff = (avg_win / avg_loss) if avg_win is not pd.NA and avg_loss is not pd.NA and avg_loss > 0 else pd.NA
        rows.append(
            {
                "name": name,
                "date": str(pd.Timestamp(latest[DATE_COL]).date()),
                "current_value": current_value,
                "bucket": bucket_label,
                "sample_count": int(len(sample)),
                "win_label": win_label,
                "win_rate": float((directional_ret > 0).mean()) if len(sample) else pd.NA,
                "payoff": payoff,
                "avg_directional_return": float(directional_ret.mean()) if len(sample) else pd.NA,
                "median_forward_return": float(sample["future_5d_return"].median()) if len(sample) else pd.NA,
            }
        )
    return rows


def _signal_counts_from_df(signals_df: pd.DataFrame, top_n_days: int = 20) -> dict[str, Any]:
    if signals_df.empty or SIGNAL_DATE_COL not in signals_df.columns:
        return {"daily": [], "top_factors": [], "top_patterns": [], "total_recent": 0}
    df = signals_df.copy()
    df[SIGNAL_DATE_COL] = pd.to_datetime(df[SIGNAL_DATE_COL], errors="coerce")
    df = df.dropna(subset=[SIGNAL_DATE_COL])
    if df.empty:
        return {"daily": [], "top_factors": [], "top_patterns": [], "total_recent": 0}
    df["_date_text"] = df[SIGNAL_DATE_COL].dt.date.astype(str)
    recent_dates = sorted(df["_date_text"].dropna().unique().tolist(), reverse=True)[:top_n_days]
    recent = df[df["_date_text"].isin(recent_dates)].copy()
    daily = []
    if not recent.empty:
        day_counts = (
            recent.assign(
                is_open=recent[SIGNAL_VALUE_COL].astype(str).eq("1"),
                is_close=recent[SIGNAL_VALUE_COL].astype(str).eq("-1"),
            )
            .groupby("_date_text", sort=False)
            .agg(open=("is_open", "sum"), close=("is_close", "sum"), factors=(SIGNAL_FACTOR_COL, "nunique"))
            .reset_index()
            .rename(columns={"_date_text": "date"})
        )
        daily = day_counts.to_dict(orient="records")
    factor_totals = (
        recent[recent[SIGNAL_VALUE_COL].astype(str).eq("1")]
        .groupby(SIGNAL_FACTOR_COL)
        .size()
        .sort_values(ascending=False)
        .head(30)
    )
    pattern_totals = (
        recent[recent[SIGNAL_VALUE_COL].astype(str).eq("1")]
        .groupby(SIGNAL_PATTERN_COL)
        .size()
        .sort_values(ascending=False)
        .head(30)
    )
    return {
        "daily": daily,
        "top_factors": list(factor_totals.items()),
        "top_patterns": list(pattern_totals.items()),
        "total_recent": int(recent[recent[SIGNAL_VALUE_COL].astype(str).eq("1")].shape[0]),
    }


def _filter_effective_signals_expanding(
    input_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    immediate_horizon: int = 5,
    delayed_horizon: int = 20,
    warmup_days: int = 252,
    win_threshold: float = 0.5,
    payoff_threshold: float = 1.0,
    edge_threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = {
        "raw_signal_count": int(len(signals_df)) if signals_df is not None else 0,
        "effective_signal_count": 0,
        "immediate_horizon": immediate_horizon,
        "delayed_horizon": delayed_horizon,
        "warmup_days": warmup_days,
        "win_threshold": win_threshold,
        "payoff_threshold": payoff_threshold,
        "edge_threshold": edge_threshold,
        "role_counts": {},
    }
    required_signals = {SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, SIGNAL_VALUE_COL}
    if input_df.empty or signals_df.empty or PRICE_COL not in input_df.columns or not required_signals.issubset(signals_df.columns):
        return signals_df.iloc[0:0].copy(), meta

    sig = signals_df.copy().reset_index(drop=True)
    sig[SIGNAL_DATE_COL] = pd.to_datetime(sig[SIGNAL_DATE_COL], errors="coerce").dt.normalize()
    sig[SIGNAL_VALUE_COL] = pd.to_numeric(sig[SIGNAL_VALUE_COL], errors="coerce")
    sig = sig.dropna(subset=[SIGNAL_DATE_COL, SIGNAL_VALUE_COL]).copy()
    sig = sig[sig[SIGNAL_VALUE_COL].isin([1, -1])].reset_index(drop=True)
    if sig.empty:
        return sig, meta

    price_cols = [DATE_COL, PRICE_COL]
    if CODE_COL in input_df.columns:
        price_cols.insert(0, CODE_COL)
    price = input_df[price_cols].copy()
    price[DATE_COL] = pd.to_datetime(price[DATE_COL], errors="coerce").dt.normalize()
    price = price.dropna(subset=[DATE_COL])
    if CODE_COL not in price.columns:
        instrument = sig[SIGNAL_INSTRUMENT_COL].astype(str).mode().iloc[0]
        price[CODE_COL] = instrument
    price[CODE_COL] = price[CODE_COL].astype(str)

    price_parts = []
    for code, group in price.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).drop_duplicates(DATE_COL).reset_index(drop=True)
        close = pd.to_numeric(group[PRICE_COL], errors="coerce")
        part = group[[CODE_COL, DATE_COL]].copy()
        part["_date_idx"] = np.arange(len(group), dtype=int)
        for horizon in (immediate_horizon, delayed_horizon):
            part[f"_known_idx_{horizon}"] = part["_date_idx"] + horizon
            part[f"_forward_return_{horizon}"] = close.shift(-horizon) / close - 1.0
        price_parts.append(part)
    if not price_parts:
        return sig.iloc[0:0].copy(), meta
    price_map = pd.concat(price_parts, ignore_index=True)

    sig[SIGNAL_INSTRUMENT_COL] = sig[SIGNAL_INSTRUMENT_COL].astype(str)
    sig = sig.merge(
        price_map,
        left_on=[SIGNAL_INSTRUMENT_COL, SIGNAL_DATE_COL],
        right_on=[CODE_COL, DATE_COL],
        how="left",
    )
    sig = sig[sig["_date_idx"].notna()].copy().reset_index(drop=True)
    if sig.empty:
        return sig.iloc[0:0].copy(), meta
    sig["_date_idx"] = sig["_date_idx"].astype(int)
    for horizon in (immediate_horizon, delayed_horizon):
        sig[f"_known_idx_{horizon}"] = pd.to_numeric(sig[f"_known_idx_{horizon}"], errors="coerce")
        sig[f"_directional_return_{horizon}"] = np.where(
            sig[SIGNAL_VALUE_COL].eq(1),
            sig[f"_forward_return_{horizon}"],
            -sig[f"_forward_return_{horizon}"],
        )

    term_role = np.full(len(sig), "", dtype=object)
    selected_horizon = np.full(len(sig), np.nan, dtype=float)
    selected_win_rate = np.full(len(sig), np.nan, dtype=float)
    selected_payoff = np.full(len(sig), np.nan, dtype=float)
    selected_mean_return = np.full(len(sig), np.nan, dtype=float)
    group_cols = [SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, SIGNAL_VALUE_COL]
    for _, group in sig.groupby(group_cols, sort=False):
        group_idx = group.index.to_numpy()
        trigger_idx = group["_date_idx"].to_numpy(dtype=float)
        metrics: dict[int, dict[str, np.ndarray]] = {}
        for horizon in (immediate_horizon, delayed_horizon):
            known_col = f"_known_idx_{horizon}"
            return_col = f"_directional_return_{horizon}"
            obs = group[group[return_col].notna() & group[known_col].notna()].copy()
            empty = np.full(len(group), np.nan, dtype=float)
            if obs.empty:
                metrics[horizon] = {
                    "count": np.zeros(len(group), dtype=float),
                    "win_rate": empty.copy(),
                    "payoff": empty.copy(),
                    "mean_return": empty.copy(),
                }
                continue

            obs = obs.sort_values(known_col)
            known_idx = obs[known_col].to_numpy(dtype=float)
            directional = obs[return_col].to_numpy(dtype=float)
            wins = directional > 0
            losses = directional < 0
            cum_count = np.r_[0, np.arange(1, len(obs) + 1)]
            cum_return_sum = np.r_[0.0, np.cumsum(directional)]
            cum_win_count = np.r_[0, np.cumsum(wins)]
            cum_loss_count = np.r_[0, np.cumsum(losses)]
            cum_win_sum = np.r_[0.0, np.cumsum(np.where(wins, directional, 0.0))]
            cum_loss_sum = np.r_[0.0, np.cumsum(np.where(losses, -directional, 0.0))]

            hist_pos = np.searchsorted(known_idx, trigger_idx, side="left")
            count = cum_count[hist_pos].astype(float)
            return_sum = cum_return_sum[hist_pos]
            win_count = cum_win_count[hist_pos].astype(float)
            loss_count = cum_loss_count[hist_pos].astype(float)
            win_sum = cum_win_sum[hist_pos]
            loss_sum = cum_loss_sum[hist_pos]
            win_rate = np.divide(win_count, count, out=empty.copy(), where=count > 0)
            mean_return = np.divide(return_sum, count, out=empty.copy(), where=count > 0)
            avg_win = np.divide(win_sum, win_count, out=empty.copy(), where=win_count > 0)
            avg_loss = np.divide(loss_sum, loss_count, out=np.zeros_like(count), where=loss_count > 0)
            payoff = np.divide(avg_win, avg_loss, out=np.full_like(count, np.inf), where=avg_loss > 0)
            metrics[horizon] = {
                "count": count,
                "win_rate": win_rate,
                "payoff": payoff,
                "mean_return": mean_return,
            }

        immediate = metrics[immediate_horizon]
        delayed = metrics[delayed_horizon]
        warm = trigger_idx >= warmup_days
        immediate_ok = (
            warm
            & (immediate["count"] > 0)
            & (immediate["win_rate"] > win_threshold)
            & (immediate["payoff"] > payoff_threshold)
            & (immediate["mean_return"] > edge_threshold)
        )
        delayed_ok = (
            warm
            & ~immediate_ok
            & (delayed["count"] > 0)
            & (delayed["win_rate"] > win_threshold)
            & (delayed["payoff"] > payoff_threshold)
            & (delayed["mean_return"] > edge_threshold)
            & ((delayed["mean_return"] - immediate["mean_return"]) > edge_threshold)
        )

        is_open = group[SIGNAL_VALUE_COL].to_numpy(dtype=float) == 1
        roles = np.full(len(group), "", dtype=object)
        roles[immediate_ok & is_open] = "追涨"
        roles[delayed_ok & is_open] = "抄底"
        roles[immediate_ok & ~is_open] = "快速逃顶"
        roles[delayed_ok & ~is_open] = "预先指示"
        term_role[group_idx] = roles

        selected_horizon[group_idx] = np.where(
            immediate_ok,
            immediate_horizon,
            np.where(delayed_ok, delayed_horizon, np.nan),
        )
        selected_win_rate[group_idx] = np.where(
            immediate_ok,
            immediate["win_rate"],
            np.where(delayed_ok, delayed["win_rate"], np.nan),
        )
        selected_payoff[group_idx] = np.where(
            immediate_ok,
            immediate["payoff"],
            np.where(delayed_ok, delayed["payoff"], np.nan),
        )
        selected_mean_return[group_idx] = np.where(
            immediate_ok,
            immediate["mean_return"],
            np.where(delayed_ok, delayed["mean_return"], np.nan),
        )

    sig["term_role"] = term_role
    sig["term_horizon"] = selected_horizon
    sig["term_win_rate"] = selected_win_rate
    sig["term_payoff"] = selected_payoff
    sig["term_mean_directional_return"] = selected_mean_return
    pass_mask = sig["term_role"].ne("").to_numpy()
    output_cols = signals_df.columns.intersection(sig.columns).tolist() + [
        "term_role",
        "term_horizon",
        "term_win_rate",
        "term_payoff",
        "term_mean_directional_return",
    ]
    filtered = sig.loc[pass_mask, output_cols].copy()
    meta["effective_signal_count"] = int(len(filtered))
    if not filtered.empty:
        meta["first_effective_date"] = str(pd.to_datetime(filtered[SIGNAL_DATE_COL]).min().date())
        meta["last_effective_date"] = str(pd.to_datetime(filtered[SIGNAL_DATE_COL]).max().date())
        meta["role_counts"] = {str(k): int(v) for k, v in filtered["term_role"].value_counts().items()}
    return filtered, meta


def _event_score_forward_stats(
    input_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    status_df: pd.DataFrame,
    rule_summary_df: pd.DataFrame,
    current_scores: dict[str, Any],
    horizon: int = 5,
) -> list[dict[str, Any]]:
    if input_df.empty or signals_df.empty or status_df.empty or PRICE_COL not in input_df.columns:
        return []
    required_status = {
        SIGNAL_INSTRUMENT_COL,
        SIGNAL_FACTOR_COL,
        "open_pattern",
        "direction_bucket",
        "factor_category",
        "frequency",
    }
    required_signals = {SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, SIGNAL_VALUE_COL}
    if not required_status.issubset(status_df.columns) or not required_signals.issubset(signals_df.columns):
        return []

    price = input_df[[DATE_COL, PRICE_COL]].copy()
    price[DATE_COL] = pd.to_datetime(price[DATE_COL], errors="coerce")
    price = price.dropna(subset=[DATE_COL]).sort_values(DATE_COL).drop_duplicates(DATE_COL).reset_index(drop=True)
    if price.empty:
        return []
    all_dates = price[DATE_COL].dt.normalize().to_numpy()
    close = pd.to_numeric(price[PRICE_COL], errors="coerce")
    future_5d = (close.shift(-horizon) / close - 1.0).to_numpy(dtype=float)
    date_to_idx = {pd.Timestamp(date).normalize(): idx for idx, date in enumerate(all_dates)}

    sig = signals_df[list(required_signals)].copy()
    sig[SIGNAL_DATE_COL] = pd.to_datetime(sig[SIGNAL_DATE_COL], errors="coerce").dt.normalize()
    sig = sig.dropna(subset=[SIGNAL_DATE_COL])
    sig["_date_idx"] = sig[SIGNAL_DATE_COL].map(date_to_idx)
    sig = sig[sig["_date_idx"].notna()].copy()
    if sig.empty:
        return []
    sig["_date_idx"] = sig["_date_idx"].astype(int)
    sig[SIGNAL_VALUE_COL] = pd.to_numeric(sig[SIGNAL_VALUE_COL], errors="coerce")

    close_idx: dict[tuple[str, str], np.ndarray] = {}
    close_sig = sig[sig[SIGNAL_VALUE_COL].eq(-1)]
    if not close_sig.empty:
        for key, group in close_sig.groupby([SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL], sort=False):
            close_idx[(str(key[0]), str(key[1]))] = np.sort(group["_date_idx"].dropna().astype(int).unique())

    open_idx: dict[tuple[str, str, str], np.ndarray] = {}
    open_sig = sig[sig[SIGNAL_VALUE_COL].eq(1)]
    if not open_sig.empty:
        for key, group in open_sig.groupby([SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL], sort=False):
            open_idx[(str(key[0]), str(key[1]), str(key[2]))] = np.sort(group["_date_idx"].dropna().astype(int).unique())

    scored = score_signal_points_for_advisor(status_df, rule_summary=rule_summary_df)
    if scored.empty:
        return []
    for col in ("category_weight", "frequency_weight", "history_multiplier"):
        scored[col] = pd.to_numeric(scored[col], errors="coerce").fillna(1.0)
    scored["_weight"] = scored["category_weight"] * scored["frequency_weight"] * scored["history_multiplier"]
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna(subset=["_weight"])

    n = len(price)
    total_num = np.zeros(n, dtype=float)
    total_den = np.zeros(n, dtype=float)
    core_num = np.zeros(n, dtype=float)
    core_den = np.zeros(n, dtype=float)

    for _, row in scored.iterrows():
        instrument = str(row.get(SIGNAL_INSTRUMENT_COL, ""))
        factor = str(row.get(SIGNAL_FACTOR_COL, ""))
        open_pattern = str(row.get("open_pattern", ""))
        weight = float(row.get("_weight", 0.0) or 0.0)
        if weight <= 0:
            continue
        _, long_base = _state_direction_score(STATE_LONG, str(row.get("direction_bucket", "")))
        _, flat_base = _state_direction_score(STATE_FLAT, str(row.get("direction_bucket", "")))
        long_base = float(long_base or 0.0)
        flat_base = float(flat_base or 0.0)
        if long_base == 0.0 and flat_base == 0.0:
            continue

        events = np.zeros(n, dtype=np.int8)
        cidx = close_idx.get((instrument, factor))
        if cidx is not None and len(cidx):
            events[cidx] = -1
        oidx = open_idx.get((instrument, factor, open_pattern))
        if oidx is not None and len(oidx):
            events[oidx] = 1
        if not np.any(events):
            continue

        change_idx = np.flatnonzero(events)
        state = np.zeros(n, dtype=np.int8)
        if len(change_idx):
            for start, end, value in zip(change_idx, np.r_[change_idx[1:], n], events[change_idx]):
                state[start:end] = value
        long_mask = state == 1
        flat_mask = state == -1
        is_core = row.get("factor_category") in CORE_CATEGORIES
        if long_base != 0.0:
            total_num[long_mask] += long_base * weight
            total_den[long_mask] += weight
            if is_core:
                core_num[long_mask] += long_base * weight
                core_den[long_mask] += weight
        if flat_base != 0.0:
            total_num[flat_mask] += flat_base * weight
            total_den[flat_mask] += weight
            if is_core:
                core_num[flat_mask] += flat_base * weight
                core_den[flat_mask] += weight

    total_score = np.divide(total_num, total_den, out=np.full(n, np.nan), where=total_den > 0)
    core_score = np.divide(core_num, core_den, out=np.full(n, np.nan), where=core_den > 0)

    rows: list[dict[str, Any]] = []
    specs = [
        ("事件核心评分", "core_score", core_score),
        ("事件总评分", "total_score", total_score),
    ]
    date_values = price[DATE_COL].dt.date.astype(str).to_numpy()
    for name, score_key, series in specs:
        try:
            current_value = float(current_scores.get(score_key))
        except Exception:
            valid = series[np.isfinite(series)]
            current_value = float(valid[-1]) if len(valid) else np.nan
        if not np.isfinite(current_value):
            continue
        lo, hi, bucket_label = _z_bucket(current_value)
        direction = 1.0 if current_value >= 0 else -1.0
        win_label = "未来5日上涨胜率" if direction > 0 else "未来5日下跌胜率"
        sample_mask = np.isfinite(series) & (series >= lo) & (series < hi) & np.isfinite(future_5d)
        directional_ret = future_5d[sample_mask] * direction
        wins = directional_ret[directional_ret > 0]
        losses = directional_ret[directional_ret < 0]
        avg_win = float(wins.mean()) if len(wins) else pd.NA
        avg_loss = float((-losses).mean()) if len(losses) else pd.NA
        payoff = (avg_win / avg_loss) if avg_win is not pd.NA and avg_loss is not pd.NA and avg_loss > 0 else pd.NA
        valid_idx = np.where(np.isfinite(series))[0]
        rows.append(
            {
                "name": name,
                "date": date_values[valid_idx[-1]] if len(valid_idx) else "",
                "current_value": current_value,
                "bucket": bucket_label,
                "sample_count": int(sample_mask.sum()),
                "win_label": win_label,
                "win_rate": float((directional_ret > 0).mean()) if len(directional_ret) else pd.NA,
                "payoff": payoff,
                "avg_directional_return": float(directional_ret.mean()) if len(directional_ret) else pd.NA,
            }
        )
    return rows


def _safe_float(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return float("nan")
    return num


def _date_text(value: Any) -> str:
    if value is None or value == "":
        return "--"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "--"
    return str(pd.Timestamp(ts).date())


def _base_factor_name(factor: Any) -> str:
    text = str(factor or "")
    for suffix in ("_月线", "_季线", "_年线"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _selected_rule_latest_signal_overview(status_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Any]:
    if status_df.empty:
        return {"total": 0, "bullish": [], "bearish": [], "bullish_count": 0, "bearish_count": 0}

    summary_by_rule: dict[str, pd.Series] = {}
    if not summary_df.empty and "rule_id" in summary_df.columns:
        for _, row in summary_df.iterrows():
            summary_by_rule[str(row.get("rule_id", ""))] = row

    records: list[dict[str, Any]] = []
    latest_dates = pd.to_datetime(status_df.get("latest_date", pd.Series(dtype=object)), errors="coerce")
    latest_ts = latest_dates.max() if len(latest_dates) else pd.NaT
    for _, row in status_df.iterrows():
        rule_id = str(row.get("rule_id", ""))
        factor = str(row.get("factor", ""))
        summary_row = summary_by_rule.get(rule_id, pd.Series(dtype=object))
        current_state = str(row.get("current_state", "空"))
        is_long = current_state == "多"
        last_signal_date = row.get("last_open_signal_date") if is_long else row.get("last_close_signal_date")
        last_signal_date_text = _date_text(last_signal_date)
        if last_signal_date_text == "--":
            state_age_days: Any = pd.NA
        else:
            state_age_days = int((pd.Timestamp(latest_ts) - pd.Timestamp(last_signal_date_text)).days) if pd.notna(latest_ts) else pd.NA
        records.append(
            {
                "factor": factor,
                "base_factor": _base_factor_name(factor),
                "current_view": "看多" if is_long else "看空",
                "state_class": "pill-bullish" if is_long else "pill-bearish",
                "position": 1.0 if is_long else 0.0,
                "latest_date": _date_text(row.get("latest_date")),
                "last_signal_type": "开仓" if is_long and last_signal_date_text != "--" else "平仓" if last_signal_date_text != "--" else "无",
                "last_signal_date": last_signal_date_text,
                "state_age_days": state_age_days,
                "open_rule": str(row.get("open_condition", "")),
                "close_rule": str(row.get("close_condition", "")),
                "excess_annual_return": _safe_float(summary_row.get("excess_annual_return")),
                "sharpe": _safe_float(summary_row.get("sharpe")),
            }
        )

    records = sorted(records, key=lambda item: (str(item.get("last_signal_date", "")) != "--", str(item.get("last_signal_date", ""))), reverse=True)
    bullish = [item for item in records if item["current_view"] == "看多"]
    bearish = [item for item in records if item["current_view"] == "看空"]
    return {
        "total": len(records),
        "latest_date": records[0]["latest_date"] if records else "",
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "bullish": bullish,
        "bearish": bearish,
    }


def _selected_rule_metric_html(summary_row: pd.Series, status_row: pd.Series) -> str:
    trade_count_value = _safe_float(summary_row.get("trade_count", 0))
    trade_count_text = "--" if not np.isfinite(trade_count_value) else str(int(trade_count_value))
    metrics = [
        ("年化超额", _format_pct_value(summary_row.get("excess_annual_return"), 1)),
        ("Sharpe", _format_float(summary_row.get("sharpe"), 2)),
        ("最大回撤", _format_pct_value(summary_row.get("max_drawdown"), 1)),
        ("交易数", trade_count_text),
        ("胜率", _format_pct_value(summary_row.get("win_rate"), 1)),
        ("当前状态", str(status_row.get("current_state", "--") or "--")),
    ]
    return "".join(
        "<span class='rule-metric-chip'>"
        f"<b>{_escape(label)}</b>{_escape(value)}"
        "</span>"
        for label, value in metrics
    )


def _filter_selected_rule_rows(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "rule_id" in out.columns:
        out = out[out["rule_id"].astype(str).eq(str(row.get("rule_id", "")))]
    if CODE_COL in out.columns and CODE_COL in row.index:
        out = out[out[CODE_COL].astype(str).eq(str(row.get(CODE_COL, "")))]
    return out.copy()


def _selected_positions_to_equity_curve(positions_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    group = _filter_selected_rule_rows(positions_df, row)
    if group.empty:
        return group
    group = group.copy()
    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    strategy_return = pd.to_numeric(group.get("strategy_return"), errors="coerce").fillna(0.0)
    benchmark_return = pd.to_numeric(group.get("benchmark_return"), errors="coerce").fillna(0.0)
    group["strategy_equity"] = (1 + strategy_return).cumprod()
    group["benchmark_equity"] = (1 + benchmark_return).cumprod()
    group["excess_equity"] = group["strategy_equity"] / group["benchmark_equity"].replace(0, np.nan)
    return group


def _prepare_composite_daily(
    daily_df: pd.DataFrame,
    strategy_id: str,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    if daily_df.empty or "strategy_id" not in daily_df.columns:
        return pd.DataFrame()
    group = daily_df[daily_df["strategy_id"].astype(str).eq(strategy_id)].copy()
    if group.empty or DATE_COL not in group.columns:
        return pd.DataFrame()
    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    numeric_columns = [
        "composite_score",
        "weekly_anchor_score",
        "exposure",
        "benchmark_return",
        "net_return",
        "benchmark_equity",
        "net_equity",
    ]
    for column in numeric_columns:
        if column in group.columns:
            group[column] = pd.to_numeric(group[column], errors="coerce")
    if "benchmark_equity" not in group.columns and "benchmark_return" in group.columns:
        group["benchmark_equity"] = group["benchmark_return"].fillna(0.0).add(1.0).cumprod()
    if "net_equity" not in group.columns and "net_return" in group.columns:
        group["net_equity"] = group["net_return"].fillna(0.0).add(1.0).cumprod()
    if {"net_equity", "benchmark_equity"}.issubset(group.columns):
        group["excess_equity"] = group["net_equity"].div(
            group["benchmark_equity"].replace(0.0, np.nan)
        )

    if PRICE_COL not in group.columns and not input_df.empty and {DATE_COL, PRICE_COL}.issubset(input_df.columns):
        price = input_df.copy()
        price[DATE_COL] = pd.to_datetime(price[DATE_COL], errors="coerce")
        if CODE_COL in price.columns and CODE_COL in group.columns and not group[CODE_COL].dropna().empty:
            instrument_code = str(group[CODE_COL].dropna().iloc[0])
            price = price[price[CODE_COL].astype(str).eq(instrument_code)]
        price = price[[DATE_COL, PRICE_COL]].dropna(subset=[DATE_COL]).drop_duplicates(DATE_COL, keep="last")
        price[PRICE_COL] = pd.to_numeric(price[PRICE_COL], errors="coerce")
        group = group.merge(price, on=DATE_COL, how="left", validate="one_to_one")
    return group


def _prepare_composite_trades(trades_df: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    if trades_df.empty or "strategy_id" not in trades_df.columns:
        return pd.DataFrame()
    group = trades_df[trades_df["strategy_id"].astype(str).eq(strategy_id)].copy()
    for column in ["signal_date", "execution_date"]:
        if column in group.columns:
            group[column] = pd.to_datetime(group[column], errors="coerce")
    if "execution_date" in group.columns:
        group = group.dropna(subset=["execution_date"]).sort_values("execution_date").reset_index(drop=True)
    if "trigger_source" in group.columns:
        group["trigger_source_cn"] = group["trigger_source"].replace(
            {
                "weekly_anchor": "周频锚",
                "intraweek_strong_event": "周中强事件",
            }
        )
    else:
        group["trigger_source_cn"] = "策略信号"
    return group


def _composite_long_spans(daily: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if daily.empty or "exposure" not in daily.columns:
        return []
    long_state = pd.to_numeric(daily["exposure"], errors="coerce").fillna(0.0).gt(0.5).to_numpy()
    if not long_state.any():
        return []
    starts = np.flatnonzero(long_state & np.r_[True, ~long_state[:-1]])
    ends = np.flatnonzero(long_state & np.r_[~long_state[1:], True])
    dates = pd.to_datetime(daily[DATE_COL]).to_numpy()
    return [(pd.Timestamp(dates[start]), pd.Timestamp(dates[end])) for start, end in zip(starts, ends)]


def _make_composite_strategy_charts(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_name: str,
    strong_threshold: float | None,
) -> str:
    if daily.empty:
        return "<div class='chart-error'>暂无可用的复合策略日度数据。</div>"
    font = dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12)
    layout_common = dict(
        template="plotly_white",
        font=font,
        hoverlabel=dict(font=font),
        hovermode="x unified",
        margin=dict(l=54, r=42, t=34, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    price_column = PRICE_COL if PRICE_COL in daily.columns and daily[PRICE_COL].notna().any() else "benchmark_equity"
    price_label = "收盘价" if price_column == PRICE_COL else "基准净值"
    fig_signal = go.Figure()
    fig_signal.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily[price_column],
            name=price_label,
            mode="lines",
            line=dict(color="#4C78A8", width=1.5),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{price_label}=%{{y:.4f}}<extra></extra>",
        )
    )
    for start, end in _composite_long_spans(daily):
        fig_signal.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(239,68,68,0.10)",
            line_width=0,
            layer="below",
        )
    if not trades.empty and "execution_date" in trades.columns:
        marker_values = daily[[DATE_COL, price_column]].rename(
            columns={DATE_COL: "execution_date", price_column: "marker_value"}
        )
        marked = trades.merge(marker_values, on="execution_date", how="left", validate="many_to_one")
        for side, name, symbol, color in [
            ("entry", "开仓点", "triangle-up", "#169B62"),
            ("exit", "平仓点", "triangle-down", "#D62728"),
        ]:
            subset = marked[marked.get("trade_side", pd.Series(index=marked.index, dtype=str)).eq(side)].copy()
            if subset.empty:
                continue
            signal_text = subset.get("signal_date", pd.Series(pd.NaT, index=subset.index)).dt.strftime("%Y-%m-%d").fillna("--")
            trigger_text = subset.get("trigger_source_cn", pd.Series("策略信号", index=subset.index)).astype(str)
            customdata = np.column_stack([signal_text.to_numpy(), trigger_text.to_numpy()])
            fig_signal.add_trace(
                go.Scatter(
                    x=subset["execution_date"],
                    y=subset["marker_value"],
                    name=name,
                    mode="markers",
                    marker=dict(symbol=symbol, size=10, color=color, line=dict(color="white", width=0.7)),
                    customdata=customdata,
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>执行值=%{y:.4f}"
                        "<br>信号日=%{customdata[0]}<br>来源=%{customdata[1]}<extra></extra>"
                    ),
                )
            )
    fig_signal.update_layout(**layout_common, height=360)
    fig_signal.update_yaxes(title_text=price_label)

    fig_score = make_subplots(specs=[[{"secondary_y": True}]])
    fig_score.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily["composite_score"],
            name="日度/复合分数",
            mode="lines",
            line=dict(color="#7C3AED", width=1.7),
            hovertemplate="%{x|%Y-%m-%d}<br>复合分数=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_score.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily["weekly_anchor_score"],
            name="周频锚分数",
            mode="lines",
            line=dict(color="#F59E0B", width=1.4, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>周频锚=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_score.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily["exposure"],
            name="仓位",
            mode="lines",
            line=dict(color="#0F766E", width=1.4, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(15,118,110,0.08)",
            hovertemplate="%{x|%Y-%m-%d}<br>仓位=%{y:.0%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_score.add_hline(y=0.0, line_width=0.9, line_dash="dot", line_color="#667085", secondary_y=False)
    if strong_threshold is not None:
        for threshold in [strong_threshold, -strong_threshold]:
            fig_score.add_hline(
                y=threshold,
                line_width=0.9,
                line_dash="dot",
                line_color="#DC2626",
                secondary_y=False,
            )
    fig_score.update_layout(**layout_common, height=360)
    fig_score.update_yaxes(title_text="复合分数", secondary_y=False)
    fig_score.update_yaxes(title_text="仓位", range=[-0.05, 1.05], tickformat=".0%", secondary_y=True)

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily["net_equity"],
            name="策略净值",
            mode="lines",
            line=dict(color="#1D3557", width=1.9),
            hovertemplate="%{x|%Y-%m-%d}<br>策略净值=%{y:.4f}<extra></extra>",
        )
    )
    fig_equity.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily["benchmark_equity"],
            name="中证全指基准",
            mode="lines",
            line=dict(color="#94A3B8", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>基准净值=%{y:.4f}<extra></extra>",
        )
    )
    fig_equity.add_hline(y=1.0, line_width=0.9, line_dash="dot", line_color="#777777")
    fig_equity.update_layout(**layout_common, height=350)
    fig_equity.update_yaxes(title_text="累计净值")

    fig_excess = go.Figure()
    fig_excess.add_trace(
        go.Scatter(
            x=daily[DATE_COL],
            y=daily["excess_equity"],
            name="超额净值",
            mode="lines",
            line=dict(color="#C1121F", width=1.9),
            fill="tozeroy",
            fillcolor="rgba(193,18,31,0.07)",
            hovertemplate="%{x|%Y-%m-%d}<br>超额净值=%{y:.4f}<extra></extra>",
        )
    )
    fig_excess.add_hline(y=1.0, line_width=0.9, line_dash="dot", line_color="#777777")
    fig_excess.update_layout(**layout_common, height=320)
    fig_excess.update_yaxes(title_text="策略 / 基准")

    return (
        "<div class='composite-chart-stack'>"
        "<div class='plot-panel'><div class='plot-panel-title'>1. 价格、持仓区间与开平仓信号</div>"
        f"{_fig_html(fig_signal, height=360)}</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>2. 复合分数、周频锚与仓位</div>"
        f"{_fig_html(fig_score, height=360)}</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>3. 历史策略净值与基准净值</div>"
        f"{_fig_html(fig_equity, height=350)}</div>"
        "<div class='plot-panel'><div class='plot-panel-title'>4. 历史超额净值曲线</div>"
        f"{_fig_html(fig_excess, height=320)}</div>"
        "</div>"
    )


def _build_composite_strategy_views(
    input_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for strategy_id, meta in COMPOSITE_STRATEGY_META.items():
        daily = _prepare_composite_daily(daily_df, strategy_id, input_df)
        if daily.empty:
            continue
        trades = _prepare_composite_trades(trades_df, strategy_id)
        summary_rows = (
            summary_df[summary_df["strategy_id"].astype(str).eq(strategy_id)]
            if not summary_df.empty and "strategy_id" in summary_df.columns
            else pd.DataFrame()
        )
        status_rows = (
            status_df[status_df["strategy_id"].astype(str).eq(strategy_id)]
            if not status_df.empty and "strategy_id" in status_df.columns
            else pd.DataFrame()
        )
        summary = summary_rows.iloc[0].to_dict() if not summary_rows.empty else {}
        status = status_rows.iloc[0].to_dict() if not status_rows.empty else daily.iloc[-1].to_dict()
        views.append(
            {
                "strategy_id": strategy_id,
                "name": meta["name"],
                "weighting": meta["weighting"],
                "rebalance": meta["rebalance"],
                "execution": meta["execution"],
                "summary": summary,
                "status": status,
                "chart_html": _make_composite_strategy_charts(
                    daily,
                    trades,
                    str(meta["name"]),
                    meta.get("strong_threshold"),
                ),
            }
        )
    return views


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
    selected_rule_summary_df = _read_csv(
        input_dir,
        [
            "results/selected_single_factor_rules/selected_rule_summary.csv",
            "selected_single_factor_rules/selected_rule_summary.csv",
            "selected_rule_summary.csv",
        ],
        optional=True,
    )
    selected_rule_status_df = _read_csv(
        input_dir,
        [
            "results/selected_single_factor_rules/selected_rule_latest_status.csv",
            "selected_single_factor_rules/selected_rule_latest_status.csv",
            "selected_rule_latest_status.csv",
        ],
        optional=True,
    )
    selected_rule_trades_df = _read_csv(
        input_dir,
        [
            "results/selected_single_factor_rules/selected_rule_trades.csv",
            "selected_single_factor_rules/selected_rule_trades.csv",
            "selected_rule_trades.csv",
        ],
        optional=True,
    )
    selected_rule_positions_df = _read_csv(
        input_dir,
        [
            "results/selected_single_factor_rules/selected_rule_daily_positions.csv",
            "selected_single_factor_rules/selected_rule_daily_positions.csv",
            "selected_rule_daily_positions.csv",
        ],
        optional=True,
    )
    composite_daily_df = _read_csv(
        input_dir,
        [
            "results/composite_timing_strategies/composite_strategy_daily.csv",
            "composite_timing_strategies/composite_strategy_daily.csv",
        ],
        optional=True,
    )
    composite_summary_df = _read_csv(
        input_dir,
        [
            "results/composite_timing_strategies/composite_strategy_summary.csv",
            "composite_timing_strategies/composite_strategy_summary.csv",
        ],
        optional=True,
    )
    composite_trades_df = _read_csv(
        input_dir,
        [
            "results/composite_timing_strategies/composite_strategy_trades.csv",
            "composite_timing_strategies/composite_strategy_trades.csv",
        ],
        optional=True,
    )
    composite_status_df = _read_csv(
        input_dir,
        [
            "results/composite_timing_strategies/composite_strategy_latest_status.csv",
            "composite_timing_strategies/composite_strategy_latest_status.csv",
        ],
        optional=True,
    )
    signal_points_state_df = _read_csv(
        input_dir,
        [
            "results/report/signal_points_state.csv",
            "report/signal_points_state.csv",
            "signal_points_state.csv",
        ],
        optional=True,
    )

    if not strategy_df.empty and DATE_COL in strategy_df.columns:
        strategy_df[DATE_COL] = pd.to_datetime(strategy_df[DATE_COL], errors="coerce")
        strategy_df = strategy_df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if not strategy_summary_df.empty and "excess_annual_return" in strategy_summary_df.columns:
        strategy_summary_df = strategy_summary_df.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)
    if not selected_rule_summary_df.empty and "excess_annual_return" in selected_rule_summary_df.columns:
        selected_rule_summary_df = selected_rule_summary_df.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)
    rule_pair_signal_overview = _selected_rule_latest_signal_overview(selected_rule_status_df, selected_rule_summary_df)
    composite_strategy_views = _build_composite_strategy_views(
        input_df,
        composite_daily_df,
        composite_summary_df,
        composite_trades_df,
        composite_status_df,
    )

    strategy_plot_html = ""
    strategy_z20_html = ""
    strategy_status: dict[str, Any] = {}
    strategy_z20_stats: list[dict[str, Any]] = []
    effective_signal_meta: dict[str, Any] = {}
    recent_signal_chart_html = ""
    if not strategy_df.empty:
        summary_row = strategy_summary_df.iloc[0] if not strategy_summary_df.empty else None
        strategy_status = _strategy_latest_status(strategy_df)
        strategy_z20_stats = _score_z20_forward_stats(strategy_df)
        strategy_plot_html = _make_strategy_html(strategy_df, summary_row=summary_row)
        strategy_z20_html = _make_score_z20_html(strategy_df)
    if not input_df.empty and not signals_df.empty:
        effective_signals_df, effective_signal_meta = _filter_effective_signals_expanding(input_df, signals_df)
        recent_signal_chart_html = _make_recent_signal_chart_html(input_df, effective_signals_df, default_visible_days=756)
    else:
        effective_signals_df = signals_df.iloc[0:0].copy()

    rule_pair_cards: list[dict[str, Any]] = []
    if not selected_rule_summary_df.empty:
        status_by_rule: dict[str, pd.Series] = {}
        if not selected_rule_status_df.empty and "rule_id" in selected_rule_status_df.columns:
            for _, status_row in selected_rule_status_df.iterrows():
                status_by_rule[str(status_row.get("rule_id", ""))] = status_row
        for _, row in selected_rule_summary_df.iterrows():
            factor = str(row.get("factor", ""))
            base_factor = _base_factor_name(factor)
            desc = dict(factor_desc_map.get(base_factor, factor_desc_map.get(factor, {})) or {})
            if not desc.get("category"):
                desc["category"] = row.get("category", "")
            if not desc.get("meaning"):
                desc["meaning"] = row.get("strategy_label", "")
            if not desc.get("note"):
                desc["note"] = row.get("notes", "")
            status_row = status_by_rule.get(str(row.get("rule_id", "")), pd.Series(dtype=object))
            row_copy = row.copy()
            row_copy.attrs["_equity_df"] = _selected_positions_to_equity_curve(selected_rule_positions_df, row)
            row_copy.attrs["_trade_df"] = _filter_selected_rule_rows(selected_rule_trades_df, row)
            try:
                chart_html = (
                    "<div class='rule-selected-metrics'>"
                    f"{_selected_rule_metric_html(row, status_row)}"
                    "</div>"
                    + _make_rule_pair_html(input_df, signals_df, row_copy, desc)
                )
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
    conclusion_color = {"偏多": "#2ecc71", "降仓": "#e74c3c", "减仓": "#e74c3c", "中性": "#f39c12", "观望": "#95a5a6"}.get(conclusion, "#95a5a6")
    bullish_structure, bearish_structure = _signal_structure_tables(advisor)

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
        "signal_info": _signal_counts_from_df(effective_signals_df, 20),
        "effective_signal_meta": effective_signal_meta,
        "recent_signal_chart_html": recent_signal_chart_html,
        "strategy_status": strategy_status,
        "strategy_z20_stats": strategy_z20_stats,
        "strategy_plot_html": strategy_plot_html,
        "strategy_z20_html": strategy_z20_html,
        "rule_pair_cards": rule_pair_cards,
        "rule_pair_signal_overview": rule_pair_signal_overview,
        "composite_strategy_views": composite_strategy_views,
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


def _format_date_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "--"
        return str(pd.Timestamp(value).date())
    except Exception:
        return "--"


def _render_composite_logic(view: dict[str, Any]) -> str:
    return f"""
<div class="composite-logic-card">
  <div class="composite-logic-title">{_escape(view.get('name'))}</div>
  <div class="composite-logic-grid">
    <div><span>加权方式</span><p>{_escape(view.get('weighting'))}</p></div>
    <div><span>调仓频率与触发</span><p>{_escape(view.get('rebalance'))}</p></div>
    <div><span>执行规则</span><p>{_escape(view.get('execution'))}</p></div>
  </div>
</div>
"""


def _render_composite_status(view: dict[str, Any]) -> str:
    status = view.get("status", {}) or {}
    exposure = _safe_float(status.get("exposure"))
    exposure = exposure if np.isfinite(exposure) else 0.0
    state_label = "持仓" if exposure > 0.5 else "空仓"
    state_class = "pill-bullish" if exposure > 0.5 else "pill-bearish"
    return f"""
<div class="composite-current-panel">
  <div class="composite-section-title">当前信号状态</div>
  <div class="composite-current-grid">
    <div class="composite-current-card"><span>最新日期</span><b>{_format_date_value(status.get(DATE_COL))}</b></div>
    <div class="composite-current-card"><span>当前方向</span><b class="keyword-pill {state_class}">{state_label}</b></div>
    <div class="composite-current-card"><span>当前仓位</span><b>{exposure:.0%}</b></div>
    <div class="composite-current-card"><span>日度/复合分数</span><b>{_format_float(status.get('composite_score'), 4)}</b></div>
    <div class="composite-current-card"><span>周频锚分数</span><b>{_format_float(status.get('weekly_anchor_score'), 4)}</b></div>
    <div class="composite-current-card"><span>最近开仓</span><b>{_format_date_value(status.get('latest_entry_date'))}</b></div>
    <div class="composite-current-card"><span>最近平仓</span><b>{_format_date_value(status.get('latest_exit_date'))}</b></div>
  </div>
</div>
"""


def _render_composite_metrics(view: dict[str, Any]) -> str:
    summary = view.get("summary", {}) or {}
    entry_count = int(_safe_float(summary.get("entry_count"))) if np.isfinite(_safe_float(summary.get("entry_count"))) else 0
    exit_count = int(_safe_float(summary.get("exit_count"))) if np.isfinite(_safe_float(summary.get("exit_count"))) else 0
    rows = [
        ("年化收益", _format_pct_value(summary.get("annual_return"), 2)),
        ("年化波动", _format_pct_value(summary.get("annual_volatility"), 2)),
        ("Sharpe", _format_float(summary.get("sharpe"), 2)),
        ("最大回撤", _format_pct_value(summary.get("max_drawdown"), 2)),
        ("Calmar", _format_float(summary.get("calmar"), 2)),
        ("年化换手", _format_float(summary.get("annual_turnover"), 2)),
        ("平均仓位", _format_pct_value(summary.get("mean_exposure"), 1)),
        ("开仓 / 平仓", f"{entry_count} / {exit_count}"),
        ("平均持有天数", _format_float(summary.get("mean_holding_days"), 1)),
        ("期末净值", _format_float(summary.get("final_equity"), 3)),
    ]
    cards = "".join(
        f"<div class='composite-metric-card'><span>{_escape(label)}</span><b>{_escape(value)}</b></div>"
        for label, value in rows
    )
    return f"""
<div class="composite-performance-panel">
  <div class="composite-section-title">历史表现</div>
  <div class="composite-metric-grid">{cards}</div>
  <p class="footnote">回测口径：策略信号在下一交易日执行，收益已扣除单边 5bp 换仓成本；历史表现不代表未来收益。</p>
</div>
"""


def _render_composite_strategy_module(views: list[dict[str, Any]]) -> str:
    if not views:
        return """
<div class="card">
  <h2>复合策略</h2>
  <div class="chart-error">尚未生成复合策略结果，请先运行 composite-strategies。</div>
</div>
"""
    buttons: list[str] = []
    panels: list[str] = []
    for index, view in enumerate(views):
        active = " active" if index == 0 else ""
        panel_id = f"composite-strategy-panel-{index}"
        buttons.append(
            f"<button class='composite-tab-btn{active}' "
            f"onclick=\"switchCompositeStrategy(event,'{panel_id}')\">{_escape(view.get('name'))}</button>"
        )
        panels.append(
            f"<div id='{panel_id}' class='composite-strategy-panel{active}'>"
            f"{_render_composite_logic(view)}"
            f"{_render_composite_status(view)}"
            f"{view.get('chart_html', '')}"
            f"{_render_composite_metrics(view)}"
            "</div>"
        )
    return (
        "<div class='composite-inner-tabs'>"
        + "".join(buttons)
        + "</div>"
        + "".join(panels)
    )


def _render_strategy_status(status: dict[str, Any]) -> str:
    if not status:
        return ""
    open_event = "今日触发开仓" if int(status.get("open_event", 0)) else "今日未触发开仓"
    close_event = "今日触发平仓" if int(status.get("close_event", 0)) else "今日未触发平仓"
    open_cls = "pill-bullish" if int(status.get("open_event", 0)) else "pill-watch"
    close_cls = "pill-bearish" if int(status.get("close_event", 0)) else "pill-watch"
    return f"""
<div class="score-current-panel">
  <div class="score-current-title">120日Z值基准策略最新状态</div>
  <div class="score-mini-grid">
    <div class="score-mini-card"><span>策略方向</span><b class="keyword-pill {status.get('position_class', 'pill-watch')}">{_escape(status.get('position_label'))}</b></div>
    <div class="score-mini-card"><span>抄底得分Z值</span><b>{_format_float(status.get('entry_z'), 3)}</b></div>
    <div class="score-mini-card"><span>逃顶得分Z值</span><b>{_format_float(status.get('exit_z'), 3)}</b></div>
    <div class="score-mini-card"><span>开仓状态</span><b class="keyword-pill {open_cls}">{open_event}</b></div>
    <div class="score-mini-card"><span>平仓状态</span><b class="keyword-pill {close_cls}">{close_event}</b></div>
  </div>
  <div class="score-rule-lines">
    <div><b>开仓规则：</b>{_escape(status.get('open_rule'))}</div>
    <div><b>平仓规则：</b>{_escape(status.get('close_rule'))}</div>
  </div>
</div>
"""


def _render_event_score_stats(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    table_rows = []
    for row in rows:
        sample = int(row.get("sample_count", 0))
        sample_note = "样本偏少" if sample < 30 else "样本充足"
        table_rows.append(
            "<tr>"
            f"<td>{_escape(row.get('name'))}</td>"
            f"<td>{_escape(row.get('bucket'))}</td>"
            f"<td>{_format_float(row.get('current_value'), 3)}</td>"
            f"<td>{sample} <span class='sample-note'>{sample_note}</span></td>"
            f"<td>{_escape(row.get('win_label'))}：{_format_pct_value(row.get('win_rate'), 1)}</td>"
            f"<td>{_format_float(row.get('payoff'), 2)}</td>"
            f"<td>{_format_pct_value(row.get('avg_directional_return'), 2)}</td>"
            "</tr>"
        )
    return f"""
<div class="score-current-panel event-objective-panel">
  <div class="score-current-title">当前事件驱动评分区间的历史未来5日统计</div>
  <div style="overflow-x:auto;">
    <table class="score-stat-table">
      <thead><tr><th>指标</th><th>当前区间</th><th>当前值</th><th>历史样本</th><th>未来5日方向胜率</th><th>赔率</th><th>平均方向收益</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </div>
  <p class="footnote">统计口径：用当前事件驱动评分所在的固定0.25分区间，回看历史同区间样本的未来5个交易日表现。当前评分为正时统计未来5日上涨胜率，当前评分为负时统计未来5日下跌胜率；赔率 = 正确方向平均收益 / 错误方向平均损失。最近尚未兑现未来5日收益的样本不参与统计。</p>
</div>
"""


def _render_z20_stats(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    table_rows = []
    for row in rows:
        sample = int(row.get("sample_count", 0))
        sample_note = "样本偏少" if sample < 30 else "样本充足"
        table_rows.append(
            "<tr>"
            f"<td>{_escape(row.get('name'))}</td>"
            f"<td>{_escape(row.get('bucket'))}</td>"
            f"<td>{_format_float(row.get('current_value'), 3)}</td>"
            f"<td>{sample} <span class='sample-note'>{sample_note}</span></td>"
            f"<td>{_escape(row.get('win_label'))}：{_format_pct_value(row.get('win_rate'), 1)}</td>"
            f"<td>{_format_float(row.get('payoff'), 2)}</td>"
            f"<td>{_format_pct_value(row.get('avg_directional_return'), 2)}</td>"
            "</tr>"
        )
    return f"""
<div class="score-current-panel z20-objective-panel">
  <p class="red-note"><b>注意：</b>20日Z值 + 3日均线仅用于看图判断和短期观察，不作为正式基准策略的开平仓规则。</p>
  <div class="score-current-title">20Z当前区间的历史未来5日统计</div>
  <div style="overflow-x:auto;">
    <table class="score-stat-table">
      <thead><tr><th>指标</th><th>当前区间</th><th>当前值</th><th>历史样本</th><th>未来5日胜率</th><th>赔率</th><th>平均方向收益</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </div>
  <p class="footnote">统计口径：对抄底得分20Z和逃顶得分20Z使用固定0.25σ区间分箱，取历史同区间样本观察未来5个交易日。抄底得分按未来5日上涨为正确方向，逃顶得分按未来5日下跌为正确方向；赔率 = 正确方向平均收益 / 错误方向平均损失。最近尚未兑现未来5日收益的样本不参与统计。</p>
</div>
"""


def _render_rule_pair_signal_rows(rows: list[dict[str, Any]], empty_text: str) -> str:
    if not rows:
        return f"<tr><td colspan='6'>{_escape(empty_text)}</td></tr>"
    html_rows = []
    for row in rows:
        age = row.get("state_age_days")
        try:
            age_text = "--" if pd.isna(age) else f"{int(age)}天"
        except Exception:
            age_text = "--"
        rule_text = row.get("open_rule") if row.get("current_view") == "看多" else row.get("close_rule")
        html_rows.append(
            "<tr>"
            f"<td><b>{_escape(row.get('base_factor'))}</b><br><span class='muted-text'>{_escape(row.get('factor'))}</span></td>"
            f"<td><span class='keyword-pill {row.get('state_class', 'pill-watch')}'>{_escape(row.get('current_view'))}</span></td>"
            f"<td>{_escape(row.get('last_signal_date'))}<br><span class='muted-text'>{_escape(row.get('last_signal_type'))} / {age_text}</span></td>"
            f"<td>{_escape(rule_text)}</td>"
            f"<td>{_format_pct_value(row.get('excess_annual_return'), 1)}</td>"
            f"<td>{_format_float(row.get('sharpe'), 2)}</td>"
            "</tr>"
        )
    return "".join(html_rows)


def _render_rule_pair_signal_overview(overview: dict[str, Any]) -> str:
    if not overview or not int(overview.get("total", 0) or 0):
        return ""
    bullish = overview.get("bullish", []) or []
    bearish = overview.get("bearish", []) or []
    return f"""
<div class="rule-signal-overview">
  <div class="rule-signal-head">
    <div>
      <h3>当前保留单因子规则多空状态</h3>
      <p>这里不再使用全量 rule pair 遍历结果，而是按 selected_single_factor_rules 中人工研究后保留的最优规则计算：当前状态为“多”表示规则仍在持仓，“空”表示空仓；日期为最近一次开仓或平仓状态切换日。</p>
    </div>
    <div class="rule-signal-counts">
      <span class="keyword-pill pill-core">合计 {int(overview.get('total', 0))}</span>
      <span class="keyword-pill pill-bullish">看多 {int(overview.get('bullish_count', 0))}</span>
      <span class="keyword-pill pill-bearish">看空 {int(overview.get('bearish_count', 0))}</span>
    </div>
  </div>
  <div class="rule-signal-columns">
    <div class="rule-signal-table-card">
      <div class="rule-signal-title bullish-title">当前看多的保留规则</div>
      <div class="compact-table-wrap">
        <table class="compact-signal-table">
          <thead><tr><th>指标</th><th>状态</th><th>最近信号</th><th>触发规则</th><th>年化超额</th><th>夏普</th></tr></thead>
          <tbody>{_render_rule_pair_signal_rows(bullish, "暂无当前看多指标")}</tbody>
        </table>
      </div>
    </div>
    <div class="rule-signal-table-card">
      <div class="rule-signal-title bearish-title">当前看空/空仓的保留规则</div>
      <div class="compact-table-wrap">
        <table class="compact-signal-table">
          <thead><tr><th>指标</th><th>状态</th><th>最近信号</th><th>触发规则</th><th>年化超额</th><th>夏普</th></tr></thead>
          <tbody>{_render_rule_pair_signal_rows(bearish, "暂无当前看空指标")}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>
"""


def _render_evidence_rows(v: dict[str, Any]) -> str:
    rows = []
    for cat in v["category_evidence"]:
        cname = str(cat["factor_category"])
        net_score = _safe_float(cat.get("net_score"))
        net_score = net_score if np.isfinite(net_score) else 0.0
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
        net_score = _safe_float(cat.get("net_score"))
        net_score = net_score if np.isfinite(net_score) else 0.0
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


def _render_rule_search(v: dict[str, Any]) -> str:
    total = len(v["rule_pair_cards"])
    return f"""
<div class="rule-search-bar">
  <input id="rule-search-input" type="search" placeholder="搜索因子名、类别、开仓规则、平仓规则..." oninput="updateRulePairSearch()" />
  <span id="rule-search-summary">显示 {total} / {total}</span>
</div>
"""


def _render_rule_pair_cards(v: dict[str, Any]) -> str:
    groups: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    sub_counts: dict[str, dict[str, int]] = {}
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
        search_text = " ".join(
            [
                str(rp.get("factor", "")),
                category,
                category_root,
                str(desc.get("meaning", "")),
                str(desc.get("direction", "")),
                str(desc.get("observation", "")),
                str(desc.get("note", "")),
                open_rule,
                close_rule,
            ]
        ).lower()
        card = (
            f"<div class='rule-pair-card' data-category-root='{_escape(category_root)}' data-category='{_escape(category)}' data-search='{_escape(search_text)}'>"
            "<div class='rule-pair-header'>"
            f"<h3>{_escape(rp['factor'])}</h3>"
            f"{meta}{rule_lines}{''.join(extra)}"
            "</div>"
            f"{rp['chart_html']}"
            "</div>"
        )
        groups.setdefault(category_root, []).append(card)
        counts[category_root] = counts.get(category_root, 0) + 1
        sub_counts.setdefault(category_root, {})
        sub_counts[category_root][category] = sub_counts[category_root].get(category, 0) + 1

    preferred = ["胜率", "赔率", "辅助", "其他"]
    roots = [root for root in preferred if root in groups] + sorted(root for root in groups if root not in preferred)
    sections = []
    for root in roots:
        total = counts[root]
        sub_options = []
        for sub_category, sub_total in sorted(sub_counts.get(root, {}).items()):
            label = sub_category or root
            sub_options.append(
                "<label class='rule-sub-filter'>"
                f"<input type='checkbox' class='rule-sub-category-check' data-root='{_escape(root)}' value='{_escape(sub_category)}' checked onchange='updateRulePairSearch()'>"
                f"<span>{_escape(label)}</span><em>{sub_total}</em>"
                "</label>"
            )
        sub_filter_html = (
            "<div class='rule-sub-filter-row'>"
            "<label class='rule-sub-filter rule-sub-filter-all'>"
            f"<input type='checkbox' class='rule-sub-category-all' data-root='{_escape(root)}' checked onchange='toggleRuleSubCategory(this)'>"
            "<span>全部</span>"
            f"<em>{total}</em>"
            "</label>"
            f"{''.join(sub_options)}"
            "</div>"
        )
        sections.append(
            f"<details class='rule-category-section' data-category-root='{_escape(root)}' open>"
            f"<summary><span>{_escape(root)}</span><em class='rule-category-count' data-total='{total}'>{total} / {total}</em></summary>"
            f"{sub_filter_html}"
            f"<div class='rule-pair-grid'>{''.join(groups[root])}</div>"
            "</details>"
        )
    return "".join(sections)


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
        net_score = _safe_float(cat.get("net_score"))
        net_score = net_score if np.isfinite(net_score) else 0.0
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
    rule_search = _render_rule_search(v)
    rule_pair_cards = _render_rule_pair_cards(v)
    bullish_rows = _render_structure_rows(v.get("bullish_structure", []), "signal_style_bucket")
    bearish_rows = _render_structure_rows(v.get("bearish_structure", []), "bearish_reason_bucket")
    strategy_status_html = _render_strategy_status(v.get("strategy_status", {}))
    z20_stats_html = _render_z20_stats(v.get("strategy_z20_stats", []))
    rule_pair_signal_overview_html = _render_rule_pair_signal_overview(v.get("rule_pair_signal_overview", {}))
    composite_views = v.get("composite_strategy_views", []) or []
    composite_module_html = _render_composite_strategy_module(composite_views)
    composite_exposures = []
    for composite_view in composite_views[:2]:
        value = _safe_float((composite_view.get("status", {}) or {}).get("exposure"))
        composite_exposures.append(value if np.isfinite(value) else 0.0)
    while len(composite_exposures) < 2:
        composite_exposures.append(0.0)
    composite_direction = (
        "方向一致"
        if len(composite_views) >= 2 and (composite_exposures[0] > 0.5) == (composite_exposures[1] > 0.5)
        else "方向分歧" if len(composite_views) >= 2 else "等待结果"
    )
    status = v.get("strategy_status", {}) or {}
    score_position_label = str(status.get("position_label", "--") or "--")
    score_position_class = str(status.get("position_class", "pill-watch") or "pill-watch")
    try:
        score_open_event = int(status.get("open_event", 0) or 0)
    except Exception:
        score_open_event = 0
    try:
        score_close_event = int(status.get("close_event", 0) or 0)
    except Exception:
        score_close_event = 0
    rule_overview = v.get("rule_pair_signal_overview", {}) or {}
    rule_total = int(rule_overview.get("total", 0) or 0)
    rule_bullish_count = int(rule_overview.get("bullish_count", 0) or 0)
    rule_bearish_count = int(rule_overview.get("bearish_count", 0) or 0)
    rule_latest_bullish = (rule_overview.get("bullish") or [{}])[0].get("base_factor", "--")
    rule_latest_bearish = (rule_overview.get("bearish") or [{}])[0].get("base_factor", "--")
    event_daily = v.get("signal_info", {}).get("daily", []) or []
    event_recent_open = int(sum(int(row.get("open", 0) or 0) for row in event_daily))
    event_recent_close = int(sum(int(row.get("close", 0) or 0) for row in event_daily))
    event_recent_net = event_recent_open - event_recent_close

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

    recent_signal_card_html = f"""
    <div class="card">
      <h2>有效信号数量 <span class="badge">近20日 {int(v['signal_info'].get('total_recent', 0))} 条多头</span></h2>
      <div class="plot-container">
        <div class="plotly-wrap">{v['recent_signal_chart_html']}</div>
      </div>
      <p class="signal-explain">净多头信号 = 当日有效多头信号数 - 当日有效空头信号数，用来看当天<span class="red-note">新增</span>方向谁更强；近20日累计净多头 = 最近20个交易日有效多头合计 - 有效空头合计，用来看<span class="red-note">一段时间</span>内多头信号是否持续占优。</p>
      <p class="footnote">数量口径：以 signals 为来源，满 1 年历史后，按 instrument + factor + pattern + signal 使用截至当日前、且对应未来收益已经兑现的 expanding 历史样本。5日上涨/下跌胜率大于50%、盈亏比大于1且平均方向收益大于1%的信号，分别归为追涨/快速逃顶；5日未达到即时标准，但20日满足相同有效性标准，且20日平均方向收益比5日高1个百分点以上的信号，分别归为抄底/预先指示。5日即时分类优先，四类互斥；不满足条件的信号不计入有效数量。六个子图依次展示净多头信号5日均线、近20日累计净多头、抄底、追涨、快速逃顶和预先指示；四类角色各自使用独立子图和独立数量轴。所有子图共享时间定位虚线和悬浮信息，并叠加收盘价右轴。图中传入全历史交易日数据，打开时默认显示最近3年约756个交易日；未触发有效信号的交易日数量记为0。</p>
    </div>
"""

    score_desc = """
<div class="strategy-desc">
<p><b>抄底得分（Entry Score）</b>：基于多因子历史信号聚合得到的开仓倾向得分。值越高，表示做多信号越强。</p>
<p><b>逃顶得分（Exit Score）</b>：基于多因子历史信号聚合得到的平仓倾向得分。值越高，表示离场或风控信号越强。</p>
<p><b>净得分（Net Score）</b>：抄底得分减去逃顶得分，正值偏多，负值偏空。</p>
<p><b>得分生成过程</b>：先用事件驱动回测统计每个信号规则的历史未来收益，月度更新可用信号白名单；日度计算时，把当日仍在有效期内的信号按历史 edge、信号期限和触发后的 age 衰减加权聚合。偏开仓、抄底、追涨的有效信号进入 Entry Score，偏平仓、逃顶、风险释放的有效信号进入 Exit Score。</p>
<p><b>策略使用方式</b>：正式基准策略使用较长窗口的 120 日 zscore 及其平滑变化来判断开仓和平仓；20 日 zscore + 3 日均线只用于短期观察和看图辅助，不作为正式基准策略的开平仓规则。</p>
</div>
"""

    css = f"""
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;color:#333;line-height:1.6}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:40px 20px;text-align:center}}
.header h1{{font-size:28px;margin-bottom:8px}}
.header .date{{font-size:14px;opacity:.86}}
.container{{max-width:1440px;margin:0 auto;padding:20px}}
.overview-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:24px}}
.overview-card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);border-top:4px solid #d0d5dd;min-height:210px;display:flex;flex-direction:column;gap:12px}}
.overview-card.event-card{{border-top-color:#7B2CBF}}
.overview-card.score-card{{border-top-color:#2563eb}}
.overview-card.rule-card{{border-top-color:#7c3aed}}
.overview-card.composite-card{{border-top-color:#0f766e}}
.overview-title{{display:flex;align-items:center;justify-content:space-between;gap:10px}}
.overview-title h2{{font-size:17px;color:#101828;margin:0}}
.overview-badge{{display:inline-block;border-radius:999px;padding:5px 13px;color:#fff;font-size:14px;font-weight:800;white-space:nowrap}}
.overview-badge.event-badge{{background:#7B2CBF}}
.overview-badge.score-badge{{background:#2563eb}}
.overview-badge.rule-badge{{background:#7c3aed}}
.overview-badge.composite-badge{{background:#0f766e}}
.overview-metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}
.overview-metric{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:9px 8px;text-align:center;min-height:62px;display:flex;flex-direction:column;justify-content:center}}
.overview-metric b{{font-size:18px;color:#1f2937}}
.overview-metric span{{font-size:11px;color:#667085;margin-top:3px}}
.overview-note{{font-size:12.5px;color:#475467;line-height:1.65;margin-top:auto}}
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
.rule-selected-metrics{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 6px 0}}
.rule-metric-chip{{display:inline-flex;align-items:center;gap:5px;background:#fff;border:1px solid #e5e7eb;border-radius:999px;padding:4px 9px;font-size:12px;color:#475467}}
.rule-metric-chip b{{color:#1f2937}}
.rule-signal-overview{{background:#fbfcff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin:14px 0 18px 0}}
.rule-signal-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:12px;flex-wrap:wrap}}
.rule-signal-head h3{{font-size:15px;color:#1f2937;margin-bottom:4px}}
.rule-signal-head p{{font-size:12.5px;color:#667085;line-height:1.6;max-width:860px}}
.rule-signal-counts{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}}
.rule-signal-columns{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.rule-signal-table-card{{background:#fff;border:1px solid #edf0f5;border-radius:8px;overflow:hidden}}
.rule-signal-title{{font-size:13px;font-weight:800;padding:9px 12px;border-bottom:1px solid #edf0f5}}
.bullish-title{{color:#169b62;background:#f0fbf5}}
.bearish-title{{color:#d62728;background:#fff5f5}}
.compact-table-wrap{{max-height:420px;overflow:auto}}
.compact-signal-table{{font-size:12px;min-width:720px}}
.compact-signal-table th,.compact-signal-table td{{padding:7px 8px;vertical-align:top;text-align:left}}
.muted-text{{font-size:11px;color:#667085}}
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
.rule-search-bar{{display:flex;align-items:center;gap:12px;margin:12px 0 16px 0;flex-wrap:wrap}}
.rule-search-bar input{{flex:1;min-width:260px;border:1px solid #d9dde7;background:#fff;border-radius:8px;padding:10px 12px;font-size:13px;color:#344054;outline:none}}
.rule-search-bar input:focus{{border-color:#94a3b8;box-shadow:0 0 0 3px rgba(148,163,184,.18)}}
.rule-search-bar span{{font-size:12.5px;color:#667085;white-space:nowrap}}
.rule-category-section{{border:1px solid #e5e7eb;border-radius:10px;background:#fff;margin-bottom:16px;overflow:hidden}}
.rule-category-section.hidden{{display:none}}
.rule-category-section summary{{cursor:pointer;list-style:none;background:#f8fafc;padding:12px 14px;display:flex;align-items:center;justify-content:space-between;font-size:14px;font-weight:800;color:#1f2937;border-bottom:1px solid #e5e7eb}}
.rule-category-section summary::-webkit-details-marker{{display:none}}
.rule-category-section summary em{{font-style:normal;font-size:12px;color:#667085;background:#eef2f7;border-radius:999px;padding:2px 9px}}
.rule-category-section .rule-pair-grid{{padding:16px}}
.rule-sub-filter-row{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid #eef2f7;background:#fff}}
.rule-sub-filter{{display:inline-flex;align-items:center;gap:6px;border:1px solid #d9dde7;border-radius:999px;padding:5px 9px;font-size:12px;color:#344054;background:#f8fafc;cursor:pointer;user-select:none}}
.rule-sub-filter input{{width:13px;height:13px}}
.rule-sub-filter em{{font-style:normal;color:#667085;font-size:11px}}
.rule-sub-filter-all{{font-weight:800;background:#eef2f7}}
.factor-meta{{font-size:12px;margin-bottom:6px;display:flex;align-items:flex-start;gap:8px;flex-wrap:wrap}}
.factor-cat{{display:inline-block;color:#fff;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}}
.factor-meaning{{color:#333;font-size:12.5px;line-height:1.5}}
.factor-extra{{font-size:11.5px;color:#666;margin-top:3px;padding-left:4px;line-height:1.5}}
.rule-pair-rules{{font-size:12.5px;color:#344054;background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:7px 10px;margin:8px 0;line-height:1.6}}
.strategy-desc{{background:#f0f4ff;border:1px solid #d0d8f0;border-radius:8px;padding:14px 16px;margin-bottom:16px;font-size:13px;line-height:1.7}}
.strategy-desc p{{margin-bottom:4px;color:#2c3e50}}
.red-note{{color:#d62728!important;font-weight:700}}
.score-current-panel{{background:#fbfcff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin:14px 0;font-size:13px;line-height:1.7}}
.score-current-title{{font-size:14px;font-weight:800;color:#1f2937;margin-bottom:10px}}
.score-mini-grid{{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:10px;margin-bottom:10px}}
.score-mini-card{{background:#fff;border:1px solid #edf0f5;border-radius:8px;padding:10px;text-align:center;min-height:68px;display:flex;flex-direction:column;justify-content:center;gap:4px}}
.score-mini-card span{{font-size:12px;color:#667085}}
.score-mini-card b{{font-size:15px;color:#1f2937}}
.score-rule-lines{{background:#fff;border:1px dashed #d0d5dd;border-radius:8px;padding:9px 12px;color:#475467}}
.score-stat-table th,.score-stat-table td{{white-space:nowrap}}
.sample-note{{font-size:11px;color:#667085;margin-left:4px}}
.signal-explain{{font-size:12.5px;color:#344054;line-height:1.7;margin:10px 0 0 0}}
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
.calc-note{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;margin:12px 0 14px 0;font-size:13px;line-height:1.7;color:#344054;text-align:left}}
.calc-note p{{margin-bottom:6px}}
.calc-note p:last-child{{margin-bottom:0}}
.calc-note b{{color:#1f2937}}
.analysis-summary{{background:#fbfcff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;margin:0 0 14px 0;font-size:13px;line-height:1.7;color:#344054}}
.analysis-summary p{{margin-bottom:8px}}
.analysis-summary ul{{margin-left:18px;margin-bottom:0}}
.analysis-summary li{{margin-bottom:5px}}
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
.composite-inner-tabs{{display:flex;gap:10px;flex-wrap:wrap;background:#eef6f5;border:1px solid #cde5e1;border-radius:10px;padding:10px;margin:0 0 16px 0}}
.composite-tab-btn{{flex:1;min-width:260px;border:1px solid #b8d8d2;background:#fff;color:#0f5f57;border-radius:8px;padding:11px 16px;font-size:14px;font-weight:800;cursor:pointer}}
.composite-tab-btn:hover{{background:#f0fdfa}}
.composite-tab-btn.active{{background:#0f766e;border-color:#0f766e;color:#fff;box-shadow:0 3px 10px rgba(15,118,110,.20)}}
.composite-strategy-panel{{display:none}}
.composite-strategy-panel.active{{display:block}}
.composite-logic-card{{background:linear-gradient(135deg,#f0fdfa,#f8fafc);border:1px solid #bfe3dc;border-left:5px solid #0f766e;border-radius:10px;padding:16px 18px;margin:0 0 16px 0}}
.composite-logic-title{{font-size:17px;font-weight:900;color:#134e4a;margin-bottom:12px}}
.composite-logic-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.composite-logic-grid>div{{background:#fff;border:1px solid #dce9e7;border-radius:8px;padding:11px 12px}}
.composite-logic-grid span{{display:block;font-size:12px;font-weight:800;color:#0f766e;margin-bottom:5px}}
.composite-logic-grid p{{font-size:13px;color:#344054;line-height:1.65;margin:0}}
.composite-current-panel,.composite-performance-panel{{background:#fbfcff;border:1px solid #e5e7eb;border-radius:10px;padding:15px 16px;margin:0 0 16px 0}}
.composite-performance-panel{{margin-top:16px}}
.composite-section-title{{font-size:15px;font-weight:900;color:#1f2937;margin-bottom:10px}}
.composite-current-grid{{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:9px}}
.composite-current-card,.composite-metric-card{{background:#fff;border:1px solid #e8edf2;border-radius:8px;padding:10px;text-align:center;min-height:68px;display:flex;flex-direction:column;justify-content:center;gap:4px}}
.composite-current-card span,.composite-metric-card span{{font-size:11.5px;color:#667085}}
.composite-current-card b,.composite-metric-card b{{font-size:15px;color:#1f2937}}
.composite-metric-grid{{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px}}
.composite-chart-stack{{display:flex;flex-direction:column;gap:14px}}
.footnote{{font-size:12px;color:#667085;line-height:1.6;margin:10px 0 0 0}}
.chart-error{{padding:16px;border:1px solid #f3c7c7;background:#fff4f4;color:#b42318;border-radius:8px;font-size:13px}}
@media(max-width:1200px){{.overview-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.composite-current-grid{{grid-template-columns:repeat(4,minmax(120px,1fr))}}.composite-metric-grid{{grid-template-columns:repeat(3,minmax(120px,1fr))}}}}
@media(max-width:900px){{.overview-grid{{grid-template-columns:1fr}}.rule-pair-grid{{grid-template-columns:1fr}}.signal-stats{{flex-direction:column;align-items:center}}.composite-logic-grid{{grid-template-columns:1fr}}.composite-current-grid,.composite-metric-grid{{grid-template-columns:repeat(2,minmax(120px,1fr))}}}}
@media(max-width:1100px){{.rule-signal-columns{{grid-template-columns:1fr}}.score-mini-grid{{grid-template-columns:repeat(2,minmax(140px,1fr))}}}}
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
  <div class="overview-grid">
    <div class="overview-card event-card">
      <div class="overview-title">
        <h2>事件驱动</h2>
        <span class="overview-badge event-badge">有效信号</span>
      </div>
      <div class="overview-metrics">
        <div class="overview-metric"><b>{event_recent_open}</b><span>近20日多头</span></div>
        <div class="overview-metric"><b>{event_recent_close}</b><span>近20日空头</span></div>
        <div class="overview-metric"><b>{event_recent_net}</b><span>净多头</span></div>
      </div>
      <div class="overview-note">
        <span class="keyword-pill pill-core">5日即时 / 20日延迟</span>
        <span class="keyword-pill pill-bullish">胜率 &gt; 50%</span>
        <span class="keyword-pill pill-bearish">盈亏比 &gt; 1，方向收益 &gt; 1%</span>
      </div>
    </div>
    <div class="overview-card score-card">
      <div class="overview-title">
        <h2>综合打分</h2>
        <span class="overview-badge score-badge">{_escape(score_position_label)}</span>
      </div>
      <div class="overview-metrics">
        <div class="overview-metric"><b>{_format_float(status.get('entry_z'), 2)}</b><span>抄底Z</span></div>
        <div class="overview-metric"><b>{_format_float(status.get('exit_z'), 2)}</b><span>逃顶Z</span></div>
        <div class="overview-metric"><b>{_format_float(status.get('position'), 2)}</b><span>仓位</span></div>
      </div>
      <div class="overview-note">
        <span class="keyword-pill {score_position_class}">120Z策略：{_escape(score_position_label)}</span>
        <span class="keyword-pill {'pill-bullish' if score_open_event else 'pill-watch'}">{'触发开仓' if score_open_event else '未触发开仓'}</span>
        <span class="keyword-pill {'pill-bearish' if score_close_event else 'pill-watch'}">{'触发平仓' if score_close_event else '未触发平仓'}</span>
      </div>
    </div>
    <div class="overview-card rule-card">
      <div class="overview-title">
        <h2>单因子</h2>
        <span class="overview-badge rule-badge">{rule_total} 个最优规则</span>
      </div>
      <div class="overview-metrics">
        <div class="overview-metric"><b>{rule_bullish_count}</b><span>当前看多</span></div>
        <div class="overview-metric"><b>{rule_bearish_count}</b><span>当前看空</span></div>
        <div class="overview-metric"><b>{rule_total}</b><span>覆盖因子</span></div>
      </div>
      <div class="overview-note">
        <span class="keyword-pill pill-bullish">最近看多：{_escape(rule_latest_bullish)}</span>
        <span class="keyword-pill pill-bearish">最近看空：{_escape(rule_latest_bearish)}</span>
      </div>
    </div>
    <div class="overview-card composite-card">
      <div class="overview-title">
        <h2>复合策略</h2>
        <span class="overview-badge composite-badge">{len(composite_views)} 个策略</span>
      </div>
      <div class="overview-metrics">
        <div class="overview-metric"><b>{composite_exposures[0]:.0%}</b><span>类别等权两速</span></div>
        <div class="overview-metric"><b>{composite_exposures[1]:.0%}</b><span>频率倒数加权</span></div>
        <div class="overview-metric"><b>{_escape(composite_direction)}</b><span>当前方向</span></div>
      </div>
      <div class="overview-note">
        <span class="keyword-pill pill-core">类别等权两速复合</span>
        <span class="keyword-pill pill-risk">开仓频率平方根倒数复合</span>
      </div>
    </div>
  </div>

  <div class="module-tabs">
    <button class="module-tab-btn active" onclick="switchModule(event,'event-module-panel')">事件驱动模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'score-module-panel')">综合打分模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'rule-module-panel')">单因子规则模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'composite-module-panel')">复合策略模块</button>
  </div>

  <div id="event-module-panel" class="module-panel active event-module">
    <h2 class="module-heading">事件驱动模块</h2>
    <p class="module-intro">事件驱动模块只展示经过历史有效性筛选后的信号数量。筛选使用满1年后的 expanding 历史样本，并根据5日与20日收益期限结构，把多头信号拆为追涨和抄底，把空头信号拆为快速逃顶和预先指示；所有统计只使用当时已经兑现的未来收益。</p>
    {recent_signal_card_html}

  </div>

  <div id="score-module-panel" class="module-panel">
    <h2 class="module-heading">综合打分模块</h2>
    <p class="module-intro">综合打分模块把历史事件回测后的有效信号汇总成抄底得分和逃顶得分，并根据得分变化生成最终择时策略。这里更关注多个信号合成以后是否形成可执行的仓位规则，因此重点观察抄底、逃顶得分的相对强弱、开平仓点以及策略相对基准的净值表现。</p>
  <div class="card">
    <h2>策略净值曲线</h2>
    {score_desc}
    {strategy_status_html}
    {z20_stats_html}
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
    <p class="module-intro">单因子规则模块保留原来的网页形式，但内容切换为当前正式保留的 selected rules。这里展示的是我们逐个因子研究后沉淀下来的开仓、闭仓规则，不再使用全量 rule pair 遍历结果。</p>
    {rule_pair_signal_overview_html}

  <div class="card">
    <h2>当前保留单因子最优规则回测 <span class="badge">{len(v['rule_pair_cards'])} 个因子</span></h2>
    <p style="font-size:13px;color:#666;margin-bottom:16px;">图中阴影为做多持仓区间：红色表示该笔交易盈利，绿色表示该笔交易亏损；绿色三角为次日入场后的开仓点，红色三角为平仓点；下方净值图展示规则净值、基准净值和超额净值。</p>
    {rule_search}
    <div class="rule-pair-grid">{rule_pair_cards}</div>
  </div>
  </div>

  <div id="composite-module-panel" class="module-panel">
    <h2 class="module-heading">复合策略模块</h2>
    <p class="module-intro">复合策略模块把正式保留的单因子持仓规则组合成两套可执行策略。两个策略地位并列，先说明加权、调仓与次日执行逻辑，再分别展示当前信号、开平仓记录、历史净值、超额曲线和绩效。</p>
    <div class="card">
      <h2>复合策略历史信号与表现</h2>
      {composite_module_html}
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
function switchCompositeStrategy(e,t) {{
  document.querySelectorAll('#composite-module-panel .composite-strategy-panel').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.querySelectorAll('#composite-module-panel .composite-tab-btn').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.getElementById(t).classList.add('active');
  e.currentTarget.classList.add('active');
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
function toggleRuleSubCategory(box) {{
  var root = box.dataset.root || '';
  document.querySelectorAll('.rule-sub-category-check[data-root="' + root + '"]').forEach(function(el) {{
    el.checked = box.checked;
  }});
  updateRulePairSearch();
}}
function updateRulePairSearch() {{
  var input = document.getElementById('rule-search-input');
  var query = input ? input.value.trim().toLowerCase() : '';
  var total = 0;
  var visible = 0;
  document.querySelectorAll('.rule-category-section').forEach(function(section) {{
    var sectionVisible = 0;
    var sectionTotal = 0;
    var root = section.dataset.categoryRoot || '';
    var checks = Array.from(section.querySelectorAll('.rule-sub-category-check'));
    var selectedCategories = checks.filter(function(el) {{ return el.checked; }}).map(function(el) {{ return el.value || ''; }});
    var selectedSet = new Set(selectedCategories);
    var allBox = section.querySelector('.rule-sub-category-all');
    if (allBox) {{
      allBox.checked = selectedCategories.length === checks.length;
      allBox.indeterminate = selectedCategories.length > 0 && selectedCategories.length < checks.length;
    }}
    section.querySelectorAll('.rule-pair-card').forEach(function(card) {{
      sectionTotal += 1;
      total += 1;
      var text = (card.dataset.search || card.textContent || '').toLowerCase();
      var category = card.dataset.category || '';
      var categoryMatch = selectedSet.has(category);
      var searchMatch = !query || text.indexOf(query) >= 0;
      var show = categoryMatch && searchMatch;
      card.classList.toggle('hidden', !show);
      if (show) {{
        sectionVisible += 1;
        visible += 1;
      }}
    }});
    var keepFilterVisible = checks.length > 0 && selectedCategories.length === 0;
    section.classList.toggle('hidden', sectionVisible === 0 && !keepFilterVisible);
    var count = section.querySelector('.rule-category-count');
    if (count) {{
      count.textContent = sectionVisible + ' / ' + sectionTotal;
    }}
  }});
  var summary = document.getElementById('rule-search-summary');
  if (summary) {{
    summary.textContent = '显示 ' + visible + ' / ' + total;
  }}
  setTimeout(function() {{
    document.querySelectorAll('#rule-module-panel .rule-pair-card:not(.hidden) .plotly-graph-div').forEach(function(el) {{
      if (window.Plotly) {{ Plotly.Plots.resize(el); }}
    }});
  }}, 80);
}}
document.addEventListener('DOMContentLoaded', updateRulePairSearch);
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
