from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any

import numpy as np
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
from baseline_score_strategy import format_rule_name_cn, rolling_zscore
from reporting import _state_direction_score, score_signal_points_for_advisor
from timing_config import (
    CODE_COL,
    CORE_CATEGORIES,
    DATE_COL,
    PRICE_COL,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_VALUE_COL,
    STATE_FLAT,
    STATE_LONG,
)


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


def _rule_pair_latest_signal_overview(rule_summary: pd.DataFrame, equity_curves: pd.DataFrame) -> dict[str, Any]:
    if rule_summary.empty or equity_curves.empty:
        return {"total": 0, "bullish": [], "bearish": [], "bullish_count": 0, "bearish_count": 0}

    eq = equity_curves.copy()
    eq[DATE_COL] = pd.to_datetime(eq[DATE_COL], errors="coerce")
    eq = eq.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    if eq.empty:
        return {"total": 0, "bullish": [], "bearish": [], "bullish_count": 0, "bearish_count": 0}

    code_col = _pick_code_col(rule_summary)
    eq_code_col = _pick_code_col(eq)
    records: list[dict[str, Any]] = []
    for _, row in rule_summary.iterrows():
        factor = str(row.get("factor", ""))
        open_condition = str(row.get("open_condition", ""))
        close_condition = str(row.get("close_condition", ""))
        base_factor = str(row.get("base_factor", factor))
        code = str(row.get(code_col, ""))
        mask = (
            eq[eq_code_col].astype(str).eq(code)
            & eq["factor"].astype(str).eq(factor)
            & eq["open_condition"].astype(str).eq(open_condition)
            & eq["close_condition"].astype(str).eq(close_condition)
        )
        if "base_factor" in eq.columns:
            mask &= eq["base_factor"].astype(str).eq(base_factor)
        group = eq.loc[mask].copy()
        if group.empty:
            continue
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        pos = pd.to_numeric(group["position"], errors="coerce").fillna(0.0)
        latest = group.iloc[-1]
        latest_position = float(pos.iloc[-1])
        prev = pos.shift(1).fillna(0.0)
        open_events = group[(prev <= 0) & (pos > 0)]
        close_events = group[(prev > 0) & (pos <= 0)]
        last_open = open_events[DATE_COL].max() if not open_events.empty else pd.NaT
        last_close = close_events[DATE_COL].max() if not close_events.empty else pd.NaT
        if latest_position > 0:
            current_view = "看多"
            state_class = "pill-bullish"
            last_signal_type = "开仓"
            last_signal_date = last_open
        else:
            current_view = "看空"
            state_class = "pill-bearish"
            last_signal_type = "平仓"
            last_signal_date = last_close
        if pd.isna(last_signal_date):
            state_age_days = pd.NA
            last_signal_date_text = "--"
        else:
            state_age_days = int((pd.Timestamp(latest[DATE_COL]) - pd.Timestamp(last_signal_date)).days)
            last_signal_date_text = str(pd.Timestamp(last_signal_date).date())
        records.append(
            {
                "factor": factor,
                "base_factor": base_factor,
                "current_view": current_view,
                "state_class": state_class,
                "position": latest_position,
                "latest_date": str(pd.Timestamp(latest[DATE_COL]).date()),
                "last_signal_type": last_signal_type if last_signal_date_text != "--" else "无",
                "last_signal_date": last_signal_date_text,
                "state_age_days": state_age_days,
                "open_rule": format_rule_name_cn(open_condition),
                "close_rule": format_rule_name_cn(close_condition),
                "excess_annual_return": _safe_float(row.get("excess_annual_return")),
                "sharpe": _safe_float(row.get("sharpe")),
            }
        )

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        date_text = str(item.get("last_signal_date", ""))
        has_date = 0 if date_text == "--" else 1
        return (has_date, date_text)

    records = sorted(records, key=sort_key, reverse=True)
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
    rule_summary_df = _read_csv(
        input_dir,
        [
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
    if not rule_best_summary_df.empty and "excess_annual_return" in rule_best_summary_df.columns:
        rule_best_summary_df = rule_best_summary_df.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)
    if not rule_best_equity_df.empty and DATE_COL in rule_best_equity_df.columns:
        rule_best_equity_df[DATE_COL] = pd.to_datetime(rule_best_equity_df[DATE_COL], errors="coerce")
        rule_best_equity_df = rule_best_equity_df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    rule_pair_signal_overview = _rule_pair_latest_signal_overview(rule_best_summary_df, rule_best_equity_df)

    strategy_plot_html = ""
    strategy_z20_html = ""
    strategy_status: dict[str, Any] = {}
    strategy_z20_stats: list[dict[str, Any]] = []
    event_score_stats: list[dict[str, Any]] = []
    recent_signal_chart_html = ""
    if not strategy_df.empty:
        summary_row = strategy_summary_df.iloc[0] if not strategy_summary_df.empty else None
        strategy_status = _strategy_latest_status(strategy_df)
        strategy_z20_stats = _score_z20_forward_stats(strategy_df)
        strategy_plot_html = _make_strategy_html(strategy_df, summary_row=summary_row)
        strategy_z20_html = _make_score_z20_html(strategy_df)
    if not input_df.empty and not signals_df.empty:
        recent_signal_chart_html = _make_recent_signal_chart_html(input_df, signals_df, default_visible_days=756)
    if not input_df.empty and not signals_df.empty and not signal_points_state_df.empty:
        event_score_stats = _event_score_forward_stats(
            input_df=input_df,
            signals_df=signals_df,
            status_df=signal_points_state_df,
            rule_summary_df=rule_summary_df,
            current_scores=advisor.get("scores", {}),
        )

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
    conclusion_color = {"偏多": "#2ecc71", "降仓": "#e74c3c", "减仓": "#e74c3c", "中性": "#f39c12", "观望": "#95a5a6"}.get(conclusion, "#95a5a6")
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
        "strategy_status": strategy_status,
        "strategy_z20_stats": strategy_z20_stats,
        "event_score_stats": event_score_stats,
        "strategy_plot_html": strategy_plot_html,
        "strategy_z20_html": strategy_z20_html,
        "rule_pair_cards": rule_pair_cards,
        "rule_pair_signal_overview": rule_pair_signal_overview,
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
      <h3>32个最优单因子当前多空状态</h3>
      <p>按每个 base 因子历史最优规则组合的最新 position 判断：position &gt; 0 为当前看多，position = 0 为当前看空/空仓；日期为最近一次开仓或平仓状态切换日。</p>
    </div>
    <div class="rule-signal-counts">
      <span class="keyword-pill pill-core">合计 {int(overview.get('total', 0))}</span>
      <span class="keyword-pill pill-bullish">看多 {int(overview.get('bullish_count', 0))}</span>
      <span class="keyword-pill pill-bearish">看空 {int(overview.get('bearish_count', 0))}</span>
    </div>
  </div>
  <div class="rule-signal-columns">
    <div class="rule-signal-table-card">
      <div class="rule-signal-title bullish-title">当前看多的最优指标</div>
      <div class="compact-table-wrap">
        <table class="compact-signal-table">
          <thead><tr><th>指标</th><th>状态</th><th>最近信号</th><th>触发规则</th><th>年化超额</th><th>夏普</th></tr></thead>
          <tbody>{_render_rule_pair_signal_rows(bullish, "暂无当前看多指标")}</tbody>
        </table>
      </div>
    </div>
    <div class="rule-signal-table-card">
      <div class="rule-signal-title bearish-title">当前看空/空仓的最优指标</div>
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
    strategy_status_html = _render_strategy_status(v.get("strategy_status", {}))
    z20_stats_html = _render_z20_stats(v.get("strategy_z20_stats", []))
    event_score_stats_html = _render_event_score_stats(v.get("event_score_stats", []))
    rule_pair_signal_overview_html = _render_rule_pair_signal_overview(v.get("rule_pair_signal_overview", {}))
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

    event_conclusion_html = f"""
    <div class="card">
      <h2>事件驱动结论解释</h2>
      <div class="interpretation">
        <p><b>事件驱动结论</b>：系统判定为 <span class="keyword-pill pill-core">{_escape(v['conclusion'])}</span>，core_score = <span class="keyword-pill pill-core">{core_score:.3f}</span>。</p>
        <div class="calc-note">
          <p><b>计算口径</b>：先把所有因子的开仓、平仓规则转成事件信号，并根据最新仍在有效期内的事件状态判断当前是多、空还是观望。</p>
          <p><b>单点打分</b>：每个信号点再按因子默认方向和历史 rule_pair 表现，折算成看多、看空、风险缓和或待确认；随后乘以因子类别权重、周期权重和历史规则表现修正，得到单点得分。</p>
          <p><b>总评分</b> = 全部单点得分加总 / 全部可解释点位权重绝对值加总，当前为 <span class="keyword-pill pill-core">{total_score:.3f}</span>；<b>核心评分</b> 只统计赔率/估值、赔率/筹码、胜率/量、胜率/资金四类核心因子，当前为 <span class="keyword-pill pill-core">{core_score:.3f}</span>。</p>
          <p><b>可解释占比</b> = 可被方向规则折算的点位数量 / 全部信号点数量，当前为 <span class="keyword-pill pill-core">{interpretable_ratio:.1f}%</span>。结论阈值固定：可解释占比低于35%或总评分接近0时观望；总评分不低于0.20且核心评分不低于0.10时偏多；总评分不高于-0.20或核心评分不高于-0.20时降仓。</p>
        </div>
        {event_score_stats_html}
      </div>
    </div>
"""

    recent_signal_card_html = f"""
    <div class="card">
      <h2>最近 20 日信号触发 <span class="badge">{int(v['signal_info'].get('total_recent', 0))} 条开仓</span></h2>
      <div class="plot-container">
        <div class="plotly-wrap">{v['recent_signal_chart_html']}</div>
      </div>
      <p class="footnote">数量口径：以 signals.csv 为来源，按交易日聚合事件条数；signal = 1 计入开仓数量，signal = -1 计入平仓数量，净开仓量 = 开仓数量 - 平仓数量。同一交易日多个因子或规则同时触发会分别计数。图中传入全历史交易日数据，打开时默认显示最近 3 年约 756 个交易日；未触发信号的交易日数量记为 0。图中展示净开仓量、开仓数量和平仓数量的 5 日均线，hover 中保留当日原始数量。</p>
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
.overview-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:24px}}
.overview-card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.08);border-top:4px solid #d0d5dd;min-height:210px;display:flex;flex-direction:column;gap:12px}}
.overview-card.event-card{{border-top-color:{v['conclusion_color']}}}
.overview-card.score-card{{border-top-color:#2563eb}}
.overview-card.rule-card{{border-top-color:#7c3aed}}
.overview-title{{display:flex;align-items:center;justify-content:space-between;gap:10px}}
.overview-title h2{{font-size:17px;color:#101828;margin:0}}
.overview-badge{{display:inline-block;border-radius:999px;padding:5px 13px;color:#fff;font-size:14px;font-weight:800;white-space:nowrap}}
.overview-badge.event-badge{{background:{v['conclusion_color']}}}
.overview-badge.score-badge{{background:#2563eb}}
.overview-badge.rule-badge{{background:#7c3aed}}
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
@media(max-width:900px){{.overview-grid{{grid-template-columns:1fr}}.rule-pair-grid{{grid-template-columns:1fr}}.signal-stats{{flex-direction:column;align-items:center}}}}
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
        <span class="overview-badge event-badge">{_escape(v['conclusion'])}</span>
      </div>
      <div class="overview-metrics">
        <div class="overview-metric"><b>{core_score:.2f}</b><span>核心评分</span></div>
        <div class="overview-metric"><b>{total_score:.2f}</b><span>总评分</span></div>
        <div class="overview-metric"><b>{interpretable_ratio:.0f}%</b><span>可解释占比</span></div>
      </div>
      <div class="overview-note">
        基于 <span class="keyword-pill pill-core">{v['total_sig']} 个信号点</span>
        <span class="keyword-pill pill-bullish">多 {bullish_count}</span>
        <span class="keyword-pill pill-bearish">空 {bearish_count}</span>
        <span class="keyword-pill pill-watch">观望 {watch_count}</span>
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
  </div>

  <div class="module-tabs">
    <button class="module-tab-btn active" onclick="switchModule(event,'event-module-panel')">事件驱动模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'score-module-panel')">综合打分模块</button>
    <button class="module-tab-btn" onclick="switchModule(event,'rule-module-panel')">单因子规则模块</button>
  </div>

  <div id="event-module-panel" class="module-panel active event-module">
    <h2 class="module-heading">事件驱动模块</h2>
    <p class="module-intro">事件驱动模块从每个因子的开仓、平仓事件出发，统计当前市场中看多、看空和观望信号的分布。它不直接给出最终仓位，而是回答最近哪些类型的因子正在触发交易事件、这些事件偏向抄底还是风险释放、不同证据之间是否一致。适合用来解释当下择时观点的来源和结构。</p>
    {event_conclusion_html}
    {recent_signal_card_html}

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
    <p class="module-intro">单因子规则模块逐个展示每个 base 因子历史上表现最好的开仓和平仓规则组合。它用于回答某个因子单独使用时，什么事件规则最有效、交易点是否合理、超额收益是否稳定。这些结果主要作为规则解释和因子筛选参考，不等同于最终综合策略。</p>
    {rule_pair_signal_overview_html}

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
