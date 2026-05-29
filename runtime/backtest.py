from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data_cleaning import get_factor_columns
from io_utils import write_table
from signal_generation import (
    _build_factor_event_cache_from_signal_table,
    build_event_conditions,
    generate_signal_table,
)
from timing_config import (
    CODE_COL,
    DATE_COL,
    DEFAULT_HORIZONS,
    NAME_COL,
    PRICE_COL,
    SIGNAL_DATE_COL,
    TRADING_DAYS,
    EventCondition,
    _split_factor_frequency,
)


def _summarize_returns(returns: Iterable[float]) -> dict[str, float]:
    values = pd.Series(list(returns), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "count": 0,
            "mean_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "p05_return": np.nan,
            "p25_return": np.nan,
            "p75_return": np.nan,
            "p95_return": np.nan,
            "min_return": np.nan,
            "max_return": np.nan,
        }

    return {
        "count": int(values.size),
        "mean_return": float(values.mean()),
        "median_return": float(values.median()),
        "win_rate": float((values > 0).mean()),
        "p05_return": float(values.quantile(0.05)),
        "p25_return": float(values.quantile(0.25)),
        "p75_return": float(values.quantile(0.75)),
        "p95_return": float(values.quantile(0.95)),
        "min_return": float(values.min()),
        "max_return": float(values.max()),
    }


def _event_forward_rows_from_cache(
    factor: str,
    condition: EventCondition,
    factor_cache: list[dict[str, Any]],
    horizons: Iterable[int],
) -> list[dict[str, Any]]:
    rows = []
    if condition.requires_position:
        return rows

    for item in factor_cache:
        code = item[CODE_COL]
        prices = item["prices"]
        event_idx = np.flatnonzero(item["events"][condition.name])

        for horizon in horizons:
            entry_idx = event_idx + 1
            exit_idx = entry_idx + int(horizon)
            valid_range = exit_idx < len(prices)
            safe_entry = entry_idx[valid_range]
            safe_exit = exit_idx[valid_range]
            valid_price = np.isfinite(prices[safe_entry]) & np.isfinite(prices[safe_exit])
            valid_entry = safe_entry[valid_price]
            valid_exit = safe_exit[valid_price]
            returns = prices[valid_exit] / prices[valid_entry] - 1 if len(valid_entry) else []
            row = {
                CODE_COL: code,
                "factor": factor,
                "event_side": condition.side,
                "condition": condition.name,
                "horizon": int(horizon),
            }
            row.update(_summarize_returns(returns))
            rows.append(row)
    return rows


def _open_event_frequency_stats(
    factor_cache: list[dict[str, Any]],
    open_condition: EventCondition,
) -> dict[str, float]:
    """统计一个开仓规则在完整样本里的季度触发频率。"""
    total_events = 0
    total_quarters = 0
    min_quarter_events = np.nan

    for item in factor_cache:
        dates = pd.to_datetime(item["dates"])
        if len(dates) == 0:
            continue

        quarters = pd.PeriodIndex(dates, freq="Q")
        event_counts = pd.Series(
            item["events"][open_condition.name].astype(int),
            index=quarters,
            dtype=int,
        ).groupby(level=0).sum()
        if event_counts.empty:
            continue

        total_events += int(event_counts.sum())
        total_quarters += int(event_counts.size)
        current_min = float(event_counts.min())
        min_quarter_events = (
            current_min
            if not np.isfinite(min_quarter_events)
            else min(float(min_quarter_events), current_min)
        )

    events_per_quarter = total_events / total_quarters if total_quarters else np.nan
    return {
        "open_event_count": float(total_events),
        "open_quarter_count": float(total_quarters),
        "open_events_per_quarter": float(events_per_quarter),
        "min_quarter_open_events": float(min_quarter_events) if np.isfinite(min_quarter_events) else np.nan,
    }


def _eligible_open_conditions_for_factor(
    factor: str,
    open_conditions: list[EventCondition],
    factor_cache: list[dict[str, Any]],
    min_raw_open_events_per_quarter: float | None,
) -> tuple[list[EventCondition], dict[str, dict[str, float]]]:
    """
    原始指标规则较多且噪声较大，默认只保留平均每季度触发足够多的开仓规则。

    季线、年线不套这个过滤，因为它们本身就是慢周期信号。
    """
    _, frequency = _split_factor_frequency(factor)
    stats_by_condition = {
        condition.name: _open_event_frequency_stats(factor_cache, condition)
        for condition in open_conditions
    }
    if frequency != "原始" or min_raw_open_events_per_quarter is None:
        return open_conditions, stats_by_condition

    eligible = [
        condition
        for condition in open_conditions
        if stats_by_condition[condition.name]["open_events_per_quarter"] >= min_raw_open_events_per_quarter
    ]
    return eligible, stats_by_condition


def _condition_applicable_to_factor(factor: str, condition: EventCondition) -> bool:
    if condition.kind == "cross_factor_resonance":
        return _split_factor_frequency(str(factor))[0] == str(condition.params.get("anchor_factor", ""))
    return True


def _should_dynamic_close(
    condition: EventCondition,
    current_return: float,
    holding_days: int,
) -> bool:
    if condition.kind == "stop_loss":
        return current_return <= condition.params["return"]
    if condition.kind == "take_profit":
        return current_return >= condition.params["return"]
    if condition.kind == "time_exit":
        return holding_days >= condition.params["days"]
    return False


def _calc_annualized_trade_return(trade_return: float, holding_days: int | float) -> float:
    """把单笔整段收益按实际持仓天数折算成年化收益。"""
    if holding_days <= 0 or not np.isfinite(trade_return):
        return np.nan
    if trade_return <= -1:
        return -1.0
    return float((1 + trade_return) ** (TRADING_DAYS / holding_days) - 1)


def _calc_trade_max_drawdown(prices: np.ndarray, entry_idx: int, exit_idx: int) -> float:
    """计算单笔交易从入场到出场期间的最大回撤。"""
    path = np.asarray(prices[entry_idx : exit_idx + 1], dtype=float)
    path = path[np.isfinite(path)]
    if path.size == 0:
        return np.nan
    running_max = np.maximum.accumulate(path)
    drawdown = path / running_max - 1
    return float(drawdown.min())


def _trade_records_from_group(
    code: str,
    factor: str,
    group: pd.DataFrame,
    prices: np.ndarray,
    dates: np.ndarray,
    open_condition: EventCondition,
    close_condition: EventCondition,
    open_events: np.ndarray,
    close_events: np.ndarray,
) -> list[dict[str, Any]]:
    trades = []
    i = 0
    while i < len(group) - 2:
        if not open_events[i]:
            i += 1
            continue

        entry_signal_idx = i
        entry_idx = entry_signal_idx + 1
        if entry_idx >= len(group) - 1 or not np.isfinite(prices[entry_idx]):
            break

        exit_signal_idx = None
        exit_idx = None
        close_reason = close_condition.name
        j = entry_idx
        while j < len(group):
            if not np.isfinite(prices[j]):
                j += 1
                continue

            current_return = prices[j] / prices[entry_idx] - 1
            holding_days = j - entry_idx
            if close_condition.requires_position:
                should_close = _should_dynamic_close(close_condition, current_return, holding_days)
            else:
                should_close = bool(close_events[j])

            if should_close:
                exit_signal_idx = j
                exit_idx = min(j + 1, len(group) - 1)
                break
            j += 1

        if exit_idx is None:
            exit_signal_idx = len(group) - 1
            exit_idx = len(group) - 1
            close_reason = "end"

        if exit_idx > entry_idx and np.isfinite(prices[exit_idx]):
            trade_return = prices[exit_idx] / prices[entry_idx] - 1
            holding_days = exit_idx - entry_idx
            max_drawdown = _calc_trade_max_drawdown(prices, entry_idx, exit_idx)
            trades.append(
                {
                    CODE_COL: code,
                    "factor": factor,
                    "open_condition": open_condition.name,
                    "close_condition": close_condition.name,
                    "close_reason": close_reason,
                    "entry_signal_date": dates[entry_signal_idx],
                    "entry_date": dates[entry_idx],
                    "exit_signal_date": dates[exit_signal_idx],
                    "exit_date": dates[exit_idx],
                    "entry_signal_idx": entry_signal_idx,
                    "exit_signal_idx": exit_signal_idx,
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_price": prices[entry_idx],
                    "exit_price": prices[exit_idx],
                    "trade_return": trade_return,
                    "annualized_trade_return": _calc_annualized_trade_return(trade_return, holding_days),
                    "max_drawdown": max_drawdown,
                    "holding_days": holding_days,
                    "forced_exit": close_reason == "end",
                }
            )

        i = max(exit_idx, entry_signal_idx + 1)
    return trades


def _trade_records_from_cache(
    factor: str,
    open_condition: EventCondition,
    close_condition: EventCondition,
    factor_cache: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for item in factor_cache:
        close_events = (
            np.zeros(len(item["group"]), dtype=bool)
            if close_condition.requires_position
            else item["events"][close_condition.name]
        )
        records.extend(
            _trade_records_from_group(
                code=item[CODE_COL],
                factor=factor,
                group=item["group"],
                prices=item["prices"],
                dates=item["dates"],
                open_condition=open_condition,
                close_condition=close_condition,
                open_events=item["events"][open_condition.name],
                close_events=close_events,
            )
        )
    return records


def _completed_trade_records(records: list[dict[str, Any]], latest_idx: int) -> list[dict[str, Any]]:
    completed = []
    for record in records:
        if record["forced_exit"]:
            continue
        if int(record["exit_idx"]) == latest_idx and int(record["exit_signal_idx"]) == latest_idx:
            continue
        completed.append(record)
    return completed


def _summarize_trade_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "trade_count": 0,
            "mean_trade_return": np.nan,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "min_trade_return": np.nan,
            "max_trade_return": np.nan,
            "mean_annualized_trade_return": np.nan,
            "median_annualized_trade_return": np.nan,
            "max_drawdown": np.nan,
            "mean_holding_days": np.nan,
            "forced_exit_count": 0,
        }

    returns = pd.Series([record["trade_return"] for record in records], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    annualized_returns = pd.Series(
        [record["annualized_trade_return"] for record in records],
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    max_drawdowns = pd.Series([record["max_drawdown"] for record in records], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    holding_days = pd.Series([record["holding_days"] for record in records], dtype=float)
    return {
        "trade_count": int(len(returns)),
        "mean_trade_return": float(returns.mean()) if not returns.empty else np.nan,
        "median_trade_return": float(returns.median()) if not returns.empty else np.nan,
        "win_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
        "min_trade_return": float(returns.min()) if not returns.empty else np.nan,
        "max_trade_return": float(returns.max()) if not returns.empty else np.nan,
        "mean_annualized_trade_return": float(annualized_returns.mean()) if not annualized_returns.empty else np.nan,
        "median_annualized_trade_return": float(annualized_returns.median()) if not annualized_returns.empty else np.nan,
        "max_drawdown": float(max_drawdowns.min()) if not max_drawdowns.empty else np.nan,
        "mean_holding_days": float(holding_days.mean()),
        "forced_exit_count": int(sum(record["forced_exit"] for record in records)),
    }


def _current_position_state(
    prices: np.ndarray,
    dates: np.ndarray,
    open_events: np.ndarray,
    close_condition: EventCondition,
    close_events: np.ndarray,
) -> dict[str, Any]:
    latest_idx = len(prices) - 1
    state = "空"
    pending_signal = ""
    pending_signal_date = pd.NaT
    entry_signal_idx = None
    entry_idx = None
    last_open_signal_idx = None
    last_close_signal_idx = None

    for i in range(len(prices)):
        if state == "空":
            if open_events[i]:
                last_open_signal_idx = i
                if i + 1 <= latest_idx:
                    state = "多"
                    entry_signal_idx = i
                    entry_idx = i + 1
                    pending_signal = ""
                    pending_signal_date = pd.NaT
                else:
                    pending_signal = "待开仓"
                    pending_signal_date = dates[i]
            continue

        if entry_idx is None or not np.isfinite(prices[i]) or not np.isfinite(prices[entry_idx]):
            continue

        current_return = prices[i] / prices[entry_idx] - 1
        holding_days = i - entry_idx
        if close_condition.requires_position:
            should_close = _should_dynamic_close(close_condition, current_return, holding_days)
        else:
            should_close = bool(close_events[i])

        if should_close:
            last_close_signal_idx = i
            if i + 1 <= latest_idx:
                state = "空"
                entry_signal_idx = None
                entry_idx = None
                pending_signal = ""
                pending_signal_date = pd.NaT
            else:
                pending_signal = "待闭仓"
                pending_signal_date = dates[i]

    current_holding_days = np.nan
    entry_signal_date = pd.NaT
    entry_date = pd.NaT
    if state == "多" and entry_idx is not None:
        current_holding_days = latest_idx - entry_idx
        entry_signal_date = dates[entry_signal_idx]
        entry_date = dates[entry_idx]

    return {
        "current_state": state,
        "pending_signal": pending_signal,
        "pending_signal_date": pending_signal_date,
        "entry_signal_date": entry_signal_date,
        "entry_date": entry_date,
        "current_holding_days": current_holding_days,
        "last_open_signal_date": dates[last_open_signal_idx] if last_open_signal_idx is not None else pd.NaT,
        "last_close_signal_date": dates[last_close_signal_idx] if last_close_signal_idx is not None else pd.NaT,
    }


def _rule_status_from_cache(
    factor: str,
    open_condition: EventCondition,
    close_condition: EventCondition,
    factor_cache: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in factor_cache:
        group = item["group"]
        prices = item["prices"]
        dates = item["dates"]
        latest_idx = len(group) - 1
        close_events = (
            np.zeros(len(group), dtype=bool)
            if close_condition.requires_position
            else item["events"][close_condition.name]
        )
        open_events = item["events"][open_condition.name]
        records = _trade_records_from_group(
            code=item[CODE_COL],
            factor=factor,
            group=group,
            prices=prices,
            dates=dates,
            open_condition=open_condition,
            close_condition=close_condition,
            open_events=open_events,
            close_events=close_events,
        )
        completed = _completed_trade_records(records, latest_idx)
        summary = _summarize_trade_records(completed)
        state = _current_position_state(prices, dates, open_events, close_condition, close_events)

        mean_holding = summary["mean_holding_days"]
        current_holding = state["current_holding_days"]
        if state["current_state"] == "多" and np.isfinite(mean_holding) and np.isfinite(current_holding):
            expected_remaining = max(float(mean_holding) - float(current_holding), 0.0)
        else:
            expected_remaining = np.nan

        rows.append(
            {
                CODE_COL: item[CODE_COL],
                NAME_COL: group[NAME_COL].iloc[-1] if NAME_COL in group else "",
                "latest_date": dates[-1],
                "factor": factor,
                "open_condition": open_condition.name,
                "close_condition": close_condition.name,
                "current_state": state["current_state"],
                "pending_signal": state["pending_signal"],
                "pending_signal_date": state["pending_signal_date"],
                "entry_signal_date": state["entry_signal_date"],
                "entry_date": state["entry_date"],
                "current_holding_days": current_holding,
                "historical_mean_holding_days": mean_holding,
                "historical_median_holding_days": _median_holding_days(completed),
                "expected_remaining_days": expected_remaining,
                "last_open_signal_date": state["last_open_signal_date"],
                "last_close_signal_date": state["last_close_signal_date"],
                **summary,
            }
        )
    return rows


def _median_holding_days(records: list[dict[str, Any]]) -> float:
    if not records:
        return np.nan
    values = pd.Series([record["holding_days"] for record in records], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else np.nan


def run_event_study(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    factors: Iterable[str] | None = None,
    signal_table: pd.DataFrame | None = None,
    min_raw_open_events_per_quarter: float | None = 2.0,
    include_trade_summary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_factors = list(factors) if factors is not None else get_factor_columns(df)
    conditions = build_event_conditions()
    if signal_table is None:
        signal_table = generate_signal_table(df, factors=selected_factors, conditions=conditions)

    event_rows = []
    trade_rows = []
    for factor in selected_factors:
        factor_cache = _build_factor_event_cache_from_signal_table(df, factor, conditions, signal_table)
        open_conditions, open_frequency_stats = _eligible_open_conditions_for_factor(
            factor,
            [condition for condition in conditions["open"] if _condition_applicable_to_factor(factor, condition)],
            factor_cache,
            min_raw_open_events_per_quarter,
        )
        static_close_conditions = [
            condition
            for condition in conditions["close"]
            if not condition.requires_position and _condition_applicable_to_factor(factor, condition)
        ]
        for condition in open_conditions + static_close_conditions:
            rows = _event_forward_rows_from_cache(factor, condition, factor_cache, horizons)
            if condition.side == "open":
                for row in rows:
                    row.update(open_frequency_stats[condition.name])
            event_rows.extend(rows)

        if include_trade_summary:
            for open_condition in open_conditions:
                for close_condition in [
                    condition for condition in conditions["close"] if _condition_applicable_to_factor(factor, condition)
                ]:
                    records = _trade_records_from_cache(factor, open_condition, close_condition, factor_cache)
                    row = {
                        "factor": factor,
                        "open_condition": open_condition.name,
                        "close_condition": close_condition.name,
                    }
                    row.update(open_frequency_stats[open_condition.name])
                    row.update(_summarize_trade_records(records))
                    trade_rows.append(row)

    event_summary = pd.DataFrame(event_rows)
    trade_summary = pd.DataFrame(trade_rows)
    write_table(event_summary, output_path / "event_forward_returns.csv")
    if include_trade_summary:
        write_table(trade_summary, output_path / "open_close_trades.csv")
    return event_summary, trade_summary


def combine_open_close_rules(open_rule: EventCondition, close_rule: EventCondition) -> tuple[EventCondition, EventCondition]:
    if open_rule.side != "open":
        raise ValueError(f"open_rule must have side='open': {open_rule}")
    if close_rule.side != "close":
        raise ValueError(f"close_rule must have side='close': {close_rule}")
    return open_rule, close_rule


def _max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min()) if not drawdown.empty else np.nan


def _annual_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) <= 1 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return np.nan
    years = (len(equity) - 1) / TRADING_DAYS
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan


def _sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    std = returns.std(ddof=1)
    if returns.empty or not np.isfinite(std) or std == 0:
        return np.nan
    return float(returns.mean() / std * math.sqrt(TRADING_DAYS))


def _backtest_rule_pair_from_cache(
    factor: str,
    open_rule: EventCondition,
    close_rule: EventCondition,
    factor_cache: list[dict[str, Any]],
    include_equity: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    equity_rows = []
    summaries = []

    for item in factor_cache:
        code = item[CODE_COL]
        group = item["group"]
        prices = pd.Series(item["prices"], dtype=float)
        close_events = (
            np.zeros(len(group), dtype=bool)
            if close_rule.requires_position
            else item["events"][close_rule.name]
        )
        records = _trade_records_from_group(
            code=code,
            factor=factor,
            group=group,
            prices=item["prices"],
            dates=item["dates"],
            open_condition=open_rule,
            close_condition=close_rule,
            open_events=item["events"][open_rule.name],
            close_events=close_events,
        )

        benchmark_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        position = np.zeros(len(group), dtype=float)
        for record in records:
            start = min(int(record["entry_idx"]) + 1, len(position))
            stop = min(int(record["exit_idx"]) + 1, len(position))
            if start < stop:
                position[start:stop] = 1.0

        strategy_returns = benchmark_returns * position
        strategy_equity = (1 + strategy_returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()
        excess_equity = strategy_equity / benchmark_equity.replace(0, np.nan)
        turnover = float(np.abs(np.diff(np.r_[0.0, position])).sum())

        summaries.append(
            {
                CODE_COL: code,
                "factor": factor,
                "open_condition": open_rule.name,
                "close_condition": close_rule.name,
                "annual_return": _annual_return(strategy_equity),
                "benchmark_annual_return": _annual_return(benchmark_equity),
                "excess_annual_return": _annual_return(excess_equity),
                "max_drawdown": _max_drawdown(strategy_equity),
                "benchmark_max_drawdown": _max_drawdown(benchmark_equity),
                "excess_max_drawdown": _max_drawdown(excess_equity),
                "sharpe": _sharpe(strategy_returns),
                "turnover": turnover,
                "holding_ratio": float(position.mean()),
                "trade_count": int(len(records)),
                "final_equity": float(strategy_equity.iloc[-1]),
                "benchmark_final_equity": float(benchmark_equity.iloc[-1]),
                "excess_final_equity": float(excess_equity.iloc[-1]),
            }
        )

        if include_equity:
            equity_rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: code,
                        DATE_COL: group[DATE_COL],
                        "factor": factor,
                        "open_condition": open_rule.name,
                        "close_condition": close_rule.name,
                        "position": position,
                        "strategy_return": strategy_returns,
                        "benchmark_return": benchmark_returns,
                        "strategy_equity": strategy_equity,
                        "benchmark_equity": benchmark_equity,
                        "excess_equity": excess_equity,
                    }
                )
            )

    summary = pd.DataFrame(summaries)
    if summary.empty:
        summary_row: dict[str, Any] = {
            "factor": factor,
            "open_condition": open_rule.name,
            "close_condition": close_rule.name,
        }
    else:
        summary_row = summary.iloc[0].to_dict() if len(summary) == 1 else _aggregate_backtest_summary(summary)

    equity = pd.concat(equity_rows, ignore_index=True) if equity_rows else None
    return summary_row, equity


def _aggregate_backtest_summary(summary: pd.DataFrame) -> dict[str, Any]:
    row = summary.iloc[0][["factor", "open_condition", "close_condition"]].to_dict()
    numeric_cols = [c for c in summary.columns if c not in {CODE_COL, "factor", "open_condition", "close_condition"}]
    for col in numeric_cols:
        row[col] = float(summary[col].mean())
    row[CODE_COL] = "ALL"
    return row




def run_rule_pair_backtest(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    factors: Iterable[str] | None = None,
    signal_table: pd.DataFrame | None = None,
    max_equity_curves: int | None = 200,
    min_raw_open_events_per_quarter: float | None = 2.0,
    min_excess_annual_return: float | None = 0.05,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_factors = list(factors) if factors is not None else get_factor_columns(df)
    conditions = build_event_conditions()
    if signal_table is None:
        signal_table = generate_signal_table(df, factors=selected_factors, conditions=conditions)

    summary_rows = []

    for factor in selected_factors:
        factor_cache = _build_factor_event_cache_from_signal_table(df, factor, conditions, signal_table)
        open_conditions, open_frequency_stats = _eligible_open_conditions_for_factor(
            factor,
            [condition for condition in conditions["open"] if _condition_applicable_to_factor(factor, condition)],
            factor_cache,
            min_raw_open_events_per_quarter,
        )
        for open_rule in open_conditions:
            for close_rule in [
                condition for condition in conditions["close"] if _condition_applicable_to_factor(factor, condition)
            ]:
                open_rule, close_rule = combine_open_close_rules(open_rule, close_rule)
                summary, _ = _backtest_rule_pair_from_cache(
                    factor,
                    open_rule,
                    close_rule,
                    factor_cache,
                    include_equity=False,
                )
                summary.update(open_frequency_stats[open_rule.name])
                summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).replace([np.inf, -np.inf], np.nan)
    if (
        min_excess_annual_return is not None
        and not summary_df.empty
        and "excess_annual_return" in summary_df.columns
    ):
        summary_df = summary_df[
            pd.to_numeric(summary_df["excess_annual_return"], errors="coerce") > min_excess_annual_return
        ].copy()
    sort_col = "excess_annual_return" if "excess_annual_return" in summary_df else "annual_return"
    if sort_col in summary_df:
        summary_df = summary_df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    if save_outputs:
        write_table(summary_df, output_path / "rule_pair_summary.csv")

    equity_df = pd.DataFrame()
    if max_equity_curves is None:
        equity_specs = summary_df
    else:
        equity_specs = summary_df.head(max_equity_curves)

    equity_rows = []
    by_name = {rule.name: rule for rule in conditions["open"] + conditions["close"]}
    equity_cache_by_factor: dict[str, list[dict[str, Any]]] = {}
    for spec in equity_specs.itertuples(index=False):
        open_rule = by_name[spec.open_condition]
        close_rule = by_name[spec.close_condition]
        if spec.factor not in equity_cache_by_factor:
            equity_cache_by_factor[spec.factor] = _build_factor_event_cache_from_signal_table(
                df,
                spec.factor,
                conditions,
                signal_table,
            )
        _, equity = _backtest_rule_pair_from_cache(
            spec.factor,
            open_rule,
            close_rule,
            equity_cache_by_factor[spec.factor],
            include_equity=True,
        )
        if equity is not None:
            equity_rows.append(equity)

    if equity_rows:
        equity_df = pd.concat(equity_rows, ignore_index=True)
    if save_outputs:
        write_table(equity_df, output_path / "equity_curves.csv")
    return summary_df, equity_df


def run_rule_pair_backtest_by_year_end(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    factors: Iterable[str] | None = None,
    signal_table: pd.DataFrame | None = None,
    min_history_years: int = 2,
    min_raw_open_events_per_quarter: float | None = 2.0,
    min_excess_annual_return: float | None = 0.05,
    target_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Generate rule-pair summaries using only history available up to each prior year-end.

    Example:
    - target_year = 2026 -> train_end_date = 2025-12-31
    - target_year = 2025 -> train_end_date = 2024-12-31

    A snapshot is produced only when the in-sample history up to train_end_date
    is at least `min_history_years`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    if data.empty:
        empty = pd.DataFrame()
        write_table(empty, output_path / "rule_pair_summary_by_year_end.csv")
        return empty

    if signal_table is None:
        signal_table = generate_signal_table(data, factors=list(factors) if factors is not None else None)
    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL], errors="coerce")
    selected_factors = list(factors) if factors is not None else get_factor_columns(data)
    conditions = build_event_conditions()

    def _truncate_factor_cache(
        factor_cache: list[dict[str, Any]],
        cutoff_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        cutoff_np = np.datetime64(cutoff_date)
        truncated: list[dict[str, Any]] = []
        for item in factor_cache:
            dates = pd.to_datetime(item["dates"]).to_numpy(dtype="datetime64[ns]")
            keep_len = int(np.searchsorted(dates, cutoff_np, side="right"))
            if keep_len <= 0:
                continue
            truncated.append(
                {
                    CODE_COL: item[CODE_COL],
                    "group": item["group"].iloc[:keep_len].copy(),
                    "prices": np.asarray(item["prices"][:keep_len], dtype=float),
                    "dates": dates[:keep_len],
                    "events": {
                        name: np.asarray(values[:keep_len], dtype=bool)
                        for name, values in item["events"].items()
                    },
                }
            )
        return truncated

    factor_cache_by_factor = {
        factor: _build_factor_event_cache_from_signal_table(data, factor, conditions, signals)
        for factor in selected_factors
    }

    global_start = pd.Timestamp(data[DATE_COL].min())
    available_years = sorted(pd.to_datetime(data[DATE_COL]).dt.year.dropna().astype(int).unique().tolist())
    selected_years = available_years if target_years is None else sorted(set(int(y) for y in target_years))
    rows: list[dict[str, Any]] = []

    for target_year in selected_years:
        if target_year not in available_years:
            continue
        train_end_date = pd.Timestamp(year=target_year - 1, month=12, day=31)
        min_required_date = global_start + pd.DateOffset(years=min_history_years)
        if train_end_date < min_required_date:
            continue

        for factor in selected_factors:
            factor_cache = _truncate_factor_cache(factor_cache_by_factor[factor], train_end_date)
            if not factor_cache:
                continue

            train_start_date = min(pd.Timestamp(item["group"][DATE_COL].min()) for item in factor_cache)
            train_max_date = max(pd.Timestamp(item["group"][DATE_COL].max()) for item in factor_cache)
            train_years = (train_max_date - train_start_date).days / 365.25

            open_conditions, open_frequency_stats = _eligible_open_conditions_for_factor(
                factor,
                [condition for condition in conditions["open"] if _condition_applicable_to_factor(factor, condition)],
                factor_cache,
                min_raw_open_events_per_quarter,
            )
            close_conditions = [
                condition for condition in conditions["close"] if _condition_applicable_to_factor(factor, condition)
            ]
            for open_rule in open_conditions:
                for close_rule in close_conditions:
                    open_rule, close_rule = combine_open_close_rules(open_rule, close_rule)
                    summary, _ = _backtest_rule_pair_from_cache(
                        factor,
                        open_rule,
                        close_rule,
                        factor_cache,
                        include_equity=False,
                    )
                    summary.update(open_frequency_stats[open_rule.name])
                    summary["target_year"] = int(target_year)
                    summary["train_end_date"] = train_end_date.strftime("%Y-%m-%d")
                    summary["train_start_date"] = train_start_date.strftime("%Y-%m-%d")
                    summary["train_years"] = float(train_years)
                    rows.append(summary)

    result = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan) if rows else pd.DataFrame()
    if (
        min_excess_annual_return is not None
        and not result.empty
        and "excess_annual_return" in result.columns
    ):
        result = result[
            pd.to_numeric(result["excess_annual_return"], errors="coerce") > min_excess_annual_return
        ].copy()
    if not result.empty:
        order_cols = ["target_year", "excess_annual_return", CODE_COL, "factor", "open_condition", "close_condition"]
        existing = [c for c in order_cols if c in result.columns]
        ascending = [True, False, True, True, True, True][: len(existing)]
        result = result.sort_values(existing, ascending=ascending, na_position="last").reset_index(drop=True)
    write_table(result, output_path / "rule_pair_summary_by_year_end.csv")
    return result


def _best_rule_row_for_base_factor(
    rule_summary: pd.DataFrame,
    base_factor: str,
    instrument: str | None = None,
) -> pd.Series | None:
    if rule_summary is None or rule_summary.empty:
        return None
    required = {"factor", "open_condition", "close_condition", "excess_annual_return"}
    if not required.issubset(rule_summary.columns):
        return None

    factor_candidates = [base_factor, f"{base_factor}_季线", f"{base_factor}_年线"]
    candidates = rule_summary[rule_summary["factor"].isin(factor_candidates)].copy()
    if candidates.empty:
        return None
    if instrument is not None and CODE_COL in candidates.columns:
        exact = candidates[candidates[CODE_COL].astype(str).eq(str(instrument))]
        if not exact.empty:
            candidates = exact
        else:
            all_rows = candidates[candidates[CODE_COL].astype(str).eq("ALL")]
            if not all_rows.empty:
                candidates = all_rows
    candidates["excess_annual_return"] = pd.to_numeric(candidates["excess_annual_return"], errors="coerce")
    candidates = candidates[candidates["excess_annual_return"].notna()]
    if "trade_count" in candidates.columns:
        candidates["trade_count"] = pd.to_numeric(candidates["trade_count"], errors="coerce")
        candidates = candidates[candidates["trade_count"].fillna(0) > 0]
    if candidates.empty:
        return None

    sort_cols = ["excess_annual_return"]
    ascending = [False]
    for col in ["sharpe", "excess_final_equity", "trade_count"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
            sort_cols.append(col)
            ascending.append(False)
    return candidates.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]


def run_best_rule_pair_backtest(
    df: pd.DataFrame,
    reference_rule_summary: pd.DataFrame,
    output_dir: str | Path = "results",
    factors: Iterable[str] | None = None,
    signal_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run only the currently best rule pair for each base factor.

    Uses an existing rule summary as the selector, then refreshes those selected
    pairs on the latest data and signals.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if reference_rule_summary is None or reference_rule_summary.empty:
        empty_summary = pd.DataFrame()
        empty_equity = pd.DataFrame()
        write_table(empty_summary, output_path / "rule_pair_best_base_summary.csv")
        write_table(empty_equity, output_path / "rule_pair_best_base_equity_curves.csv")
        return empty_summary, empty_equity

    selected_factors = list(factors) if factors is not None else get_factor_columns(df)
    base_factors = []
    seen: set[str] = set()
    for factor in selected_factors:
        base, _ = _split_factor_frequency(str(factor))
        if base in seen:
            continue
        seen.add(base)
        base_factors.append(base)

    conditions = build_event_conditions()
    if signal_table is None:
        signal_table = generate_signal_table(df, factors=selected_factors, conditions=conditions)

    by_name = {rule.name: rule for rule in conditions["open"] + conditions["close"]}
    factor_cache_by_factor: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    equity_rows: list[pd.DataFrame] = []
    instruments = df[CODE_COL].dropna().astype(str).drop_duplicates().tolist()

    for instrument in instruments:
        for base_factor in base_factors:
            best_row = _best_rule_row_for_base_factor(reference_rule_summary, base_factor, instrument=instrument)
            if best_row is None:
                continue
            factor = str(best_row["factor"])
            open_name = str(best_row["open_condition"])
            close_name = str(best_row["close_condition"])
            if open_name not in by_name or close_name not in by_name:
                continue
            if factor not in factor_cache_by_factor:
                factor_cache_by_factor[factor] = _build_factor_event_cache_from_signal_table(
                    df,
                    factor,
                    conditions,
                    signal_table,
                )
            summary, equity = _backtest_rule_pair_from_cache(
                factor=factor,
                open_rule=by_name[open_name],
                close_rule=by_name[close_name],
                factor_cache=factor_cache_by_factor[factor],
                include_equity=True,
            )
            summary["base_factor"] = base_factor
            summary["selected_from_excess_annual_return"] = pd.to_numeric(best_row.get("excess_annual_return"), errors="coerce")
            summary_rows.append(summary)
            if equity is not None and not equity.empty:
                equity = equity[equity[CODE_COL].astype(str).eq(str(instrument))].copy()
                if not equity.empty:
                    equity["base_factor"] = base_factor
                    equity_rows.append(equity)

    summary_df = pd.DataFrame(summary_rows).replace([np.inf, -np.inf], np.nan)
    if not summary_df.empty:
        sort_cols = [col for col in ["base_factor", CODE_COL, "excess_annual_return", "sharpe"] if col in summary_df.columns]
        ascending = [True, True, False, False][: len(sort_cols)]
        summary_df = summary_df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    equity_df = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()

    write_table(summary_df, output_path / "rule_pair_best_base_summary.csv")
    write_table(equity_df, output_path / "rule_pair_best_base_equity_curves.csv")
    return summary_df, equity_df


