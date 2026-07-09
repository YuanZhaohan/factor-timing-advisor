from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import _annual_return, _calc_trade_max_drawdown, _max_drawdown, _sharpe
from io_utils import read_run_table, write_table
from signal_generation import _build_factor_event_cache_from_signal_table, build_event_conditions
from timing_config import CODE_COL, DATE_COL, NAME_COL, PRICE_COL, SKILL_ROOT, TRADING_DAYS


DEFAULT_RUN_DIR = SKILL_ROOT / "workspace" / "runs" / "default"
OUTPUT_SUBDIR = "selected_single_factor_rules"


@dataclass(frozen=True)
class SelectedRuleSpec:
    rule_id: str
    factor: str
    category: str
    strategy_label: str
    base_open_condition: str
    close_condition: str
    open_condition_label: str | None = None
    close_condition_label: str | None = None
    open_transform: str = "identity"
    close_transform: str = "identity"
    open_transform_params: dict[str, Any] = field(default_factory=dict)
    close_transform_params: dict[str, Any] = field(default_factory=dict)
    min_hold_days_for_close: int = 0
    stop_loss_return: float | None = None
    stop_loss_window: int | None = None
    notes: str = ""


SELECTED_RULES: tuple[SelectedRuleSpec, ...] = (
    SelectedRuleSpec(
        rule_id="chip_profit_index",
        factor="指数筹码盈利",
        category="赔率/筹码",
        strategy_label="指数筹码盈利_自身年线过滤_等待确认低频版",
        base_open_condition="开仓_低位均值回复_20日",
        open_condition_label="开仓_低位均值回复_20日 + 年线<=0.5 + 5日内收涨0.3%确认 + 20日信号间隔",
        close_condition="闭仓_加速度转负_10日",
        close_condition_label="闭仓_加速度转负_10日(持仓满5日后生效) OR 2%/5日止损",
        open_transform="chip_profit_index",
        min_hold_days_for_close=5,
        stop_loss_return=-0.02,
        stop_loss_window=5,
        notes="低位修复买点，先用自身年线过滤，再等待短期价格确认，避免频繁开仓。",
    ),
    SelectedRuleSpec(
        rule_id="chip_profit_center",
        factor="筹码盈利中枢_年线",
        category="赔率/筹码",
        strategy_label="筹码盈利中枢_年线候选_季线等待修复_月线短确认",
        base_open_condition="开仓_累计上升_5日_0.5sigma",
        open_condition_label="开仓_累计上升_5日_0.5sigma；季线弱化时等待40日修复并要求月线一日回升",
        close_condition="闭仓_上方拐点_1sigma",
        close_condition_label="闭仓_上方拐点_1sigma(持仓满10日后生效) OR 4%/5日止损",
        open_transform="chip_profit_center",
        min_hold_days_for_close=10,
        stop_loss_return=-0.04,
        stop_loss_window=5,
        notes="年线开仓为主；若季线仍下行，等待季线修复和月线回升。",
    ),
    SelectedRuleSpec(
        rule_id="chip_deviation",
        factor="筹码乖离率_季线",
        category="赔率/筹码",
        strategy_label="筹码乖离率_季线累计修复且价格同日上涨",
        base_open_condition="开仓_累计上升_5日_0.5sigma",
        open_condition_label="开仓_累计上升_5日_0.5sigma + 信号日收涨",
        close_condition="闭仓_下穿_1.5sigma",
        open_transform="same_day_price_up",
        notes="筹码乖离季线修复，同时要求价格同日确认。",
    ),
    SelectedRuleSpec(
        rule_id="forward_valuation",
        factor="远期估值",
        category="赔率/估值",
        strategy_label="远期估值_价格底背离60日",
        base_open_condition="开仓_价格新低因子未确认_60日",
        close_condition="闭仓_原始季线年线空头共振",
        notes="非常左侧的估值底背离：价格创新低但估值因子不确认。",
    ),
    SelectedRuleSpec(
        rule_id="historical_valuation",
        factor="历史估值",
        category="赔率/估值",
        strategy_label="历史估值_价格底背离60日_闭仓持仓满3日",
        base_open_condition="开仓_价格新低因子未确认_60日",
        close_condition="闭仓_原始季线年线空头共振",
        close_condition_label="闭仓_原始季线年线空头共振(持仓满3日后生效)",
        min_hold_days_for_close=3,
        notes="历史估值底背离；闭仓信号持仓满3日后才生效。",
    ),
    SelectedRuleSpec(
        rule_id="price_diffusion",
        factor="年价格扩散系数_季线",
        category="胜率/量",
        strategy_label="价格扩散系数_最优规则",
        base_open_condition="开仓_价格下跌因子上升_20日",
        close_condition="闭仓_高位钝化转弱_5日_1sigma",
        notes="价格仍弱但长期扩散开始修复，属于底部修复/背离开仓。",
    ),
    SelectedRuleSpec(
        rule_id="pb_skew",
        factor="pb偏度_季线",
        category="赔率/估值",
        strategy_label="pb偏度_上穿负0.5后10日内上穿MA10",
        base_open_condition="开仓_上穿_-0.5sigma",
        open_condition_label="开仓_上穿_-0.5sigma后10日内价格上穿MA10",
        close_condition="闭仓_20日振幅放大向下_1sigma",
        open_transform="wait_price_cross_ma10_10d",
        notes="估值偏度修复后，等待价格站上MA10做右侧确认。",
    ),
    SelectedRuleSpec(
        rule_id="sentiment_entropy",
        factor="情绪熵",
        category="胜率/量",
        strategy_label="情绪熵_扩散起点收涨确认_扩散终点退出",
        base_open_condition="开仓_加速度转正_10日",
        open_condition_label="开仓_加速度转正_10日后5日内收涨确认",
        close_condition="闭仓_加速度转负_5日",
        open_transform="wait_price_up_5d",
        notes="买扩散起点，卖扩散终点；开仓需要价格收涨承接。",
    ),
    SelectedRuleSpec(
        rule_id="turnover_diffusion",
        factor="季换手扩散系数_季线",
        category="胜率/量",
        strategy_label="换手扩散系数_季线加速度转正后3日收涨确认",
        base_open_condition="开仓_加速度转正_5日",
        open_condition_label="开仓_加速度转正_5日后3日内收涨确认",
        close_condition="闭仓_10日振幅放大向下_1sigma",
        open_transform="wait_price_up_3d",
        notes="中期换手参与面扩散后，等待短期价格确认。",
    ),
    SelectedRuleSpec(
        rule_id="volume_factor",
        factor="放量因子",
        category="胜率/量",
        strategy_label="放量因子_上穿1sigma且放量下跌等待修复",
        base_open_condition="开仓_上穿_1sigma",
        open_condition_label="开仓_上穿_1sigma；若信号日跌幅<=-1%则3日内等收复信号日收盘价",
        close_condition="闭仓_加速度转负_3日",
        open_transform="volume_down_wait_recover_3d",
        open_transform_params={"wait_days": 3, "down_return": -0.01},
        notes="躲避放量下跌；若信号日大跌，等待价格收复信号日收盘价。",
    ),
    SelectedRuleSpec(
        rule_id="fund_flow",
        factor="资金流因子",
        category="胜率/资金",
        strategy_label="资金流因子_低位修复后收涨确认_资金流转弱退出",
        base_open_condition="开仓_低位均值回复_20日",
        open_condition_label="开仓_低位均值回复_20日后5日内收涨确认",
        close_condition="闭仓_累计下降_5日_0.5sigma",
        open_transform="wait_price_up_5d",
        notes="资金低位修复后等价格收涨，资金转弱退出。",
    ),
    SelectedRuleSpec(
        rule_id="institutional_net_buy",
        factor="机构主动净买入",
        category="胜率/资金",
        strategy_label="机构主动净买入_季线低位修复_月线强势退潮退出",
        base_open_condition="开仓_低位均值回复_20日",
        open_condition_label="机构主动净买入_季线<=-0.5 + 季线5日斜率刚转正 + 信号日不大跌",
        close_condition="闭仓_下穿_1.5sigma",
        open_transform="institutional_net_buy_low_season_turn",
        open_transform_params={
            "season_factor": "机构主动净买入_季线",
            "season_max": -0.5,
            "slope_days": 5,
            "min_daily_return": -0.005,
        },
        notes="季线低位后等待5日斜率转正，并要求信号日不大跌；月线下穿1.5sigma退出。",
    ),
    SelectedRuleSpec(
        rule_id="margin_trading_ratio",
        factor="融资成交占比_季线",
        category="胜率/资金",
        strategy_label="融资成交占比_季线扩张后收涨确认",
        base_open_condition="开仓_累计上升_10日_1sigma",
        open_condition_label="开仓_累计上升_10日_1sigma后5日内收涨确认",
        close_condition="闭仓_下穿均线_20日",
        open_transform="wait_price_up_5d",
        notes="融资参与扩张后等价格确认，融资参与退潮退出。",
    ),
    SelectedRuleSpec(
        rule_id="small_cap_turnover",
        factor="小市值换手因子_季线",
        category="胜率/量",
        strategy_label="小市值换手因子_季线低位修复与价因子退潮",
        base_open_condition="开仓_低位均值回复_60日",
        close_condition="闭仓_价因子同步下降_20日",
        close_condition_label="闭仓_价因子同步下降_20日 OR 5日价因子同步转弱",
        close_transform="small_cap_turnover_close",
        notes="小市值换手低位修复开仓；价格和因子同步退潮退出。",
    ),
    SelectedRuleSpec(
        rule_id="up_down_volume_diff",
        factor="上涨下跌量差_季线",
        category="胜率/量",
        strategy_label="上涨下跌量差_季线快版阳量优势修复且价格确认",
        base_open_condition="开仓_上穿_0sigma",
        open_condition_label="开仓_上穿_0sigma且3日价格动量为正",
        close_condition="闭仓_下穿_1sigma",
        open_transform="price_momentum_up_3d",
        notes="阳线量能优势重新转正，并要求3日价格动量为正。",
    ),
)


def _rule_specs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": spec.rule_id,
                "factor": spec.factor,
                "category": spec.category,
                "strategy_label": spec.strategy_label,
                "open_condition": _open_label(spec),
                "base_open_condition": spec.base_open_condition,
                "close_condition": _close_label(spec),
                "base_close_condition": spec.close_condition,
                "open_transform": spec.open_transform,
                "close_transform": spec.close_transform,
                "min_hold_days_for_close": spec.min_hold_days_for_close,
                "stop_loss_return": spec.stop_loss_return,
                "stop_loss_window": spec.stop_loss_window,
                "notes": spec.notes,
            }
            for spec in SELECTED_RULES
        ]
    )


def _open_label(spec: SelectedRuleSpec) -> str:
    return spec.open_condition_label or spec.base_open_condition


def _close_label(spec: SelectedRuleSpec) -> str:
    return spec.close_condition_label or spec.close_condition


def _condition_by_name(conditions: dict[str, list[Any]], name: str) -> Any:
    by_name = {condition.name: condition for condition in conditions["open"] + conditions["close"]}
    if name not in by_name:
        raise KeyError(f"Unknown event condition: {name}")
    return by_name[name]


def _events_by_code(cache: list[dict[str, Any]], condition_name: str) -> dict[str, np.ndarray]:
    return {str(item[CODE_COL]): np.asarray(item["events"][condition_name], dtype=bool) for item in cache}


def _factor_cache(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    conditions: dict[str, list[Any]],
    factor: str,
    memo: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if factor not in memo:
        memo[factor] = _build_factor_event_cache_from_signal_table(data, factor, conditions, signals)
    return memo[factor]


def _price_up(group: pd.DataFrame) -> np.ndarray:
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    return price.pct_change().gt(0).fillna(False).to_numpy(dtype=bool)


def _price_momentum_up(group: pd.DataFrame, days: int) -> np.ndarray:
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    return price.pct_change(days).gt(0).fillna(False).to_numpy(dtype=bool)


def _wait_for_confirm(base_open: np.ndarray, confirm: np.ndarray, wait_days: int) -> np.ndarray:
    base_open = np.asarray(base_open, dtype=bool)
    confirm = np.asarray(confirm, dtype=bool)
    out = np.zeros(len(base_open), dtype=bool)
    pending_idx: int | None = None
    for idx in range(len(base_open)):
        if pending_idx is not None and idx - pending_idx > wait_days:
            pending_idx = None
        if base_open[idx]:
            pending_idx = idx
        if pending_idx is not None and idx >= pending_idx and confirm[idx]:
            out[idx] = True
            pending_idx = None
    return out


def _wait_confirm_first_hit(base: np.ndarray, confirm: np.ndarray, max_wait: int) -> np.ndarray:
    base = np.asarray(base, dtype=bool)
    confirm = np.asarray(confirm, dtype=bool)
    out = np.zeros_like(base, dtype=bool)
    i = 0
    while i < len(base):
        if not base[i]:
            i += 1
            continue
        end = min(i + max_wait, len(base) - 1)
        hits = np.flatnonzero(confirm[i : end + 1])
        if hits.size:
            hit = i + int(hits[0])
            out[hit] = True
            i = hit + 1
        else:
            i = end + 1
    return out


def _thin_events(events: np.ndarray, min_gap_days: int) -> np.ndarray:
    events = np.asarray(events, dtype=bool)
    if min_gap_days <= 0:
        return events.copy()
    out = np.zeros_like(events, dtype=bool)
    last = -10**9
    for idx in np.flatnonzero(events):
        if idx - last >= min_gap_days:
            out[idx] = True
            last = idx
    return out


def _price_cross_above_ma(group: pd.DataFrame, window: int) -> np.ndarray:
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    ma = price.rolling(window, min_periods=window).mean()
    event = price.shift(1).le(ma.shift(1)) & price.gt(ma)
    return event.fillna(False).to_numpy(dtype=bool)


def _required_param(params: dict[str, Any], key: str) -> Any:
    if key not in params:
        raise KeyError(f"Missing selected-rule transform parameter: {key}")
    return params[key]


def _wait_after_down_volume_signal(group: pd.DataFrame, base_open: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    wait_days = int(_required_param(params, "wait_days"))
    down_return = float(_required_param(params, "down_return"))
    price = pd.to_numeric(group[PRICE_COL], errors="coerce").reset_index(drop=True)
    ret1 = price.pct_change()
    base_open = np.asarray(base_open, dtype=bool)
    selected = np.zeros(len(base_open), dtype=bool)
    pending_idx: int | None = None
    signal_price: float | None = None
    for idx in range(len(base_open)):
        if pending_idx is not None and idx - pending_idx > wait_days:
            pending_idx = None
            signal_price = None
        if base_open[idx]:
            if pd.notna(ret1.iloc[idx]) and ret1.iloc[idx] <= down_return:
                pending_idx = idx
                signal_price = float(price.iloc[idx]) if pd.notna(price.iloc[idx]) else None
            else:
                selected[idx] = True
                pending_idx = None
                signal_price = None
        if pending_idx is not None and idx > pending_idx and signal_price is not None:
            if pd.notna(price.iloc[idx]) and float(price.iloc[idx]) > signal_price:
                selected[idx] = True
                pending_idx = None
                signal_price = None
    return selected


def _institutional_net_buy_low_season_turn_open(group: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    season_factor = str(_required_param(params, "season_factor"))
    season_max = float(_required_param(params, "season_max"))
    slope_days = int(_required_param(params, "slope_days"))
    min_daily_return = float(_required_param(params, "min_daily_return"))
    season = pd.to_numeric(group[season_factor], errors="coerce")
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    slope = season.diff(slope_days)
    event = season.le(season_max) & slope.gt(0) & slope.shift(1).le(0) & price.pct_change().gt(min_daily_return)
    return event.fillna(False).to_numpy(dtype=bool)


def _five_day_price_factor_sync_down(group: pd.DataFrame, factor: str) -> np.ndarray:
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    value = pd.to_numeric(group[factor], errors="coerce")
    return (price.pct_change(5).lt(0) & value.diff(5).lt(0)).fillna(False).to_numpy(dtype=bool)


def _selected_open_events(
    spec: SelectedRuleSpec,
    cache: list[dict[str, Any]],
    base_open_events: dict[str, np.ndarray],
    cache_by_factor: dict[str, list[dict[str, Any]]],
    data_by_code: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    if spec.open_transform == "chip_profit_index":
        year_by_code = {
            str(item[CODE_COL]): item
            for item in cache_by_factor["指数筹码盈利_年线"]
        }
        for item in cache:
            code = str(item[CODE_COL])
            group = item["group"].reset_index(drop=True)
            base_open = np.asarray(base_open_events[code], dtype=bool)
            price_confirm = pd.to_numeric(group[PRICE_COL], errors="coerce").pct_change().fillna(0.0).to_numpy(dtype=float) > 0.003
            year_value = pd.to_numeric(year_by_code[code]["group"]["指数筹码盈利_年线"], errors="coerce").to_numpy(dtype=float)
            waited = _wait_confirm_first_hit(base_open & (year_value <= 0.5), price_confirm, 5)
            selected[code] = _thin_events(waited, 20)
        return selected

    if spec.open_transform == "chip_profit_center":
        season_by_code = {
            str(item[CODE_COL]): item
            for item in cache_by_factor["筹码盈利中枢_季线"]
        }
        for item in cache:
            code = str(item[CODE_COL])
            base_open = np.asarray(base_open_events[code], dtype=bool)
            season = pd.to_numeric(season_by_code[code]["group"]["筹码盈利中枢_季线"], errors="coerce").to_numpy(dtype=float)
            month = pd.to_numeric(data_by_code[code]["筹码盈利中枢"], errors="coerce").to_numpy(dtype=float)
            season_delta20 = pd.Series(season).diff(20).to_numpy(dtype=float)
            season_repaired = season_delta20 >= -0.05
            month_rebound = pd.Series(month).diff(1).to_numpy(dtype=float) > 0
            out = np.zeros_like(base_open, dtype=bool)
            i = 0
            while i < len(base_open):
                if not base_open[i]:
                    i += 1
                    continue
                if not np.isfinite(season_delta20[i]) or season_delta20[i] >= -0.05:
                    out[i] = True
                    i += 1
                    continue
                end = min(i + 40, len(base_open) - 1)
                confirmed = None
                for j in range(i + 1, end + 1):
                    if season_repaired[j] and month_rebound[j]:
                        confirmed = j
                        break
                if confirmed is not None:
                    out[confirmed] = True
                    i = confirmed + 1
                else:
                    i = end + 1
            selected[code] = out
        return selected

    for item in cache:
        code = str(item[CODE_COL])
        group = item["group"].reset_index(drop=True)
        base_open = np.asarray(base_open_events[code], dtype=bool)
        if spec.open_transform == "identity":
            selected[code] = base_open
        elif spec.open_transform == "same_day_price_up":
            selected[code] = base_open & _price_up(group)
        elif spec.open_transform == "wait_price_up_3d":
            selected[code] = _wait_for_confirm(base_open, _price_up(group), 3)
        elif spec.open_transform == "wait_price_up_5d":
            selected[code] = _wait_for_confirm(base_open, _price_up(group), 5)
        elif spec.open_transform == "wait_price_cross_ma10_10d":
            selected[code] = _wait_for_confirm(base_open, _price_cross_above_ma(group, 10), 10)
        elif spec.open_transform == "volume_down_wait_recover_3d":
            selected[code] = _wait_after_down_volume_signal(group, base_open, spec.open_transform_params)
        elif spec.open_transform == "institutional_net_buy_low_season_turn":
            selected[code] = _institutional_net_buy_low_season_turn_open(data_by_code.get(code, group), spec.open_transform_params)
        elif spec.open_transform == "price_momentum_up_3d":
            selected[code] = base_open & _price_momentum_up(group, 3)
        else:
            raise ValueError(f"Unsupported open transform: {spec.open_transform}")
    return selected


def _selected_close_events(
    spec: SelectedRuleSpec,
    cache: list[dict[str, Any]],
    base_close_events: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if spec.close_transform == "identity":
        return base_close_events
    if spec.close_transform == "small_cap_turnover_close":
        selected: dict[str, np.ndarray] = {}
        for item in cache:
            code = str(item[CODE_COL])
            group = item["group"].reset_index(drop=True)
            selected[code] = np.asarray(base_close_events[code], dtype=bool) | _five_day_price_factor_sync_down(group, spec.factor)
        return selected
    raise ValueError(f"Unsupported close transform: {spec.close_transform}")


def _calc_annualized_trade_return(trade_return: float, holding_days: int | float) -> float:
    if holding_days <= 0 or not np.isfinite(trade_return):
        return np.nan
    if trade_return <= -1:
        return -1.0
    return float((1 + trade_return) ** (TRADING_DAYS / holding_days) - 1)


def _simulate_one_group(
    spec: SelectedRuleSpec,
    item: dict[str, Any],
    open_events: np.ndarray,
    close_events: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    code = str(item[CODE_COL])
    group = item["group"].reset_index(drop=True)
    prices = np.asarray(item["prices"], dtype=float)
    dates = np.asarray(item["dates"])
    records: list[dict[str, Any]] = []
    latest_idx = len(group) - 1

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
        close_reason = _close_label(spec)
        for j in range(entry_idx, len(group)):
            if not np.isfinite(prices[j]):
                continue
            current_return = prices[j] / prices[entry_idx] - 1
            holding_days = j - entry_idx
            hit_stop = (
                spec.stop_loss_return is not None
                and spec.stop_loss_window is not None
                and holding_days <= spec.stop_loss_window
                and current_return <= spec.stop_loss_return
            )
            hit_close = holding_days >= spec.min_hold_days_for_close and bool(close_events[j])
            if hit_stop or hit_close:
                exit_signal_idx = j
                exit_idx = min(j + 1, len(group) - 1)
                close_reason = f"止损{spec.stop_loss_return:.0%}/{spec.stop_loss_window}日" if hit_stop else _close_label(spec)
                break

        if exit_idx is None:
            exit_signal_idx = len(group) - 1
            exit_idx = len(group) - 1
            close_reason = "end"

        if exit_idx > entry_idx and np.isfinite(prices[exit_idx]):
            trade_return = prices[exit_idx] / prices[entry_idx] - 1
            holding_days = exit_idx - entry_idx
            records.append(
                {
                    CODE_COL: code,
                    NAME_COL: group[NAME_COL].iloc[-1] if NAME_COL in group else "",
                    "rule_id": spec.rule_id,
                    "factor": spec.factor,
                    "category": spec.category,
                    "strategy_label": spec.strategy_label,
                    "open_condition": _open_label(spec),
                    "close_condition": _close_label(spec),
                    "close_reason": close_reason,
                    "entry_signal_date": dates[entry_signal_idx],
                    "entry_date": dates[entry_idx],
                    "exit_signal_date": dates[exit_signal_idx],
                    "exit_date": dates[exit_idx],
                    "entry_signal_idx": entry_signal_idx,
                    "entry_idx": entry_idx,
                    "exit_signal_idx": exit_signal_idx,
                    "exit_idx": exit_idx,
                    "entry_price": prices[entry_idx],
                    "exit_price": prices[exit_idx],
                    "trade_return": trade_return,
                    "annualized_trade_return": _calc_annualized_trade_return(trade_return, holding_days),
                    "max_drawdown": _calc_trade_max_drawdown(prices, entry_idx, exit_idx),
                    "holding_days": holding_days,
                    "forced_exit": close_reason == "end",
                }
            )
        i = max(int(exit_idx), entry_signal_idx + 1)

    position = np.zeros(len(group), dtype=float)
    for record in records:
        start = min(int(record["entry_idx"]) + 1, len(position))
        stop = min(int(record["exit_idx"]) + 1, len(position))
        if start < stop:
            position[start:stop] = 1.0

    status = _current_status_from_events(spec, group, prices, dates, open_events, close_events)
    status.update(
        {
            CODE_COL: code,
            NAME_COL: group[NAME_COL].iloc[-1] if NAME_COL in group else "",
            "latest_date": dates[-1] if len(dates) else pd.NaT,
            "latest_price": prices[-1] if len(prices) else np.nan,
            "latest_factor_value": pd.to_numeric(group[spec.factor], errors="coerce").iloc[-1] if spec.factor in group else np.nan,
            "latest_open_signal": bool(open_events[latest_idx]) if latest_idx >= 0 else False,
            "latest_close_signal": bool(close_events[latest_idx]) if latest_idx >= 0 else False,
            "open_event_count": int(np.asarray(open_events, dtype=bool).sum()),
            "close_event_count": int(np.asarray(close_events, dtype=bool).sum()),
        }
    )
    return records, position, status


def _current_status_from_events(
    spec: SelectedRuleSpec,
    group: pd.DataFrame,
    prices: np.ndarray,
    dates: np.ndarray,
    open_events: np.ndarray,
    close_events: np.ndarray,
) -> dict[str, Any]:
    latest_idx = len(prices) - 1
    state = "空"
    pending_signal = ""
    pending_signal_date = pd.NaT
    entry_signal_idx: int | None = None
    entry_idx: int | None = None
    last_open_signal_idx: int | None = None
    last_close_signal_idx: int | None = None

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
        hit_stop = (
            spec.stop_loss_return is not None
            and spec.stop_loss_window is not None
            and holding_days <= spec.stop_loss_window
            and current_return <= spec.stop_loss_return
        )
        hit_close = holding_days >= spec.min_hold_days_for_close and bool(close_events[i])
        if hit_stop or hit_close:
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

    entry_signal_date = pd.NaT
    entry_date = pd.NaT
    current_holding_days = np.nan
    current_return = np.nan
    if state == "多" and entry_idx is not None:
        entry_signal_date = dates[entry_signal_idx] if entry_signal_idx is not None else pd.NaT
        entry_date = dates[entry_idx]
        current_holding_days = latest_idx - entry_idx
        current_return = prices[latest_idx] / prices[entry_idx] - 1 if np.isfinite(prices[latest_idx]) and np.isfinite(prices[entry_idx]) else np.nan

    return {
        "current_state": state,
        "pending_signal": pending_signal,
        "pending_signal_date": pending_signal_date,
        "entry_signal_date": entry_signal_date,
        "entry_date": entry_date,
        "current_holding_days": current_holding_days,
        "current_return": current_return,
        "last_open_signal_date": dates[last_open_signal_idx] if last_open_signal_idx is not None else pd.NaT,
        "last_close_signal_date": dates[last_close_signal_idx] if last_close_signal_idx is not None else pd.NaT,
    }


def _summarize_rule(
    spec: SelectedRuleSpec,
    item: dict[str, Any],
    records: list[dict[str, Any]],
    position: np.ndarray,
    open_events: np.ndarray,
) -> dict[str, Any]:
    group = item["group"].reset_index(drop=True)
    prices = np.asarray(item["prices"], dtype=float)
    benchmark_returns = pd.Series(prices, dtype=float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    strategy_returns = benchmark_returns * position
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    excess_equity = strategy_equity / benchmark_equity.replace(0, np.nan)
    completed = [record for record in records if not record["forced_exit"]]
    returns = pd.Series([record["trade_return"] for record in completed], dtype=float)
    return {
        CODE_COL: item[CODE_COL],
        NAME_COL: group[NAME_COL].iloc[-1] if NAME_COL in group else "",
        "rule_id": spec.rule_id,
        "factor": spec.factor,
        "category": spec.category,
        "strategy_label": spec.strategy_label,
        "open_condition": _open_label(spec),
        "close_condition": _close_label(spec),
        "annual_return": _annual_return(strategy_equity),
        "benchmark_annual_return": _annual_return(benchmark_equity),
        "excess_annual_return": _annual_return(excess_equity),
        "max_drawdown": _max_drawdown(strategy_equity),
        "excess_max_drawdown": _max_drawdown(excess_equity),
        "sharpe": _sharpe(strategy_returns),
        "holding_ratio": float(position.mean()) if len(position) else np.nan,
        "open_event_count": int(np.asarray(open_events, dtype=bool).sum()),
        "trade_count": int(len(completed)),
        "win_rate": float(returns.gt(0).mean()) if not returns.empty else np.nan,
        "mean_trade_return": float(returns.mean()) if not returns.empty else np.nan,
        "mean_holding_days": float(np.mean([record["holding_days"] for record in completed])) if completed else np.nan,
        "final_equity": float(strategy_equity.iloc[-1]) if len(strategy_equity) else np.nan,
        "benchmark_final_equity": float(benchmark_equity.iloc[-1]) if len(benchmark_equity) else np.nan,
        "excess_final_equity": float(excess_equity.iloc[-1]) if len(excess_equity) else np.nan,
    }


def _position_rows(spec: SelectedRuleSpec, item: dict[str, Any], position: np.ndarray) -> pd.DataFrame:
    group = item["group"].reset_index(drop=True)
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    benchmark_return = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame(
        {
            CODE_COL: item[CODE_COL],
            NAME_COL: group[NAME_COL].iloc[-1] if NAME_COL in group else "",
            DATE_COL: group[DATE_COL],
            "rule_id": spec.rule_id,
            "factor": spec.factor,
            "position": position,
            "strategy_return": benchmark_return.to_numpy(dtype=float) * position,
            "benchmark_return": benchmark_return,
        }
    )


def _write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    written: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        clean = table.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
        written[name] = write_table(clean, output_dir / f"{name}.csv")
        clean.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return written


def run_selected_single_factor_rules(
    input_dir: str | Path = DEFAULT_RUN_DIR,
    output_dir: str | Path | None = None,
) -> dict[str, int]:
    run_dir = Path(input_dir)
    out_root = Path(output_dir) if output_dir is not None else run_dir
    output_path = out_root / "results" / OUTPUT_SUBDIR

    data = read_run_table(run_dir, ["data/input_snapshot.csv", "input_snapshot.csv"]).sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    signals = read_run_table(run_dir, ["results/signals/signals.csv", "signals.csv"])
    conditions = build_event_conditions()
    condition_names = {condition.name for condition in conditions["open"] + conditions["close"]}
    data_by_code = {str(code): group.reset_index(drop=True) for code, group in data.groupby(CODE_COL, sort=False)}
    cache_by_factor: dict[str, list[dict[str, Any]]] = {}

    summary_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    position_tables: list[pd.DataFrame] = []

    for spec in SELECTED_RULES:
        if spec.base_open_condition not in condition_names:
            raise KeyError(f"{spec.rule_id}: unknown open condition {spec.base_open_condition}")
        if spec.close_condition not in condition_names:
            raise KeyError(f"{spec.rule_id}: unknown close condition {spec.close_condition}")

        cache = _factor_cache(data, signals, conditions, spec.factor, cache_by_factor)
        if spec.open_transform == "chip_profit_index":
            _factor_cache(data, signals, conditions, "指数筹码盈利_年线", cache_by_factor)
        if spec.open_transform == "chip_profit_center":
            _factor_cache(data, signals, conditions, "筹码盈利中枢_季线", cache_by_factor)

        base_open_events = _events_by_code(cache, spec.base_open_condition)
        base_close_events = _events_by_code(cache, spec.close_condition)
        open_events_by_code = _selected_open_events(spec, cache, base_open_events, cache_by_factor, data_by_code)
        close_events_by_code = _selected_close_events(spec, cache, base_close_events)

        for item in cache:
            code = str(item[CODE_COL])
            records, position, status = _simulate_one_group(
                spec,
                item,
                open_events_by_code[code],
                close_events_by_code[code],
            )
            summary_rows.append(_summarize_rule(spec, item, records, position, open_events_by_code[code]))
            trade_rows.extend(records)
            status.update(
                {
                    "rule_id": spec.rule_id,
                    "factor": spec.factor,
                    "category": spec.category,
                    "strategy_label": spec.strategy_label,
                    "open_condition": _open_label(spec),
                    "close_condition": _close_label(spec),
                    "base_open_condition": spec.base_open_condition,
                    "base_close_condition": spec.close_condition,
                    "notes": spec.notes,
                }
            )
            status_rows.append(status)
            position_tables.append(_position_rows(spec, item, position))

    summary = pd.DataFrame(summary_rows)
    status = pd.DataFrame(status_rows)
    trades = pd.DataFrame(trade_rows)
    positions = pd.concat(position_tables, ignore_index=True) if position_tables else pd.DataFrame()
    specs = _rule_specs_frame()

    _write_outputs(
        output_path,
        {
            "selected_rule_specs": specs,
            "selected_rule_latest_status": status,
            "selected_rule_summary": summary,
            "selected_rule_trades": trades,
            "selected_rule_daily_positions": positions,
        },
    )
    return {
        "selected_rule_count": len(specs),
        "selected_rule_status_rows": len(status),
        "selected_rule_summary_rows": len(summary),
        "selected_rule_trade_rows": len(trades),
        "selected_rule_position_rows": len(positions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh selected single-factor rules.")
    parser.add_argument("--input-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = run_selected_single_factor_rules(input_dir=args.input_dir, output_dir=args.output_dir)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
