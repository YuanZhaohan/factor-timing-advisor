from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data_cleaning import get_factor_columns
from io_utils import write_table
from timing_config import (
    CODE_COL,
    DATE_COL,
    NAME_COL,
    PRICE_COL,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_NAME_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_VALUE_COL,
    EventCondition,
    _format_pct,
    _format_sigma,
    _split_factor_frequency,
)

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:  # pragma: no cover - used only when scipy is unavailable
    scipy_find_peaks = None


def _find_peaks(values: pd.Series | np.ndarray, height: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if scipy_find_peaks is not None:
        kwargs = {} if height is None else {"height": height}
        peaks, _ = scipy_find_peaks(arr, **kwargs)
        return peaks

    valid = np.isfinite(arr)
    if arr.size < 3:
        return np.array([], dtype=int)
    left = arr[1:-1] > arr[:-2]
    right = arr[1:-1] > arr[2:]
    height_ok = np.ones(arr.size - 2, dtype=bool) if height is None else arr[1:-1] >= height
    peak_mask = valid[1:-1] & valid[:-2] & valid[2:] & left & right & height_ok
    return np.flatnonzero(peak_mask) + 1


def process_signal(
    group: pd.DataFrame,
    lead: int = 2,
    score_col: str = "score",
    threshold: float = 0.5,
    adj_window: int = 10,
    adj_value: float = 0.5,
) -> pd.DataFrame:
    """
    Detect turning-point signals for one instrument.

    `peak_signal == 1` is an open event and `peak_signal == -1` is a close event.
    The signal at row t only uses values available up to row t.

    Turning points need right-side confirmation. For example, to confirm a valley
    at t-lead, the function must have already seen later observations up to t.
    This is intentional: the trade signal is recorded on the confirmation day,
    and the backtest enters no earlier than the next trading day, so the rule is
    delayed but does not leak future information.
    """
    group = group.copy()
    score = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=float)
    n = len(score)
    peaks = np.full(n, np.nan)
    if n < 3:
        group["peak_signal"] = peaks
        return group

    valid_triplet = np.isfinite(score[:-2]) & np.isfinite(score[1:-1]) & np.isfinite(score[2:])
    local_peak = np.zeros(n, dtype=bool)
    local_valley = np.zeros(n, dtype=bool)
    local_peak_any = np.zeros(n, dtype=bool)
    local_valley_any = np.zeros(n, dtype=bool)
    local_peak[1:-1] = valid_triplet & (score[1:-1] > score[:-2]) & (score[1:-1] > score[2:]) & (score[1:-1] >= threshold)
    local_valley[1:-1] = valid_triplet & (score[1:-1] < score[:-2]) & (score[1:-1] < score[2:]) & (score[1:-1] <= -threshold)
    local_peak_any[1:-1] = valid_triplet & (score[1:-1] > score[:-2]) & (score[1:-1] > score[2:])
    local_valley_any[1:-1] = valid_triplet & (score[1:-1] < score[:-2]) & (score[1:-1] < score[2:])

    t_idx = np.arange(max(int(lead), 2), n, dtype=int)
    if len(t_idx) == 0:
        group["peak_signal"] = peaks
        return group
    pivot_idx = t_idx - int(lead)
    finite_now = np.isfinite(score[t_idx]) & np.isfinite(score[t_idx - 1])

    close_mask = local_peak[pivot_idx] & finite_now & (score[t_idx] < score[t_idx - 1]) & (score[t_idx] >= 0)
    open_mask = local_valley[pivot_idx] & finite_now & (score[t_idx] > score[t_idx - 1]) & (score[t_idx] <= threshold)
    peaks[t_idx[close_mask]] = -1
    peaks[t_idx[open_mask]] = 1

    if adj_window > 0:
        close_hits = (peaks == -1).astype(int)
        open_hits = (peaks == 1).astype(int)
        close_cumsum = np.concatenate([[0], np.cumsum(close_hits)])
        open_cumsum = np.concatenate([[0], np.cumsum(open_hits)])
        left = np.maximum(0, t_idx - int(adj_window))
        recent_close = (close_cumsum[t_idx] - close_cumsum[left]) > 0
        recent_open = (open_cumsum[t_idx] - open_cumsum[left]) > 0
        can_adjust = t_idx >= int(adj_window)
        not_extreme = ~np.isfinite(peaks[t_idx]) | (np.abs(peaks[t_idx]) != 1)
        trend_up = (
            np.isfinite(score[t_idx - 2])
            & np.isfinite(score[t_idx - 1])
            & np.isfinite(score[t_idx])
            & (score[t_idx - 2] < score[t_idx - 1])
            & (score[t_idx - 1] < score[t_idx])
        )
        trend_down = (
            np.isfinite(score[t_idx - 2])
            & np.isfinite(score[t_idx - 1])
            & np.isfinite(score[t_idx])
            & (score[t_idx - 2] > score[t_idx - 1])
            & (score[t_idx - 1] > score[t_idx])
        )
        adj_close_mask = (
            can_adjust
            & not_extreme
            & local_valley_any[pivot_idx]
            & np.isfinite(score[pivot_idx])
            & (score[pivot_idx] > threshold)
            & recent_close
            & trend_up
        )
        peaks[t_idx[adj_close_mask]] = -adj_value

        not_extreme = ~np.isfinite(peaks[t_idx]) | (np.abs(peaks[t_idx]) != 1)
        adj_open_mask = (
            can_adjust
            & not_extreme
            & local_peak_any[pivot_idx]
            & np.isfinite(score[pivot_idx])
            & (score[pivot_idx] < -threshold)
            & recent_open
            & trend_down
        )
        peaks[t_idx[adj_open_mask]] = adj_value

    group["peak_signal"] = peaks
    return group


def build_event_conditions() -> dict[str, list[EventCondition]]:
    open_rules: list[EventCondition] = []
    close_rules: list[EventCondition] = []

    # 下方拐点开仓：
    # 因子先跌到指定 sigma 以下，然后在 lead 天后确认出现局部谷值且开始向上。
    # 适合测试“低位企稳/均值回复”类买点，例如 开仓_下方拐点_-1sigma。
    for level in (-0.5, -1, -1.5):
        open_rules.append(
            EventCondition(f"开仓_下方拐点_{_format_sigma(level)}", "open", "lower_turn", {"level": level})
        )

    # 上穿阈值开仓：
    # 因子从阈值下方上穿到阈值上方时触发。
    # 负阈值上穿偏“低位修复”，0 轴上穿偏“状态转正”，正阈值上穿偏“强势突破”。
    for level in (-1.5, -1, -0.5, 0, 0.5, 1, 1.5):
        open_rules.append(
            EventCondition(f"开仓_上穿_{_format_sigma(level)}", "open", "cross_up", {"level": level})
        )

    # 连续上升开仓：
    # 因子连续 days 天环比上升时触发，只在刚满足连续上升的第一天触发一次。
    # 适合测试“短期动量正在改善”的买点。
    #
    # 斜率转正开仓：
    # 因子当前值相对 days 天前的差分由负转正时触发。
    # 比连续上升更宽松，用来捕捉中短期方向从下行切到上行。
    for days in (3, 5, 10):
        open_rules.append(EventCondition(f"开仓_连续上升_{days}日", "open", "rise_streak", {"days": days}))
        open_rules.append(EventCondition(f"开仓_斜率转正_{days}日", "open", "slope_turn_pos", {"days": days}))
        open_rules.append(EventCondition(f"开仓_加速度转正_{days}日", "open", "accel_turn_pos", {"days": days}))
        for sigma in (0.5, 1, 1.5):
            open_rules.append(
                EventCondition(
                    f"开仓_累计上升_{days}日_{_format_sigma(sigma)}",
                    "open",
                    "fast_rise",
                    {"days": days, "sigma": sigma},
                )
            )

    # N 日振幅放大：
    # 因子过去 N 日 max-min 突然超过阈值，且当前位于窗口上半区。
    # 它不是单纯方向信号，更像“波动放大后向上选择”的买点。
    for days in (5, 10, 20):
        for sigma in (1, 1.5):
            open_rules.append(
                EventCondition(
                    f"开仓_{days}日振幅放大向上_{_format_sigma(sigma)}",
                    "open",
                    "range_expansion_up",
                    {"days": days, "sigma": sigma},
                )
            )

    # 低位持续后回升：
    # 因子此前连续 N 日处于低位阈值以下，今天重新向上脱离该低位区。
    for days in (5, 10, 20):
        for level in (-1, -1.5):
            open_rules.append(
                EventCondition(
                    f"开仓_低位持续{days}日后回升_{_format_sigma(level)}",
                    "open",
                    "low_stay_rebound",
                    {"days": days, "level": level},
                )
            )

    for target in ("季线", "年线"):
        open_rules.append(
            EventCondition(
                f"开仓_指标上穿{target}",
                "open",
                "factor_cross_smooth_up",
                {"target": target},
            )
        )
    open_rules.append(EventCondition("开仓_原始季线年线多头共振", "open", "raw_season_year_bull", {}))

    # 价格因子底背离：
    # 指数价格创窗口新低，但因子未同步创新低且正在回升，作为左侧修复线索。
    for window in (20, 60):
        open_rules.append(
            EventCondition(
                f"开仓_价格因子底背离_{window}日",
                "open",
                "price_factor_bottom_divergence",
                {"window": window},
            )
        )

    # 价因子联动开仓：
    # 价格和因子同步改善偏趋势确认；价格下跌但因子上升偏左侧修复。
    for days in (5, 20):
        open_rules.append(
            EventCondition(
                f"开仓_价因子同步上升_{days}日",
                "open",
                "price_factor_sync_up",
                {"days": days},
            )
        )
        open_rules.append(
            EventCondition(
                f"开仓_价格下跌因子上升_{days}日",
                "open",
                "price_down_factor_up",
                {"days": days},
            )
        )

    # 价格突破/新低未确认：
    # 价格突破且因子确认，偏右侧趋势；价格新低但因子不确认，偏抄底观察。
    for window in (20, 60):
        open_rules.append(
            EventCondition(
                f"开仓_价格突破因子确认_{window}日",
                "open",
                "price_breakout_factor_confirm",
                {"window": window},
            )
        )
        open_rules.append(
            EventCondition(
                f"开仓_价格新低因子未确认_{window}日",
                "open",
                "price_new_low_factor_not_confirm",
                {"window": window},
            )
        )

    # 均线上穿开仓：
    # 因子上穿自身 window 日均线时触发，用因子自己的趋势线判断状态改善。
    #
    # 低位均值回复开仓：
    # 因子前一日处在过去 window 日低分位附近，且当日开始回升时触发。
    # 默认 quantile=0.2，即过去窗口内较低的 20% 区域。
    for window in (20, 60):
        open_rules.append(EventCondition(f"开仓_上穿均线_{window}日", "open", "ma_cross_up", {"window": window}))
        open_rules.append(
            EventCondition(
                f"开仓_低位均值回复_{window}日",
                "open",
                "low_reversion",
                {"window": window, "quantile": 0.2},
            )
        )

    # process_signal 开仓：
    # 复用你原先的拐点检测函数，peak_signal == 1 作为开仓事件。
    # 它比简单 lower_turn 多了 find_peaks 和 lead 确认逻辑。
    open_rules.append(EventCondition("开仓_process_signal买点", "open", "process_signal_open", {"threshold": 0.5}))

    for level in (0.5, 1, 1.5):
        close_rules.append(EventCondition(f"闭仓_上方拐点_{_format_sigma(level)}", "close", "upper_turn", {"level": level}))
    for level in (1.5, 1, 0.5, 0, -0.5, -1, -1.5):
        close_rules.append(EventCondition(f"闭仓_下穿_{_format_sigma(level)}", "close", "cross_down", {"level": level}))
    for days in (3, 5, 10):
        close_rules.append(EventCondition(f"闭仓_连续下降_{days}日", "close", "fall_streak", {"days": days}))
        close_rules.append(EventCondition(f"闭仓_斜率转负_{days}日", "close", "slope_turn_neg", {"days": days}))
        close_rules.append(EventCondition(f"闭仓_加速度转负_{days}日", "close", "accel_turn_neg", {"days": days}))
        for sigma in (0.5, 1, 1.5):
            close_rules.append(
                EventCondition(
                    f"闭仓_累计下降_{days}日_{_format_sigma(sigma)}",
                    "close",
                    "fast_fall",
                    {"days": days, "sigma": sigma},
                )
            )
    for days in (5, 10, 20):
        for sigma in (1, 1.5):
            close_rules.append(
                EventCondition(
                    f"闭仓_{days}日振幅放大向下_{_format_sigma(sigma)}",
                    "close",
                    "range_expansion_down",
                    {"days": days, "sigma": sigma},
                )
            )
    for days in (5, 10, 20):
        for level in (1, 1.5):
            close_rules.append(
                EventCondition(
                    f"闭仓_高位钝化转弱_{days}日_{_format_sigma(level)}",
                    "close",
                    "high_stall_weak",
                    {"days": days, "level": level},
                )
            )
            close_rules.append(
                EventCondition(
                    f"闭仓_高位持续{days}日后回落_{_format_sigma(level)}",
                    "close",
                    "high_stay_fall",
                    {"days": days, "level": level},
                )
            )
    for target in ("季线", "年线"):
        close_rules.append(
            EventCondition(
                f"闭仓_指标下穿{target}",
                "close",
                "factor_cross_smooth_down",
                {"target": target},
            )
        )
    close_rules.append(EventCondition("闭仓_原始季线年线空头共振", "close", "raw_season_year_bear", {}))
    for window in (20, 60):
        close_rules.append(
            EventCondition(
                f"闭仓_价格因子顶背离_{window}日",
                "close",
                "price_factor_top_divergence",
                {"window": window},
            )
        )
    for days in (5, 20):
        close_rules.append(
            EventCondition(
                f"闭仓_价因子同步下降_{days}日",
                "close",
                "price_factor_sync_down",
                {"days": days},
            )
        )
        close_rules.append(
            EventCondition(
                f"闭仓_价格上涨因子下降_{days}日",
                "close",
                "price_up_factor_down",
                {"days": days},
            )
        )
    for window in (20, 60):
        close_rules.append(
            EventCondition(
                f"闭仓_价格破位因子确认_{window}日",
                "close",
                "price_breakdown_factor_confirm",
                {"window": window},
            )
        )
        close_rules.append(
            EventCondition(
                f"闭仓_价格新高因子未确认_{window}日",
                "close",
                "price_new_high_factor_not_confirm",
                {"window": window},
            )
        )
    for window in (20, 60):
        close_rules.append(EventCondition(f"闭仓_下穿均线_{window}日", "close", "ma_cross_down", {"window": window}))
        close_rules.append(EventCondition(f"闭仓_高位均值回复_{window}日", "close", "high_reversion", {"window": window, "quantile": 0.8}))
    for loss in (-0.03, -0.05, -0.08):
        close_rules.append(EventCondition(f"闭仓_止损_{_format_pct(loss)}", "close", "stop_loss", {"return": loss}, True))
    for gain in (0.05, 0.08, 0.12):
        close_rules.append(EventCondition(f"闭仓_止盈_{_format_pct(gain)}", "close", "take_profit", {"return": gain}, True))
    for days in (5, 10, 20, 60):
        close_rules.append(EventCondition(f"闭仓_持仓满_{days}日", "close", "time_exit", {"days": days}, True))
    close_rules.append(EventCondition("闭仓_process_signal卖点", "close", "process_signal_close", {"threshold": 0.5}))

    open_rules.extend(
        [
            EventCondition(
                "开仓_资金流资金分歧共振_3日",
                "open",
                "cross_factor_resonance",
                {
                    "anchor_factor": "资金流因子",
                    "confirm_factor": "资金分歧度因子",
                    "window": 3,
                    "anchor_families": ("上穿", "连续上升", "斜率转正", "低位均值回复", "下方拐点", "process_signal买点"),
                    "confirm_families": ("上穿", "连续上升", "斜率转正", "低位均值回复", "下方拐点"),
                },
            ),
            EventCondition(
                "开仓_筹码峰指数筹码盈利拐点共振_5日",
                "open",
                "cross_factor_resonance",
                {
                    "anchor_factor": "指数筹码盈利",
                    "confirm_factor": "筹码峰支撑",
                    "window": 5,
                    "anchor_families": ("下方拐点", "低位均值回复", "process_signal买点"),
                    "anchor_side": "open",
                    "confirm_side": "close",
                    "confirm_families": ("上方拐点", "高位均值回复", "process_signal卖点"),
                },
            ),
            EventCondition(
                "开仓_筹码峰筹码盈利中枢拐点共振_5日",
                "open",
                "cross_factor_resonance",
                {
                    "anchor_factor": "筹码盈利中枢",
                    "confirm_factor": "筹码峰支撑",
                    "window": 5,
                    "anchor_families": ("下方拐点", "低位均值回复", "process_signal买点"),
                    "anchor_side": "open",
                    "confirm_side": "close",
                    "confirm_families": ("上方拐点", "高位均值回复", "process_signal卖点"),
                },
            ),
        ]
    )
    close_rules.extend(
        [
            EventCondition(
                "闭仓_资金流资金分歧共振_3日",
                "close",
                "cross_factor_resonance",
                {
                    "anchor_factor": "资金流因子",
                    "confirm_factor": "资金分歧度因子",
                    "window": 3,
                    "anchor_families": ("下穿", "连续下降", "斜率转负", "高位均值回复", "上方拐点", "process_signal卖点"),
                    "confirm_families": ("下穿", "连续下降", "斜率转负", "高位均值回复", "上方拐点"),
                },
            ),
            EventCondition(
                "闭仓_筹码峰指数筹码盈利拐点共振_5日",
                "close",
                "cross_factor_resonance",
                {
                    "anchor_factor": "指数筹码盈利",
                    "confirm_factor": "筹码峰支撑",
                    "window": 5,
                    "anchor_families": ("上方拐点", "高位均值回复", "process_signal卖点"),
                    "anchor_side": "close",
                    "confirm_side": "open",
                    "confirm_families": ("下方拐点", "低位均值回复", "process_signal买点"),
                },
            ),
            EventCondition(
                "闭仓_筹码峰筹码盈利中枢拐点共振_5日",
                "close",
                "cross_factor_resonance",
                {
                    "anchor_factor": "筹码盈利中枢",
                    "confirm_factor": "筹码峰支撑",
                    "window": 5,
                    "anchor_families": ("上方拐点", "高位均值回复", "process_signal卖点"),
                    "anchor_side": "close",
                    "confirm_side": "open",
                    "confirm_families": ("下方拐点", "低位均值回复", "process_signal买点"),
                },
            ),
        ]
    )

    funding_open_families = ("上穿", "连续上升", "斜率转正", "低位均值回复", "下方拐点", "process_signal买点")
    funding_close_families = ("下穿", "连续下降", "斜率转负", "高位均值回复", "上方拐点", "process_signal卖点")
    for confirm_factor in ("机构主动净买入", "融资余额因子", "融资成交占比", "放量因子"):
        open_rules.append(
            EventCondition(
                f"开仓_资金流{confirm_factor}共振_3日",
                "open",
                "cross_factor_resonance",
                {
                    "anchor_factor": "资金流因子",
                    "confirm_factor": confirm_factor,
                    "window": 3,
                    "anchor_families": funding_open_families,
                    "confirm_families": funding_open_families,
                },
            )
        )
        close_rules.append(
            EventCondition(
                f"闭仓_资金流{confirm_factor}共振_3日",
                "close",
                "cross_factor_resonance",
                {
                    "anchor_factor": "资金流因子",
                    "confirm_factor": confirm_factor,
                    "window": 3,
                    "anchor_families": funding_close_families,
                    "confirm_families": funding_close_families,
                },
            )
        )

    open_rules.append(
        EventCondition(
            "开仓_波动率下行协偏度风险缓和共振_5日",
            "open",
            "cross_factor_resonance",
            {
                "anchor_factor": "真实波动率",
                "confirm_factor": "下行协偏度",
                "window": 5,
                "anchor_side": "close",
                "confirm_side": "open",
                "anchor_families": ("上方拐点", "高位均值回复", "process_signal卖点"),
                "confirm_families": ("下方拐点", "低位均值回复", "process_signal买点"),
            },
        )
    )
    close_rules.append(
        EventCondition(
            "闭仓_波动率下行协偏度风险放大共振_5日",
            "close",
            "cross_factor_resonance",
            {
                "anchor_factor": "真实波动率",
                "confirm_factor": "下行协偏度",
                "window": 5,
                "anchor_side": "open",
                "confirm_side": "close",
                "anchor_families": ("上穿", "连续上升", "斜率转正", "加速度转正"),
                "confirm_families": ("下穿", "连续下降", "斜率转负", "加速度转负"),
            },
        )
    )

    return {"open": open_rules, "close": close_rules}


def _cross_up(series: pd.Series, level: float) -> pd.Series:
    """因子从 level 下方上穿到 level 上方时触发。"""
    return (series.shift(1) <= level) & (series > level)


def _event_body_name(pattern: str) -> str:
    text = str(pattern)
    for prefix in ("开仓_", "闭仓_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _factor_with_same_frequency(target_base: str, factor: str) -> str:
    _, frequency = _split_factor_frequency(str(factor))
    if frequency == "季线":
        return f"{target_base}_季线"
    if frequency == "年线":
        return f"{target_base}_年线"
    return target_base


@lru_cache(maxsize=2)
def _non_resonance_static_conditions(side: str) -> tuple[EventCondition, ...]:
    conditions = build_event_conditions()
    return tuple(
        condition
        for condition in conditions[side]
        if not condition.requires_position and condition.kind != "cross_factor_resonance"
    )


def _pattern_matches_families(pattern: str, families: Iterable[str]) -> bool:
    body = _event_body_name(pattern)
    return any(body.startswith(str(family)) for family in families)


def _family_event_union(
    group: pd.DataFrame,
    factor: str,
    side: str,
    families: Iterable[str],
) -> pd.Series:
    if factor not in group.columns:
        return pd.Series(False, index=group.index, dtype=bool)

    matched = []
    for condition in _non_resonance_static_conditions(side):
        if _pattern_matches_families(condition.name, families):
            matched.append(_detect_events_single(group, factor, condition))
    if not matched:
        return pd.Series(False, index=group.index, dtype=bool)
    return pd.concat(matched, axis=1).any(axis=1).reindex(group.index, fill_value=False).astype(bool)


def _cross_factor_resonance_event(
    group: pd.DataFrame,
    factor: str,
    side: str,
    params: dict[str, Any],
) -> pd.Series:
    current_base, _ = _split_factor_frequency(str(factor))
    anchor_base = str(params["anchor_factor"])
    if current_base != anchor_base:
        return pd.Series(False, index=group.index, dtype=bool)

    anchor_factor = factor
    confirm_factor = _factor_with_same_frequency(str(params["confirm_factor"]), factor)
    anchor_side = str(params.get("anchor_side", side))
    confirm_side = str(params.get("confirm_side", side))
    anchor_events = _family_event_union(group, anchor_factor, anchor_side, params["anchor_families"]).to_numpy(dtype=bool)
    confirm_events = _family_event_union(group, confirm_factor, confirm_side, params["confirm_families"]).to_numpy(dtype=bool)
    window = max(int(params.get("window", 3)), 0)

    event = np.zeros(len(group), dtype=bool)
    for idx in range(len(group)):
        left = max(0, idx - window)
        if anchor_events[idx] and confirm_events[left : idx + 1].any():
            event[idx] = True
            continue
        if confirm_events[idx] and anchor_events[left:idx].any():
            event[idx] = True
    return pd.Series(event, index=group.index, dtype=bool)


def _cross_down(series: pd.Series, level: float) -> pd.Series:
    """因子从 level 上方下穿到 level 下方时触发。"""
    return (series.shift(1) >= level) & (series < level)


def _turn_event(series: pd.Series, level: float, lead: int, direction: str) -> pd.Series:
    """
    拐点事件。

    拐点一定要用右侧信息确认：例如某天是局部谷值，至少要看到它后面一天
    开始反弹，才能知道它真的是谷值。因此信号不能发在拐点当天，只能滞后发出。

    这里没有信息泄露：在 t 日发出信号时，只确认 t-lead 那一天是不是局部峰/谷。
    例如 lead=2 时，今天只判断 2 天前是否已经形成拐点；回测里还会从下一交易日
    才开始计算收益/持仓，所以是“确认后再晚一天交易”的保守口径。

    lower:
        t-lead 那天低于 level，且比前后相邻点都低；同时今天继续向上确认。
    upper:
        t-lead 那天高于 level，且比前后相邻点都高；同时今天继续向下确认。
    """
    # candidate 是被确认的历史点；left/right 是它前后相邻点。
    # 因为 right=series.shift(lead - 1)，在 t 日最多只用到 t-lead+1，
    # 仍然早于当前 t 日。也就是说，我们用右侧确认，但信号发在确认日，
    # 后续交易再滞后一日执行，不把未来信息放到过去交易。
    candidate = series.shift(lead)
    left = series.shift(lead + 1)
    right = series.shift(lead - 1)
    if direction == "lower":
        # 低位拐点：先确认历史点是局部谷值，再要求当前因子仍在向上。
        is_turn = (candidate <= level) & (candidate < left) & (candidate < right)
        confirm = series > series.shift(1)
    else:
        # 高位拐点：先确认历史点是局部峰值，再要求当前因子仍在向下。
        is_turn = (candidate >= level) & (candidate > left) & (candidate > right)
        confirm = series < series.shift(1)
    return is_turn & confirm


def _streak_event(series: pd.Series, days: int, direction: str) -> pd.Series:
    """
    连续上涨/下跌事件。

    只有“刚刚满足连续 days 天”的第一天触发，后面如果继续连续上涨/下跌，
    不会每天重复触发，避免同一段趋势产生过多重复事件。
    """
    up = series.diff() > 0
    step = up if direction == "up" else ~up
    # 第一个 diff 是空值，不能算作上涨或下跌。
    step = step & series.diff().notna()
    streak = step.rolling(days, min_periods=days).sum().eq(days)
    previous_streak = streak.shift(1, fill_value=False).astype(bool)
    return streak & ~previous_streak


def _fast_change_event(series: pd.Series, days: int, sigma: float, direction: str) -> pd.Series:
    """
    N 日快速变化事件。

    open 用累计上升：当前因子 - N 日前因子 >= sigma。
    close 用累计下降：当前因子 - N 日前因子 <= -sigma。
    只在差分刚越过阈值时触发，避免同一段急涨/急跌每天重复触发。
    """
    delta = series.diff(days)
    if direction == "up":
        event = (delta >= sigma) & (delta.shift(1) < sigma)
    else:
        event = (delta <= -sigma) & (delta.shift(1) > -sigma)
    return event & delta.notna()


def _range_expansion_event(series: pd.Series, days: int, sigma: float, direction: str) -> pd.Series:
    """
    N 日振幅放大事件。

    使用过去 days 日因子的 max-min 作为振幅。只有振幅刚超过阈值时触发。
    direction=up 要求当前值在窗口中位数上方，direction=down 要求当前值在窗口中位数下方。
    """
    rolling_high = series.rolling(days, min_periods=days).max()
    rolling_low = series.rolling(days, min_periods=days).min()
    rolling_mid = (rolling_high + rolling_low) / 2
    amplitude = rolling_high - rolling_low
    just_expanded = (amplitude >= sigma) & (amplitude.shift(1) < sigma)
    if direction == "up":
        direction_ok = series >= rolling_mid
    else:
        direction_ok = series <= rolling_mid
    return just_expanded & direction_ok & amplitude.notna()


def _acceleration_event(series: pd.Series, days: int, direction: str) -> pd.Series:
    """
    N 日动量的加速度转向。

    open: N 日动量为正，且动量的日变化刚从非正转为正。
    close: N 日动量为负，且动量的日变化刚从非负转为负。
    """
    momentum = series.diff(days)
    acceleration = momentum.diff()
    if direction == "up":
        event = (momentum > 0) & (acceleration > 0) & (acceleration.shift(1) <= 0)
    else:
        event = (momentum < 0) & (acceleration < 0) & (acceleration.shift(1) >= 0)
    return event & momentum.notna() & acceleration.notna()


def _high_stall_weak_event(series: pd.Series, days: int, level: float) -> pd.Series:
    """
    高位钝化后转弱。

    前 days 天都维持在 level 以上，说明高位钝化；今天跌破 level 时触发闭仓。
    """
    previous_high_stall = series.shift(1).rolling(days, min_periods=days).min().ge(level)
    return previous_high_stall & series.lt(level)


def _low_stay_rebound_event(series: pd.Series, days: int, level: float) -> pd.Series:
    """
    低位持续后回升。

    前 days 天都低于 level，今天向上突破 level 时触发。
    """
    previous_low_stay = series.shift(1).rolling(days, min_periods=days).max().le(level)
    return previous_low_stay & series.gt(level)


def _high_stay_fall_event(series: pd.Series, days: int, level: float) -> pd.Series:
    """
    高位持续后回落。

    前 days 天都高于 level，今天向下跌破 level 时触发。
    """
    previous_high_stay = series.shift(1).rolling(days, min_periods=days).min().ge(level)
    return previous_high_stay & series.lt(level)


def _price_factor_divergence_event(
    group: pd.DataFrame,
    factor: str,
    window: int,
    direction: str,
) -> pd.Series:
    """
    价格-因子背离事件。

    bottom:
        价格创 window 日新低，但因子没有同步创新低，且因子当天回升。
    top:
        价格创 window 日新高，但因子没有同步创新高，且因子当天回落。

    只使用当日及以前的价格和因子值。
    """
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    s = pd.to_numeric(group[factor], errors="coerce")
    if direction == "bottom":
        price_break = price <= price.rolling(window, min_periods=window).min()
        factor_not_break = s > s.rolling(window, min_periods=window).min()
        factor_turn = s > s.shift(1)
    else:
        price_break = price >= price.rolling(window, min_periods=window).max()
        factor_not_break = s < s.rolling(window, min_periods=window).max()
        factor_turn = s < s.shift(1)
    return price_break & factor_not_break & factor_turn & price.notna() & s.notna()


def _price_factor_relation_event(
    group: pd.DataFrame,
    factor: str,
    kind: str,
    days: int | None = None,
    window: int | None = None,
) -> pd.Series:
    """
    价格和因子的联动/确认事件。

    sync_up/sync_down:
        N 日价格收益和 N 日因子变化同向。
    price_down_factor_up / price_up_factor_down:
        价格与因子短期背离。
    breakout/breakdown confirm:
        价格突破或破位时，因子方向同步确认。
    new_low/new_high not_confirm:
        价格创新低/新高，但因子没有同步创新低/新高。

    所有事件只在关系刚成立时触发一次。
    """
    price = pd.to_numeric(group[PRICE_COL], errors="coerce")
    s = pd.to_numeric(group[factor], errors="coerce")

    if days is not None:
        price_return = price.pct_change(days)
        factor_delta = s.diff(days)
        if kind == "sync_up":
            relation = (price_return > 0) & (factor_delta > 0)
        elif kind == "sync_down":
            relation = (price_return < 0) & (factor_delta < 0)
        elif kind == "price_down_factor_up":
            relation = (price_return < 0) & (factor_delta > 0)
        elif kind == "price_up_factor_down":
            relation = (price_return > 0) & (factor_delta < 0)
        else:
            raise ValueError(f"Unknown price-factor days relation: {kind}")
        relation = relation & price_return.notna() & factor_delta.notna()
    elif window is not None:
        rolling_price_high = price.rolling(window, min_periods=window).max()
        rolling_price_low = price.rolling(window, min_periods=window).min()
        rolling_factor_high = s.rolling(window, min_periods=window).max()
        rolling_factor_low = s.rolling(window, min_periods=window).min()
        factor_ma = s.rolling(window, min_periods=window).mean()
        factor_delta = s.diff(window)

        price_new_high = price >= rolling_price_high
        price_new_low = price <= rolling_price_low
        factor_not_new_high = s < rolling_factor_high
        factor_not_new_low = s > rolling_factor_low

        if kind == "breakout_confirm":
            relation = price_new_high & ((s > 0) | (factor_delta > 0))
        elif kind == "breakdown_confirm":
            relation = price_new_low & ((s < 0) | (factor_delta < 0))
        elif kind == "new_low_not_confirm":
            relation = price_new_low & (factor_not_new_low | (s > factor_ma))
        elif kind == "new_high_not_confirm":
            relation = price_new_high & (factor_not_new_high | (s < factor_ma))
        else:
            raise ValueError(f"Unknown price-factor window relation: {kind}")
        relation = relation & price.notna() & s.notna()
    else:
        raise ValueError("Either days or window must be provided")

    previous_relation = relation.shift(1, fill_value=False).astype(bool)
    return relation & ~previous_relation


def _raw_season_year_resonance_event(group: pd.DataFrame, factor: str, direction: str) -> pd.Series:
    """
    原始/季线/年线三周期共振。

    只在原始指标字段上触发：
    bull: 原始、季线、年线同处上行，且原始 > 季线 > 年线。
    bear: 原始、季线、年线同处下行，且原始 < 季线 < 年线。
    """
    base_factor, frequency = _split_factor_frequency(factor)
    if frequency != "原始":
        return pd.Series(False, index=group.index)

    season_col = f"{base_factor}_季线"
    year_col = f"{base_factor}_年线"
    if season_col not in group.columns or year_col not in group.columns:
        return pd.Series(False, index=group.index)

    raw = pd.to_numeric(group[base_factor], errors="coerce")
    season = pd.to_numeric(group[season_col], errors="coerce")
    year = pd.to_numeric(group[year_col], errors="coerce")
    if direction == "bull":
        structure = (raw > season) & (season > year)
        trend = (raw.diff() > 0) & (season.diff() > 0) & (year.diff() > 0)
    else:
        structure = (raw < season) & (season < year)
        trend = (raw.diff() < 0) & (season.diff() < 0) & (year.diff() < 0)
    event = structure & trend
    previous_event = event.shift(1, fill_value=False).astype(bool)
    return event & ~previous_event & raw.notna() & season.notna() & year.notna()


def _factor_smooth_cross_event(
    group: pd.DataFrame,
    factor: str,
    target: str,
    direction: str,
) -> pd.Series:
    """
    原始指标相对季线/年线的上穿或下穿。

    只在原始指标上生成事件，避免同一组字段在 _季线/_年线 行重复触发。
    """
    base_factor, frequency = _split_factor_frequency(factor)
    if frequency != "原始":
        return pd.Series(False, index=group.index)

    target_col = f"{base_factor}_{target}"
    if target_col not in group.columns:
        return pd.Series(False, index=group.index)

    raw = pd.to_numeric(group[base_factor], errors="coerce")
    smooth = pd.to_numeric(group[target_col], errors="coerce")
    if direction == "up":
        event = (raw.shift(1) <= smooth.shift(1)) & (raw > smooth)
    else:
        event = (raw.shift(1) >= smooth.shift(1)) & (raw < smooth)
    return event & raw.notna() & smooth.notna()


def _detect_events_single(group: pd.DataFrame, factor: str, condition: EventCondition) -> pd.Series:
    """
    在单个标的、单个因子上，把一个 EventCondition 翻译成事件布尔序列。

    返回值与 group 等长：
        True  表示当天触发该事件；
        False 表示当天没有触发。

    注意这里全部是“事件检测”，不是持仓。主流程会先生成 signals.csv，
    第二步再从信号表还原事件并做开平仓配对。
    """
    s = pd.to_numeric(group[factor], errors="coerce")
    kind = condition.kind
    params = condition.params

    if kind == "lower_turn":
        # 下方拐点：低于指定 sigma 后形成局部谷值，并向上确认。
        event = _turn_event(s, params["level"], params.get("lead", 2), "lower")
    elif kind == "upper_turn":
        # 上方拐点：高于指定 sigma 后形成局部峰值，并向下确认。
        event = _turn_event(s, params["level"], params.get("lead", 2), "upper")
    elif kind == "cross_up":
        # 上穿阈值：昨天在阈值下方或等于阈值，今天站上阈值。
        event = _cross_up(s, params["level"])
    elif kind == "cross_down":
        # 下穿阈值：昨天在阈值上方或等于阈值，今天跌破阈值。
        event = _cross_down(s, params["level"])
    elif kind == "rise_streak":
        # 连续上升：因子连续 days 天上涨，且只在刚满足时触发。
        event = _streak_event(s, params["days"], "up")
    elif kind == "fall_streak":
        # 连续下降：因子连续 days 天下跌，且只在刚满足时触发。
        event = _streak_event(s, params["days"], "down")
    elif kind == "slope_turn_pos":
        # 斜率转正：当前因子 - days 天前因子，从非正转为正。
        slope = s.diff(params["days"])
        event = (slope > 0) & (slope.shift(1) <= 0)
    elif kind == "slope_turn_neg":
        # 斜率转负：当前因子 - days 天前因子，从非负转为负。
        slope = s.diff(params["days"])
        event = (slope < 0) & (slope.shift(1) >= 0)
    elif kind == "accel_turn_pos":
        # 加速度转正：N 日动量为正，且动量改善速度刚转正。
        event = _acceleration_event(s, params["days"], "up")
    elif kind == "accel_turn_neg":
        # 加速度转负：N 日动量为负，且动量恶化速度刚转负。
        event = _acceleration_event(s, params["days"], "down")
    elif kind == "fast_rise":
        # 累计上升：N 天内因子抬升超过指定 sigma，且只在刚越过阈值时触发。
        event = _fast_change_event(s, params["days"], params["sigma"], "up")
    elif kind == "fast_fall":
        # 累计下降：N 天内因子下跌超过指定 sigma，且只在刚跌破阈值时触发。
        event = _fast_change_event(s, params["days"], params["sigma"], "down")
    elif kind == "range_expansion_up":
        # N 日振幅放大向上：振幅刚放大，且当前值位于窗口上半区。
        event = _range_expansion_event(s, params["days"], params["sigma"], "up")
    elif kind == "range_expansion_down":
        # N 日振幅放大向下：振幅刚放大，且当前值位于窗口下半区。
        event = _range_expansion_event(s, params["days"], params["sigma"], "down")
    elif kind == "low_stay_rebound":
        # 低位持续后回升：连续 N 日低于阈值后，今天重新站上阈值。
        event = _low_stay_rebound_event(s, params["days"], params["level"])
    elif kind == "high_stall_weak":
        # 高位钝化转弱：前 N 天维持在高位阈值以上，今天跌破该阈值。
        event = _high_stall_weak_event(s, params["days"], params["level"])
    elif kind == "high_stay_fall":
        # 高位持续后回落：连续 N 日高于阈值后，今天跌破阈值。
        event = _high_stay_fall_event(s, params["days"], params["level"])
    elif kind == "factor_cross_smooth_up":
        # 原始指标上穿季线/年线：只在原始指标字段上触发。
        event = _factor_smooth_cross_event(group, factor, params["target"], "up")
    elif kind == "factor_cross_smooth_down":
        # 原始指标下穿季线/年线：只在原始指标字段上触发。
        event = _factor_smooth_cross_event(group, factor, params["target"], "down")
    elif kind == "raw_season_year_bull":
        # 原始/季线/年线三周期多头共振：只在原始指标字段上触发。
        event = _raw_season_year_resonance_event(group, factor, "bull")
    elif kind == "raw_season_year_bear":
        # 原始/季线/年线三周期空头共振：只在原始指标字段上触发。
        event = _raw_season_year_resonance_event(group, factor, "bear")
    elif kind == "price_factor_bottom_divergence":
        # 价格因子底背离：价格创窗口新低，但因子没有同步新低并开始回升。
        event = _price_factor_divergence_event(group, factor, params["window"], "bottom")
    elif kind == "price_factor_top_divergence":
        # 价格因子顶背离：价格创窗口新高，但因子没有同步新高并开始回落。
        event = _price_factor_divergence_event(group, factor, params["window"], "top")
    elif kind == "price_factor_sync_up":
        # 价因子同步上升：N 日价格收益和因子变化同时为正。
        event = _price_factor_relation_event(group, factor, "sync_up", days=params["days"])
    elif kind == "price_factor_sync_down":
        # 价因子同步下降：N 日价格收益和因子变化同时为负。
        event = _price_factor_relation_event(group, factor, "sync_down", days=params["days"])
    elif kind == "price_down_factor_up":
        # 价格下跌但因子上升：价格仍弱，因子先修复。
        event = _price_factor_relation_event(group, factor, "price_down_factor_up", days=params["days"])
    elif kind == "price_up_factor_down":
        # 价格上涨但因子下降：价格仍强，因子先转弱。
        event = _price_factor_relation_event(group, factor, "price_up_factor_down", days=params["days"])
    elif kind == "price_breakout_factor_confirm":
        # 价格突破因子确认：价格创新高，因子也偏强。
        event = _price_factor_relation_event(group, factor, "breakout_confirm", window=params["window"])
    elif kind == "price_breakdown_factor_confirm":
        # 价格破位因子确认：价格创新低，因子也偏弱。
        event = _price_factor_relation_event(group, factor, "breakdown_confirm", window=params["window"])
    elif kind == "price_new_low_factor_not_confirm":
        # 价格新低因子未确认：价格创新低，但因子没有同步恶化。
        event = _price_factor_relation_event(group, factor, "new_low_not_confirm", window=params["window"])
    elif kind == "price_new_high_factor_not_confirm":
        # 价格新高因子未确认：价格创新高，但因子没有同步走强。
        event = _price_factor_relation_event(group, factor, "new_high_not_confirm", window=params["window"])
    elif kind == "ma_cross_up":
        # 上穿均线：因子从自身 window 日均线下方上穿到均线上方。
        ma = s.rolling(params["window"], min_periods=params["window"]).mean()
        event = (s.shift(1) <= ma.shift(1)) & (s > ma)
    elif kind == "ma_cross_down":
        # 下穿均线：因子从自身 window 日均线上方下穿到均线下方。
        ma = s.rolling(params["window"], min_periods=params["window"]).mean()
        event = (s.shift(1) >= ma.shift(1)) & (s < ma)
    elif kind == "low_reversion":
        # 低位均值回复：昨天还处于过去 window 日低分位附近，今天开始回升。
        q = s.rolling(params["window"], min_periods=params["window"]).quantile(params["quantile"])
        event = (s.shift(1) <= q.shift(1)) & (s > s.shift(1))
    elif kind == "high_reversion":
        # 高位均值回复：昨天还处于过去 window 日高分位附近，今天开始回落。
        q = s.rolling(params["window"], min_periods=params["window"]).quantile(params["quantile"])
        event = (s.shift(1) >= q.shift(1)) & (s < s.shift(1))
    elif kind == "process_signal_open":
        # 复用原 process_signal：peak_signal == 1 作为买点事件。
        event = process_signal(group, score_col=factor, threshold=params.get("threshold", 0.5))["peak_signal"].eq(1)
    elif kind == "process_signal_close":
        # 复用原 process_signal：peak_signal == -1 作为卖点事件。
        event = process_signal(group, score_col=factor, threshold=params.get("threshold", 0.5))["peak_signal"].eq(-1)
    elif kind == "cross_factor_resonance":
        event = _cross_factor_resonance_event(group, factor, condition.side, params)
    elif condition.requires_position:
        # 止损、止盈、持仓满 N 日这类规则必须知道入场价或持仓天数，
        # 不能单独从因子序列检测，所以这里先返回全 False。
        # 真正触发逻辑在 _should_dynamic_close 里处理。
        event = pd.Series(False, index=group.index)
    else:
        raise ValueError(f"Unknown condition kind: {kind}")

    # 保证返回 index 与原 group 对齐，并统一成 bool，方便后续缓存和配对。
    return event.reindex(group.index, fill_value=False).astype(bool)


def detect_events(df: pd.DataFrame, factor: str, condition: EventCondition) -> pd.Series:
    """对全市场数据按代码分组检测事件，再拼回原始 df 的索引顺序。"""
    events = []
    for _, group in df.groupby(CODE_COL, sort=False):
        events.append(_detect_events_single(group, factor, condition))
    if not events:
        return pd.Series(False, index=df.index)
    return pd.concat(events).reindex(df.index, fill_value=False).astype(bool)


def generate_signal_table(
    df: pd.DataFrame,
    factors: Iterable[str] | None = None,
    conditions: dict[str, list[EventCondition]] | None = None,
) -> pd.DataFrame:
    """
    第一步：生成通用信号长表。

    输出列固定为：
        date, instrument, instrument_name, factor, pattern, signal

    signal:
        1  表示该 pattern 在当天触发开仓事件；
        -1 表示该 pattern 在当天触发闭仓事件。

    这个函数只生成“可以单独从因子序列判断”的事件信号。
    止损、止盈、持仓满 N 日这类规则依赖入场价或持仓天数，
    需要在第二步回测中根据持仓动态判断，因此不会出现在这里。
    """
    selected_factors = list(factors) if factors is not None else get_factor_columns(df)
    rule_sets = conditions if conditions is not None else build_event_conditions()
    static_conditions = rule_sets["open"] + [
        condition for condition in rule_sets["close"] if not condition.requires_position
    ]

    rows = []
    for factor in selected_factors:
        for code, group in df.groupby(CODE_COL, sort=False):
            group = group.reset_index(drop=True)
            for condition in static_conditions:
                events = _detect_events_single(group, factor, condition).to_numpy(dtype=bool)
                hit_idx = np.flatnonzero(events)
                if len(hit_idx) == 0:
                    continue
                signal_value = 1 if condition.side == "open" else -1
                hit_rows = group.iloc[hit_idx]
                rows.append(
                    pd.DataFrame(
                        {
                            SIGNAL_DATE_COL: hit_rows[DATE_COL].to_numpy(),
                            SIGNAL_INSTRUMENT_COL: code,
                            SIGNAL_NAME_COL: hit_rows[NAME_COL].to_numpy(),
                            SIGNAL_FACTOR_COL: factor,
                            SIGNAL_PATTERN_COL: condition.name,
                            SIGNAL_VALUE_COL: signal_value,
                        }
                    )
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                SIGNAL_DATE_COL,
                SIGNAL_INSTRUMENT_COL,
                SIGNAL_NAME_COL,
                SIGNAL_FACTOR_COL,
                SIGNAL_PATTERN_COL,
                SIGNAL_VALUE_COL,
            ]
        )

    signal_table = pd.concat(rows, ignore_index=True)
    return signal_table.sort_values(
        [SIGNAL_FACTOR_COL, SIGNAL_INSTRUMENT_COL, SIGNAL_DATE_COL, SIGNAL_PATTERN_COL]
    ).reset_index(drop=True)


def save_signal_table(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    factors: Iterable[str] | None = None,
    conditions: dict[str, list[EventCondition]] | None = None,
) -> pd.DataFrame:
    signal_table = generate_signal_table(df, factors=factors, conditions=conditions)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_table(signal_table, output_path / "signals.csv")
    return signal_table


def _build_factor_event_cache_from_signal_table(
    df: pd.DataFrame,
    factor: str,
    conditions: dict[str, list[EventCondition]],
    signal_table: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    从第一步生成的信号长表还原成第二步回测使用的事件缓存。

    输入：
        df:
            原始行情和因子数据，至少包含 日期、代码、指数名称、收盘价。
            这里用它提供完整交易日序列、价格和标的信息。
        factor:
            当前要回测的因子名。函数只会读取 signal_table 中这个因子的信号。
        conditions:
            build_event_conditions() 返回的规则字典，用来知道有哪些合法 pattern。
            其中 requires_position=True 的动态闭仓规则不会从信号表还原，
            因为止损、止盈、持仓满 N 日需要在持仓后动态判断。
        signal_table:
            第一步生成的信号表，标准列为：
            date, instrument, instrument_name, factor, pattern, signal。

    输出：
        list[dict[str, Any]]
            每个元素对应一个 instrument，结构类似：
            {
                "代码": "000985.CSI",
                "group": 该 instrument 的原始 df 子表，已按日期升序并重置 index,
                "prices": 收盘价 numpy 数组,
                "dates": 日期 numpy 数组,
                "events": {
                    "开仓_上穿_-1sigma": bool 数组,
                    "闭仓_下穿_0sigma": bool 数组,
                    ...
                },
            }

    用途：
        第一阶段的 signal_table 是方便查看和落库的长表；
        第二阶段回测需要快速按日期索引判断某个 pattern 是否触发。
        这个函数就是两者之间的格式转换层，让回测不再关心每个
        pattern 背后到底是 _turn_event、均线规则还是 process_signal。
    """
    all_conditions = conditions["open"] + [
        condition for condition in conditions["close"] if not condition.requires_position
    ]
    known_patterns = {condition.name for condition in all_conditions}
    factor_signals = signal_table[signal_table[SIGNAL_FACTOR_COL].eq(factor)].copy()
    if not factor_signals.empty:
        factor_signals[SIGNAL_DATE_COL] = pd.to_datetime(factor_signals[SIGNAL_DATE_COL])
        factor_signals = factor_signals[factor_signals[SIGNAL_PATTERN_COL].isin(known_patterns)]

    cache = []
    for code, group in df.groupby(CODE_COL, sort=False):
        group = group.reset_index(drop=True)
        group = group.copy()
        group[DATE_COL] = pd.to_datetime(group[DATE_COL])
        events = {condition.name: np.zeros(len(group), dtype=bool) for condition in all_conditions}
        code_signals = factor_signals[factor_signals[SIGNAL_INSTRUMENT_COL].eq(code)]
        if not code_signals.empty:
            date_to_idx = pd.Series(group.index.to_numpy(), index=group[DATE_COL]).to_dict()
            for row in code_signals.itertuples(index=False):
                pattern = getattr(row, SIGNAL_PATTERN_COL)
                dt = pd.Timestamp(getattr(row, SIGNAL_DATE_COL))
                idx = date_to_idx.get(dt)
                if idx is not None and pattern in events:
                    events[pattern][idx] = True

        cache.append(
            {
                CODE_COL: code,
                "group": group,
                "prices": group[PRICE_COL].to_numpy(dtype=float),
                "dates": group[DATE_COL].to_numpy(),
                "events": events,
            }
        )
    return cache


