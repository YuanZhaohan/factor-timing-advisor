from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from io_utils import read_table, write_table
from momentum_context import (
    CROSS_ABOVE_MID_BUCKET,
    CROSS_BELOW_MID_BUCKET,
    DEFAULT_BLACKLIST_EVIDENCE_MODE,
    PRICE_BB_STD,
    PRICE_BB_WINDOW,
    SYNC_WINDOW,
    UNKNOWN_BUCKET,
    build_monthly_momentum_context_blacklist,
    build_monthly_momentum_context_blacklist_filter,
    build_monthly_momentum_context_filter,
    build_monthly_momentum_context_whitelist,
    build_signal_context_table,
)
from signal_generation import process_signal
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
    TRADING_DAYS,
)


DEFAULT_LOOKBACKS = (20, 63, 126, 252)
DEFAULT_SMOOTHS = (1, 5, 10)
DEFAULT_THRESHOLDS = (0.5, 1.0, 1.5)
DEFAULT_MARGINS = (0.0, 0.5)
DEFAULT_TURN_WINDOWS = (2, 3, 5)
PRICE_BUCKET_COL = "price_band_bucket"
DOMAIN_BUCKETS = (
    CROSS_ABOVE_MID_BUCKET,
    CROSS_BELOW_MID_BUCKET,
)
DEFAULT_CHILD_SIGNAL_HISTORY_YEARS = 3.0
DEFAULT_MIN_CHILD_SIGNAL_EVENTS_3Y = 12
CHILD_SIGNAL_KEY_COLS = [
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_PATTERN_COL,
    "event_side",
    PRICE_BUCKET_COL,
]


@dataclass(frozen=True)
class SignalCountRulePair:
    rule_family: str
    lookback: int
    smooth: int
    entry_threshold: float
    exit_threshold: float
    margin: float | None = None
    turn_window: int | None = None

    @property
    def key(self) -> str:
        margin_text = "" if self.margin is None else f"_m{self.margin:g}"
        turn_text = "" if self.turn_window is None else f"_tw{self.turn_window:g}"
        return (
            f"{self.rule_family}_lb{self.lookback}_sm{self.smooth}"
            f"_open{self.entry_threshold:g}_close{self.exit_threshold:g}{margin_text}{turn_text}"
        )

    @property
    def open_rule(self) -> str:
        if self.rule_family == "independent_z":
            return f"open_z>={self.entry_threshold:g}"
        if self.rule_family == "spread_z":
            return f"open_z-close_z>={self.entry_threshold:g}"
        if self.rule_family == "dominance_z":
            return f"open_z>={self.entry_threshold:g} and open_z-close_z>={self.margin:g}"
        if self.rule_family == "turn_up_z":
            return f"open_z turn_up tw{self.turn_window:g} and open_z>={self.entry_threshold:g}"
        if self.rule_family == "cross_ma_z":
            return f"open_z cross_above_ma tw{self.turn_window:g} and open_z>={self.entry_threshold:g}"
        if self.rule_family == "spread_turn_z":
            return f"spread_z turn_up tw{self.turn_window:g} and spread_z>={self.entry_threshold:g}"
        if self.rule_family == "reversal_after_extreme_z":
            return f"open_z recovers after <=-{self.entry_threshold:g} in tw{self.turn_window:g}"
        if self.rule_family == "process_signal_open_z":
            return f"process_signal(open_z)==1 lead{self.turn_window:g} threshold{self.entry_threshold:g}"
        if self.rule_family == "process_signal_close_z":
            return f"process_signal(close_z)==-1 lead{self.turn_window:g} threshold{self.entry_threshold:g}"
        if self.rule_family == "process_signal_pair_z":
            return (
                f"process_signal(open_z)==1 or process_signal(close_z)==-1 "
                f"lead{self.turn_window:g} threshold{self.entry_threshold:g}"
            )
        if self.rule_family == "process_signal_spread_z":
            return f"process_signal(spread_z)==1 lead{self.turn_window:g} threshold{self.entry_threshold:g}"
        return self.rule_family

    @property
    def close_rule(self) -> str:
        if self.rule_family == "independent_z":
            return f"close_z>={self.exit_threshold:g}"
        if self.rule_family == "spread_z":
            return f"open_z-close_z<=-{self.exit_threshold:g}"
        if self.rule_family == "dominance_z":
            return f"close_z>={self.exit_threshold:g} and close_z-open_z>={self.margin:g}"
        if self.rule_family == "turn_up_z":
            return f"close_z turn_up tw{self.turn_window:g} and close_z>={self.exit_threshold:g}"
        if self.rule_family == "cross_ma_z":
            return f"close_z cross_above_ma tw{self.turn_window:g} and close_z>={self.exit_threshold:g}"
        if self.rule_family == "spread_turn_z":
            return f"spread_z turn_down tw{self.turn_window:g} and spread_z<=-{self.exit_threshold:g}"
        if self.rule_family == "reversal_after_extreme_z":
            return f"close_z recovers after <=-{self.exit_threshold:g} in tw{self.turn_window:g}"
        if self.rule_family == "process_signal_open_z":
            return f"process_signal(open_z)==-1 lead{self.turn_window:g} threshold{self.exit_threshold:g}"
        if self.rule_family == "process_signal_close_z":
            return f"process_signal(close_z)==1 lead{self.turn_window:g} threshold{self.exit_threshold:g}"
        if self.rule_family == "process_signal_pair_z":
            return (
                f"process_signal(close_z)==1 or process_signal(open_z)==-1 "
                f"lead{self.turn_window:g} threshold{self.exit_threshold:g}"
            )
        if self.rule_family == "process_signal_spread_z":
            return f"process_signal(spread_z)==-1 lead{self.turn_window:g} threshold{self.exit_threshold:g}"
        return self.rule_family


def generate_signal_count_rule_pairs(
    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
    smooths: Iterable[int] = DEFAULT_SMOOTHS,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    margins: Iterable[float] = DEFAULT_MARGINS,
    turn_windows: Iterable[int] = DEFAULT_TURN_WINDOWS,
    include_turning: bool = False,
) -> list[SignalCountRulePair]:
    rules: list[SignalCountRulePair] = []
    for lookback in lookbacks:
        for smooth in smooths:
            for entry_threshold in thresholds:
                for exit_threshold in thresholds:
                    rules.append(
                        SignalCountRulePair(
                            rule_family="independent_z",
                            lookback=int(lookback),
                            smooth=int(smooth),
                            entry_threshold=float(entry_threshold),
                            exit_threshold=float(exit_threshold),
                        )
                    )
                    rules.append(
                        SignalCountRulePair(
                            rule_family="spread_z",
                            lookback=int(lookback),
                            smooth=int(smooth),
                            entry_threshold=float(entry_threshold),
                            exit_threshold=float(exit_threshold),
                        )
                    )
                    for margin in margins:
                        rules.append(
                            SignalCountRulePair(
                                rule_family="dominance_z",
                                lookback=int(lookback),
                                smooth=int(smooth),
                                entry_threshold=float(entry_threshold),
                                exit_threshold=float(exit_threshold),
                                margin=float(margin),
                        )
                    )
                    if include_turning:
                        for turn_window in turn_windows:
                            for family in (
                                "turn_up_z",
                                "cross_ma_z",
                                "spread_turn_z",
                                "reversal_after_extreme_z",
                                "process_signal_open_z",
                                "process_signal_close_z",
                                "process_signal_pair_z",
                                "process_signal_spread_z",
                            ):
                                rules.append(
                                    SignalCountRulePair(
                                        rule_family=family,
                                        lookback=int(lookback),
                                        smooth=int(smooth),
                                        entry_threshold=float(entry_threshold),
                                        exit_threshold=float(exit_threshold),
                                        turn_window=int(turn_window),
                                    )
                                )
    return rules


def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    min_periods = min(int(lookback), max(20, int(lookback) // 3))
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(int(lookback), min_periods=min_periods).mean()
    std = values.rolling(int(lookback), min_periods=min_periods).std(ddof=0)
    z = (values - mean) / std.replace(0.0, np.nan)
    z = z.mask(std.eq(0.0) & mean.notna(), 0.0)
    return z.replace([np.inf, -np.inf], np.nan)


def aggregate_daily_signal_counts(signal_table: pd.DataFrame) -> pd.DataFrame:
    if signal_table.empty:
        return pd.DataFrame(columns=[SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, SIGNAL_NAME_COL, "open_count", "close_count"])

    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])
    signals[SIGNAL_INSTRUMENT_COL] = signals[SIGNAL_INSTRUMENT_COL].astype(str)
    signals[SIGNAL_VALUE_COL] = pd.to_numeric(signals[SIGNAL_VALUE_COL], errors="coerce")
    signals["event_side"] = np.select(
        [signals[SIGNAL_VALUE_COL].eq(1), signals[SIGNAL_VALUE_COL].eq(-1)],
        ["open", "close"],
        default="other",
    )
    signals = signals[signals["event_side"].isin(["open", "close"])].copy()
    if signals.empty:
        return pd.DataFrame(columns=[SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, SIGNAL_NAME_COL, "open_count", "close_count"])

    names = (
        signals[[SIGNAL_INSTRUMENT_COL, SIGNAL_NAME_COL]]
        .dropna()
        .drop_duplicates(subset=[SIGNAL_INSTRUMENT_COL], keep="last")
    )
    counts = (
        signals.groupby([SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, "event_side"], observed=True)
        .size()
        .unstack("event_side", fill_value=0)
        .reset_index()
    )
    for col in ["open", "close"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.rename(columns={"open": "open_count", "close": "close_count"})
    counts = counts.merge(names, on=SIGNAL_INSTRUMENT_COL, how="left")
    return counts[[SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, SIGNAL_NAME_COL, "open_count", "close_count"]]


def aggregate_daily_signal_counts_by_bucket(
    signal_context: pd.DataFrame,
    bucket_col: str = PRICE_BUCKET_COL,
    drop_unknown: bool = True,
) -> pd.DataFrame:
    columns = [
        SIGNAL_DATE_COL,
        SIGNAL_INSTRUMENT_COL,
        SIGNAL_NAME_COL,
        bucket_col,
        "open_count",
        "close_count",
    ]
    if signal_context.empty or bucket_col not in signal_context.columns:
        return pd.DataFrame(columns=columns)

    signals = signal_context.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL], errors="coerce")
    signals[SIGNAL_INSTRUMENT_COL] = signals[SIGNAL_INSTRUMENT_COL].astype(str)
    signals[SIGNAL_VALUE_COL] = pd.to_numeric(signals[SIGNAL_VALUE_COL], errors="coerce")
    signals[bucket_col] = signals[bucket_col].fillna(UNKNOWN_BUCKET).astype(str)
    if drop_unknown:
        signals = signals[signals[bucket_col].ne(UNKNOWN_BUCKET)].copy()
    signals["event_side"] = np.select(
        [signals[SIGNAL_VALUE_COL].eq(1), signals[SIGNAL_VALUE_COL].eq(-1)],
        ["open", "close"],
        default="other",
    )
    signals = signals[signals["event_side"].isin(["open", "close"])].copy()
    if signals.empty:
        return pd.DataFrame(columns=columns)

    names = (
        signals[[SIGNAL_INSTRUMENT_COL, SIGNAL_NAME_COL]]
        .dropna()
        .drop_duplicates(subset=[SIGNAL_INSTRUMENT_COL], keep="last")
    )
    counts = (
        signals.groupby([SIGNAL_DATE_COL, SIGNAL_INSTRUMENT_COL, bucket_col, "event_side"], observed=True)
        .size()
        .unstack("event_side", fill_value=0)
        .reset_index()
    )
    for col in ["open", "close"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.rename(columns={"open": "open_count", "close": "close_count"})
    counts = counts.merge(names, on=SIGNAL_INSTRUMENT_COL, how="left")
    return counts[columns]


def align_signal_counts_to_price(
    price_df: pd.DataFrame,
    signal_table: pd.DataFrame,
    source: str,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    prices = price_df[[DATE_COL, CODE_COL, NAME_COL, PRICE_COL]].copy()
    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL])
    prices[CODE_COL] = prices[CODE_COL].astype(str)
    prices = prices.sort_values([CODE_COL, DATE_COL]).drop_duplicates([CODE_COL, DATE_COL])
    if start_date is not None:
        prices = prices[prices[DATE_COL].ge(pd.Timestamp(start_date))].copy()

    counts = aggregate_daily_signal_counts(signal_table)
    if not counts.empty:
        counts = counts.rename(columns={SIGNAL_DATE_COL: DATE_COL, SIGNAL_INSTRUMENT_COL: CODE_COL})
        counts[DATE_COL] = pd.to_datetime(counts[DATE_COL])
        counts[CODE_COL] = counts[CODE_COL].astype(str)
        counts = counts[[DATE_COL, CODE_COL, "open_count", "close_count"]]

    aligned = prices.merge(counts, on=[DATE_COL, CODE_COL], how="left")
    aligned[["open_count", "close_count"]] = aligned[["open_count", "close_count"]].fillna(0.0)
    aligned["open_count"] = pd.to_numeric(aligned["open_count"], errors="coerce").fillna(0.0)
    aligned["close_count"] = pd.to_numeric(aligned["close_count"], errors="coerce").fillna(0.0)
    aligned["net_count"] = aligned["open_count"] - aligned["close_count"]
    aligned["signal_source"] = source
    return aligned


def align_bucket_signal_counts_to_price(
    price_df: pd.DataFrame,
    signal_context: pd.DataFrame,
    source: str = "bucket_raw",
    bucket_col: str = PRICE_BUCKET_COL,
    start_date: str | pd.Timestamp | None = None,
    drop_unknown: bool = True,
) -> pd.DataFrame:
    prices = price_df[[DATE_COL, CODE_COL, NAME_COL, PRICE_COL]].copy()
    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL])
    prices[CODE_COL] = prices[CODE_COL].astype(str)
    prices = prices.sort_values([CODE_COL, DATE_COL]).drop_duplicates([CODE_COL, DATE_COL])
    if start_date is not None:
        prices = prices[prices[DATE_COL].ge(pd.Timestamp(start_date))].copy()

    counts = aggregate_daily_signal_counts_by_bucket(
        signal_context,
        bucket_col=bucket_col,
        drop_unknown=drop_unknown,
    )
    columns = list(prices.columns) + [bucket_col, "open_count", "close_count", "net_count", "signal_source"]
    if counts.empty:
        return pd.DataFrame(columns=columns)

    counts = counts.rename(columns={SIGNAL_DATE_COL: DATE_COL, SIGNAL_INSTRUMENT_COL: CODE_COL})
    counts[DATE_COL] = pd.to_datetime(counts[DATE_COL])
    counts[CODE_COL] = counts[CODE_COL].astype(str)
    counts[bucket_col] = counts[bucket_col].astype(str)
    bucket_map = counts[[CODE_COL, bucket_col]].drop_duplicates()
    skeleton = prices.merge(bucket_map, on=CODE_COL, how="inner")
    aligned = skeleton.merge(
        counts[[DATE_COL, CODE_COL, bucket_col, "open_count", "close_count"]],
        on=[DATE_COL, CODE_COL, bucket_col],
        how="left",
    )
    aligned[["open_count", "close_count"]] = aligned[["open_count", "close_count"]].fillna(0.0)
    aligned["open_count"] = pd.to_numeric(aligned["open_count"], errors="coerce").fillna(0.0)
    aligned["close_count"] = pd.to_numeric(aligned["close_count"], errors="coerce").fillna(0.0)
    aligned["net_count"] = aligned["open_count"] - aligned["close_count"]
    aligned["signal_source"] = source
    return aligned[columns]


def build_signal_count_inputs(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    filtered_signals: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if start_date is None and not filtered_signals.empty:
        start_date = pd.to_datetime(filtered_signals[SIGNAL_DATE_COL]).min()
    frames = [
        align_signal_counts_to_price(price_df, raw_signals, source="raw", start_date=start_date),
        align_signal_counts_to_price(price_df, filtered_signals, source="filtered", start_date=start_date),
    ]
    return pd.concat(frames, ignore_index=True)


def _child_signal_id(frame: pd.DataFrame, key_cols: list[str] = CHILD_SIGNAL_KEY_COLS) -> pd.Series:
    key = frame[key_cols[0]].astype(str)
    for col in key_cols[1:]:
        key = key + "|" + frame[col].astype(str)
    return key


def _attach_trailing_child_signal_counts(
    signal_context: pd.DataFrame,
    history_years: float = DEFAULT_CHILD_SIGNAL_HISTORY_YEARS,
    bucket_col: str = PRICE_BUCKET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_context.empty:
        return signal_context.copy(), pd.DataFrame()

    key_cols = [
        SIGNAL_INSTRUMENT_COL,
        SIGNAL_FACTOR_COL,
        SIGNAL_PATTERN_COL,
        "event_side",
        bucket_col,
    ]
    data = signal_context.copy()
    data[SIGNAL_DATE_COL] = pd.to_datetime(data[SIGNAL_DATE_COL], errors="coerce")
    data = data.dropna(subset=[SIGNAL_DATE_COL])
    data[SIGNAL_INSTRUMENT_COL] = data[SIGNAL_INSTRUMENT_COL].astype(str)
    data[bucket_col] = data[bucket_col].fillna(UNKNOWN_BUCKET).astype(str)
    data = data[data[bucket_col].ne(UNKNOWN_BUCKET)].copy()
    if data.empty:
        return data, pd.DataFrame()

    daily = (
        data.groupby(key_cols + [SIGNAL_DATE_COL], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "child_signal_daily_count"})
        .sort_values(key_cols + [SIGNAL_DATE_COL])
        .reset_index(drop=True)
    )
    history_days = max(int(round(float(history_years) * 365.25)), 1)

    pieces: list[pd.DataFrame] = []
    for _, group in daily.groupby(key_cols, dropna=False, sort=False):
        group = group.sort_values(SIGNAL_DATE_COL).copy()
        dates = group[SIGNAL_DATE_COL].to_numpy(dtype="datetime64[ns]")
        counts = group["child_signal_daily_count"].to_numpy(dtype=float)
        window_starts = dates - np.timedelta64(history_days, "D")
        left = np.searchsorted(dates, window_starts, side="left")
        right = np.arange(len(group))
        cumulative = np.concatenate([[0.0], np.cumsum(counts)])
        group["prior_child_signal_count_3y"] = cumulative[right] - cumulative[left]
        pieces.append(group)

    daily_counts = pd.concat(pieces, ignore_index=True) if pieces else daily.iloc[0:0].copy()
    data = data.merge(
        daily_counts[key_cols + [SIGNAL_DATE_COL, "child_signal_daily_count", "prior_child_signal_count_3y"]],
        on=key_cols + [SIGNAL_DATE_COL],
        how="left",
    )
    data["child_signal_id"] = _child_signal_id(data, key_cols)
    daily_counts["child_signal_id"] = _child_signal_id(daily_counts, key_cols)
    return data, daily_counts


def build_domain_child_signal_context(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    history_years: float = DEFAULT_CHILD_SIGNAL_HISTORY_YEARS,
    min_child_signal_events: int = DEFAULT_MIN_CHILD_SIGNAL_EVENTS_3Y,
    bucket_col: str = PRICE_BUCKET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_context = build_signal_context_table(price_df, raw_signals)
    if signal_context.empty:
        return signal_context, pd.DataFrame(), pd.DataFrame()

    context, daily_counts = _attach_trailing_child_signal_counts(
        signal_context,
        history_years=history_years,
        bucket_col=bucket_col,
    )
    if context.empty:
        return context, daily_counts, pd.DataFrame()

    data_start = pd.to_datetime(price_df[DATE_COL], errors="coerce").dropna().min()
    history_days = max(int(round(float(history_years) * 365.25)), 1)
    min_history_date = pd.Timestamp(data_start) + pd.Timedelta(days=history_days) if pd.notna(data_start) else pd.NaT
    context["child_signal_history_years"] = (
        (context[SIGNAL_DATE_COL] - pd.Timestamp(data_start)).dt.days / 365.25 if pd.notna(data_start) else np.nan
    )
    context["min_child_signal_events_3y"] = int(min_child_signal_events)
    context["valid_child_signal"] = (
        context[SIGNAL_DATE_COL].ge(min_history_date)
        & pd.to_numeric(context["prior_child_signal_count_3y"], errors="coerce").ge(int(min_child_signal_events))
    )

    key_base_cols = [
        SIGNAL_INSTRUMENT_COL,
        SIGNAL_FACTOR_COL,
        SIGNAL_PATTERN_COL,
        "event_side",
    ]
    base_defs = context[key_base_cols].drop_duplicates().reset_index(drop=True)
    bucket_defs = pd.DataFrame({bucket_col: list(DOMAIN_BUCKETS)})
    base_defs["_join_key"] = 1
    bucket_defs["_join_key"] = 1
    universe = base_defs.merge(bucket_defs, on="_join_key", how="inner").drop(columns="_join_key")
    universe["child_signal_id"] = _child_signal_id(universe, key_base_cols + [bucket_col])

    latest_date = context[SIGNAL_DATE_COL].max()
    recent_start = pd.Timestamp(latest_date) - pd.Timedelta(days=history_days)
    recent_counts = (
        context[context[SIGNAL_DATE_COL].gt(recent_start) & context[SIGNAL_DATE_COL].le(latest_date)]
        .groupby(key_base_cols + [bucket_col], dropna=False)
        .size()
        .rename("observed_child_signal_count_3y")
        .reset_index()
    )
    valid_latest = (
        context[context["valid_child_signal"]]
        .groupby(key_base_cols + [bucket_col], dropna=False)
        .size()
        .rename("valid_event_count_all_history")
        .reset_index()
    )
    universe = universe.merge(recent_counts, on=key_base_cols + [bucket_col], how="left")
    universe = universe.merge(valid_latest, on=key_base_cols + [bucket_col], how="left")
    universe["observed_child_signal_count_3y"] = universe["observed_child_signal_count_3y"].fillna(0).astype(int)
    universe["valid_event_count_all_history"] = universe["valid_event_count_all_history"].fillna(0).astype(int)
    universe["valid_child_signal_latest"] = universe["observed_child_signal_count_3y"].ge(int(min_child_signal_events))
    universe["history_years"] = float(history_years)
    universe["min_child_signal_events_3y"] = int(min_child_signal_events)
    return context, daily_counts, universe


def _attach_standardized_counts(group: pd.DataFrame, lookback: int, smooth: int) -> pd.DataFrame:
    work = group.sort_values(DATE_COL).reset_index(drop=True).copy()
    work["open_z"] = _rolling_zscore(work["open_count"], lookback).rolling(int(smooth), min_periods=1).mean()
    work["close_z"] = _rolling_zscore(work["close_count"], lookback).rolling(int(smooth), min_periods=1).mean()
    work["spread_z"] = work["open_z"] - work["close_z"]
    return work


def _turn_up(series: pd.Series, window: int) -> pd.Series:
    delta = series - series.shift(int(window))
    return delta.gt(0.0) & delta.shift(1).le(0.0)


def _turn_down(series: pd.Series, window: int) -> pd.Series:
    delta = series - series.shift(int(window))
    return delta.lt(0.0) & delta.shift(1).ge(0.0)


def _cross_above_ma(series: pd.Series, window: int) -> pd.Series:
    ma = series.rolling(int(window), min_periods=max(2, int(window))).mean()
    return series.ge(ma) & series.shift(1).lt(ma.shift(1))


def _recover_after_extreme(series: pd.Series, threshold: float, window: int) -> pd.Series:
    prior_min = series.shift(1).rolling(int(window), min_periods=1).min()
    delta = series - series.shift(1)
    return prior_min.le(-float(threshold)) & delta.gt(0.0)


def _process_signal_peaks(series: pd.Series, threshold: float, lead: int) -> pd.Series:
    temp = pd.DataFrame({"score": pd.to_numeric(series, errors="coerce")})
    peaks = process_signal(
        temp,
        lead=int(lead),
        score_col="score",
        threshold=float(threshold),
    )["peak_signal"]
    return pd.to_numeric(peaks, errors="coerce")


def _detect_rule_events(work: pd.DataFrame, rule: SignalCountRulePair) -> tuple[pd.Series, pd.Series]:
    if rule.rule_family == "independent_z":
        open_event = work["open_z"].ge(rule.entry_threshold)
        close_event = work["close_z"].ge(rule.exit_threshold)
    elif rule.rule_family == "spread_z":
        open_event = work["spread_z"].ge(rule.entry_threshold)
        close_event = work["spread_z"].le(-rule.exit_threshold)
    elif rule.rule_family == "dominance_z":
        margin = float(rule.margin or 0.0)
        open_event = work["open_z"].ge(rule.entry_threshold) & work["open_z"].sub(work["close_z"]).ge(margin)
        close_event = work["close_z"].ge(rule.exit_threshold) & work["close_z"].sub(work["open_z"]).ge(margin)
    elif rule.rule_family == "turn_up_z":
        window = int(rule.turn_window or 3)
        open_event = _turn_up(work["open_z"], window) & work["open_z"].ge(rule.entry_threshold)
        close_event = _turn_up(work["close_z"], window) & work["close_z"].ge(rule.exit_threshold)
    elif rule.rule_family == "cross_ma_z":
        window = int(rule.turn_window or 3)
        open_event = _cross_above_ma(work["open_z"], window) & work["open_z"].ge(rule.entry_threshold)
        close_event = _cross_above_ma(work["close_z"], window) & work["close_z"].ge(rule.exit_threshold)
    elif rule.rule_family == "spread_turn_z":
        window = int(rule.turn_window or 3)
        open_event = _turn_up(work["spread_z"], window) & work["spread_z"].ge(rule.entry_threshold)
        close_event = _turn_down(work["spread_z"], window) & work["spread_z"].le(-rule.exit_threshold)
    elif rule.rule_family == "reversal_after_extreme_z":
        window = int(rule.turn_window or 3)
        open_event = _recover_after_extreme(work["open_z"], rule.entry_threshold, window)
        close_event = _recover_after_extreme(work["close_z"], rule.exit_threshold, window)
    elif rule.rule_family == "process_signal_open_z":
        lead = int(rule.turn_window or 2)
        open_peaks = _process_signal_peaks(work["open_z"], rule.entry_threshold, lead)
        close_peaks = _process_signal_peaks(work["open_z"], rule.exit_threshold, lead)
        open_event = open_peaks.eq(1.0)
        close_event = close_peaks.eq(-1.0)
    elif rule.rule_family == "process_signal_close_z":
        lead = int(rule.turn_window or 2)
        open_peaks = _process_signal_peaks(work["close_z"], rule.entry_threshold, lead)
        close_peaks = _process_signal_peaks(work["close_z"], rule.exit_threshold, lead)
        open_event = open_peaks.eq(-1.0)
        close_event = close_peaks.eq(1.0)
    elif rule.rule_family == "process_signal_pair_z":
        lead = int(rule.turn_window or 2)
        open_z_entry_peaks = _process_signal_peaks(work["open_z"], rule.entry_threshold, lead)
        close_z_entry_peaks = _process_signal_peaks(work["close_z"], rule.entry_threshold, lead)
        close_z_exit_peaks = _process_signal_peaks(work["close_z"], rule.exit_threshold, lead)
        open_z_exit_peaks = _process_signal_peaks(work["open_z"], rule.exit_threshold, lead)
        open_event = open_z_entry_peaks.eq(1.0) | close_z_entry_peaks.eq(-1.0)
        close_event = close_z_exit_peaks.eq(1.0) | open_z_exit_peaks.eq(-1.0)
    elif rule.rule_family == "process_signal_spread_z":
        lead = int(rule.turn_window or 2)
        open_peaks = _process_signal_peaks(work["spread_z"], rule.entry_threshold, lead)
        close_peaks = _process_signal_peaks(work["spread_z"], rule.exit_threshold, lead)
        open_event = open_peaks.eq(1.0)
        close_event = close_peaks.eq(-1.0)
    else:
        raise ValueError(f"Unknown rule family: {rule.rule_family}")
    return open_event.fillna(False), close_event.fillna(False)


def _build_position(open_events: np.ndarray, close_events: np.ndarray) -> np.ndarray:
    position = np.zeros(len(open_events), dtype=float)
    state = 0.0
    for i in range(1, len(position)):
        if state <= 0 and bool(open_events[i - 1]):
            state = 1.0
        elif state > 0 and bool(close_events[i - 1]):
            state = 0.0
        position[i] = state
    return position


def _annual_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) <= 1 or float(equity.iloc[0]) <= 0 or float(equity.iloc[-1]) <= 0:
        return np.nan
    years = (len(equity) - 1) / TRADING_DAYS
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan


def _max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    vol = float(values.std(ddof=0))
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(values.mean() / vol * np.sqrt(TRADING_DAYS))


def _make_equity_frame(
    work: pd.DataFrame,
    rule: SignalCountRulePair,
    open_events: np.ndarray,
    close_events: np.ndarray,
    position: np.ndarray,
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    excess_equity: pd.Series,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            DATE_COL: work[DATE_COL],
            CODE_COL: work[CODE_COL],
            NAME_COL: work[NAME_COL],
            "signal_source": work["signal_source"],
            "rule_key": rule.key,
            "rule_family": rule.rule_family,
            "open_count": work["open_count"],
            "close_count": work["close_count"],
            "open_z": work["open_z"],
            "close_z": work["close_z"],
            "spread_z": work["spread_z"],
            "open_event": open_events.astype(int),
            "close_event": close_events.astype(int),
            "position": position,
            "strategy_equity": strategy_equity,
            "benchmark_equity": benchmark_equity,
            "excess_equity": excess_equity,
        }
    )
    if PRICE_BUCKET_COL in work.columns:
        result[PRICE_BUCKET_COL] = work[PRICE_BUCKET_COL].iloc[0]
    return result


def backtest_signal_count_rule_pairs(
    signal_count_data: pd.DataFrame,
    rule_pairs: Iterable[SignalCountRulePair] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_count_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    rules = list(rule_pairs) if rule_pairs is not None else generate_signal_count_rule_pairs()
    rule_groups: dict[tuple[int, int], list[SignalCountRulePair]] = {}
    for rule in rules:
        rule_groups.setdefault((int(rule.lookback), int(rule.smooth)), []).append(rule)

    rows: list[dict[str, object]] = []
    best_equity_frames: list[pd.DataFrame] = []
    best_by_source: dict[str, float] = {}

    data = signal_count_data.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    data[CODE_COL] = data[CODE_COL].astype(str)
    data[PRICE_COL] = pd.to_numeric(data[PRICE_COL], errors="coerce")

    group_cols = ["signal_source", CODE_COL]
    if PRICE_BUCKET_COL in data.columns:
        data[PRICE_BUCKET_COL] = data[PRICE_BUCKET_COL].fillna("all").astype(str)
        group_cols.append(PRICE_BUCKET_COL)

    for group_key, group in data.groupby(group_cols, sort=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_map = {col: value for col, value in zip(group_cols, key_values)}
        source = str(key_map["signal_source"])
        code = str(key_map[CODE_COL])
        bucket = str(key_map.get(PRICE_BUCKET_COL, "all"))
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        if group.empty:
            continue
        benchmark_returns = group[PRICE_COL].pct_change().fillna(0.0)
        benchmark_equity = (1.0 + benchmark_returns).cumprod()
        benchmark_annual_return = _annual_return(benchmark_equity)
        benchmark_max_drawdown = _max_drawdown(benchmark_equity)
        benchmark_sharpe = _sharpe(benchmark_returns)

        for (lookback, smooth), grouped_rules in rule_groups.items():
            work = _attach_standardized_counts(group, lookback=lookback, smooth=smooth)
            for rule in grouped_rules:
                open_event, close_event = _detect_rule_events(work, rule)
                open_events = open_event.to_numpy(dtype=bool)
                close_events = close_event.to_numpy(dtype=bool)
                position = _build_position(open_events, close_events)
                strategy_returns = benchmark_returns.reset_index(drop=True) * position
                strategy_equity = (1.0 + strategy_returns).cumprod()
                excess_equity = strategy_equity / benchmark_equity.reset_index(drop=True).replace(0, np.nan)
                trade_count = int(((pd.Series(position).shift(1).fillna(0.0) <= 0) & (pd.Series(position) > 0)).sum())
                summary = {
                    CODE_COL: code,
                    NAME_COL: group[NAME_COL].dropna().iloc[-1] if group[NAME_COL].notna().any() else np.nan,
                    "signal_source": source,
                    PRICE_BUCKET_COL: bucket,
                    "rule_key": rule.key,
                    "rule_family": rule.rule_family,
                    "open_rule": rule.open_rule,
                    "close_rule": rule.close_rule,
                    "lookback": int(rule.lookback),
                    "smooth": int(rule.smooth),
                    "entry_threshold": float(rule.entry_threshold),
                    "exit_threshold": float(rule.exit_threshold),
                    "margin": np.nan if rule.margin is None else float(rule.margin),
                    "annual_return": _annual_return(strategy_equity),
                    "benchmark_annual_return": benchmark_annual_return,
                    "excess_annual_return": _annual_return(excess_equity),
                    "max_drawdown": _max_drawdown(strategy_equity),
                    "benchmark_max_drawdown": benchmark_max_drawdown,
                    "excess_max_drawdown": _max_drawdown(excess_equity),
                    "sharpe": _sharpe(strategy_returns),
                    "benchmark_sharpe": benchmark_sharpe,
                    "holding_ratio": float(np.mean(position)) if len(position) else np.nan,
                    "trade_count": trade_count,
                    "final_equity": float(strategy_equity.iloc[-1]) if len(strategy_equity) else np.nan,
                    "benchmark_final_equity": float(benchmark_equity.iloc[-1]) if len(benchmark_equity) else np.nan,
                    "excess_final_equity": float(excess_equity.dropna().iloc[-1]) if not excess_equity.dropna().empty else np.nan,
                    "open_event_count": int(open_events.sum()),
                    "close_event_count": int(close_events.sum()),
                    "avg_open_count": float(work["open_count"].mean()),
                    "avg_close_count": float(work["close_count"].mean()),
                    "start_date": group[DATE_COL].min(),
                    "end_date": group[DATE_COL].max(),
                }
                rows.append(summary)

                excess = summary["excess_annual_return"]
                current = float(excess) if np.isfinite(excess) else -np.inf
                best_key = "\x1f".join(str(key_map[col]) for col in group_cols)
                if current > best_by_source.get(best_key, -np.inf):
                    best_by_source[best_key] = current
                    best_equity_frames = [
                        item
                        for item in best_equity_frames
                        if "\x1f".join(
                            str(item[col].iloc[0])
                            for col in group_cols
                            if col in item.columns
                        )
                        != best_key
                    ]
                    best_equity_frames.append(
                        _make_equity_frame(
                            work,
                            rule,
                            open_events,
                            close_events,
                            position,
                            strategy_equity,
                            benchmark_equity.reset_index(drop=True),
                            excess_equity,
                        )
                    )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["signal_source", "excess_annual_return", "sharpe", "annual_return"],
            ascending=[True, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    equity_df = pd.concat(best_equity_frames, ignore_index=True) if best_equity_frames else pd.DataFrame()
    return summary_df, equity_df


def compare_filtered_signal_count_rules(rule_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rule_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_cols = [
        "annual_return",
        "excess_annual_return",
        "max_drawdown",
        "excess_max_drawdown",
        "sharpe",
        "holding_ratio",
        "trade_count",
        "final_equity",
        "excess_final_equity",
        "open_event_count",
        "close_event_count",
    ]
    base_cols = [
        CODE_COL,
        NAME_COL,
        "rule_key",
        "rule_family",
        "open_rule",
        "close_rule",
        "lookback",
        "smooth",
        "entry_threshold",
        "exit_threshold",
        "margin",
    ]
    raw = rule_summary[rule_summary["signal_source"].eq("raw")][base_cols + metric_cols].copy()
    filtered = rule_summary[rule_summary["signal_source"].eq("filtered")][base_cols + metric_cols].copy()
    paired = raw.merge(
        filtered,
        on=base_cols,
        how="inner",
        suffixes=("_raw", "_filtered"),
    )
    for col in metric_cols:
        paired[f"{col}_delta"] = paired[f"{col}_filtered"] - paired[f"{col}_raw"]
    paired = paired.sort_values(
        ["excess_annual_return_delta", "sharpe_delta"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    variant_rows: list[dict[str, object]] = []
    for source, group in rule_summary.groupby("signal_source", sort=False):
        best = group.sort_values("excess_annual_return", ascending=False, na_position="last").head(1)
        best_row = best.iloc[0] if not best.empty else pd.Series(dtype=object)
        variant_rows.append(
            {
                "signal_source": source,
                "rule_count": int(len(group)),
                "positive_excess_rule_count": int(pd.to_numeric(group["excess_annual_return"], errors="coerce").gt(0).sum()),
                "positive_excess_rule_share": float(pd.to_numeric(group["excess_annual_return"], errors="coerce").gt(0).mean()),
                "median_excess_annual_return": float(pd.to_numeric(group["excess_annual_return"], errors="coerce").median()),
                "mean_excess_annual_return": float(pd.to_numeric(group["excess_annual_return"], errors="coerce").mean()),
                "best_excess_annual_return": best_row.get("excess_annual_return", np.nan),
                "best_annual_return": best_row.get("annual_return", np.nan),
                "best_sharpe": best_row.get("sharpe", np.nan),
                "best_holding_ratio": best_row.get("holding_ratio", np.nan),
                "best_trade_count": best_row.get("trade_count", np.nan),
                "best_rule_key": best_row.get("rule_key", ""),
                "best_open_rule": best_row.get("open_rule", ""),
                "best_close_rule": best_row.get("close_rule", ""),
            }
        )

    if not paired.empty:
        variant_rows.append(
            {
                "signal_source": "filtered_minus_raw_same_rules",
                "rule_count": int(len(paired)),
                "positive_excess_rule_count": int(pd.to_numeric(paired["excess_annual_return_delta"], errors="coerce").gt(0).sum()),
                "positive_excess_rule_share": float(pd.to_numeric(paired["excess_annual_return_delta"], errors="coerce").gt(0).mean()),
                "median_excess_annual_return": float(pd.to_numeric(paired["excess_annual_return_delta"], errors="coerce").median()),
                "mean_excess_annual_return": float(pd.to_numeric(paired["excess_annual_return_delta"], errors="coerce").mean()),
                "best_excess_annual_return": float(pd.to_numeric(paired["excess_annual_return_delta"], errors="coerce").max()),
                "best_annual_return": float(pd.to_numeric(paired["annual_return_delta"], errors="coerce").max()),
                "best_sharpe": float(pd.to_numeric(paired["sharpe_delta"], errors="coerce").max()),
                "best_holding_ratio": np.nan,
                "best_trade_count": np.nan,
                "best_rule_key": paired.iloc[0].get("rule_key", ""),
                "best_open_rule": paired.iloc[0].get("open_rule", ""),
                "best_close_rule": paired.iloc[0].get("close_rule", ""),
            }
        )
    variant_summary = pd.DataFrame(variant_rows)
    return paired, variant_summary


def summarize_recent_bucket_signal_counts(
    bucket_count_data: pd.DataFrame,
    lookback_days: int = 20,
    bucket_col: str = PRICE_BUCKET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_columns = [
        DATE_COL,
        CODE_COL,
        NAME_COL,
        bucket_col,
        "open_count",
        "close_count",
        "net_count",
    ]
    aggregate_columns = [
        CODE_COL,
        NAME_COL,
        bucket_col,
        "lookback_days",
        "start_date",
        "end_date",
        "open_count",
        "close_count",
        "net_count",
        "open_share",
        "close_share",
    ]
    if bucket_count_data.empty or bucket_col not in bucket_count_data.columns:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=aggregate_columns)

    data = bucket_count_data.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.dropna(subset=[DATE_COL])
    if data.empty:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=aggregate_columns)

    recent_dates = pd.DatetimeIndex(data[DATE_COL].drop_duplicates().sort_values())[-max(int(lookback_days), 1) :]
    recent = data[data[DATE_COL].isin(recent_dates)].copy()
    recent["open_count"] = pd.to_numeric(recent["open_count"], errors="coerce").fillna(0.0)
    recent["close_count"] = pd.to_numeric(recent["close_count"], errors="coerce").fillna(0.0)
    recent["net_count"] = recent["open_count"] - recent["close_count"]
    daily = recent[daily_columns].sort_values([CODE_COL, bucket_col, DATE_COL]).reset_index(drop=True)

    aggregate = (
        recent.groupby([CODE_COL, NAME_COL, bucket_col], dropna=False, as_index=False)
        .agg(open_count=("open_count", "sum"), close_count=("close_count", "sum"))
        .sort_values([CODE_COL, bucket_col])
        .reset_index(drop=True)
    )
    aggregate["net_count"] = aggregate["open_count"] - aggregate["close_count"]
    aggregate["lookback_days"] = len(recent_dates)
    aggregate["start_date"] = recent_dates.min() if len(recent_dates) else pd.NaT
    aggregate["end_date"] = recent_dates.max() if len(recent_dates) else pd.NaT
    aggregate["open_share"] = aggregate["open_count"] / aggregate.groupby(CODE_COL)["open_count"].transform("sum").replace(0, np.nan)
    aggregate["close_share"] = aggregate["close_count"] / aggregate.groupby(CODE_COL)["close_count"].transform("sum").replace(0, np.nan)
    aggregate = aggregate[aggregate_columns]
    return daily, aggregate


def summarize_recent_signal_counts(
    signal_count_data: pd.DataFrame,
    lookback_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_columns = [
        DATE_COL,
        CODE_COL,
        NAME_COL,
        "signal_source",
        "open_count",
        "close_count",
        "net_count",
    ]
    aggregate_columns = [
        CODE_COL,
        NAME_COL,
        "signal_source",
        "lookback_days",
        "start_date",
        "end_date",
        "open_count",
        "close_count",
        "net_count",
    ]
    if signal_count_data.empty:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=aggregate_columns)

    data = signal_count_data.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.dropna(subset=[DATE_COL])
    if data.empty:
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=aggregate_columns)

    recent_dates = pd.DatetimeIndex(data[DATE_COL].drop_duplicates().sort_values())[-max(int(lookback_days), 1) :]
    recent = data[data[DATE_COL].isin(recent_dates)].copy()
    recent["open_count"] = pd.to_numeric(recent["open_count"], errors="coerce").fillna(0.0)
    recent["close_count"] = pd.to_numeric(recent["close_count"], errors="coerce").fillna(0.0)
    recent["net_count"] = recent["open_count"] - recent["close_count"]
    daily = recent[daily_columns].sort_values(["signal_source", CODE_COL, DATE_COL]).reset_index(drop=True)

    aggregate = (
        recent.groupby([CODE_COL, NAME_COL, "signal_source"], dropna=False, as_index=False)
        .agg(open_count=("open_count", "sum"), close_count=("close_count", "sum"))
        .sort_values([CODE_COL, "signal_source"])
        .reset_index(drop=True)
    )
    aggregate["net_count"] = aggregate["open_count"] - aggregate["close_count"]
    aggregate["lookback_days"] = len(recent_dates)
    aggregate["start_date"] = recent_dates.min() if len(recent_dates) else pd.NaT
    aggregate["end_date"] = recent_dates.max() if len(recent_dates) else pd.NaT
    return daily, aggregate[aggregate_columns]


def best_global_rule_pairs(rule_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rule_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = rule_summary.copy()
    data["excess_annual_return"] = pd.to_numeric(data["excess_annual_return"], errors="coerce")
    best = (
        data.sort_values(
            ["signal_source", "excess_annual_return", "sharpe", "annual_return"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        .groupby(["signal_source", CODE_COL], dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    by_family = (
        data.sort_values(
            ["signal_source", CODE_COL, "rule_family", "excess_annual_return", "sharpe", "annual_return"],
            ascending=[True, True, True, False, False, False],
            na_position="last",
        )
        .groupby(["signal_source", CODE_COL, "rule_family"], dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best, by_family


def summarize_global_rule_quality(rule_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "signal_source",
        CODE_COL,
        NAME_COL,
        "rule_count",
        "positive_excess_rule_count",
        "positive_excess_rule_share",
        "median_excess_annual_return",
        "mean_excess_annual_return",
        "best_excess_annual_return",
        "best_sharpe",
        "best_rule_key",
        "best_rule_family",
    ]
    if rule_summary.empty:
        return pd.DataFrame(columns=columns)

    data = rule_summary.copy()
    data["excess_annual_return"] = pd.to_numeric(data["excess_annual_return"], errors="coerce")
    data["_positive_excess"] = data["excess_annual_return"].gt(0.0)
    group_cols = ["signal_source", CODE_COL, NAME_COL]
    quality = data.groupby(group_cols, dropna=False, as_index=False).agg(
        rule_count=("rule_key", "size"),
        positive_excess_rule_count=("_positive_excess", "sum"),
        positive_excess_rule_share=("_positive_excess", "mean"),
        median_excess_annual_return=("excess_annual_return", "median"),
        mean_excess_annual_return=("excess_annual_return", "mean"),
        best_excess_annual_return=("excess_annual_return", "max"),
        best_sharpe=("sharpe", "max"),
    )
    best = (
        data.sort_values(group_cols + ["excess_annual_return", "sharpe"], ascending=[True, True, True, False, False])
        .groupby(group_cols, dropna=False, as_index=False)
        .head(1)[group_cols + ["rule_key", "rule_family"]]
        .rename(columns={"rule_key": "best_rule_key", "rule_family": "best_rule_family"})
    )
    quality = quality.merge(best, on=group_cols, how="left")
    return quality[columns].sort_values(
        ["best_excess_annual_return", "positive_excess_rule_share"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def best_bucket_rule_pairs(
    rule_summary: pd.DataFrame,
    bucket_col: str = PRICE_BUCKET_COL,
    min_excess_annual_return: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rule_summary.empty or bucket_col not in rule_summary.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    metric = pd.to_numeric(rule_summary["excess_annual_return"], errors="coerce")
    data = rule_summary.assign(_selection_metric=metric).copy()
    sort_cols = ["signal_source", CODE_COL, bucket_col, "_selection_metric", "sharpe", "annual_return"]
    by_bucket = (
        data.sort_values(sort_cols, ascending=[True, True, True, False, False, False], na_position="last")
        .groupby(["signal_source", CODE_COL, bucket_col], dropna=False, as_index=False)
        .head(1)
        .drop(columns=["_selection_metric"])
        .reset_index(drop=True)
    )
    by_bucket["selected_bucket"] = pd.to_numeric(by_bucket["excess_annual_return"], errors="coerce").gt(
        float(min_excess_annual_return)
    )

    by_family = (
        data.sort_values(
            ["signal_source", CODE_COL, bucket_col, "rule_family", "_selection_metric", "sharpe", "annual_return"],
            ascending=[True, True, True, True, False, False, False],
            na_position="last",
        )
        .groupby(["signal_source", CODE_COL, bucket_col, "rule_family"], dropna=False, as_index=False)
        .head(1)
        .drop(columns=["_selection_metric"])
        .reset_index(drop=True)
    )

    selection = by_bucket[
        [
            "signal_source",
            CODE_COL,
            NAME_COL,
            bucket_col,
            "selected_bucket",
            "rule_key",
            "rule_family",
            "open_rule",
            "close_rule",
            "annual_return",
            "benchmark_annual_return",
            "excess_annual_return",
            "sharpe",
            "holding_ratio",
            "trade_count",
            "final_equity",
            "benchmark_final_equity",
            "excess_final_equity",
            "open_event_count",
            "close_event_count",
        ]
    ].copy()
    return by_bucket, by_family, selection


def summarize_bucket_rule_quality(
    rule_summary: pd.DataFrame,
    bucket_col: str = PRICE_BUCKET_COL,
) -> pd.DataFrame:
    columns = [
        "signal_source",
        CODE_COL,
        NAME_COL,
        bucket_col,
        "rule_count",
        "positive_excess_rule_count",
        "positive_excess_rule_share",
        "median_excess_annual_return",
        "mean_excess_annual_return",
        "best_excess_annual_return",
        "best_sharpe",
        "best_rule_key",
        "best_rule_family",
    ]
    if rule_summary.empty or bucket_col not in rule_summary.columns:
        return pd.DataFrame(columns=columns)

    data = rule_summary.copy()
    data["excess_annual_return"] = pd.to_numeric(data["excess_annual_return"], errors="coerce")
    data["_positive_excess"] = data["excess_annual_return"].gt(0.0)
    group_cols = ["signal_source", CODE_COL, NAME_COL, bucket_col]
    quality = data.groupby(group_cols, dropna=False, as_index=False).agg(
        rule_count=("rule_key", "size"),
        positive_excess_rule_count=("_positive_excess", "sum"),
        positive_excess_rule_share=("_positive_excess", "mean"),
        median_excess_annual_return=("excess_annual_return", "median"),
        mean_excess_annual_return=("excess_annual_return", "mean"),
        best_excess_annual_return=("excess_annual_return", "max"),
        best_sharpe=("sharpe", "max"),
    )
    best = (
        data.sort_values(group_cols + ["excess_annual_return", "sharpe"], ascending=[True, True, True, True, False, False])
        .groupby(group_cols, dropna=False, as_index=False)
        .head(1)[group_cols + ["rule_key", "rule_family"]]
        .rename(columns={"rule_key": "best_rule_key", "rule_family": "best_rule_family"})
    )
    quality = quality.merge(best, on=group_cols, how="left")
    return quality[columns].sort_values(
        ["best_excess_annual_return", "positive_excess_rule_share"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def filter_signal_context_by_selected_buckets(
    signal_context: pd.DataFrame,
    bucket_selection: pd.DataFrame,
    bucket_col: str = PRICE_BUCKET_COL,
) -> pd.DataFrame:
    if signal_context.empty or bucket_selection.empty or bucket_col not in signal_context.columns:
        return signal_context.iloc[0:0].copy()

    selected = bucket_selection[bucket_selection["selected_bucket"]].copy()
    if selected.empty:
        return signal_context.iloc[0:0].copy()
    selected_keys = selected[[SIGNAL_INSTRUMENT_COL if SIGNAL_INSTRUMENT_COL in selected.columns else CODE_COL, bucket_col]]
    selected_keys = selected_keys.rename(columns={CODE_COL: SIGNAL_INSTRUMENT_COL}).drop_duplicates()

    signals = signal_context.copy()
    signals[SIGNAL_INSTRUMENT_COL] = signals[SIGNAL_INSTRUMENT_COL].astype(str)
    selected_keys[SIGNAL_INSTRUMENT_COL] = selected_keys[SIGNAL_INSTRUMENT_COL].astype(str)
    return signals.merge(selected_keys, on=[SIGNAL_INSTRUMENT_COL, bucket_col], how="inner")


def run_signal_count_timing_shadow(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    filtered_signals: pd.DataFrame,
    output_dir: str | Path | None = None,
    start_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_data = build_signal_count_inputs(
        price_df=price_df,
        raw_signals=raw_signals,
        filtered_signals=filtered_signals,
        start_date=start_date,
    )
    rule_summary, best_equity = backtest_signal_count_rule_pairs(count_data)
    paired_comparison, variant_summary = compare_filtered_signal_count_rules(rule_summary)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(rule_summary, output_path / "rule_backtest.parquet")
        write_table(paired_comparison, output_path / "rule_comparison.parquet")
        write_table(variant_summary, output_path / "variant_summary.parquet")
        write_table(best_equity, output_path / "best_equity.parquet")
    return rule_summary, paired_comparison, variant_summary, best_equity


def run_event_filter_signal_count_timing_shadow(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    output_dir: str | Path | None = None,
    start_date: str | pd.Timestamp | None = None,
    include_turning: bool = True,
    bb_window: int = PRICE_BB_WINDOW,
    bb_std: float = PRICE_BB_STD,
    sync_window: int = SYNC_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_whitelist = build_monthly_momentum_context_whitelist(
        price_df,
        raw_signals,
        bb_window=bb_window,
        bb_std=bb_std,
        sync_window=sync_window,
    )
    filtered_events, filtered_events_with_context = build_monthly_momentum_context_filter(
        price_df,
        raw_signals,
        event_whitelist,
        bb_window=bb_window,
        bb_std=bb_std,
        sync_window=sync_window,
    )

    if start_date is None and not filtered_events.empty:
        start_date = pd.to_datetime(filtered_events[SIGNAL_DATE_COL], errors="coerce").min()
    raw_counts = align_signal_counts_to_price(price_df, raw_signals, source="raw", start_date=start_date)
    filtered_counts = align_signal_counts_to_price(
        price_df,
        filtered_events,
        source="event_filtered",
        start_date=start_date,
    )
    count_data = pd.concat([raw_counts, filtered_counts], ignore_index=True)
    rule_pairs = generate_signal_count_rule_pairs(include_turning=include_turning)
    rule_summary, best_equity = backtest_signal_count_rule_pairs(count_data, rule_pairs)
    best_rule, best_by_family = best_global_rule_pairs(rule_summary)
    rule_quality = summarize_global_rule_quality(rule_summary)
    comparable = rule_summary.replace({"signal_source": {"event_filtered": "filtered"}})
    paired_comparison, variant_summary = compare_filtered_signal_count_rules(comparable)
    _, recent_summary = summarize_recent_signal_counts(filtered_counts, lookback_days=20)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata = pd.DataFrame(
            [
                {
                    "bb_window": int(bb_window),
                    "bb_std": float(bb_std),
                    "sync_window": int(sync_window),
                }
            ]
        )
        write_table(metadata, output_path / "run_metadata.parquet")
        write_table(event_whitelist, output_path / "event_whitelist.parquet")
        write_table(filtered_events, output_path / "filtered_events.parquet")
        write_table(filtered_events_with_context, output_path / "filtered_events_with_context.parquet")
        write_table(event_whitelist, output_path / "domain_event_whitelist.parquet")
        write_table(filtered_events, output_path / "domain_filtered_events.parquet")
        write_table(filtered_events_with_context, output_path / "domain_filtered_events_with_context.parquet")
        write_table(rule_summary, output_path / "rule_pair_backtest.parquet")
        write_table(best_rule, output_path / "best_rule_pair.parquet")
        write_table(best_by_family, output_path / "best_rule_pair_by_family.parquet")
        write_table(rule_quality, output_path / "rule_pair_quality.parquet")
        write_table(best_equity, output_path / "best_rule_pair_equity.parquet")
        write_table(paired_comparison, output_path / "raw_vs_event_filtered_rule_comparison.parquet")
        write_table(variant_summary, output_path / "raw_vs_event_filtered_variant_summary.parquet")
        write_table(recent_summary, output_path / "recent20_filtered_event_counts_summary.parquet")
    return event_whitelist, filtered_events, rule_summary, best_rule, variant_summary


def run_event_blacklist_signal_count_timing_shadow(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    output_dir: str | Path | None = None,
    start_date: str | pd.Timestamp | None = None,
    include_turning: bool = True,
    bb_window: int = PRICE_BB_WINDOW,
    bb_std: float = PRICE_BB_STD,
    sync_window: int = SYNC_WINDOW,
    blacklist_evidence_mode: str = DEFAULT_BLACKLIST_EVIDENCE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_blacklist = build_monthly_momentum_context_blacklist(
        price_df,
        raw_signals,
        bb_window=bb_window,
        bb_std=bb_std,
        sync_window=sync_window,
        blacklist_evidence_mode=blacklist_evidence_mode,
    )
    filtered_events, filtered_events_with_context, blacklisted_events_with_context = (
        build_monthly_momentum_context_blacklist_filter(
            price_df,
            raw_signals,
            event_blacklist,
            bb_window=bb_window,
            bb_std=bb_std,
            sync_window=sync_window,
        )
    )

    if start_date is None and not filtered_events.empty:
        start_date = pd.to_datetime(filtered_events[SIGNAL_DATE_COL], errors="coerce").min()
    raw_counts = align_signal_counts_to_price(price_df, raw_signals, source="raw", start_date=start_date)
    filtered_counts = align_signal_counts_to_price(
        price_df,
        filtered_events,
        source="filtered",
        start_date=start_date,
    )
    count_data = pd.concat([raw_counts, filtered_counts], ignore_index=True)
    rule_pairs = generate_signal_count_rule_pairs(include_turning=include_turning)
    rule_summary, best_equity = backtest_signal_count_rule_pairs(count_data, rule_pairs)
    best_rule, best_by_family = best_global_rule_pairs(rule_summary)
    rule_quality = summarize_global_rule_quality(rule_summary)
    paired_comparison, variant_summary = compare_filtered_signal_count_rules(rule_summary)
    _, recent_summary = summarize_recent_signal_counts(filtered_counts, lookback_days=20)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata = pd.DataFrame(
            [
                {
                    "filter_mode": "blacklist",
                    "bb_window": int(bb_window),
                    "bb_std": float(bb_std),
                    "sync_window": int(sync_window),
                    "blacklist_evidence_mode": str(blacklist_evidence_mode),
                }
            ]
        )
        write_table(metadata, output_path / "run_metadata.parquet")
        write_table(event_blacklist, output_path / "event_blacklist.parquet")
        write_table(filtered_events, output_path / "filtered_events.parquet")
        write_table(best_rule, output_path / "best_rule_pair.parquet")
        write_table(best_equity, output_path / "best_rule_pair_equity.parquet")
        write_table(variant_summary, output_path / "raw_vs_filtered_variant_summary.parquet")
        write_table(recent_summary, output_path / "recent20_filtered_event_counts_summary.parquet")
    return event_blacklist, filtered_events, rule_summary, best_rule, variant_summary


def run_bucket_signal_count_timing_shadow(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    output_dir: str | Path | None = None,
    start_date: str | pd.Timestamp | None = None,
    lookback_days: int = 20,
    include_turning: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_context = build_signal_context_table(price_df, raw_signals)
    bucket_count_data = align_bucket_signal_counts_to_price(
        price_df=price_df,
        signal_context=signal_context,
        source="bucket_raw",
        start_date=start_date,
        drop_unknown=True,
    )
    recent_daily, recent_summary = summarize_recent_bucket_signal_counts(
        bucket_count_data,
        lookback_days=lookback_days,
    )
    rule_pairs = generate_signal_count_rule_pairs(include_turning=include_turning)
    rule_summary, best_equity = backtest_signal_count_rule_pairs(bucket_count_data, rule_pairs)
    best_by_bucket, best_by_family, bucket_selection = best_bucket_rule_pairs(rule_summary)
    bucket_quality = summarize_bucket_rule_quality(rule_summary)
    selected_signals = filter_signal_context_by_selected_buckets(signal_context, bucket_selection)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(recent_daily, output_path / "recent20_bucket_counts_daily.parquet")
        write_table(recent_summary, output_path / "recent20_bucket_counts_summary.parquet")
        write_table(rule_summary, output_path / "bucket_rule_backtest.parquet")
        write_table(best_by_bucket, output_path / "bucket_best_rule.parquet")
        write_table(best_by_family, output_path / "bucket_best_rule_by_family.parquet")
        write_table(bucket_quality, output_path / "bucket_rule_quality.parquet")
        write_table(bucket_selection, output_path / "bucket_selection.parquet")
        write_table(selected_signals, output_path / "selected_bucket_signals.parquet")
        write_table(best_equity, output_path / "bucket_best_equity.parquet")
    return rule_summary, best_by_bucket, best_by_family, bucket_selection, recent_summary, best_equity


def run_domain_child_signal_count_timing_shadow(
    price_df: pd.DataFrame,
    raw_signals: pd.DataFrame,
    output_dir: str | Path | None = None,
    start_date: str | pd.Timestamp | None = None,
    lookback_days: int = 20,
    history_years: float = DEFAULT_CHILD_SIGNAL_HISTORY_YEARS,
    min_child_signal_events: int = DEFAULT_MIN_CHILD_SIGNAL_EVENTS_3Y,
    include_turning: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    child_context, child_daily_counts, child_universe = build_domain_child_signal_context(
        price_df=price_df,
        raw_signals=raw_signals,
        history_years=history_years,
        min_child_signal_events=min_child_signal_events,
    )
    if "valid_child_signal" in child_context.columns:
        valid_child_signals = child_context[child_context["valid_child_signal"]].copy()
    else:
        valid_child_signals = child_context.iloc[0:0].copy()
    count_data = align_signal_counts_to_price(
        price_df=price_df,
        signal_table=valid_child_signals,
        source="domain_child_valid",
        start_date=start_date,
    )
    bucket_count_data = align_bucket_signal_counts_to_price(
        price_df=price_df,
        signal_context=valid_child_signals,
        source="domain_child_valid",
        start_date=start_date,
        drop_unknown=True,
    )
    recent_daily, recent_summary = summarize_recent_signal_counts(
        count_data,
        lookback_days=lookback_days,
    )
    recent_bucket_daily, recent_bucket_summary = summarize_recent_bucket_signal_counts(
        bucket_count_data,
        lookback_days=lookback_days,
    )
    rule_pairs = generate_signal_count_rule_pairs(include_turning=include_turning)
    rule_summary, best_equity = backtest_signal_count_rule_pairs(count_data, rule_pairs)
    best_rule, best_by_family = best_global_rule_pairs(rule_summary)
    rule_quality = summarize_global_rule_quality(rule_summary)
    selected_signals = valid_child_signals.copy()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(child_universe, output_path / "child_signal_universe.parquet")
        write_table(child_daily_counts, output_path / "child_signal_daily_counts.parquet")
        write_table(valid_child_signals, output_path / "valid_child_signals.parquet")
        write_table(recent_daily, output_path / "recent20_domain_child_counts_daily.parquet")
        write_table(recent_summary, output_path / "recent20_domain_child_counts_summary.parquet")
        write_table(recent_bucket_daily, output_path / "recent20_domain_child_bucket_counts_daily.parquet")
        write_table(recent_bucket_summary, output_path / "recent20_domain_child_bucket_counts_summary.parquet")
        write_table(rule_summary, output_path / "domain_child_rule_backtest.parquet")
        write_table(best_rule, output_path / "domain_child_best_rule.parquet")
        write_table(best_by_family, output_path / "domain_child_best_rule_by_family.parquet")
        write_table(rule_quality, output_path / "domain_child_rule_quality.parquet")
        write_table(
            pd.DataFrame([{"selection_scope": "global_rule_pair", "bucket_selection_used": False}]),
            output_path / "domain_child_bucket_selection.parquet",
        )
        write_table(selected_signals, output_path / "selected_domain_child_signals.parquet")
        write_table(best_equity, output_path / "domain_child_best_equity.parquet")
    return rule_summary, best_rule, best_by_family, rule_quality, recent_summary, best_equity


def run_default_signal_count_timing_shadow(
    run_dir: str | Path = "skills/factor-timing-advisor/workspace/runs/default",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    price_df = read_table(run_path / "data/input_snapshot.parquet")
    raw_signals = read_table(run_path / "results/signals/signals.parquet")
    filtered_signals = read_table(run_path / "results/momentum_context_shadow/filtered_signals.parquet")
    return run_signal_count_timing_shadow(
        price_df=price_df,
        raw_signals=raw_signals,
        filtered_signals=filtered_signals,
        output_dir=run_path / "results/signal_count_timing_shadow",
    )


def run_default_event_filter_signal_count_timing_shadow(
    run_dir: str | Path = "skills/factor-timing-advisor/workspace/runs/default",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    price_df = read_table(run_path / "data/input_snapshot.parquet")
    raw_signals = read_table(run_path / "results/signals/signals.parquet")
    return run_event_filter_signal_count_timing_shadow(
        price_df=price_df,
        raw_signals=raw_signals,
        output_dir=run_path / "results/price_mid_cross_event_filter_shadow",
    )


def run_default_event_blacklist_signal_count_timing_shadow(
    run_dir: str | Path = "skills/factor-timing-advisor/workspace/runs/default",
    blacklist_evidence_mode: str = DEFAULT_BLACKLIST_EVIDENCE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    price_df = read_table(run_path / "data/input_snapshot.parquet")
    raw_signals = read_table(run_path / "results/signals/signals.parquet")
    return run_event_blacklist_signal_count_timing_shadow(
        price_df=price_df,
        raw_signals=raw_signals,
        output_dir=run_path / "results/price_mid_cross_event_blacklist_shadow",
        blacklist_evidence_mode=blacklist_evidence_mode,
    )


def run_event_filter_window_sweep_shadow(
    run_dir: str | Path = "skills/factor-timing-advisor/workspace/runs/default",
    schemes: Iterable[tuple[str, int, float]] | None = None,
) -> pd.DataFrame:
    run_path = Path(run_dir)
    price_df = read_table(run_path / "data/input_snapshot.parquet")
    raw_signals = read_table(run_path / "results/signals/signals.parquet")
    output_root = run_path / "results/price_mid_cross_event_filter_window_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    if schemes is None:
        schemes = (
            ("bb_mid_20d", 20, 2.0),
            ("bb_mid_40d", 40, 2.0),
            ("bb_mid_60d", 60, 2.0),
        )

    rows: list[dict[str, object]] = []
    for scheme_name, bb_window, bb_std in schemes:
        scheme_dir = output_root / scheme_name
        _, filtered_events, _, best_rule, variant_summary = run_event_filter_signal_count_timing_shadow(
            price_df=price_df,
            raw_signals=raw_signals,
            output_dir=scheme_dir,
            bb_window=int(bb_window),
            bb_std=float(bb_std),
            sync_window=SYNC_WINDOW,
        )
        filtered_context_path = scheme_dir / "domain_filtered_events_with_context.parquet"
        recent_path = scheme_dir / "recent20_filtered_event_counts_summary.parquet"
        filtered_context = read_table(filtered_context_path) if filtered_context_path.exists() else pd.DataFrame()
        recent = read_table(recent_path) if recent_path.exists() else pd.DataFrame()
        filtered_best = best_rule[best_rule["signal_source"].eq("event_filtered")].head(1)
        raw_best = best_rule[best_rule["signal_source"].eq("raw")].head(1)
        filtered_variant = variant_summary[variant_summary["signal_source"].eq("filtered")].head(1)
        same_rule_variant = variant_summary[variant_summary["signal_source"].eq("filtered_minus_raw_same_rules")].head(1)
        bucket_counts = (
            filtered_context["price_band_bucket"].value_counts().to_dict()
            if "price_band_bucket" in filtered_context.columns
            else {}
        )
        recent_row = recent.iloc[0] if not recent.empty else pd.Series(dtype=object)
        filtered_best_row = filtered_best.iloc[0] if not filtered_best.empty else pd.Series(dtype=object)
        raw_best_row = raw_best.iloc[0] if not raw_best.empty else pd.Series(dtype=object)
        filtered_variant_row = filtered_variant.iloc[0] if not filtered_variant.empty else pd.Series(dtype=object)
        same_rule_row = same_rule_variant.iloc[0] if not same_rule_variant.empty else pd.Series(dtype=object)
        rows.append(
            {
                "scheme": scheme_name,
                "bb_window": int(bb_window),
                "bb_std": float(bb_std),
                "sync_window": int(SYNC_WINDOW),
                "raw_event_count": int(len(raw_signals)),
                "filtered_event_count": int(len(filtered_events)),
                "event_retention_rate": float(len(filtered_events) / len(raw_signals)) if len(raw_signals) else np.nan,
                "recent20_open_count": recent_row.get("open_count", np.nan),
                "recent20_close_count": recent_row.get("close_count", np.nan),
                "recent20_net_count": recent_row.get("net_count", np.nan),
                "filtered_best_rule_key": filtered_best_row.get("rule_key", ""),
                "filtered_best_rule_family": filtered_best_row.get("rule_family", ""),
                "filtered_best_annual_return": filtered_best_row.get("annual_return", np.nan),
                "filtered_best_excess_annual_return": filtered_best_row.get("excess_annual_return", np.nan),
                "filtered_best_sharpe": filtered_best_row.get("sharpe", np.nan),
                "filtered_best_final_equity": filtered_best_row.get("final_equity", np.nan),
                "filtered_best_trade_count": filtered_best_row.get("trade_count", np.nan),
                "raw_best_excess_annual_return": raw_best_row.get("excess_annual_return", np.nan),
                "raw_best_sharpe": raw_best_row.get("sharpe", np.nan),
                "positive_excess_rule_share": filtered_variant_row.get("positive_excess_rule_share", np.nan),
                "median_excess_annual_return": filtered_variant_row.get("median_excess_annual_return", np.nan),
                "mean_excess_annual_return": filtered_variant_row.get("mean_excess_annual_return", np.nan),
                "same_rule_improvement_share": same_rule_row.get("positive_excess_rule_share", np.nan),
                "same_rule_median_improvement": same_rule_row.get("median_excess_annual_return", np.nan),
                "cross_above_mid_event_count": int(bucket_counts.get(CROSS_ABOVE_MID_BUCKET, 0)),
                "cross_below_mid_event_count": int(bucket_counts.get(CROSS_BELOW_MID_BUCKET, 0)),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["filtered_best_excess_annual_return", "positive_excess_rule_share"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    write_table(comparison, output_root / "scheme_comparison.parquet")
    return comparison


def run_default_bucket_signal_count_timing_shadow(
    run_dir: str | Path = "skills/factor-timing-advisor/workspace/runs/default",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    price_df = read_table(run_path / "data/input_snapshot.parquet")
    raw_signals = read_table(run_path / "results/signals/signals.parquet")
    return run_bucket_signal_count_timing_shadow(
        price_df=price_df,
        raw_signals=raw_signals,
        output_dir=run_path / "results/signal_count_timing_shadow/bucket_domain",
    )


def run_default_domain_child_signal_count_timing_shadow(
    run_dir: str | Path = "skills/factor-timing-advisor/workspace/runs/default",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir)
    price_df = read_table(run_path / "data/input_snapshot.parquet")
    raw_signals = read_table(run_path / "results/signals/signals.parquet")
    return run_domain_child_signal_count_timing_shadow(
        price_df=price_df,
        raw_signals=raw_signals,
        output_dir=run_path / "results/signal_count_timing_shadow/domain_child",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Shadow backtest for event-count timing rules.")
    parser.add_argument("--run-dir", default="skills/factor-timing-advisor/workspace/runs/default")
    parser.add_argument("--event-filter", action="store_true")
    parser.add_argument("--event-blacklist", action="store_true")
    parser.add_argument("--blacklist-evidence-mode", default=DEFAULT_BLACKLIST_EVIDENCE_MODE)
    parser.add_argument("--event-filter-sweep", action="store_true")
    parser.add_argument("--bucket-domain", action="store_true")
    parser.add_argument("--domain-child", action="store_true")
    args = parser.parse_args()
    if args.event_filter_sweep:
        comparison = run_event_filter_window_sweep_shadow(args.run_dir)
        if not comparison.empty:
            print(comparison.to_string(index=False))
    elif args.event_filter:
        _, filtered_events, _, best_rule, variant_summary = run_default_event_filter_signal_count_timing_shadow(args.run_dir)
        print(f"filtered_events={len(filtered_events)}")
        if not best_rule.empty:
            print(best_rule.to_string(index=False))
        if not variant_summary.empty:
            print(variant_summary.to_string(index=False))
    elif args.event_blacklist:
        blacklist, filtered_events, _, best_rule, variant_summary = run_default_event_blacklist_signal_count_timing_shadow(
            args.run_dir,
            blacklist_evidence_mode=args.blacklist_evidence_mode,
        )
        print(f"blacklist_contexts={len(blacklist)}")
        print(f"filtered_events={len(filtered_events)}")
        if not best_rule.empty:
            print(best_rule.to_string(index=False))
        if not variant_summary.empty:
            print(variant_summary.to_string(index=False))
    elif args.domain_child:
        _, best_by_bucket, _, _, recent_summary, _ = run_default_domain_child_signal_count_timing_shadow(args.run_dir)
        if not recent_summary.empty:
            print(recent_summary.to_string(index=False))
        if not best_by_bucket.empty:
            print(best_by_bucket.to_string(index=False))
    elif args.bucket_domain:
        _, best_by_bucket, _, _, recent_summary, _ = run_default_bucket_signal_count_timing_shadow(args.run_dir)
        if not recent_summary.empty:
            print(recent_summary.to_string(index=False))
        if not best_by_bucket.empty:
            print(best_by_bucket.to_string(index=False))
    else:
        _, _, variant_summary, _ = run_default_signal_count_timing_shadow(args.run_dir)
        if not variant_summary.empty:
            print(variant_summary.to_string(index=False))


if __name__ == "__main__":
    main()

