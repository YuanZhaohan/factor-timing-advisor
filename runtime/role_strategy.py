# -*- coding: utf-8 -*-
from __future__ import annotations

"""信号用途识别与边际信息衰减状态分数。

当前版本保留：
1. build_factor_signal_utility：主观规则分类与期限结构量化分类。
2. build_signal_edge_decay：计算信号触发后第 k 天仍有多少边际信息。
3. build_daily_signal_state_score：汇总有效历史信号为每日 entry/exit/net score。
4. run_expanding_edge_decay_timing：expanding window 样本外择时曲线。
"""

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from io_utils import read_table, resolve_table_file, table_candidates, write_table

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - optional acceleration
    Parallel = None
    delayed = None

from timing_config import (
    CODE_COL,
    DATE_COL,
    NAME_COL,
    PRICE_COL,
    TRADING_DAYS,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_VALUE_COL,
    _split_factor_frequency,
)

UTILITY_HORIZONS = (1, 3, 5, 10, 15, 20, 60)
EDGE_AGE_GRID = (0, 1, 3, 5, 10, 15, 20)
EDGE_FORWARD_HORIZON = 60
EDGE_MIN_COUNT = 12
EDGE_T_FULL_CONFIDENCE = 2.0
DEFAULT_MIN_EVENTS = 12
DEFAULT_MIN_ABS_SCORE = 0.0
TERM_STRUCTURE_EDGE = 0.01
DEFAULT_MIN_EVENTS_PER_YEAR = 2.0
DEFAULT_SCORE_RANK_PCT_THRESHOLD = 0.7
DEFAULT_AUX_SCORE_RANK_PCT_THRESHOLD = 0.9
DEFAULT_MAX_RULES_PER_BUCKET = 1
DEFAULT_SCORE_THRESHOLD_GRID = (0.0, 0.002, 0.005, 0.01)
DEFAULT_SCORE_DOMINANCE_GRID = (1.0, 1.2, 1.5)

ROLE_ORDER = [
    "bottom_entry",
    "trend_entry",
    "momentum_entry",
    "top_exit",
    "trend_exit",
    "risk_exit",
    "aux_filter",
]

ROLE_SCORE_COLS = [
    "bottom_entry_score_mean",
    "trend_entry_score_mean",
    "momentum_entry_score_mean",
    "top_exit_score_mean",
    "trend_exit_score_mean",
    "risk_exit_score_mean",
]

ROLE_COUNT_COLS = [
    "bottom_entry_signal_count",
    "trend_entry_signal_count",
    "momentum_entry_signal_count",
    "top_exit_signal_count",
    "trend_exit_signal_count",
    "risk_exit_signal_count",
]


def _pattern_family(pattern: str) -> str:
    name = str(pattern)
    for prefix in ("开仓_", "闭仓_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    for sep in ("_", "-"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name


def _role_from_pattern(pattern: Any, event_side: str) -> str:
    name = str(pattern)
    if str(event_side) == "open":
        if any(token in name for token in ("下方拐点", "低位", "底背离", "价格下跌因子上升", "价格新低", "process_signal买点")):
            return "bottom_entry"
        if any(token in name for token in ("连续上升", "累计上升", "振幅放大向上", "加速度转正")):
            return "momentum_entry"
        if any(token in name for token in ("上穿", "同步上升", "价格突破", "多头共振", "斜率转正")):
            return "trend_entry"
        return "aux_filter"

    if any(token in name for token in ("上方拐点", "高位", "顶背离", "价格上涨因子下降", "价格新高", "process_signal卖点")):
        return "top_exit"
    if any(token in name for token in ("下穿", "连续下降", "累计下降", "振幅放大向下", "加速度转负", "同步下降", "价格破位", "空头共振", "斜率转负")):
        return "trend_exit"
    return "risk_exit"


def _term_value(row: pd.Series, horizon: int) -> float:
    try:
        return float(row.get(f"return_{horizon}d", np.nan))
    except (TypeError, ValueError):
        return np.nan


def _term_best_horizon(row: pd.Series, event_side: str) -> float:
    pairs = [(h, _term_value(row, h)) for h in UTILITY_HORIZONS]
    pairs = [(h, v) for h, v in pairs if np.isfinite(v)]
    if not pairs:
        return np.nan
    if str(event_side) == "open":
        return float(max(pairs, key=lambda item: item[1])[0])
    return float(min(pairs, key=lambda item: item[1])[0])


def _classify_term_structure(row: pd.Series) -> tuple[str, str]:
    """根据未来收益期限结构做量化分类。

    量化口径：
    - `early = return_5d`
    - `mid = return_10d`
    - `best_later = max(return_15d, return_20d, return_60d)`
    - `worst_later = min(return_15d, return_20d, return_60d)`
    - `delayed = mid - early`
    - `TERM_STRUCTURE_EDGE = 0.01`
    """
    event_side = str(row.get("event_side", ""))
    early = _term_value(row, 5)
    mid = _term_value(row, 10)
    later_values = [_term_value(row, h) for h in (15, 20, 60)]
    later_values = [v for v in later_values if np.isfinite(v)]
    if not later_values or not np.isfinite(early):
        pattern = row.get("condition", row.get("pattern", ""))
        return "期限结构不清晰", _role_from_pattern(pattern, event_side)

    best_later = max(later_values)
    worst_later = min(later_values)
    delayed = mid - early if np.isfinite(mid) else np.nan
    edge = TERM_STRUCTURE_EDGE

    if event_side == "open":
        if early > edge and (not np.isfinite(delayed) or early >= delayed):
            return "追涨/即时兑现", "momentum_entry"
        if early <= edge and np.isfinite(delayed) and delayed > edge:
            return "超跌/延迟修复", "bottom_entry"
        if early <= 0 and best_later > edge:
            return "超跌/中期修复", "bottom_entry"
        if best_later > edge:
            return "趋势/持续兑现", "trend_entry"
        return "弱收益/不清晰", "aux_filter"

    if early < -edge:
        return "趋势转弱/立即下跌", "trend_exit"
    if early >= -edge and worst_later < -edge:
        return "逃顶/延迟走弱", "top_exit"
    if worst_later < -edge:
        return "风险退出/持续走弱", "risk_exit"
    return "弱风险/不清晰", "risk_exit"


def _term_structure_frame(data: pd.DataFrame, rule_col: str) -> pd.DataFrame:
    key_cols = [col for col in (CODE_COL, "factor", rule_col, "event_side") if col in data.columns]
    if not key_cols or "horizon" not in data.columns or "mean_return" not in data.columns:
        return pd.DataFrame()

    curve = data[key_cols + ["horizon", "mean_return"]].copy()
    curve["horizon"] = pd.to_numeric(curve["horizon"], errors="coerce")
    curve["mean_return"] = pd.to_numeric(curve["mean_return"], errors="coerce")
    curve = curve.dropna(subset=["horizon"])
    if curve.empty:
        return pd.DataFrame()

    pivot = curve.pivot_table(index=key_cols, columns="horizon", values="mean_return", aggfunc="mean").reset_index()
    pivot = pivot.rename(
        columns={
            col: f"return_{int(col)}d"
            for col in pivot.columns
            if isinstance(col, (int, float, np.integer, np.floating)) and np.isfinite(col)
        }
    )
    for horizon in UTILITY_HORIZONS:
        col = f"return_{horizon}d"
        if col not in pivot.columns:
            pivot[col] = np.nan

    labels_roles = pivot.apply(_classify_term_structure, axis=1)
    pivot["term_structure_label"] = [item[0] for item in labels_roles]
    pivot["term_structure_role"] = [item[1] for item in labels_roles]
    pivot["term_best_horizon"] = pivot.apply(lambda row: _term_best_horizon(row, str(row.get("event_side", ""))), axis=1)
    return pivot


def _score_utility_row(row: pd.Series) -> float:
    mean_return = float(row.get("mean_return", np.nan))
    median_return = float(row.get("median_return", np.nan))
    if str(row.get("event_side")) == "open":
        raw_edge = min(mean_return, median_return) if np.isfinite(median_return) else mean_return
    else:
        raw_edge = -max(mean_return, median_return) if np.isfinite(median_return) else -mean_return
    return float(max(raw_edge, 0.0)) if np.isfinite(raw_edge) else 0.0


def _utility_label(score: float, count: float, min_events: int, min_abs_score: float) -> str:
    if not np.isfinite(count) or count < min_events:
        return "invalid"
    if not np.isfinite(score) or score <= min_abs_score:
        return "invalid"
    return "valid"


def build_factor_signal_utility(
    event_summary: pd.DataFrame,
    output_dir: str | Path | None = None,
    horizons: Iterable[int] = UTILITY_HORIZONS,
    min_events: int = DEFAULT_MIN_EVENTS,
    min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
) -> pd.DataFrame:
    """从事件研究结果生成分类、期限结构和有效性标签。"""
    columns = [
        CODE_COL,
        "factor",
        "factor_base",
        "frequency",
        "pattern",
        "event_side",
        "signal_family",
        "subjective_usage_role",
        "term_structure_label",
        "quantitative_usage_role",
        "usage_role",
        "utility_label",
        "score",
        "event_count",
        "best_horizon",
        "term_best_horizon",
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_15d",
        "return_20d",
        "return_60d",
        "mean_return",
        "median_return",
        "win_rate",
        "p25_return",
        "p75_return",
    ]
    if event_summary.empty:
        utility = pd.DataFrame(columns=columns)
    else:
        data = event_summary.copy()
        data = data[data["horizon"].isin(tuple(horizons))].copy()
        for col in ("count", "mean_return", "median_return", "win_rate", "p25_return", "p75_return"):
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        data["score"] = data.apply(_score_utility_row, axis=1)
        term_frame = _term_structure_frame(data, "condition")
        data = data.sort_values(
            ["factor", "condition", "event_side", "score", "count"],
            ascending=[True, True, True, False, False],
            na_position="last",
        )
        best = data.groupby(["factor", "condition", "event_side"], as_index=False).head(1).copy()
        best["signal_family"] = best["condition"].map(_pattern_family)
        best["subjective_usage_role"] = best.apply(lambda row: _role_from_pattern(row["condition"], row["event_side"]), axis=1)
        merge_keys = [col for col in (CODE_COL, "factor", "condition", "event_side") if col in best.columns and col in term_frame.columns]
        if not term_frame.empty and merge_keys:
            best = best.merge(term_frame, on=merge_keys, how="left")
        else:
            best["term_structure_label"] = "期限结构不清晰"
            best["term_structure_role"] = np.nan
            best["term_best_horizon"] = np.nan
            for horizon in UTILITY_HORIZONS:
                best[f"return_{horizon}d"] = np.nan
        best["quantitative_usage_role"] = best["term_structure_role"]
        best["usage_role"] = best["quantitative_usage_role"].where(best["quantitative_usage_role"].notna(), best["subjective_usage_role"])
        best["utility_label"] = best.apply(
            lambda row: _utility_label(float(row["score"]), float(row["count"]), min_events, min_abs_score),
            axis=1,
        )
        best.loc[best["utility_label"].eq("invalid"), "usage_role"] = "invalid"
        best["factor_base"] = best["factor"].map(lambda value: _split_factor_frequency(str(value))[0])
        best["frequency"] = best["factor"].map(lambda value: _split_factor_frequency(str(value))[1])
        utility = best.rename(
            columns={"condition": "pattern", "count": "event_count", "horizon": "best_horizon"}
        )
        for col in columns:
            if col not in utility.columns:
                utility[col] = np.nan
        utility = utility[columns].reset_index(drop=True)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(utility, output_path / "factor_signal_utility.csv")
    return utility


def _resolve_signal_window(
    signal_table: pd.DataFrame,
    start_date=None,
    end_date=None,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, float, int]:
    if signal_table.empty or SIGNAL_DATE_COL not in signal_table.columns:
        return pd.NaT, pd.NaT, np.nan, 0

    signal_dates = pd.to_datetime(signal_table[SIGNAL_DATE_COL], errors="coerce").dropna()
    if signal_dates.empty:
        return pd.NaT, pd.NaT, np.nan, 0

    start_ts = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp(signal_dates.min())
    end_ts = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(signal_dates.max())
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    window_days = max((end_ts - start_ts).days + 1, 1)
    window_years = window_days / 365.25
    return start_ts, end_ts, window_years, window_days


def filter_signal_candidates(
    utility: pd.DataFrame,
    signal_table: pd.DataFrame,
    output_dir: str | Path | None = None,
    min_events_per_year: float = DEFAULT_MIN_EVENTS_PER_YEAR,
    score_rank_pct_threshold: float = DEFAULT_SCORE_RANK_PCT_THRESHOLD,
    aux_score_rank_pct_threshold: float = DEFAULT_AUX_SCORE_RANK_PCT_THRESHOLD,
    max_rules_per_bucket: int = DEFAULT_MAX_RULES_PER_BUCKET,
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter rules and signal points before expensive expanding backtests.

    Assumption:
    `utility` and `signal_table` should come from the same history window. For an
    expanding snapshot, pass the snapshot utility plus the signal table clipped to
    that snapshot's date range.
    """
    if utility.empty:
        empty_utility = utility.copy()
        empty_signals = signal_table.iloc[0:0].copy()
        return empty_utility, empty_signals

    start_ts, end_ts, window_years, window_days = _resolve_signal_window(
        signal_table,
        start_date=start_date,
        end_date=end_date,
    )
    dynamic_min_event_count = int(math.ceil(max(window_years, 0.0) * float(min_events_per_year))) if np.isfinite(window_years) else 0

    scored = utility.copy()
    scored["event_count"] = pd.to_numeric(scored.get("event_count"), errors="coerce")
    scored["score"] = pd.to_numeric(scored.get("score"), errors="coerce")
    scored["window_start"] = start_ts
    scored["window_end"] = end_ts
    scored["window_days"] = window_days
    scored["window_years"] = window_years
    scored["dynamic_min_event_count"] = dynamic_min_event_count
    scored["passes_utility"] = scored.get("utility_label", pd.Series(index=scored.index, dtype=object)).eq("valid")
    scored["passes_event_density"] = scored["event_count"].ge(dynamic_min_event_count)
    scored["passes_term_structure"] = scored.get(
        "term_structure_label",
        pd.Series(index=scored.index, dtype=object),
    ).ne("期限结构不清晰")

    rank_group_cols = [
        col
        for col in (CODE_COL, "factor_base", "event_side", "frequency")
        if col in scored.columns
    ]
    if rank_group_cols:
        scored["score_rank_pct"] = scored.groupby(rank_group_cols)["score"].rank(
            method="average",
            pct=True,
        )
    else:
        scored["score_rank_pct"] = scored["score"].rank(method="average", pct=True)

    score_threshold = np.where(
        scored.get("usage_role", pd.Series(index=scored.index, dtype=object)).eq("aux_filter"),
        aux_score_rank_pct_threshold,
        score_rank_pct_threshold,
    )
    scored["score_rank_pct_threshold"] = score_threshold
    scored["passes_score_rank"] = scored["score_rank_pct"].ge(scored["score_rank_pct_threshold"])
    scored["selected_flag"] = (
        scored["passes_utility"]
        & scored["passes_event_density"]
        & scored["passes_term_structure"]
        & scored["passes_score_rank"]
    )

    selected = scored[scored["selected_flag"]].copy()
    if not selected.empty:
        bucket_cols = [
            col
            for col in (CODE_COL, "factor_base", "usage_role", "event_side", "frequency")
            if col in selected.columns
        ]
        selected = selected.sort_values(
            ["score", "event_count", "factor", "pattern"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        if bucket_cols:
            selected["bucket_rank"] = selected.groupby(bucket_cols).cumcount() + 1
        else:
            selected["bucket_rank"] = np.arange(len(selected)) + 1
        selected = selected[selected["bucket_rank"].le(int(max_rules_per_bucket))].copy()
    else:
        selected["bucket_rank"] = pd.Series(dtype=float)

    filtered_signals = signal_table.iloc[0:0].copy()
    if not selected.empty and not signal_table.empty:
        merge_left = signal_table.copy()
        merge_right = selected.copy()
        if SIGNAL_INSTRUMENT_COL in merge_left.columns and CODE_COL in merge_right.columns:
            merge_left["_instrument_key"] = merge_left[SIGNAL_INSTRUMENT_COL].astype(str)
            merge_right["_instrument_key"] = merge_right[CODE_COL].astype(str)
            key_cols = ["_instrument_key", SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL]
        else:
            key_cols = [SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL]
        filtered_signals = merge_left.merge(
            merge_right[key_cols].drop_duplicates(),
            on=key_cols,
            how="inner",
        ).drop(columns=["_instrument_key"], errors="ignore")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(scored, output_path / "factor_signal_filter_diagnostics.csv")
        write_table(selected, output_path / "filtered_factor_signal_utility.csv")
        write_table(filtered_signals, output_path / "filtered_signals.csv")

    return selected.reset_index(drop=True), filtered_signals.reset_index(drop=True)


def _precompute_utility_observations(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    horizons: Iterable[int] = UTILITY_HORIZONS,
) -> pd.DataFrame:
    """预计算 signal x horizon 的未来收益明细，用于 expanding utility 快照。"""
    if df.empty or signal_table.empty:
        return pd.DataFrame()

    horizon_values = tuple(int(h) for h in horizons)
    rows: list[pd.DataFrame] = []
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])

    for instrument, group in data.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        dates = pd.to_datetime(group[DATE_COL])
        prices = pd.to_numeric(group[PRICE_COL], errors="coerce").to_numpy()
        date_to_pos = pd.Series(np.arange(len(dates)), index=dates).to_dict()
        sig = signals[signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))].copy()
        if sig.empty:
            continue
        sig["_signal_pos"] = sig[SIGNAL_DATE_COL].map(date_to_pos)
        sig = sig.dropna(subset=["_signal_pos"]).copy()
        if sig.empty:
            continue
        sig["_signal_pos"] = sig["_signal_pos"].astype(int)
        sig["event_side"] = [
            _event_side_from_signal(value, pattern)
            for value, pattern in zip(sig[SIGNAL_VALUE_COL], sig[SIGNAL_PATTERN_COL])
        ]
        base_cols = [
            SIGNAL_FACTOR_COL,
            SIGNAL_PATTERN_COL,
            SIGNAL_DATE_COL,
            "event_side",
            "_signal_pos",
        ]

        signal_pos = sig["_signal_pos"].to_numpy()
        entry_idx = signal_pos + 1
        entry_valid = (entry_idx >= 0) & (entry_idx < len(prices))
        if entry_valid.any():
            valid_pos = np.flatnonzero(entry_valid)
            price_valid = np.isfinite(prices[entry_idx[valid_pos]]) & (prices[entry_idx[valid_pos]] > 0)
            entry_valid[valid_pos] = price_valid
        if not entry_valid.any():
            continue

        for horizon in horizon_values:
            exit_idx = entry_idx + int(horizon)
            valid = entry_valid & (exit_idx < len(prices))
            if valid.any():
                valid_pos = np.flatnonzero(valid)
                price_valid = np.isfinite(prices[exit_idx[valid_pos]]) & (prices[exit_idx[valid_pos]] > 0)
                valid[valid_pos] = price_valid
            if not valid.any():
                continue
            part = sig.loc[valid, base_cols].copy()
            part[CODE_COL] = instrument
            part["horizon"] = int(horizon)
            part["forward_return"] = prices[exit_idx[valid]] / prices[entry_idx[valid]] - 1
            part["entry_date"] = dates.iloc[entry_idx[valid]].to_numpy()
            part["exit_date"] = dates.iloc[exit_idx[valid]].to_numpy()
            rows.append(part)

    if not rows:
        return pd.DataFrame()

    observations = pd.concat(rows, ignore_index=True)
    observations["entry_date"] = pd.to_datetime(observations["entry_date"])
    observations["exit_date"] = pd.to_datetime(observations["exit_date"])
    return observations


def _snapshot_utility_from_observations(
    utility_observations: pd.DataFrame,
    max_exit_date,
    min_signal_date=None,
    min_events: int = DEFAULT_MIN_EVENTS,
    min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
) -> pd.DataFrame:
    """从已完成收益观测中构建截至某日的 utility 快照。"""
    if utility_observations.empty:
        return pd.DataFrame()

    valid = utility_observations[
        pd.to_datetime(utility_observations["exit_date"]) <= pd.Timestamp(max_exit_date)
    ].copy()
    if min_signal_date is not None and not valid.empty:
        valid = valid[pd.to_datetime(valid[SIGNAL_DATE_COL]) >= pd.Timestamp(min_signal_date)].copy()
    if valid.empty:
        return pd.DataFrame()

    group_cols = [CODE_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, "event_side", "horizon"]
    valid["_positive_return"] = pd.to_numeric(valid["forward_return"], errors="coerce").gt(0).astype(float)
    event_summary = valid.groupby(group_cols, as_index=False).agg(
        count=("forward_return", "count"),
        mean_return=("forward_return", "mean"),
        median_return=("forward_return", "median"),
        win_rate=("_positive_return", "mean"),
    )
    quantiles = (
        valid.groupby(group_cols)["forward_return"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "p25_return", 0.75: "p75_return"})
        .reset_index()
    )
    event_summary = event_summary.merge(quantiles, on=group_cols, how="left")
    event_summary = event_summary.rename(
        columns={
            SIGNAL_FACTOR_COL: "factor",
            SIGNAL_PATTERN_COL: "condition",
        }
    )
    return build_factor_signal_utility(
        event_summary,
        output_dir=None,
        horizons=UTILITY_HORIZONS,
        min_events=min_events,
        min_abs_score=min_abs_score,
    )


def _event_side_from_signal(value: Any, pattern: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return "open" if float(numeric) > 0 else "close"
    return "open" if str(pattern).startswith("开仓_") else "close"


def _edge_score_frame(summary: pd.DataFrame, min_count: int) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    data = summary.copy()
    data["raw_edge"] = np.where(
        data["event_side"].eq("open"),
        np.minimum(data["mean_return"], data["median_return"]),
        -np.maximum(data["mean_return"], data["median_return"]),
    )
    data["edge"] = pd.to_numeric(data["raw_edge"], errors="coerce").clip(lower=0)
    data["standard_error"] = data["std_return"] / np.sqrt(data["count"].replace(0, np.nan))
    zero_se = data["standard_error"].eq(0) & data["edge"].gt(0)
    data["t_value"] = data["edge"] / data["standard_error"]
    data.loc[zero_se, "t_value"] = np.inf
    data["confidence"] = (data["t_value"].abs() / EDGE_T_FULL_CONFIDENCE).clip(lower=0, upper=1)
    invalid = data["count"].lt(min_count) | data["edge"].isna() | data["standard_error"].isna() | (data["standard_error"].le(0) & ~zero_se)
    data.loc[invalid, ["confidence", "t_value"]] = 0.0
    data["score"] = data["edge"] * data["confidence"]
    return data.replace([np.inf, -np.inf], np.nan)


def _precompute_edge_observations(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    age_grid: Iterable[int] = EDGE_AGE_GRID,
    forward_horizon: int = EDGE_FORWARD_HORIZON,
) -> pd.DataFrame:
    """预计算所有 signal x age x forward_horizon 的 forward_return 明细。

    返回结果包含 `exit_date`，后续 expanding window 只使用
    `exit_date <= eval_date` 的已完成收益观测，避免未来函数。
    """
    if df.empty or signal_table.empty:
        return pd.DataFrame()

    ages = tuple(int(age) for age in age_grid)
    rows: list[pd.DataFrame] = []
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])

    for instrument, group in data.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        dates = pd.to_datetime(group[DATE_COL])
        prices = pd.to_numeric(group[PRICE_COL], errors="coerce").to_numpy()
        date_to_pos = pd.Series(np.arange(len(dates)), index=dates).to_dict()
        sig = signals[signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))].copy()
        if sig.empty:
            continue
        sig["_signal_pos"] = sig[SIGNAL_DATE_COL].map(date_to_pos)
        sig = sig.dropna(subset=["_signal_pos"]).copy()
        if sig.empty:
            continue
        sig["_signal_pos"] = sig["_signal_pos"].astype(int)
        sig["event_side"] = [
            _event_side_from_signal(value, pattern)
            for value, pattern in zip(sig[SIGNAL_VALUE_COL], sig[SIGNAL_PATTERN_COL])
        ]
        base_cols = [SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, "event_side", "_signal_pos", SIGNAL_DATE_COL]

        for age in ages:
            base_idx = sig["_signal_pos"].to_numpy() + 1 + age
            exit_idx = base_idx + int(forward_horizon)
            valid = (base_idx >= 0) & (exit_idx < len(prices))
            if valid.any():
                valid_pos = np.flatnonzero(valid)
                price_valid = (
                    np.isfinite(prices[base_idx[valid_pos]])
                    & np.isfinite(prices[exit_idx[valid_pos]])
                    & (prices[base_idx[valid_pos]] > 0)
                    & (prices[exit_idx[valid_pos]] > 0)
                )
                valid[valid_pos] = price_valid
            if not valid.any():
                continue
            part = sig.loc[valid, base_cols].copy()
            part[CODE_COL] = instrument
            part["age"] = age
            part["forward_horizon"] = int(forward_horizon)
            part["forward_return"] = prices[exit_idx[valid]] / prices[base_idx[valid]] - 1
            part["exit_date"] = dates.iloc[exit_idx[valid]].to_numpy()
            part["base_date"] = dates.iloc[base_idx[valid]].to_numpy()
            rows.append(part)

    if rows:
        observations = pd.concat(rows, ignore_index=True)
        observations["exit_date"] = pd.to_datetime(observations["exit_date"])
        observations["base_date"] = pd.to_datetime(observations["base_date"])
        return observations
    return pd.DataFrame()


def _snapshot_edge_decay(
    observations: pd.DataFrame,
    max_exit_date,
    min_signal_date=None,
    min_count: int = EDGE_MIN_COUNT,
) -> pd.DataFrame:
    """从预计算观测中截取已完成样本，生成 edge decay 统计。

    `max_exit_date` 是防未来函数的关键参数，只有
    `exit_date <= max_exit_date` 的观测才能参与统计。
    """
    if observations.empty:
        return pd.DataFrame()

    valid = observations[pd.to_datetime(observations["exit_date"]) <= pd.Timestamp(max_exit_date)].copy()
    if min_signal_date is not None and not valid.empty:
        valid = valid[pd.to_datetime(valid[SIGNAL_DATE_COL]) >= pd.Timestamp(min_signal_date)]
    if valid.empty:
        return pd.DataFrame()

    group_cols = [CODE_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, "event_side", "age", "forward_horizon"]
    valid["_positive_return"] = pd.to_numeric(valid["forward_return"], errors="coerce").gt(0).astype(float)
    edge_decay = valid.groupby(group_cols, as_index=False).agg(
        count=("forward_return", "count"),
        mean_return=("forward_return", "mean"),
        median_return=("forward_return", "median"),
        std_return=("forward_return", "std"),
        win_rate=("_positive_return", "mean"),
    )
    quantiles = (
        valid.groupby(group_cols)["forward_return"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "p25_return", 0.75: "p75_return"})
        .reset_index()
    )
    edge_decay = edge_decay.merge(quantiles, on=group_cols, how="left")
    edge_decay = edge_decay.rename(columns={SIGNAL_FACTOR_COL: "factor", SIGNAL_PATTERN_COL: "pattern"})
    edge_decay = _edge_score_frame(edge_decay, min_count=min_count)
    return edge_decay


def _snapshot_score_block(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    decay_daily: pd.DataFrame,
    eval_dates: Iterable[Any],
) -> pd.DataFrame:
    """Vectorized score for a block of eval dates using the same decay table.

    Recent trigger definition:
    age = eval_pos - signal_pos - 1.
    Active signals satisfy 0 <= age <= max_age, so same-day signals are not used.
    """
    eval_index = pd.DatetimeIndex(pd.to_datetime(list(eval_dates))).dropna().unique().sort_values()
    if len(eval_index) == 0:
        return pd.DataFrame()

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    skeleton_rows: list[pd.DataFrame] = []
    for instrument, group in data.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(group[DATE_COL]))
        eval_dates_inst = eval_index.intersection(dates)
        if len(eval_dates_inst) == 0:
            continue
        skeleton_rows.append(
            pd.DataFrame(
                {
                    CODE_COL: instrument,
                    DATE_COL: eval_dates_inst.to_numpy(),
                }
            )
        )
    if not skeleton_rows:
        return pd.DataFrame()

    skeleton = pd.concat(skeleton_rows, ignore_index=True)
    skeleton["entry_score"] = 0.0
    skeleton["exit_score"] = 0.0
    skeleton["active_entry_signal_count"] = 0
    skeleton["active_exit_signal_count"] = 0

    if decay_daily.empty or signal_table.empty:
        skeleton["net_score"] = 0.0
        return skeleton

    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])
    max_age = int(decay_daily["age"].max())
    rows: list[pd.DataFrame] = []

    for instrument, group in data.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(group[DATE_COL]))
        eval_dates_inst = eval_index.intersection(dates)
        if len(eval_dates_inst) == 0:
            continue

        date_to_pos = pd.Series(np.arange(len(dates)), index=dates).to_dict()
        eval_pos = np.array([date_to_pos[d] for d in eval_dates_inst], dtype=int)
        max_eval_date = eval_dates_inst.max()
        max_eval_pos = int(eval_pos.max())

        sig = signals[
            signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))
            & signals[SIGNAL_DATE_COL].le(max_eval_date)
        ].copy()
        if sig.empty:
            continue
        sig["_signal_pos"] = sig[SIGNAL_DATE_COL].map(date_to_pos)
        sig = sig.dropna(subset=["_signal_pos"]).copy()
        if sig.empty:
            continue
        sig["_signal_pos"] = sig["_signal_pos"].astype(int)
        sig = sig[sig["_signal_pos"].lt(max_eval_pos)]
        if sig.empty:
            continue
        sig["event_side"] = [
            _event_side_from_signal(v, p)
            for v, p in zip(sig[SIGNAL_VALUE_COL], sig[SIGNAL_PATTERN_COL])
        ]
        sig["factor"] = sig[SIGNAL_FACTOR_COL]
        sig["pattern"] = sig[SIGNAL_PATTERN_COL]
        sig[CODE_COL] = instrument

        base_cols = [CODE_COL, "factor", "pattern", "event_side", "_signal_pos"]
        signal_pos = sig["_signal_pos"].to_numpy()
        for age in range(max_age + 1):
            current_pos = signal_pos + 1 + age
            valid = np.isin(current_pos, eval_pos)
            if not valid.any():
                continue
            part = sig.loc[valid, base_cols].copy()
            part["age"] = age
            part[DATE_COL] = dates[current_pos[valid]].to_numpy()
            part = part.merge(
                decay_daily[[CODE_COL, "factor", "pattern", "event_side", "age", "score"]],
                on=[CODE_COL, "factor", "pattern", "event_side", "age"],
                how="inner",
            )
            part = part[part["score"].gt(0)]
            if not part.empty:
                rows.append(part)

    if not rows:
        skeleton["net_score"] = 0.0
        return skeleton

    contrib = pd.concat(rows, ignore_index=True)
    grouped = contrib.groupby([CODE_COL, DATE_COL, "event_side"]).agg(
        score=("score", "sum"),
        signal_count=("score", "size"),
    ).reset_index()
    pivot_score = grouped.pivot_table(
        index=[CODE_COL, DATE_COL],
        columns="event_side",
        values="score",
        fill_value=0.0,
    )
    pivot_count = grouped.pivot_table(
        index=[CODE_COL, DATE_COL],
        columns="event_side",
        values="signal_count",
        fill_value=0,
    )
    block = pivot_score.reset_index().rename(columns={"open": "entry_score", "close": "exit_score"})
    block_count = pivot_count.reset_index().rename(
        columns={"open": "active_entry_signal_count", "close": "active_exit_signal_count"}
    )
    block = skeleton.merge(block, on=[CODE_COL, DATE_COL], how="left", suffixes=("_base", ""))
    block = block.drop(
        columns=[
            "entry_score_base",
            "exit_score_base",
            "active_entry_signal_count_base",
            "active_exit_signal_count_base",
        ],
        errors="ignore",
    )
    block = block.merge(block_count, on=[CODE_COL, DATE_COL], how="left")
    for col in ("entry_score", "exit_score", "active_entry_signal_count", "active_exit_signal_count"):
        if col not in block:
            block[col] = 0
        block[col] = pd.to_numeric(block[col], errors="coerce").fillna(0.0)
    block["net_score"] = block["entry_score"] - block["exit_score"]
    return block


def _snapshot_score_block_matrix(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    decay_daily: pd.DataFrame,
    eval_dates: Iterable[Any],
    return_contrib: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Matrix-based daily score aggregation for a block of eval dates."""
    eval_index = pd.DatetimeIndex(pd.to_datetime(list(eval_dates))).dropna().unique().sort_values()
    if len(eval_index) == 0:
        return pd.DataFrame()

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    skeleton_rows: list[pd.DataFrame] = []
    for instrument, group in data.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(group[DATE_COL]))
        eval_dates_inst = eval_index.intersection(dates)
        if len(eval_dates_inst) == 0:
            continue
        skeleton_rows.append(
            pd.DataFrame(
                {
                    CODE_COL: instrument,
                    DATE_COL: eval_dates_inst.to_numpy(),
                }
            )
        )
    if not skeleton_rows:
        return pd.DataFrame()

    skeleton = pd.concat(skeleton_rows, ignore_index=True)
    skeleton["entry_score"] = 0.0
    skeleton["exit_score"] = 0.0
    skeleton["active_entry_signal_count"] = 0.0
    skeleton["active_exit_signal_count"] = 0.0
    contrib_frames: list[pd.DataFrame] = []

    if decay_daily.empty or signal_table.empty:
        skeleton["net_score"] = 0.0
        if return_contrib:
            return skeleton, pd.DataFrame(
                columns=[CODE_COL, DATE_COL, "signal_date", "factor", "pattern", "event_side", "age", "score"]
            )
        return skeleton

    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])
    rows: list[pd.DataFrame] = []

    for instrument, group in data.groupby(CODE_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(group[DATE_COL]))
        eval_dates_inst = eval_index.intersection(dates)
        if len(eval_dates_inst) == 0:
            continue

        date_to_pos = pd.Series(np.arange(len(dates)), index=dates).to_dict()
        start_pos = int(date_to_pos[eval_dates_inst[0]])
        end_pos = int(date_to_pos[eval_dates_inst[-1]])
        max_eval_date = eval_dates_inst[-1]

        sig = signals[
            signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))
            & signals[SIGNAL_DATE_COL].le(max_eval_date)
        ].copy()
        if sig.empty:
            rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: instrument,
                        DATE_COL: eval_dates_inst.to_numpy(),
                        "entry_score": 0.0,
                        "exit_score": 0.0,
                        "active_entry_signal_count": 0.0,
                        "active_exit_signal_count": 0.0,
                    }
                )
            )
            continue

        sig["_signal_pos"] = sig[SIGNAL_DATE_COL].map(date_to_pos)
        sig = sig.dropna(subset=["_signal_pos"]).copy()
        if sig.empty:
            rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: instrument,
                        DATE_COL: eval_dates_inst.to_numpy(),
                        "entry_score": 0.0,
                        "exit_score": 0.0,
                        "active_entry_signal_count": 0.0,
                        "active_exit_signal_count": 0.0,
                    }
                )
            )
            continue

        sig["_signal_pos"] = sig["_signal_pos"].astype(int)
        sig = sig[sig["_signal_pos"].lt(end_pos)].copy()
        if sig.empty:
            rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: instrument,
                        DATE_COL: eval_dates_inst.to_numpy(),
                        "entry_score": 0.0,
                        "exit_score": 0.0,
                        "active_entry_signal_count": 0.0,
                        "active_exit_signal_count": 0.0,
                    }
                )
            )
            continue

        sig["event_side"] = [
            _event_side_from_signal(v, p)
            for v, p in zip(sig[SIGNAL_VALUE_COL], sig[SIGNAL_PATTERN_COL])
        ]
        sig["factor"] = sig[SIGNAL_FACTOR_COL]
        sig["pattern"] = sig[SIGNAL_PATTERN_COL]
        sig[CODE_COL] = instrument

        decay_part = decay_daily[decay_daily[CODE_COL].astype(str).eq(str(instrument))].copy()
        if decay_part.empty:
            rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: instrument,
                        DATE_COL: eval_dates_inst.to_numpy(),
                        "entry_score": 0.0,
                        "exit_score": 0.0,
                        "active_entry_signal_count": 0.0,
                        "active_exit_signal_count": 0.0,
                    }
                )
            )
            continue

        max_age = int(pd.to_numeric(decay_part["age"], errors="coerce").max())
        min_signal_pos = max(0, start_pos - max_age - 1)
        sig = sig[sig["_signal_pos"].between(min_signal_pos, end_pos - 1)].copy()
        if sig.empty:
            rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: instrument,
                        DATE_COL: eval_dates_inst.to_numpy(),
                        "entry_score": 0.0,
                        "exit_score": 0.0,
                        "active_entry_signal_count": 0.0,
                        "active_exit_signal_count": 0.0,
                    }
                )
            )
            continue

        rule_keys = ["factor", "pattern", "event_side"]
        rule_map = decay_part[rule_keys].drop_duplicates().reset_index(drop=True)
        rule_map["rule_id"] = np.arange(len(rule_map), dtype=int)
        decay_part = decay_part.merge(rule_map, on=rule_keys, how="left")
        sig = sig.merge(rule_map, on=rule_keys, how="inner")
        if sig.empty:
            rows.append(
                pd.DataFrame(
                    {
                        CODE_COL: instrument,
                        DATE_COL: eval_dates_inst.to_numpy(),
                        "entry_score": 0.0,
                        "exit_score": 0.0,
                        "active_entry_signal_count": 0.0,
                        "active_exit_signal_count": 0.0,
                    }
                )
            )
            continue

        score_matrix = np.zeros((len(rule_map), max_age + 1), dtype=float)
        rule_idx = decay_part["rule_id"].to_numpy(dtype=int)
        age_idx = pd.to_numeric(decay_part["age"], errors="coerce").to_numpy(dtype=int)
        score_vals = pd.to_numeric(decay_part["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        score_matrix[rule_idx, age_idx] = score_vals

        signal_pos = sig["_signal_pos"].to_numpy(dtype=int)
        signal_rule_id = sig["rule_id"].to_numpy(dtype=int)
        signal_side = sig["event_side"].to_numpy(dtype=object)
        ages = np.arange(max_age + 1, dtype=int)
        pos_matrix = signal_pos[:, None] + 1 + ages[None, :]
        score_lookup = score_matrix[signal_rule_id]

        valid = (pos_matrix >= start_pos) & (pos_matrix <= end_pos) & (score_lookup > 0)
        open_mask = valid & (signal_side[:, None] == "open")
        close_mask = valid & (signal_side[:, None] == "close")

        block_len = len(eval_dates_inst)
        if open_mask.any():
            open_local_idx = pos_matrix[open_mask] - start_pos
            entry_score = np.bincount(open_local_idx, weights=score_lookup[open_mask], minlength=block_len)
            entry_count = np.bincount(open_local_idx, minlength=block_len)
        else:
            entry_score = np.zeros(block_len, dtype=float)
            entry_count = np.zeros(block_len, dtype=float)

        if close_mask.any():
            close_local_idx = pos_matrix[close_mask] - start_pos
            exit_score = np.bincount(close_local_idx, weights=score_lookup[close_mask], minlength=block_len)
            exit_count = np.bincount(close_local_idx, minlength=block_len)
        else:
            exit_score = np.zeros(block_len, dtype=float)
            exit_count = np.zeros(block_len, dtype=float)

        if return_contrib:
            signal_dates = pd.to_datetime(sig[SIGNAL_DATE_COL]).to_numpy()
            factor_values = sig["factor"].to_numpy()
            pattern_values = sig["pattern"].to_numpy()
            side_values = sig["event_side"].to_numpy()

            if open_mask.any():
                signal_idx, age_idx = np.nonzero(open_mask)
                eval_pos_values = pos_matrix[open_mask]
                contrib_frames.append(
                    pd.DataFrame(
                        {
                            CODE_COL: instrument,
                            DATE_COL: dates[eval_pos_values].to_numpy(),
                            "signal_date": signal_dates[signal_idx],
                            "factor": factor_values[signal_idx],
                            "pattern": pattern_values[signal_idx],
                            "event_side": side_values[signal_idx],
                            "age": age_idx,
                            "score": score_lookup[open_mask],
                        }
                    )
                )
            if close_mask.any():
                signal_idx, age_idx = np.nonzero(close_mask)
                eval_pos_values = pos_matrix[close_mask]
                contrib_frames.append(
                    pd.DataFrame(
                        {
                            CODE_COL: instrument,
                            DATE_COL: dates[eval_pos_values].to_numpy(),
                            "signal_date": signal_dates[signal_idx],
                            "factor": factor_values[signal_idx],
                            "pattern": pattern_values[signal_idx],
                            "event_side": side_values[signal_idx],
                            "age": age_idx,
                            "score": score_lookup[close_mask],
                        }
                    )
                )

        rows.append(
            pd.DataFrame(
                {
                    CODE_COL: instrument,
                    DATE_COL: eval_dates_inst.to_numpy(),
                    "entry_score": entry_score,
                    "exit_score": exit_score,
                    "active_entry_signal_count": entry_count,
                    "active_exit_signal_count": exit_count,
                }
            )
        )

    if not rows:
        skeleton["net_score"] = 0.0
        if return_contrib:
            return skeleton, pd.DataFrame(
                columns=[CODE_COL, DATE_COL, "signal_date", "factor", "pattern", "event_side", "age", "score"]
            )
        return skeleton

    block = pd.concat(rows, ignore_index=True)
    block = skeleton[[CODE_COL, DATE_COL]].merge(block, on=[CODE_COL, DATE_COL], how="left")
    for col in ("entry_score", "exit_score", "active_entry_signal_count", "active_exit_signal_count"):
        block[col] = pd.to_numeric(block[col], errors="coerce").fillna(0.0)
    block["net_score"] = block["entry_score"] - block["exit_score"]
    if return_contrib:
        contrib = pd.concat(contrib_frames, ignore_index=True) if contrib_frames else pd.DataFrame(
            columns=[CODE_COL, DATE_COL, "signal_date", "factor", "pattern", "event_side", "age", "score"]
        )
        return block, contrib
    return block


def _attach_score_variants(timing: pd.DataFrame) -> pd.DataFrame:
    """Keep sum-based score as default and add mean / sqrt-adjusted variants."""
    if timing.empty:
        return timing

    work = timing.copy()
    entry_sum = pd.to_numeric(work.get("entry_score"), errors="coerce").fillna(0.0)
    exit_sum = pd.to_numeric(work.get("exit_score"), errors="coerce").fillna(0.0)
    entry_count = pd.to_numeric(work.get("active_entry_signal_count"), errors="coerce").fillna(0.0)
    exit_count = pd.to_numeric(work.get("active_exit_signal_count"), errors="coerce").fillna(0.0)

    entry_count_safe = entry_count.where(entry_count > 0, 1.0)
    exit_count_safe = exit_count.where(exit_count > 0, 1.0)
    entry_sqrt_safe = np.sqrt(entry_count_safe)
    exit_sqrt_safe = np.sqrt(exit_count_safe)

    work["entry_score_sum"] = entry_sum
    work["exit_score_sum"] = exit_sum
    work["net_score_sum"] = entry_sum - exit_sum

    work["entry_score_mean"] = entry_sum / entry_count_safe
    work.loc[entry_count.le(0), "entry_score_mean"] = 0.0
    work["exit_score_mean"] = exit_sum / exit_count_safe
    work.loc[exit_count.le(0), "exit_score_mean"] = 0.0
    work["net_score_mean"] = work["entry_score_mean"] - work["exit_score_mean"]

    work["entry_score_sqrtadj"] = entry_sum / entry_sqrt_safe
    work.loc[entry_count.le(0), "entry_score_sqrtadj"] = 0.0
    work["exit_score_sqrtadj"] = exit_sum / exit_sqrt_safe
    work.loc[exit_count.le(0), "exit_score_sqrtadj"] = 0.0
    work["net_score_sqrtadj"] = work["entry_score_sqrtadj"] - work["exit_score_sqrtadj"]

    return work


def _save_frame_artifact(df: pd.DataFrame, path_stem: Path) -> Path | None:
    """Save dataframe as parquet when available, fallback to pickle."""
    if df is None or df.empty:
        return None
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path_stem.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        pickle_path = path_stem.with_suffix(".pkl")
        df.to_pickle(pickle_path)
        return pickle_path


def _load_frame_artifact(path_stem: Path) -> pd.DataFrame:
    parquet_path = path_stem.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    pickle_path = path_stem.with_suffix(".pkl")
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    return pd.DataFrame()


def _build_role_score_frame(
    signal_contributions: pd.DataFrame,
    utility_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate daily role scores from single-signal contributions using monthly role labels."""
    if signal_contributions.empty or utility_snapshots.empty:
        return pd.DataFrame()

    contrib = signal_contributions.copy()
    utility = utility_snapshots.copy()
    contrib[DATE_COL] = pd.to_datetime(contrib[DATE_COL], errors="coerce")
    contrib["refresh_date"] = pd.to_datetime(contrib["refresh_date"], errors="coerce")
    utility["refresh_date"] = pd.to_datetime(utility["refresh_date"], errors="coerce")

    merge_cols = [CODE_COL, "factor", "pattern", "event_side", "refresh_date"]
    util_cols = merge_cols + ["usage_role"]
    merged = contrib.merge(
        utility[util_cols].drop_duplicates(),
        on=merge_cols,
        how="left",
    )
    merged["usage_role"] = merged["usage_role"].fillna("invalid")
    merged = merged[merged["usage_role"].isin(ROLE_ORDER[:6])].copy()
    if merged.empty:
        return pd.DataFrame()

    grouped = merged.groupby([CODE_COL, DATE_COL, "usage_role"], as_index=False).agg(
        score_mean=("score", "mean"),
        signal_count=("score", "size"),
    )

    score_pivot = grouped.pivot_table(
        index=[CODE_COL, DATE_COL],
        columns="usage_role",
        values="score_mean",
        fill_value=0.0,
    )
    count_pivot = grouped.pivot_table(
        index=[CODE_COL, DATE_COL],
        columns="usage_role",
        values="signal_count",
        fill_value=0,
    )

    score_rename = {role: f"{role}_score_mean" for role in ROLE_ORDER[:6]}
    count_rename = {role: f"{role}_signal_count" for role in ROLE_ORDER[:6]}
    score_frame = score_pivot.reset_index().rename(columns=score_rename)
    count_frame = count_pivot.reset_index().rename(columns=count_rename)
    role_frame = score_frame.merge(count_frame, on=[CODE_COL, DATE_COL], how="outer")

    for col in ROLE_SCORE_COLS:
        if col not in role_frame.columns:
            role_frame[col] = 0.0
        role_frame[col] = pd.to_numeric(role_frame[col], errors="coerce").fillna(0.0)
    for col in ROLE_COUNT_COLS:
        if col not in role_frame.columns:
            role_frame[col] = 0
        role_frame[col] = pd.to_numeric(role_frame[col], errors="coerce").fillna(0.0)

    return role_frame[[CODE_COL, DATE_COL, *ROLE_SCORE_COLS, *ROLE_COUNT_COLS]]


def _rolling_eval_score(
    eval_ts,
    data: pd.DataFrame,
    signal_table: pd.DataFrame,
    observations: pd.DataFrame,
    utility_observations: pd.DataFrame,
    min_count: int,
    filter_min_events: int,
    filter_min_abs_score: float,
    filter_min_events_per_year: float,
    filter_score_rank_pct_threshold: float,
    filter_aux_score_rank_pct_threshold: float,
    filter_max_rules_per_bucket: int,
) -> pd.DataFrame:
    eval_ts = pd.Timestamp(eval_ts)
    history_signals = signal_table[
        (pd.to_datetime(signal_table[SIGNAL_DATE_COL]) <= eval_ts)
    ].copy()
    if history_signals.empty:
        return _snapshot_score_block(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), [eval_ts])

    utility_snapshot = _snapshot_utility_from_observations(
        utility_observations,
        max_exit_date=eval_ts,
        min_events=filter_min_events,
        min_abs_score=filter_min_abs_score,
    )
    if utility_snapshot.empty:
        return _snapshot_score_block(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), [eval_ts])

    _, score_signal_table = filter_signal_candidates(
        utility_snapshot,
        history_signals,
        output_dir=None,
        min_events_per_year=filter_min_events_per_year,
        score_rank_pct_threshold=filter_score_rank_pct_threshold,
        aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
        max_rules_per_bucket=filter_max_rules_per_bucket,
        end_date=eval_ts,
    )
    if score_signal_table.empty:
        return _snapshot_score_block(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), [eval_ts])

    score_signal_table = score_signal_table.copy()
    score_signal_table[SIGNAL_DATE_COL] = pd.to_datetime(score_signal_table[SIGNAL_DATE_COL])
    allowed_rules = score_signal_table[
        [SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL]
    ].drop_duplicates().rename(columns={SIGNAL_INSTRUMENT_COL: CODE_COL})
    decay_observations = observations.merge(
        allowed_rules,
        on=[CODE_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL],
        how="inner",
    )
    if decay_observations.empty:
        return _snapshot_score_block(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), [eval_ts])

    decay = _snapshot_edge_decay(
        decay_observations,
        max_exit_date=eval_ts,
        min_count=min_count,
    )
    if decay.empty:
        return _snapshot_score_block(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), [eval_ts])

    decay_daily = _interpolate_edge_decay(decay)
    return _snapshot_score_block(data, score_signal_table, decay_daily, [eval_ts])


def _monthly_refresh_block_score(
    refresh_ts,
    block_dates: list[pd.Timestamp],
    data: pd.DataFrame,
    signal_table: pd.DataFrame,
    observations: pd.DataFrame,
    utility_observations: pd.DataFrame,
    min_count: int,
    filter_min_events: int,
    filter_min_abs_score: float,
    filter_min_events_per_year: float,
    filter_score_rank_pct_threshold: float,
    filter_aux_score_rank_pct_threshold: float,
    filter_max_rules_per_bucket: int,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    refresh_ts = pd.Timestamp(refresh_ts)
    block_end = pd.Timestamp(block_dates[-1])

    history_signals = signal_table[
        pd.to_datetime(signal_table[SIGNAL_DATE_COL]).le(refresh_ts)
    ].copy()
    if history_signals.empty:
        empty_block = _snapshot_score_block_matrix(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), block_dates)
        if return_diagnostics:
            return empty_block, {}
        return empty_block

    utility_snapshot = _snapshot_utility_from_observations(
        utility_observations,
        max_exit_date=refresh_ts,
        min_events=filter_min_events,
        min_abs_score=filter_min_abs_score,
    )
    if utility_snapshot.empty:
        empty_block = _snapshot_score_block_matrix(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), block_dates)
        if return_diagnostics:
            return empty_block, {}
        return empty_block

    selected_utility, _ = filter_signal_candidates(
        utility_snapshot,
        history_signals,
        output_dir=None,
        min_events_per_year=filter_min_events_per_year,
        score_rank_pct_threshold=filter_score_rank_pct_threshold,
        aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
        max_rules_per_bucket=filter_max_rules_per_bucket,
        end_date=refresh_ts,
    )
    if selected_utility.empty:
        empty_block = _snapshot_score_block_matrix(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), block_dates)
        if return_diagnostics:
            diagnostics = {
                "utility_snapshot": utility_snapshot.assign(refresh_date=refresh_ts, block_end_date=block_end),
                "selected_rules": selected_utility.assign(refresh_date=refresh_ts, block_end_date=block_end),
            }
            return empty_block, diagnostics
        return empty_block

    allowed_rules = selected_utility[
        [CODE_COL, "factor", "pattern"]
    ].drop_duplicates().rename(columns={"factor": SIGNAL_FACTOR_COL, "pattern": SIGNAL_PATTERN_COL})

    score_signal_table = signal_table.copy()
    score_signal_table[SIGNAL_DATE_COL] = pd.to_datetime(score_signal_table[SIGNAL_DATE_COL])
    score_signal_table = score_signal_table[
        score_signal_table[SIGNAL_DATE_COL].le(block_end)
    ].merge(
        allowed_rules.rename(columns={CODE_COL: SIGNAL_INSTRUMENT_COL}),
        on=[SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL],
        how="inner",
    )
    if score_signal_table.empty:
        empty_block = _snapshot_score_block_matrix(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), block_dates)
        if return_diagnostics:
            diagnostics = {
                "utility_snapshot": utility_snapshot.assign(refresh_date=refresh_ts, block_end_date=block_end),
                "selected_rules": selected_utility.assign(refresh_date=refresh_ts, block_end_date=block_end),
            }
            return empty_block, diagnostics
        return empty_block

    decay_observations = observations.merge(
        allowed_rules,
        on=[CODE_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL],
        how="inner",
    )
    if decay_observations.empty:
        empty_block = _snapshot_score_block_matrix(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), block_dates)
        if return_diagnostics:
            diagnostics = {
                "utility_snapshot": utility_snapshot.assign(refresh_date=refresh_ts, block_end_date=block_end),
                "selected_rules": selected_utility.assign(refresh_date=refresh_ts, block_end_date=block_end),
            }
            return empty_block, diagnostics
        return empty_block

    decay = _snapshot_edge_decay(
        decay_observations,
        max_exit_date=refresh_ts,
        min_count=min_count,
    )
    if decay.empty:
        empty_block = _snapshot_score_block_matrix(data, signal_table.iloc[0:0].copy(), pd.DataFrame(), block_dates)
        if return_diagnostics:
            diagnostics = {
                "utility_snapshot": utility_snapshot.assign(refresh_date=refresh_ts, block_end_date=block_end),
                "selected_rules": selected_utility.assign(refresh_date=refresh_ts, block_end_date=block_end),
            }
            return empty_block, diagnostics
        return empty_block

    decay_daily = _interpolate_edge_decay(decay)
    if return_diagnostics:
        block_score, block_contrib = _snapshot_score_block_matrix(
            data,
            score_signal_table,
            decay_daily,
            block_dates,
            return_contrib=True,
        )
        diagnostics = {
            "utility_snapshot": utility_snapshot.assign(refresh_date=refresh_ts, block_end_date=block_end),
            "selected_rules": selected_utility.assign(refresh_date=refresh_ts, block_end_date=block_end),
            "decay_snapshot": decay.assign(refresh_date=refresh_ts, block_end_date=block_end),
            "signal_contributions": block_contrib.assign(refresh_date=refresh_ts, block_end_date=block_end),
        }
        return block_score, diagnostics
    return _snapshot_score_block_matrix(data, score_signal_table, decay_daily, block_dates)


def run_daily_rolling_edge_timing(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    min_history_years: int = 2,
    warmup_years: int = 3,
    parallel_n_jobs: int = 1,
    batch_size: int = 20,
    age_grid: Iterable[int] = EDGE_AGE_GRID,
    forward_horizon: int = EDGE_FORWARD_HORIZON,
    min_count: int = EDGE_MIN_COUNT,
    filter_min_events: int = DEFAULT_MIN_EVENTS,
    filter_min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
    filter_min_events_per_year: float = DEFAULT_MIN_EVENTS_PER_YEAR,
    filter_score_rank_pct_threshold: float = DEFAULT_SCORE_RANK_PCT_THRESHOLD,
    filter_aux_score_rank_pct_threshold: float = DEFAULT_AUX_SCORE_RANK_PCT_THRESHOLD,
    filter_max_rules_per_bucket: int = DEFAULT_MAX_RULES_PER_BUCKET,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Daily rolling score with trailing-history rule selection.

    For each evaluation day:
    - use all history from the start date up to eval_date
    - evaluation starts only after at least `min_history_years` history is available
    - only use observations with completed `ret_60d` (`exit_date <= eval_date`)
    - generate a same-day score row even if no signal survives the filter
    """
    if df.empty or signal_table.empty:
        return pd.DataFrame()

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])

    print("预计算 edge observations ...")
    observations = _precompute_edge_observations(
        data,
        signals,
        age_grid=age_grid,
        forward_horizon=forward_horizon,
    )
    print(f"  共 {len(observations)} 条观测")
    if observations.empty:
        return pd.DataFrame()

    print("预计算 utility observations ...")
    utility_observations = _precompute_utility_observations(data, signals, horizons=UTILITY_HORIZONS)
    print(f"  共 {len(utility_observations)} 条 utility 观测")
    if utility_observations.empty:
        return pd.DataFrame()

    all_dates = sorted(data[DATE_COL].dropna().unique())
    if not all_dates:
        return pd.DataFrame()
    warmup_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=warmup_years)
    min_history_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=min_history_years)
    eval_start_date = max(warmup_date, min_history_date)
    eval_dates = [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d) >= eval_start_date]
    if not eval_dates:
        return pd.DataFrame()

    print(
        f"逐日滚动打分: {len(eval_dates)} 个交易日 "
        f"({pd.Timestamp(eval_dates[0]).date()} ~ {pd.Timestamp(eval_dates[-1]).date()})"
    )

    def _run_chunk(chunk_dates: list[pd.Timestamp]) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for eval_ts in chunk_dates:
            rows.append(
                _rolling_eval_score(
                    eval_ts=eval_ts,
                    data=data,
                    signal_table=signals,
                    observations=observations,
                    utility_observations=utility_observations,
                    min_count=min_count,
                    filter_min_events=filter_min_events,
                    filter_min_abs_score=filter_min_abs_score,
                    filter_min_events_per_year=filter_min_events_per_year,
                    filter_score_rank_pct_threshold=filter_score_rank_pct_threshold,
                    filter_aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
                    filter_max_rules_per_bucket=filter_max_rules_per_bucket,
                )
            )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    chunk_size = max(1, int(batch_size))
    chunks = [eval_dates[i : i + chunk_size] for i in range(0, len(eval_dates), chunk_size)]
    if Parallel is not None and int(parallel_n_jobs) not in (0, 1):
        results = Parallel(n_jobs=parallel_n_jobs, prefer="threads")(
            delayed(_run_chunk)(chunk) for chunk in chunks
        )
    else:
        results = []
        for idx, chunk in enumerate(chunks, start=1):
            if idx == 1 or idx % max(1, len(chunks) // 10) == 0:
                print(f"  处理批次 {idx}/{len(chunks)}")
            results.append(_run_chunk(chunk))

    timing = pd.concat([item for item in results if item is not None and not item.empty], ignore_index=True)
    if timing.empty:
        return pd.DataFrame()

    timing = timing.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    timing = _attach_score_variants(timing)
    if save_intermediates and diagnostics_list:
        utility_frames = [d["utility_snapshot"] for d in diagnostics_list if "utility_snapshot" in d and not d["utility_snapshot"].empty]
        contrib_frames = [d["signal_contributions"] for d in diagnostics_list if "signal_contributions" in d and not d["signal_contributions"].empty]
        if utility_frames and contrib_frames:
            role_frame = _build_role_score_frame(
                pd.concat(contrib_frames, ignore_index=True),
                pd.concat(utility_frames, ignore_index=True),
            )
            if not role_frame.empty:
                timing = timing.merge(role_frame, on=[CODE_COL, DATE_COL], how="left")
    for col in ROLE_SCORE_COLS:
        if col not in timing.columns:
            timing[col] = 0.0
        timing[col] = pd.to_numeric(timing[col], errors="coerce").fillna(0.0)
    for col in ROLE_COUNT_COLS:
        if col not in timing.columns:
            timing[col] = 0.0
        timing[col] = pd.to_numeric(timing[col], errors="coerce").fillna(0.0)
    timing["position_signal"] = np.select(
        [timing["net_score"].gt(0), timing["net_score"].lt(0)],
        ["多", "空"],
        default="观望",
    )
    names = data.groupby(CODE_COL)[NAME_COL].last().reset_index()
    timing = timing.merge(names, on=CODE_COL, how="left")
    timing = timing[
        [
            DATE_COL,
            CODE_COL,
            NAME_COL,
            "entry_score",
            "exit_score",
            "net_score",
            "entry_score_sum",
            "exit_score_sum",
            "net_score_sum",
            "entry_score_mean",
            "exit_score_mean",
            "net_score_mean",
            "entry_score_sqrtadj",
            "exit_score_sqrtadj",
            "net_score_sqrtadj",
            *ROLE_SCORE_COLS,
            *ROLE_COUNT_COLS,
            "active_entry_signal_count",
            "active_exit_signal_count",
            "position_signal",
        ]
    ]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(timing, output_path / "rolling_edge_timing.csv")
        print(f"  已写入 {output_path / 'rolling_edge_timing.csv'} (共 {len(timing)} 行)")

    print(f"完成：{len(timing)} 条逐日滚动记录")
    return timing


def run_monthly_refresh_daily_score(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    min_history_years: int = 2,
    warmup_years: int = 3,
    parallel_n_jobs: int = 1,
    age_grid: Iterable[int] = EDGE_AGE_GRID,
    forward_horizon: int = EDGE_FORWARD_HORIZON,
    min_count: int = EDGE_MIN_COUNT,
    filter_min_events: int = DEFAULT_MIN_EVENTS,
    filter_min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
    filter_min_events_per_year: float = DEFAULT_MIN_EVENTS_PER_YEAR,
    filter_score_rank_pct_threshold: float = DEFAULT_SCORE_RANK_PCT_THRESHOLD,
    filter_aux_score_rank_pct_threshold: float = DEFAULT_AUX_SCORE_RANK_PCT_THRESHOLD,
    filter_max_rules_per_bucket: int = DEFAULT_MAX_RULES_PER_BUCKET,
    save_intermediates: bool = False,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Monthly whitelist refresh with daily matrix-scored output."""
    if df.empty or signal_table.empty:
        return pd.DataFrame()

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])

    print("预计算 edge observations ...")
    observations = _precompute_edge_observations(
        data,
        signals,
        age_grid=age_grid,
        forward_horizon=forward_horizon,
    )
    print(f"  共 {len(observations)} 条观测")
    if observations.empty:
        return pd.DataFrame()

    print("预计算 utility observations ...")
    utility_observations = _precompute_utility_observations(data, signals, horizons=UTILITY_HORIZONS)
    print(f"  共 {len(utility_observations)} 条 utility 观测")
    if utility_observations.empty:
        return pd.DataFrame()

    output_path = Path(output_dir) if output_dir is not None else None
    if save_intermediates and output_path is not None:
        cache_dir = output_path / "cache"
        _save_frame_artifact(observations, cache_dir / "edge_observations")
        _save_frame_artifact(utility_observations, cache_dir / "utility_observations")

    all_dates = sorted(data[DATE_COL].dropna().unique())
    if not all_dates:
        return pd.DataFrame()
    warmup_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=warmup_years)
    min_history_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=min_history_years)
    eval_start_date = max(warmup_date, min_history_date)
    eval_dates = [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d) >= eval_start_date]
    if not eval_dates:
        return pd.DataFrame()

    print(
        f"月度刷新白名单、日度打分: {len(eval_dates)} 个交易日 "
        f"({pd.Timestamp(eval_dates[0]).date()} ~ {pd.Timestamp(eval_dates[-1]).date()})"
    )

    period_index = pd.PeriodIndex(eval_dates, freq="M")
    blocks: list[list[pd.Timestamp]] = []
    current_block: list[pd.Timestamp] = []
    current_period = None
    for dt, period in zip(eval_dates, period_index):
        if current_period is None or period == current_period:
            current_block.append(dt)
            current_period = period
            continue
        blocks.append(current_block)
        current_block = [dt]
        current_period = period
    if current_block:
        blocks.append(current_block)

    def _run_block(block_dates: list[pd.Timestamp]):
        return _monthly_refresh_block_score(
            refresh_ts=block_dates[0],
            block_dates=block_dates,
            data=data,
            signal_table=signals,
            observations=observations,
            utility_observations=utility_observations,
            min_count=min_count,
            filter_min_events=filter_min_events,
            filter_min_abs_score=filter_min_abs_score,
            filter_min_events_per_year=filter_min_events_per_year,
            filter_score_rank_pct_threshold=filter_score_rank_pct_threshold,
            filter_aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
            filter_max_rules_per_bucket=filter_max_rules_per_bucket,
            return_diagnostics=save_intermediates,
        )

    if Parallel is not None and int(parallel_n_jobs) not in (0, 1):
        results = Parallel(n_jobs=parallel_n_jobs, prefer="threads")(
            delayed(_run_block)(block) for block in blocks
        )
    else:
        results = []
        for idx, block in enumerate(blocks, start=1):
            if idx == 1 or idx % max(1, len(blocks) // 10) == 0:
                print(f"  处理月份块 {idx}/{len(blocks)}")
            results.append(_run_block(block))

    if save_intermediates:
        score_frames = [item[0] for item in results if item and item[0] is not None and not item[0].empty]
        diagnostics_list = [item[1] for item in results if item and len(item) > 1 and isinstance(item[1], dict)]
    else:
        score_frames = [item for item in results if item is not None and not item.empty]
        diagnostics_list = []

    timing = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    if timing.empty:
        return pd.DataFrame()

    timing = timing.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    timing = _attach_score_variants(timing)
    timing["position_signal"] = np.select(
        [timing["net_score"].gt(0), timing["net_score"].lt(0)],
        ["多", "空"],
        default="观望",
    )
    names = data.groupby(CODE_COL)[NAME_COL].last().reset_index()
    timing = timing.merge(names, on=CODE_COL, how="left")
    timing = timing[
        [
            DATE_COL,
            CODE_COL,
            NAME_COL,
            "entry_score",
            "exit_score",
            "net_score",
            "entry_score_sum",
            "exit_score_sum",
            "net_score_sum",
            "entry_score_mean",
            "exit_score_mean",
            "net_score_mean",
            "entry_score_sqrtadj",
            "exit_score_sqrtadj",
            "net_score_sqrtadj",
            "active_entry_signal_count",
            "active_exit_signal_count",
            "position_signal",
        ]
    ]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(timing, output_path / "monthly_refresh_daily_score.csv")
        print(f"  已写入 {output_path / 'monthly_refresh_daily_score.csv'} (共 {len(timing)} 行)")
        if save_intermediates and diagnostics_list:
            diagnostics_dir = output_path / "intermediates"
            utility_frames = [d["utility_snapshot"] for d in diagnostics_list if "utility_snapshot" in d and not d["utility_snapshot"].empty]
            selected_frames = [d["selected_rules"] for d in diagnostics_list if "selected_rules" in d and not d["selected_rules"].empty]
            decay_frames = [d["decay_snapshot"] for d in diagnostics_list if "decay_snapshot" in d and not d["decay_snapshot"].empty]
            contrib_frames = [d["signal_contributions"] for d in diagnostics_list if "signal_contributions" in d and not d["signal_contributions"].empty]
            if utility_frames:
                _save_frame_artifact(pd.concat(utility_frames, ignore_index=True), diagnostics_dir / "monthly_utility_snapshots")
            if selected_frames:
                _save_frame_artifact(pd.concat(selected_frames, ignore_index=True), diagnostics_dir / "monthly_selected_rules")
            if decay_frames:
                _save_frame_artifact(pd.concat(decay_frames, ignore_index=True), diagnostics_dir / "monthly_decay_snapshots")
            if contrib_frames:
                _save_frame_artifact(pd.concat(contrib_frames, ignore_index=True), diagnostics_dir / "monthly_signal_contributions")

    print(f"完成：{len(timing)} 条月度刷新日度记录")
    return timing


def update_monthly_refresh_daily_score_incremental(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    output_dir: str | Path,
    min_history_years: int = 2,
    warmup_years: int = 3,
    age_grid: Iterable[int] = EDGE_AGE_GRID,
    forward_horizon: int = EDGE_FORWARD_HORIZON,
    min_count: int = EDGE_MIN_COUNT,
    filter_min_events: int = DEFAULT_MIN_EVENTS,
    filter_min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
    filter_min_events_per_year: float = DEFAULT_MIN_EVENTS_PER_YEAR,
    filter_score_rank_pct_threshold: float = DEFAULT_SCORE_RANK_PCT_THRESHOLD,
    filter_aux_score_rank_pct_threshold: float = DEFAULT_AUX_SCORE_RANK_PCT_THRESHOLD,
    filter_max_rules_per_bucket: int = DEFAULT_MAX_RULES_PER_BUCKET,
    save_intermediates: bool = False,
) -> pd.DataFrame:
    """Incrementally append trailing monthly-refresh daily scores.

    Behavior:
    - If no existing score file exists, fallback to full `run_monthly_refresh_daily_score`.
    - If new data stays in the same month, only append missing tail dates in that month.
    - If new data crosses into a new month, only recompute the missing month blocks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_path = output_path / "monthly_refresh_daily_score.csv"
    existing_score_path = next((candidate for candidate in table_candidates(score_path) if candidate.exists()), None)

    if existing_score_path is None:
        return run_monthly_refresh_daily_score(
            df=df,
            signal_table=signal_table,
            min_history_years=min_history_years,
            warmup_years=warmup_years,
            parallel_n_jobs=1,
            age_grid=age_grid,
            forward_horizon=forward_horizon,
            min_count=min_count,
            filter_min_events=filter_min_events,
            filter_min_abs_score=filter_min_abs_score,
            filter_min_events_per_year=filter_min_events_per_year,
            filter_score_rank_pct_threshold=filter_score_rank_pct_threshold,
            filter_aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
            filter_max_rules_per_bucket=filter_max_rules_per_bucket,
            save_intermediates=save_intermediates,
            output_dir=output_path,
        )

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.dropna(subset=[DATE_COL]).sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    signals = signal_table.copy()
    signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL], errors="coerce")
    signals = signals.dropna(subset=[SIGNAL_DATE_COL]).reset_index(drop=True)

    existing = read_table(existing_score_path)
    existing[DATE_COL] = pd.to_datetime(existing[DATE_COL], errors="coerce")
    existing = existing.dropna(subset=[DATE_COL]).sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    if existing.empty:
        return run_monthly_refresh_daily_score(
            df=data,
            signal_table=signals,
            min_history_years=min_history_years,
            warmup_years=warmup_years,
            parallel_n_jobs=1,
            age_grid=age_grid,
            forward_horizon=forward_horizon,
            min_count=min_count,
            filter_min_events=filter_min_events,
            filter_min_abs_score=filter_min_abs_score,
            filter_min_events_per_year=filter_min_events_per_year,
            filter_score_rank_pct_threshold=filter_score_rank_pct_threshold,
            filter_aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
            filter_max_rules_per_bucket=filter_max_rules_per_bucket,
            save_intermediates=save_intermediates,
            output_dir=output_path,
        )

    all_dates = sorted(data[DATE_COL].dropna().unique())
    if not all_dates:
        return existing
    warmup_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=warmup_years)
    min_history_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=min_history_years)
    eval_start_date = max(warmup_date, min_history_date)
    eval_dates = [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d) >= eval_start_date]
    if not eval_dates:
        return existing

    latest_scored_date = pd.Timestamp(existing[DATE_COL].max())
    missing_eval_dates = [dt for dt in eval_dates if dt > latest_scored_date]
    if not missing_eval_dates:
        print("monthly_refresh_daily_score.csv 已是最新，无需增量更新。")
        return existing

    eval_index = pd.DatetimeIndex(eval_dates)
    month_refresh_map: dict[pd.Period, pd.Timestamp] = {}
    for period, dates_in_period in pd.Series(eval_index, index=eval_index).groupby(eval_index.to_period("M")):
        month_refresh_map[period] = pd.Timestamp(dates_in_period.iloc[0])

    missing_blocks: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []
    missing_index = pd.DatetimeIndex(missing_eval_dates)
    for period, dates_in_period in pd.Series(missing_index, index=missing_index).groupby(missing_index.to_period("M")):
        block_dates = [pd.Timestamp(x) for x in dates_in_period.tolist()]
        refresh_ts = month_refresh_map[period]
        missing_blocks.append((refresh_ts, block_dates))

    required_cache_exit_date = max(refresh_ts for refresh_ts, _ in missing_blocks)
    cache_dir = output_path / "cache"
    observations = _load_frame_artifact(cache_dir / "edge_observations")
    utility_observations = _load_frame_artifact(cache_dir / "utility_observations")

    def _cache_covers_required_date(frame: pd.DataFrame) -> bool:
        if frame.empty or "exit_date" not in frame.columns:
            return False
        max_exit_date = pd.to_datetime(frame["exit_date"], errors="coerce").max()
        return pd.notna(max_exit_date) and pd.Timestamp(max_exit_date) >= required_cache_exit_date

    if not _cache_covers_required_date(observations):
        print(f"重建 edge observations ... (需要覆盖至 {required_cache_exit_date.date()})")
        observations = _precompute_edge_observations(
            data,
            signals,
            age_grid=age_grid,
            forward_horizon=forward_horizon,
        )
        _save_frame_artifact(observations, cache_dir / "edge_observations")
    else:
        print(f"复用 edge observations cache (已覆盖本次刷新日 {required_cache_exit_date.date()})")

    if not _cache_covers_required_date(utility_observations):
        print(f"重建 utility observations ... (需要覆盖至 {required_cache_exit_date.date()})")
        utility_observations = _precompute_utility_observations(data, signals, horizons=UTILITY_HORIZONS)
        _save_frame_artifact(utility_observations, cache_dir / "utility_observations")
    else:
        print(f"复用 utility observations cache (已覆盖本次刷新日 {required_cache_exit_date.date()})")

    print(f"增量更新 score: 补 {len(missing_eval_dates)} 个交易日，涉及 {len(missing_blocks)} 个月份块")
    new_frames: list[pd.DataFrame] = []
    diagnostics_list: list[dict[str, pd.DataFrame]] = []
    for refresh_ts, block_dates in missing_blocks:
        result = _monthly_refresh_block_score(
            refresh_ts=refresh_ts,
            block_dates=block_dates,
            data=data,
            signal_table=signals,
            observations=observations,
            utility_observations=utility_observations,
            min_count=min_count,
            filter_min_events=filter_min_events,
            filter_min_abs_score=filter_min_abs_score,
            filter_min_events_per_year=filter_min_events_per_year,
            filter_score_rank_pct_threshold=filter_score_rank_pct_threshold,
            filter_aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
            filter_max_rules_per_bucket=filter_max_rules_per_bucket,
            return_diagnostics=save_intermediates,
        )
        if save_intermediates:
            block_score, diagnostics = result
            diagnostics_list.append(diagnostics)
        else:
            block_score = result
        if block_score is not None and not block_score.empty:
            new_frames.append(block_score)

    if not new_frames:
        print("没有新增的 score 行。")
        return existing

    new_timing = pd.concat(new_frames, ignore_index=True)
    new_timing = _attach_score_variants(new_timing)
    new_timing["position_signal"] = np.select(
        [new_timing["net_score"].gt(0), new_timing["net_score"].lt(0)],
        ["多", "空"],
        default="观望",
    )
    names = data.groupby(CODE_COL)[NAME_COL].last().reset_index()
    new_timing = new_timing.merge(names, on=CODE_COL, how="left")
    new_timing = new_timing[
        [
            DATE_COL,
            CODE_COL,
            NAME_COL,
            "entry_score",
            "exit_score",
            "net_score",
            "entry_score_sum",
            "exit_score_sum",
            "net_score_sum",
            "entry_score_mean",
            "exit_score_mean",
            "net_score_mean",
            "entry_score_sqrtadj",
            "exit_score_sqrtadj",
            "net_score_sqrtadj",
            "active_entry_signal_count",
            "active_exit_signal_count",
            "position_signal",
        ]
    ]

    merged = pd.concat([existing, new_timing], ignore_index=True)
    merged = merged.sort_values([CODE_COL, DATE_COL]).drop_duplicates(subset=[CODE_COL, DATE_COL], keep="last").reset_index(drop=True)
    score_path = write_table(merged, score_path)
    print(f"  已增量写入 {score_path} (新增 {len(new_timing)} 行, 总计 {len(merged)} 行)")

    if save_intermediates and diagnostics_list:
        diagnostics_dir = output_path / "intermediates"
        utility_frames = [d["utility_snapshot"] for d in diagnostics_list if "utility_snapshot" in d and not d["utility_snapshot"].empty]
        selected_frames = [d["selected_rules"] for d in diagnostics_list if "selected_rules" in d and not d["selected_rules"].empty]
        decay_frames = [d["decay_snapshot"] for d in diagnostics_list if "decay_snapshot" in d and not d["decay_snapshot"].empty]
        contrib_frames = [d["signal_contributions"] for d in diagnostics_list if "signal_contributions" in d and not d["signal_contributions"].empty]
        if utility_frames:
            _save_frame_artifact(pd.concat(utility_frames, ignore_index=True), diagnostics_dir / "monthly_utility_snapshots_incremental")
        if selected_frames:
            _save_frame_artifact(pd.concat(selected_frames, ignore_index=True), diagnostics_dir / "monthly_selected_rules_incremental")
        if decay_frames:
            _save_frame_artifact(pd.concat(decay_frames, ignore_index=True), diagnostics_dir / "monthly_decay_snapshots_incremental")
        if contrib_frames:
            _save_frame_artifact(pd.concat(contrib_frames, ignore_index=True), diagnostics_dir / "monthly_signal_contributions_incremental")

    return merged


def run_expanding_edge_decay_timing(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    warmup_years: int = 3,
    step_days: int = 20,
    age_grid: Iterable[int] = EDGE_AGE_GRID,
    forward_horizon: int = EDGE_FORWARD_HORIZON,
    min_count: int = EDGE_MIN_COUNT,
    dynamic_filter: bool = False,
    filter_min_events: int = DEFAULT_MIN_EVENTS,
    filter_min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
    filter_min_events_per_year: float = DEFAULT_MIN_EVENTS_PER_YEAR,
    filter_score_rank_pct_threshold: float = DEFAULT_SCORE_RANK_PCT_THRESHOLD,
    filter_aux_score_rank_pct_threshold: float = DEFAULT_AUX_SCORE_RANK_PCT_THRESHOLD,
    filter_max_rules_per_bucket: int = DEFAULT_MAX_RULES_PER_BUCKET,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Optimized expanding-window edge timing.

    Recent signal definition:
    - signal at position t is effective from t+1.
    - for evaluation position e, age = e - t - 1.
    - active signals satisfy 0 <= age <= max_age.

    The edge model is rebuilt every `step_days` trading days. Within each rebuild
    block, all daily scores are calculated in one vectorized expansion instead of
    scanning the full signal table day by day.
    """
    if df.empty or signal_table.empty:
        return pd.DataFrame()

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])

    print("预计算 edge observations ...")
    observations = _precompute_edge_observations(
        df,
        signal_table,
        age_grid=age_grid,
        forward_horizon=forward_horizon,
    )
    print(f"  共 {len(observations)} 条观测")
    if observations.empty:
        return pd.DataFrame()

    utility_observations = pd.DataFrame()
    if dynamic_filter:
        print("预计算 utility observations ...")
        utility_observations = _precompute_utility_observations(df, signal_table, horizons=UTILITY_HORIZONS)
        print(f"  共 {len(utility_observations)} 条 utility 观测")
        if utility_observations.empty:
            return pd.DataFrame()

    all_dates = sorted(data[DATE_COL].dropna().unique())
    if not all_dates:
        return pd.DataFrame()
    warmup_date = pd.Timestamp(all_dates[0]) + pd.DateOffset(years=warmup_years)
    eval_dates = [d for d in all_dates if pd.Timestamp(d) >= warmup_date]
    if len(eval_dates) <= 1:
        print(f"  警告：评估日期不足，warmup={warmup_years}年")
        return pd.DataFrame()

    print(f"逐日打分: {len(eval_dates)} 个交易日 ({pd.Timestamp(eval_dates[0]).date()} ~ {pd.Timestamp(eval_dates[-1]).date()})")
    results: list[pd.DataFrame] = []
    step = max(1, int(step_days))

    for start in range(0, len(eval_dates), step):
        block_dates = eval_dates[start:start + step]
        if not block_dates:
            continue
        eval_ts = pd.Timestamp(block_dates[0])
        score_signal_table = signal_table
        decay_observations = observations
        if dynamic_filter:
            history_signals = signal_table[pd.to_datetime(signal_table[SIGNAL_DATE_COL]) <= eval_ts].copy()
            utility_snapshot = _snapshot_utility_from_observations(
                utility_observations,
                max_exit_date=eval_ts,
                min_events=filter_min_events,
                min_abs_score=filter_min_abs_score,
            )
            _, score_signal_table = filter_signal_candidates(
                utility_snapshot,
                history_signals,
                output_dir=None,
                min_events_per_year=filter_min_events_per_year,
                score_rank_pct_threshold=filter_score_rank_pct_threshold,
                aux_score_rank_pct_threshold=filter_aux_score_rank_pct_threshold,
                max_rules_per_bucket=filter_max_rules_per_bucket,
                end_date=eval_ts,
            )
            if score_signal_table.empty:
                continue
            score_signal_table = score_signal_table.copy()
            score_signal_table[SIGNAL_DATE_COL] = pd.to_datetime(score_signal_table[SIGNAL_DATE_COL])
            allowed_rules = score_signal_table[
                [SIGNAL_INSTRUMENT_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL]
            ].drop_duplicates().rename(columns={SIGNAL_INSTRUMENT_COL: CODE_COL})
            decay_observations = observations.merge(
                allowed_rules,
                on=[CODE_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL],
                how="inner",
            )
            if decay_observations.empty:
                continue

        decay = _snapshot_edge_decay(decay_observations, max_exit_date=eval_ts, min_count=min_count)
        if decay.empty:
            continue
        decay_daily = _interpolate_edge_decay(decay)
        if decay_daily.empty:
            continue
        if start % max(1, len(eval_dates) // 10) == 0:
            print(f"  重建 edge_decay @ {eval_ts.date()} ({start + 1}/{len(eval_dates)})")

        score = _snapshot_score_block(data, score_signal_table, decay_daily, block_dates)
        if not score.empty:
            results.append(score)

    if not results:
        print("  无有效结果")
        return pd.DataFrame()

    timing = pd.concat(results, ignore_index=True)
    timing = timing.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    timing = _attach_score_variants(timing)
    timing["position_signal"] = np.select(
        [timing["net_score"].gt(0), timing["net_score"].lt(0)],
        ["多", "空"],
        default="观望",
    )
    names = data.groupby(CODE_COL)[NAME_COL].last().reset_index()
    timing = timing.merge(names, on=CODE_COL, how="left")
    timing = timing[
        [
            DATE_COL,
            CODE_COL,
            NAME_COL,
            "entry_score",
            "exit_score",
            "net_score",
            "entry_score_sum",
            "exit_score_sum",
            "net_score_sum",
            "entry_score_mean",
            "exit_score_mean",
            "net_score_mean",
            "entry_score_sqrtadj",
            "exit_score_sqrtadj",
            "net_score_sqrtadj",
            "active_entry_signal_count",
            "active_exit_signal_count",
            "position_signal",
        ]
    ]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(timing, output_path / "expanding_edge_timing.csv")
        print(f"  已写入 {output_path / 'expanding_edge_timing.csv'} (共 {len(timing)} 行)")

    print(f"完成：{len(timing)} 条择时记录")
    return timing


def build_signal_edge_decay(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    output_dir: str | Path | None = None,
    age_grid: Iterable[int] = EDGE_AGE_GRID,
    forward_horizon: int = EDGE_FORWARD_HORIZON,
    min_count: int = EDGE_MIN_COUNT,
    max_exit_date=None,
) -> pd.DataFrame:
    """计算信号触发后第 k 天仍然保留的未来收益边际信息。"""
    if df.empty or signal_table.empty:
        edge_decay = pd.DataFrame()
    else:
        ages = tuple(int(age) for age in age_grid)
        rows: list[pd.DataFrame] = []
        data = df.copy()
        data[DATE_COL] = pd.to_datetime(data[DATE_COL])
        signals = signal_table.copy()
        signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])

        for instrument, group in data.groupby(CODE_COL, sort=False):
            group = group.sort_values(DATE_COL).reset_index(drop=True)
            dates = pd.to_datetime(group[DATE_COL])
            prices = pd.to_numeric(group[PRICE_COL], errors="coerce").to_numpy()
            date_to_pos = pd.Series(np.arange(len(dates)), index=dates).to_dict()
            sig = signals[signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))].copy()
            if sig.empty:
                continue
            sig["_signal_pos"] = sig[SIGNAL_DATE_COL].map(date_to_pos)
            sig = sig.dropna(subset=["_signal_pos"]).copy()
            if sig.empty:
                continue
            sig["_signal_pos"] = sig["_signal_pos"].astype(int)
            sig["event_side"] = [
                _event_side_from_signal(value, pattern)
                for value, pattern in zip(sig[SIGNAL_VALUE_COL], sig[SIGNAL_PATTERN_COL])
            ]
            base_cols = [SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, "event_side", "_signal_pos"]
            for age in ages:
                base_idx = sig["_signal_pos"].to_numpy() + 1 + age
                exit_idx = base_idx + int(forward_horizon)
                valid = (base_idx >= 0) & (exit_idx < len(prices))
                if valid.any():
                    valid_pos = np.flatnonzero(valid)
                    price_valid = (
                        np.isfinite(prices[base_idx[valid_pos]])
                        & np.isfinite(prices[exit_idx[valid_pos]])
                        & (prices[base_idx[valid_pos]] > 0)
                        & (prices[exit_idx[valid_pos]] > 0)
                    )
                    valid[valid_pos] = price_valid
                if not valid.any():
                    continue
                # max_exit_date 过滤：收益观测终点不超过指定日期。
                if max_exit_date is not None:
                    exit_dates = dates.iloc[exit_idx[valid]]
                    valid[valid] = pd.to_datetime(exit_dates) <= pd.Timestamp(max_exit_date)
                    if not valid.any():
                        continue
                part = sig.loc[valid, base_cols].copy()
                part[CODE_COL] = instrument
                part["age"] = age
                part["forward_horizon"] = int(forward_horizon)
                part["forward_return"] = prices[exit_idx[valid]] / prices[base_idx[valid]] - 1
                rows.append(part)

        if rows:
            forward = pd.concat(rows, ignore_index=True)
            edge_decay = forward.groupby(
                [CODE_COL, SIGNAL_FACTOR_COL, SIGNAL_PATTERN_COL, "event_side", "age", "forward_horizon"],
                as_index=False,
            ).agg(
                count=("forward_return", "count"),
                mean_return=("forward_return", "mean"),
                median_return=("forward_return", "median"),
                std_return=("forward_return", "std"),
                win_rate=("forward_return", lambda x: float((x > 0).mean())),
                p25_return=("forward_return", lambda x: float(x.quantile(0.25))),
                p75_return=("forward_return", lambda x: float(x.quantile(0.75))),
            )
            edge_decay = edge_decay.rename(columns={SIGNAL_FACTOR_COL: "factor", SIGNAL_PATTERN_COL: "pattern"})
            edge_decay = _edge_score_frame(edge_decay, min_count=min_count)
        else:
            edge_decay = pd.DataFrame()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(edge_decay, output_path / "signal_edge_decay.csv")
    return edge_decay


def _interpolate_edge_decay(edge_decay: pd.DataFrame) -> pd.DataFrame:
    if edge_decay.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    key_cols = [CODE_COL, "factor", "pattern", "event_side"]
    for key, group in edge_decay.groupby(key_cols, sort=False):
        group = group.sort_values("age")
        ages = pd.to_numeric(group["age"], errors="coerce").to_numpy(dtype=float)
        scores = pd.to_numeric(group["score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(ages) == 0:
            continue
        daily_age = np.arange(int(np.nanmax(ages)) + 1)
        daily_score = np.interp(daily_age, ages, scores, left=scores[0], right=0.0)
        part = pd.DataFrame({"age": daily_age, "score": daily_score})
        for col, value in zip(key_cols, key if isinstance(key, tuple) else (key,)):
            part[col] = value
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _top_contributors(data: pd.DataFrame, side: str, limit: int = 5) -> pd.Series:
    work = data[data["event_side"].eq(side)].copy()
    if work.empty:
        return pd.Series(dtype=object)
    work = work.sort_values([DATE_COL, "score"], ascending=[True, False])

    def _format(group: pd.DataFrame) -> str:
        return "; ".join(
            f"{row.factor}|{row.pattern}|age={int(row.age)}|score={row.score:.4f}"
            for row in group.head(limit).itertuples(index=False)
        )

    return work.groupby([CODE_COL, DATE_COL])[["factor", "pattern", "age", "score"]].apply(_format)


def build_daily_signal_state_score(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    edge_decay: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """把仍在有效 age 内的历史信号合成为每日 entry/exit/net score。"""
    if df.empty or signal_table.empty or edge_decay.empty:
        daily_score = pd.DataFrame()
    else:
        decay_daily = _interpolate_edge_decay(edge_decay)
        contributions: list[pd.DataFrame] = []
        data = df.copy()
        data[DATE_COL] = pd.to_datetime(data[DATE_COL])
        signals = signal_table.copy()
        signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])
        max_age = int(decay_daily["age"].max()) if not decay_daily.empty else -1

        for instrument, group in data.groupby(CODE_COL, sort=False):
            group = group.sort_values(DATE_COL).reset_index(drop=True)
            dates = pd.to_datetime(group[DATE_COL])
            date_to_pos = pd.Series(np.arange(len(dates)), index=dates).to_dict()
            sig = signals[signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))].copy()
            if sig.empty:
                continue
            sig["_signal_pos"] = sig[SIGNAL_DATE_COL].map(date_to_pos)
            sig = sig.dropna(subset=["_signal_pos"]).copy()
            if sig.empty:
                continue
            sig["_signal_pos"] = sig["_signal_pos"].astype(int)
            sig["event_side"] = [
                _event_side_from_signal(value, pattern)
                for value, pattern in zip(sig[SIGNAL_VALUE_COL], sig[SIGNAL_PATTERN_COL])
            ]
            sig_base = sig.rename(columns={SIGNAL_FACTOR_COL: "factor", SIGNAL_PATTERN_COL: "pattern"})
            sig_base[CODE_COL] = instrument
            for age in range(max_age + 1):
                current_idx = sig_base["_signal_pos"].to_numpy() + 1 + age
                valid = (current_idx >= 0) & (current_idx < len(dates))
                if not valid.any():
                    continue
                part = sig_base.loc[valid, [CODE_COL, "factor", "pattern", "event_side", "_signal_pos"]].copy()
                part["age"] = age
                part[DATE_COL] = dates.iloc[current_idx[valid]].to_numpy()
                part = part.merge(decay_daily[[CODE_COL, "factor", "pattern", "event_side", "age", "score"]], on=[CODE_COL, "factor", "pattern", "event_side", "age"], how="inner")
                part = part[part["score"].gt(0)]
                if not part.empty:
                    contributions.append(part)

        if not contributions:
            daily_score = pd.DataFrame()
        else:
            contrib = pd.concat(contributions, ignore_index=True)
            grouped = contrib.groupby([CODE_COL, DATE_COL, "event_side"]).agg(
                score=("score", "sum"),
                signal_count=("score", "size"),
            ).reset_index()
            pivot_score = grouped.pivot_table(index=[CODE_COL, DATE_COL], columns="event_side", values="score", fill_value=0.0)
            pivot_count = grouped.pivot_table(index=[CODE_COL, DATE_COL], columns="event_side", values="signal_count", fill_value=0)
            daily_score = pivot_score.reset_index().rename(columns={"open": "entry_score", "close": "exit_score"})
            daily_count = pivot_count.reset_index().rename(columns={"open": "active_entry_signal_count", "close": "active_exit_signal_count"})
            daily_score = daily_score.merge(daily_count, on=[CODE_COL, DATE_COL], how="left")
            for col in ("entry_score", "exit_score", "active_entry_signal_count", "active_exit_signal_count"):
                if col not in daily_score:
                    daily_score[col] = 0
            daily_score["net_score"] = daily_score["entry_score"] - daily_score["exit_score"]
            daily_score["position_signal"] = np.select(
                [daily_score["net_score"].gt(0), daily_score["net_score"].lt(0)],
                ["多", "空"],
                default="观望",
            )
            top_entry = _top_contributors(contrib, "open").rename("top_entry_contributors")
            top_exit = _top_contributors(contrib, "close").rename("top_exit_contributors")
            daily_score = daily_score.merge(top_entry.reset_index(), on=[CODE_COL, DATE_COL], how="left")
            daily_score = daily_score.merge(top_exit.reset_index(), on=[CODE_COL, DATE_COL], how="left")
            names = data.groupby(CODE_COL)[NAME_COL].last().reset_index()
            daily_score = daily_score.merge(names, on=CODE_COL, how="left")
            daily_score = daily_score[
                [
                    DATE_COL,
                    CODE_COL,
                    NAME_COL,
                    "entry_score",
                    "exit_score",
                    "net_score",
                    "active_entry_signal_count",
                    "active_exit_signal_count",
                    "top_entry_contributors",
                    "top_exit_contributors",
                    "position_signal",
                ]
            ].sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(daily_score, output_path / "daily_signal_state_score.csv")
    return daily_score


def strategy_vote_summary(daily_signal_state_score: pd.DataFrame, utility: pd.DataFrame) -> dict[str, Any]:
    active = utility[utility.get("utility_label", pd.Series(dtype=str)).eq("valid")].copy() if not utility.empty else pd.DataFrame()
    role_counts = active.groupby("usage_role", dropna=False).size().to_dict() if not active.empty and "usage_role" in active else {}
    term_counts = active.groupby("term_structure_label", dropna=False).size().to_dict() if not active.empty and "term_structure_label" in active else {}
    if not active.empty and {"subjective_usage_role", "quantitative_usage_role"}.issubset(active.columns):
        role_changed_count = int(active["subjective_usage_role"].ne(active["quantitative_usage_role"]).sum())
    else:
        role_changed_count = 0

    if daily_signal_state_score.empty:
        best: dict[str, Any] = {}
    else:
        row = daily_signal_state_score.sort_values(DATE_COL).iloc[-1]
        best = {
            "template_name": "信号边际信息衰减",
            "current_position": row.get("position_signal"),
            "latest_entry_score": row.get("entry_score"),
            "latest_exit_score": row.get("exit_score"),
            "latest_risk_score": row.get("exit_score"),
            "latest_net_score": row.get("net_score"),
            "latest_entry_count": row.get("active_entry_signal_count"),
            "latest_exit_count": row.get("active_exit_signal_count"),
            "latest_directional_count": row.get("active_entry_signal_count", 0) + row.get("active_exit_signal_count", 0),
        }
    return {
        "role_counts": {role: int(role_counts.get(role, 0)) for role in ROLE_ORDER},
        "term_structure_counts": {str(label): int(count) for label, count in term_counts.items()},
        "term_structure_role_changed_count": role_changed_count,
        "best_template": best,
    }


def _score_based_position(
    group: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    dominance_ratio: float,
    entry_col: str = "entry_score",
    exit_col: str = "exit_score",
) -> pd.DataFrame:
    work = group.sort_values(DATE_COL).reset_index(drop=True).copy()
    entry_values = pd.to_numeric(work[entry_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    exit_values = pd.to_numeric(work[exit_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    open_signal = (entry_values >= float(entry_threshold)) & (
        entry_values >= exit_values * float(dominance_ratio)
    )
    close_signal = (exit_values >= float(exit_threshold)) & (
        exit_values >= entry_values * float(dominance_ratio)
    )

    position = np.zeros(len(work), dtype=float)
    state = 0.0
    for i in range(1, len(work)):
        if state <= 0 and open_signal[i - 1]:
            state = 1.0
        elif state > 0 and close_signal[i - 1]:
            state = 0.0
        position[i] = state

    work["open_signal"] = open_signal.astype(int)
    work["close_signal"] = close_signal.astype(int)
    work["position"] = position
    return work


def _equity_stats(
    returns: pd.Series,
    position: pd.Series,
) -> tuple[pd.Series, float, float, float, float]:
    strategy_returns = returns.fillna(0.0) * position.fillna(0.0)
    equity = (1.0 + strategy_returns).cumprod()
    if equity.empty:
        return equity, np.nan, np.nan, np.nan, np.nan
    years = (len(equity) - 1) / TRADING_DAYS
    annual_return = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else np.nan
    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1.0).min()) if not equity.empty else np.nan
    vol = strategy_returns.std(ddof=0)
    sharpe = float(strategy_returns.mean() / vol * np.sqrt(TRADING_DAYS)) if np.isfinite(vol) and vol > 0 else np.nan
    holding_ratio = float(position.mean()) if len(position) else np.nan
    return equity, annual_return, max_drawdown, sharpe, holding_ratio


def backtest_score_rule(
    df: pd.DataFrame,
    daily_score: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    dominance_ratio: float = 1.0,
    output_dir: str | Path | None = None,
    entry_col: str = "entry_score",
    exit_col: str = "exit_score",
    include_equity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest a deterministic long/cash rule from entry_score and exit_score.

    Rule:
    - Open if `entry_score >= entry_threshold` and `entry_score >= dominance_ratio * exit_score`.
    - Close if `exit_score >= exit_threshold` and `exit_score >= dominance_ratio * entry_score`.
    - Signal observed on day t becomes position on day t+1 close-to-close returns.
    """
    if df.empty or daily_score.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    scores = daily_score.copy()
    scores[DATE_COL] = pd.to_datetime(scores[DATE_COL])

    summary_rows: list[dict[str, Any]] = []
    equity_rows: list[pd.DataFrame] = []

    for code, price_group in data.groupby(CODE_COL, sort=False):
        price_group = price_group.sort_values(DATE_COL).reset_index(drop=True)
        score_group = scores[scores[CODE_COL].astype(str).eq(str(code))].copy()
        if score_group.empty:
            continue

        merged = price_group[[DATE_COL, CODE_COL, NAME_COL, PRICE_COL]].merge(
            score_group[[DATE_COL, CODE_COL, entry_col, exit_col]],
            on=[DATE_COL, CODE_COL],
            how="left",
        ).sort_values(DATE_COL).reset_index(drop=True)
        merged[[entry_col, exit_col]] = merged[[entry_col, exit_col]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        merged = _score_based_position(
            merged,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            dominance_ratio=dominance_ratio,
            entry_col=entry_col,
            exit_col=exit_col,
        )

        benchmark_returns = pd.to_numeric(merged[PRICE_COL], errors="coerce").pct_change().fillna(0.0)
        equity, annual_return, max_drawdown, sharpe, holding_ratio = _equity_stats(
            benchmark_returns,
            merged["position"],
        )
        benchmark_equity = (1.0 + benchmark_returns).cumprod()
        benchmark_years = (len(benchmark_equity) - 1) / TRADING_DAYS
        benchmark_annual_return = (
            float(benchmark_equity.iloc[-1] ** (1 / benchmark_years) - 1)
            if benchmark_years > 0 and benchmark_equity.iloc[-1] > 0
            else np.nan
        )
        excess_equity = equity / benchmark_equity.replace(0, np.nan)
        excess_years = (len(excess_equity.dropna()) - 1) / TRADING_DAYS
        excess_annual_return = (
            float(excess_equity.dropna().iloc[-1] ** (1 / excess_years) - 1)
            if excess_years > 0 and not excess_equity.dropna().empty and excess_equity.dropna().iloc[-1] > 0
            else np.nan
        )
        trade_count = int(((merged["position"].shift(1).fillna(0.0) <= 0) & (merged["position"] > 0)).sum())

        summary_rows.append(
            {
                CODE_COL: code,
                NAME_COL: merged[NAME_COL].dropna().iloc[-1] if merged[NAME_COL].notna().any() else np.nan,
                "entry_col": entry_col,
                "exit_col": exit_col,
                "entry_threshold": float(entry_threshold),
                "exit_threshold": float(exit_threshold),
                "dominance_ratio": float(dominance_ratio),
                "annual_return": annual_return,
                "benchmark_annual_return": benchmark_annual_return,
                "excess_annual_return": excess_annual_return,
                "max_drawdown": max_drawdown,
                "sharpe": sharpe,
                "holding_ratio": holding_ratio,
                "trade_count": trade_count,
                "final_equity": float(equity.iloc[-1]) if not equity.empty else np.nan,
                "benchmark_final_equity": float(benchmark_equity.iloc[-1]) if not benchmark_equity.empty else np.nan,
                "excess_final_equity": float(excess_equity.dropna().iloc[-1]) if not excess_equity.dropna().empty else np.nan,
            }
        )

        if include_equity:
            equity_rows.append(
                pd.DataFrame(
                    {
                        DATE_COL: merged[DATE_COL],
                        CODE_COL: code,
                        NAME_COL: merged[NAME_COL],
                        "entry_score": merged[entry_col],
                        "exit_score": merged[exit_col],
                        "position": merged["position"],
                        "open_signal": merged["open_signal"],
                        "close_signal": merged["close_signal"],
                        "strategy_equity": equity,
                        "benchmark_equity": benchmark_equity,
                        "excess_equity": excess_equity,
                    }
                )
            )

    summary = pd.DataFrame(summary_rows)
    equity_df = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tag = f"entry_{entry_threshold:g}_exit_{exit_threshold:g}_dom_{dominance_ratio:g}".replace(".", "p")
        write_table(summary, output_path / f"score_rule_summary_{tag}.csv")
        if include_equity:
            write_table(equity_df, output_path / f"score_rule_equity_{tag}.csv")
    return summary, equity_df


def sweep_score_rules(
    df: pd.DataFrame,
    daily_score: pd.DataFrame,
    output_dir: str | Path | None = None,
    entry_thresholds: Iterable[float] = DEFAULT_SCORE_THRESHOLD_GRID,
    exit_thresholds: Iterable[float] = DEFAULT_SCORE_THRESHOLD_GRID,
    dominance_ratios: Iterable[float] = DEFAULT_SCORE_DOMINANCE_GRID,
    entry_col: str = "entry_score",
    exit_col: str = "exit_score",
    include_best_equity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grid-search deterministic score rules and return ranked summary."""
    summary_rows: list[pd.DataFrame] = []
    best_equity = pd.DataFrame()

    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            for dominance_ratio in dominance_ratios:
                summary, equity = backtest_score_rule(
                    df=df,
                    daily_score=daily_score,
                    entry_threshold=float(entry_threshold),
                    exit_threshold=float(exit_threshold),
                    dominance_ratio=float(dominance_ratio),
                    output_dir=None,
                    entry_col=entry_col,
                    exit_col=exit_col,
                    include_equity=include_best_equity,
                )
                if summary.empty:
                    continue
                summary_rows.append(summary)
                if include_best_equity and not equity.empty:
                    equity["entry_threshold"] = float(entry_threshold)
                    equity["exit_threshold"] = float(exit_threshold)
                    equity["dominance_ratio"] = float(dominance_ratio)
                    if best_equity.empty:
                        best_equity = equity

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["excess_annual_return", "sharpe", "annual_return"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

        if include_best_equity:
            top = summary_df.iloc[0]
            _, best_equity = backtest_score_rule(
                df=df,
                daily_score=daily_score,
                entry_threshold=float(top["entry_threshold"]),
                exit_threshold=float(top["exit_threshold"]),
                dominance_ratio=float(top["dominance_ratio"]),
                output_dir=None,
                entry_col=entry_col,
                exit_col=exit_col,
                include_equity=True,
            )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_table(summary_df, output_path / "score_rule_grid_summary.csv")
        if include_best_equity and not best_equity.empty:
            write_table(best_equity, output_path / "score_rule_best_equity.csv")
    return summary_df, best_equity
