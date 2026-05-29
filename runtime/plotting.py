from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from io_utils import read_table, table_candidates
from signal_generation import generate_signal_table
from timing_config import (
    CODE_COL,
    DATE_COL,
    NAME_COL,
    PRICE_COL,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_VALUE_COL,
    TRADING_DAYS,
    _split_factor_frequency,
)


def _load_pyplot(backend: str | None = None):
    import sys
    import matplotlib

    if backend and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use(backend)
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_dual_axis(
    df: pd.DataFrame,
    left_col: str | list[str],
    right_col: str | list[str],
    figsize: tuple[int, int] = (12, 6),
    left_linewidth: float = 2.0,
    left_alpha: float = 0.8,
    right_linewidth: float = 2.0,
    right_alpha: float = 0.8,
    left_invert: bool = False,
    right_invert: bool = False,
    x_rotation: int = 0,
    legend_loc: str = "upper left",
    fontsize: int = 12,
    right_kind: str = "line",
    right_base: float = 0.0,
    title: str | None = None,
    backend: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    ax1=None,
):
    plt = _load_pyplot(backend)
    left_cols = [left_col] if isinstance(left_col, str) else list(left_col)
    right_cols = [right_col] if isinstance(right_col, str) else list(right_col)

    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig = ax1.figure
    ax2 = ax1.twinx() if right_cols else None
    fallback_left_colors = ["#044E7E", "#FF3333", "#7B2CBF", "#81A6BE", "#000000", "#999999"]
    right_colors = ["#FF8080", "#FFB2B2", "#999999"]
    lines, labels = [], []
    color_idx = 0

    for col in left_cols:
        col_text = str(col)
        if col_text.endswith("_年线"):
            color = "#7B2CBF"
        elif col_text.endswith("_季线"):
            color = "#FF3333"
        else:
            color = fallback_left_colors[color_idx % len(fallback_left_colors)]
        line, = ax1.plot(
            df.index,
            df[col],
            color=color,
            linewidth=left_linewidth,
            alpha=left_alpha,
            label=col,
        )
        lines.append(line)
        labels.append(col)
        color_idx += 1

    if left_invert:
        ax1.invert_yaxis()

    if ax2 is not None:
        for right_idx, col in enumerate(right_cols):
            color = right_colors[right_idx % len(right_colors)]
            if right_kind == "line":
                line, = ax2.plot(
                    df.index,
                    df[col],
                    linestyle="--",
                    color=color,
                    linewidth=right_linewidth,
                    alpha=right_alpha,
                    label=f"{col} (right)",
                )
                lines.append(line)
                labels.append(f"{col} (right)")
            elif right_kind == "area":
                area = ax2.fill_between(
                    df.index,
                    right_base,
                    df[col],
                    color=color,
                    alpha=right_alpha,
                    label=f"{col} (right area)",
                )
                lines.append(area)
                labels.append(f"{col} (right area)")
            else:
                raise ValueError("right_kind must be 'line' or 'area'")
            color_idx += 1

        if right_invert:
            ax2.invert_yaxis()

    if title:
        ax1.set_title(title, fontsize=fontsize + 2)
    ax1.set_xlabel("Date")
    ax1.legend(lines, labels, loc=legend_loc, fontsize=fontsize)
    plt.xticks(rotation=x_rotation)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show and ax1 is None:
        plt.show()
    return fig, ax1, ax2


def plot_score_price_chart(
    df: pd.DataFrame,
    daily_score: pd.DataFrame,
    instrument: str,
    score_cols: str | list[str] = "net_score",
    lookback_days: int = TRADING_DAYS * 3,
    price_col: str = PRICE_COL,
    title: str | None = None,
    backend: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    plt = _load_pyplot(backend)
    score_columns = [score_cols] if isinstance(score_cols, str) else list(score_cols)

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    score = daily_score.copy()
    score[DATE_COL] = pd.to_datetime(score[DATE_COL])

    inst_key = str(instrument)
    price_part = data[data[CODE_COL].astype(str).eq(inst_key)].copy()
    score_part = score[score[CODE_COL].astype(str).eq(inst_key)].copy()
    if price_part.empty:
        raise ValueError(f"No price data found for instrument={instrument}")
    if score_part.empty:
        raise ValueError(f"No daily score data found for instrument={instrument}")

    keep_score_cols = [col for col in score_columns if col in score_part.columns]
    if not keep_score_cols:
        raise ValueError(f"None of score columns found: {score_columns}")
    if price_col not in price_part.columns:
        raise ValueError(f"Price column not found: {price_col}")

    merged = price_part[[DATE_COL, CODE_COL, NAME_COL, price_col]].merge(
        score_part[[DATE_COL, CODE_COL, *keep_score_cols, "position_signal"]],
        on=[DATE_COL, CODE_COL],
        how="left",
    )
    merged = merged.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL]).reset_index(drop=True)
    for col in keep_score_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged[price_col] = pd.to_numeric(merged[price_col], errors="coerce")
    merged["position_signal"] = merged["position_signal"].fillna("观望")

    if lookback_days is not None and lookback_days > 0 and len(merged) > lookback_days:
        merged = merged.iloc[-lookback_days:].copy()

    plot_df = merged.set_index(DATE_COL)
    default_title = f"{inst_key}"
    if NAME_COL in merged.columns and merged[NAME_COL].notna().any():
        default_title = f"{inst_key} {merged[NAME_COL].dropna().iloc[-1]}"

    fig, ax1, ax2 = plot_dual_axis(
        plot_df,
        left_col=keep_score_cols,
        right_col=price_col,
        figsize=(14, 7),
        left_linewidth=2.0,
        right_linewidth=1.8,
        right_alpha=0.9,
        title=title or f"{default_title} Score vs Price",
        backend=backend,
        show=False,
        save_path=None,
    )

    ax1.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":", alpha=0.8)

    in_long = False
    start = None
    for dt, sig in merged[[DATE_COL, "position_signal"]].itertuples(index=False):
        is_long = str(sig) == "多"
        if is_long and not in_long:
            start = pd.Timestamp(dt)
            in_long = True
        elif not is_long and in_long:
            ax1.axvspan(start, pd.Timestamp(dt), color="#f6c7c7", alpha=0.20, linewidth=0)
            in_long = False
            start = None
    if in_long and start is not None:
        ax1.axvspan(start, pd.Timestamp(merged[DATE_COL].iloc[-1]), color="#f6c7c7", alpha=0.20, linewidth=0)

    ax1.set_ylabel("Score")
    if ax2 is not None:
        ax2.set_ylabel(price_col)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax1, ax2, merged


def _factor_plot_columns(df: pd.DataFrame, factor: str) -> list[str]:
    base_factor, _ = _split_factor_frequency(factor)
    candidates = [base_factor, f"{base_factor}_季线", f"{base_factor}_年线"]
    columns = [col for col in candidates if col in df.columns]
    if not columns:
        raise ValueError(f"No factor columns found for {factor}: {candidates}")
    return columns


def _load_rule_summary_for_plot(
    rule_summary: pd.DataFrame | None = None,
    rule_summary_path: str | Path | None = "results/rule_pair_summary.csv",
) -> pd.DataFrame:
    if rule_summary is not None:
        data = rule_summary.copy()
    elif rule_summary_path is not None:
        path = next((candidate for candidate in table_candidates(rule_summary_path) if candidate.exists()), None)
        if path is None:
            return pd.DataFrame()
        data = read_table(path)
    else:
        return pd.DataFrame()

    for col in [
        "excess_annual_return",
        "sharpe",
        "trade_count",
        "holding_ratio",
        "final_equity",
        "excess_final_equity",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.replace([np.inf, -np.inf], np.nan)


def _best_rule_row(
    rule_summary: pd.DataFrame,
    factors: list[str],
    instrument: str | None = None,
) -> pd.Series | None:
    required = {"factor", "open_condition", "close_condition", "excess_annual_return"}
    if rule_summary.empty or not required.issubset(rule_summary.columns):
        return None

    candidates = rule_summary[rule_summary["factor"].isin(factors)].copy()
    candidates = candidates[
        ~candidates["close_condition"].astype(str).str.startswith("闭仓_持仓满_")
    ]
    if instrument is not None and CODE_COL in candidates.columns:
        exact = candidates[candidates[CODE_COL].astype(str).eq(str(instrument))]
        if not exact.empty:
            candidates = exact
        else:
            all_rows = candidates[candidates[CODE_COL].astype(str).eq("ALL")]
            if not all_rows.empty:
                candidates = all_rows

    candidates = candidates[candidates["excess_annual_return"].notna()]
    if "trade_count" in candidates.columns:
        candidates = candidates[candidates["trade_count"].fillna(0) > 0]
    if candidates.empty:
        return None

    sort_cols = ["excess_annual_return"]
    ascending = [False]
    for col in ["sharpe", "excess_final_equity", "trade_count"]:
        if col in candidates.columns:
            sort_cols.append(col)
            ascending.append(False)
    return candidates.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0]


def _select_backtest_plot_spec(
    df: pd.DataFrame,
    factor: str,
    rule_summary: pd.DataFrame,
    instrument: str | None,
    select_by_backtest: bool = True,
) -> dict[str, Any]:
    factor_cols = _factor_plot_columns(df, factor)
    base_factor, _ = _split_factor_frequency(factor)
    base_col = base_factor if base_factor in factor_cols else factor_cols[0]
    smooth_cols = [col for col in (f"{base_factor}_季线", f"{base_factor}_年线") if col in factor_cols]

    if not select_by_backtest:
        return {
            "display_cols": factor_cols,
            "background_factor": None,
            "open_condition": None,
            "close_condition": None,
            "base_background_factor": None,
            "base_open_condition": None,
            "base_close_condition": None,
            "base_selection_note": "",
            "selection_note": "",
            "has_backtest_rule": False,
            "has_base_backtest_rule": False,
        }

    best_all = _best_rule_row(rule_summary, factor_cols, instrument=instrument)
    best_base = _best_rule_row(rule_summary, [base_col], instrument=instrument)
    best_smooth = _best_rule_row(rule_summary, smooth_cols, instrument=instrument)
    if best_smooth is not None:
        smooth_col = str(best_smooth["factor"])
    elif smooth_cols:
        smooth_col = smooth_cols[0]
    else:
        smooth_col = None

    display_cols = [base_col]
    if smooth_col is not None and smooth_col != base_col:
        display_cols.append(smooth_col)

    if best_all is None:
        return {
            "display_cols": display_cols,
            "background_factor": None,
            "open_condition": None,
            "close_condition": None,
            "base_background_factor": None,
            "base_open_condition": None,
            "base_close_condition": None,
            "base_selection_note": "",
            "selection_note": "未找到匹配的回测规则，持仓区间回退为信号观察口径",
            "has_backtest_rule": False,
            "has_base_backtest_rule": False,
        }

    excess = float(best_all["excess_annual_return"])
    base_note = ""
    if best_base is not None:
        base_excess = float(best_base["excess_annual_return"])
        base_note = (
            f"原始指标区间: {best_base['factor']} | {best_base['open_condition']} -> "
            f"{best_base['close_condition']} | 相对超额年化 {base_excess:.2%}"
        )
    return {
        "display_cols": display_cols,
        "background_factor": str(best_all["factor"]),
        "open_condition": str(best_all["open_condition"]),
        "close_condition": str(best_all["close_condition"]),
        "base_background_factor": None if best_base is None else str(best_base["factor"]),
        "base_open_condition": None if best_base is None else str(best_base["open_condition"]),
        "base_close_condition": None if best_base is None else str(best_base["close_condition"]),
        "base_selection_note": base_note,
        "selection_note": (
            f"持仓区间: {best_all['factor']} | {best_all['open_condition']} -> "
            f"{best_all['close_condition']} | 相对超额年化 {excess:.2%}"
        ),
        "has_backtest_rule": True,
        "has_base_backtest_rule": best_base is not None,
    }


def _read_or_build_signal_table_for_plot(
    df: pd.DataFrame,
    factor_cols: list[str],
    signal_table: pd.DataFrame | None = None,
    signal_path: str | Path | None = "results/signals.csv",
) -> pd.DataFrame:
    if signal_table is not None:
        signals = signal_table.copy()
    elif signal_path is not None:
        path = next((candidate for candidate in table_candidates(signal_path) if candidate.exists()), None)
        signals = read_table(path) if path is not None else generate_signal_table(df, factors=factor_cols)
    else:
        signals = generate_signal_table(df, factors=factor_cols)

    if not signals.empty:
        signals[SIGNAL_DATE_COL] = pd.to_datetime(signals[SIGNAL_DATE_COL])
    return signals


def _filter_signal_patterns(signal_df: pd.DataFrame, signal_patterns: Iterable[str] | None) -> pd.DataFrame:
    if signal_df.empty or signal_patterns is None:
        return signal_df
    patterns = [pattern for pattern in signal_patterns if pattern]
    if not patterns:
        return signal_df
    mask = pd.Series(False, index=signal_df.index)
    for pattern in patterns:
        mask |= signal_df[SIGNAL_PATTERN_COL].astype(str).str.contains(pattern, regex=False)
    return signal_df[mask].copy()


def _safe_plot_filename(value: str) -> str:
    import re

    return re.sub(r'[\\/:*?"<>|]+', "_", value)


def factor_plot_roots(factors: Iterable[str]) -> list[str]:
    """把原始/季线/年线字段收敛成基础因子，避免同一指标重复出图。"""
    roots: list[str] = []
    seen: set[str] = set()
    for factor in factors:
        base_factor, _ = _split_factor_frequency(str(factor))
        if base_factor in seen:
            continue
        seen.add(base_factor)
        roots.append(base_factor)
    return roots


def _long_regions_from_signal_points(
    plot_dates: pd.Index,
    signal_points: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if signal_points.empty or len(plot_dates) == 0:
        return []

    events = signal_points.groupby([SIGNAL_DATE_COL, SIGNAL_VALUE_COL]).size().unstack(fill_value=0)
    event_map = events.to_dict(orient="index")
    regions: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_long = False
    start_date: pd.Timestamp | None = None

    for dt in pd.to_datetime(plot_dates):
        counts = event_map.get(dt, {})
        open_hit = counts.get(1, 0) > 0
        close_hit = counts.get(-1, 0) > 0
        if open_hit and not in_long:
            in_long = True
            start_date = dt
        elif close_hit and in_long and not open_hit:
            regions.append((start_date, dt))
            in_long = False
            start_date = None

    if in_long and start_date is not None:
        regions.append((start_date, pd.Timestamp(plot_dates[-1])))
    return regions


def _long_regions_from_position(
    position_df: pd.DataFrame,
    plot_dates: pd.Index,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if position_df.empty or len(plot_dates) == 0:
        return []

    pos = position_df[[DATE_COL, "position"]].copy()
    pos[DATE_COL] = pd.to_datetime(pos[DATE_COL])
    pos = pos[pos[DATE_COL].between(pd.Timestamp(plot_dates.min()), pd.Timestamp(plot_dates.max()))]
    if pos.empty:
        return []

    regions: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_long = False
    start_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    for dt, value in zip(pos[DATE_COL], pos["position"]):
        dt = pd.Timestamp(dt)
        is_long = bool(value > 0)
        if is_long and not in_long:
            in_long = True
            start_date = dt
        elif not is_long and in_long:
            regions.append((start_date, last_date or dt))
            in_long = False
            start_date = None
        last_date = dt

    if in_long and start_date is not None:
        regions.append((start_date, pd.Timestamp(pos[DATE_COL].iloc[-1])))
    return regions


def _completed_trade_pairs_from_position(
    position_df: pd.DataFrame,
    plot_dates: pd.Index,
    major_regions: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if position_df.empty or len(plot_dates) == 0:
        return []

    pos = position_df[[DATE_COL, "position"]].copy()
    pos[DATE_COL] = pd.to_datetime(pos[DATE_COL])
    pos = pos[pos[DATE_COL].between(pd.Timestamp(plot_dates.min()), pd.Timestamp(plot_dates.max()))]
    if pos.empty:
        return []

    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_long = False
    buy_date: pd.Timestamp | None = None
    for dt, value in zip(pos[DATE_COL], pos["position"]):
        dt = pd.Timestamp(dt)
        is_long = bool(value > 0)
        if is_long and not in_long:
            buy_date = dt
            in_long = True
        elif not is_long and in_long:
            if buy_date is not None:
                pairs.append((buy_date, dt))
            buy_date = None
            in_long = False

    if not major_regions:
        return pairs

    filtered: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for buy_dt, sell_dt in pairs:
        for start_dt, end_dt in major_regions:
            start_dt = pd.Timestamp(start_dt)
            end_dt = pd.Timestamp(end_dt)
            if start_dt <= buy_dt <= end_dt and start_dt <= sell_dt <= end_dt:
                filtered.append((buy_dt, sell_dt))
                break
    return filtered


def _open_buy_dates_from_position(
    position_df: pd.DataFrame,
    plot_dates: pd.Index,
) -> list[pd.Timestamp]:
    if position_df.empty or len(plot_dates) == 0:
        return []

    pos = position_df[[DATE_COL, "position"]].copy()
    pos[DATE_COL] = pd.to_datetime(pos[DATE_COL])
    pos = pos[pos[DATE_COL].between(pd.Timestamp(plot_dates.min()), pd.Timestamp(plot_dates.max()))]
    if pos.empty:
        return []

    open_buy_dates: list[pd.Timestamp] = []
    in_long = False
    buy_date: pd.Timestamp | None = None
    for dt, value in zip(pos[DATE_COL], pos["position"]):
        dt = pd.Timestamp(dt)
        is_long = bool(value > 0)
        if is_long and not in_long:
            buy_date = dt
            in_long = True
        elif not is_long and in_long:
            buy_date = None
            in_long = False

    if in_long and buy_date is not None:
        open_buy_dates.append(buy_date)
    return open_buy_dates


def _backtest_position_for_plot(
    df: pd.DataFrame,
    signal_table: pd.DataFrame,
    instrument: str,
    factor: str,
    open_condition: str,
    close_condition: str,
) -> pd.DataFrame:
    from backtest import _backtest_rule_pair_from_cache
    from signal_generation import _build_factor_event_cache_from_signal_table, build_event_conditions

    conditions = build_event_conditions()
    all_conditions = conditions["open"] + conditions["close"]
    by_name = {condition.name: condition for condition in all_conditions}
    if open_condition not in by_name or close_condition not in by_name:
        return pd.DataFrame()

    factor_cache = _build_factor_event_cache_from_signal_table(df, factor, conditions, signal_table)
    _, equity = _backtest_rule_pair_from_cache(
        factor=factor,
        open_rule=by_name[open_condition],
        close_rule=by_name[close_condition],
        factor_cache=factor_cache,
        include_equity=True,
    )
    if equity is None or equity.empty:
        return pd.DataFrame()
    return equity[equity[CODE_COL].astype(str).eq(str(instrument))].copy()


def _signal_points_for_plot(
    group: pd.DataFrame,
    plot_df: pd.DataFrame,
    signals: pd.DataFrame,
    selected_instrument: str,
    factor_cols: list[str],
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    signal_points = signals[
        signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(selected_instrument))
        & signals[SIGNAL_FACTOR_COL].isin(factor_cols)
    ].copy()
    signal_points = signal_points[
        signal_points[SIGNAL_DATE_COL].between(plot_df.index.min(), plot_df.index.max())
    ]
    if signal_points.empty:
        return signal_points

    values = (
        group[[DATE_COL, *factor_cols]]
        .melt(id_vars=DATE_COL, var_name=SIGNAL_FACTOR_COL, value_name="factor_value")
        .rename(columns={DATE_COL: SIGNAL_DATE_COL})
    )
    values[SIGNAL_DATE_COL] = pd.to_datetime(values[SIGNAL_DATE_COL], errors="coerce")
    signal_points = signal_points.merge(
        values,
        on=[SIGNAL_DATE_COL, SIGNAL_FACTOR_COL],
        how="left",
    ).dropna(subset=["factor_value"])
    return (
        signal_points.groupby([SIGNAL_DATE_COL, SIGNAL_FACTOR_COL, SIGNAL_VALUE_COL], as_index=False)
        .agg(
            factor_value=("factor_value", "first"),
            pattern_count=(SIGNAL_PATTERN_COL, "size"),
            patterns=(SIGNAL_PATTERN_COL, lambda x: "，".join(list(dict.fromkeys(map(str, x)))[:3])),
        )
    )


def _refresh_legend(ax1, ax2=None, note: str | None = None, fontsize: int = 10):
    from matplotlib.lines import Line2D

    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
    else:
        handles2, labels2 = [], []

    handles = handles1 + handles2
    labels = labels1 + labels2
    if note:
        handles.append(Line2D([], [], linestyle="none", marker=None, color="none"))
        labels.append(note)

    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    ax1.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=fontsize)


def plot_factor_signal_chart(
    df: pd.DataFrame,
    factor: str,
    instrument: str | None = None,
    signal_table: pd.DataFrame | None = None,
    signal_path: str | Path | None = "results/signals.csv",
    rule_summary: pd.DataFrame | None = None,
    rule_summary_path: str | Path | None = "results/rule_pair_summary.csv",
    select_by_backtest: bool = True,
    price_col: str = PRICE_COL,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    lookback_days: int | None = TRADING_DAYS * 3,
    signal_patterns: Iterable[str] | None = None,
    figsize: tuple[int, int] = (14, 7),
    annotate_signals: bool = False,
    annotate_open_signals: bool = False,
    show_open_markers: bool = False,
    show_close_markers: bool = False,
    max_signal_labels: int = 80,
    shade_long_regions: bool = True,
    long_region_color: str = "#FF9999",
    long_region_alpha: float = 0.10,
    shade_base_best_regions: bool = True,
    show_base_best_trade_marks: bool = True,
    base_buy_color: str = "#169B62",
    base_sell_color: str = "#111111",
    show_base_factor_points: bool = True,
    base_factor_point_size: float = 13,
    base_factor_point_alpha: float = 0.55,
    backend: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    plt = _load_pyplot(backend)
    all_factor_cols = _factor_plot_columns(df, factor)
    selected_instrument = instrument or df[CODE_COL].dropna().iloc[0]
    rule_summary_df = _load_rule_summary_for_plot(rule_summary, rule_summary_path)
    plot_spec = _select_backtest_plot_spec(
        df=df,
        factor=factor,
        rule_summary=rule_summary_df,
        instrument=str(selected_instrument),
        select_by_backtest=select_by_backtest,
    )
    display_cols = [col for col in plot_spec["display_cols"] if col in df.columns]

    group = df[df[CODE_COL].astype(str).eq(str(selected_instrument))].copy()
    if group.empty:
        raise ValueError(f"Instrument not found: {selected_instrument}")

    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.sort_values(DATE_COL).reset_index(drop=True)
    if start_date is not None:
        group = group[group[DATE_COL].ge(pd.Timestamp(start_date))]
    if end_date is not None:
        group = group[group[DATE_COL].le(pd.Timestamp(end_date))]
    if start_date is None and lookback_days is not None and lookback_days > 0:
        group = group.tail(int(lookback_days))
    if group.empty:
        raise ValueError("No rows left after date filtering")

    plot_df = group[[DATE_COL, price_col, *display_cols]].set_index(DATE_COL)
    title_name = group[NAME_COL].iloc[-1] if NAME_COL in group else selected_instrument
    fig, ax1, ax2 = plot_dual_axis(
        plot_df,
        left_col=display_cols,
        right_col=price_col,
        figsize=figsize,
        title=f"{title_name} - {factor}",
        backend=backend,
        show=False,
    )
    ax1.axhline(0, color="#666666", linewidth=0.8, alpha=0.5)
    ax1.axhline(1, color="#999999", linewidth=0.7, alpha=0.25)
    ax1.axhline(-1, color="#999999", linewidth=0.7, alpha=0.25)
    ax1.set_ylabel("因子值 / sigma")
    if ax2 is not None:
        ax2.set_ylabel(price_col)
    base_factor, _ = _split_factor_frequency(factor)
    if show_base_factor_points and base_factor in plot_df.columns:
        ax1.scatter(
            plot_df.index,
            plot_df[base_factor],
            s=base_factor_point_size,
            color="#044E7E",
            alpha=base_factor_point_alpha,
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
            label=f"{base_factor} 数据点",
        )

    signal_factors = list(dict.fromkeys(all_factor_cols + display_cols))
    background_factor = plot_spec.get("background_factor")
    if background_factor:
        signal_factors = list(dict.fromkeys(signal_factors + [background_factor]))
    base_background_factor = plot_spec.get("base_background_factor")
    if base_background_factor:
        signal_factors = list(dict.fromkeys(signal_factors + [base_background_factor]))
    signals = _read_or_build_signal_table_for_plot(
        df=df,
        factor_cols=signal_factors,
        signal_table=signal_table,
        signal_path=signal_path,
    )
    filtered_signals = _filter_signal_patterns(signals, signal_patterns)
    signal_points = _signal_points_for_plot(
        group=group,
        plot_df=plot_df,
        signals=filtered_signals,
        selected_instrument=str(selected_instrument),
        factor_cols=display_cols,
    )

    if shade_long_regions:
        base_regions = []
        base_position_df = pd.DataFrame()
        if (
            shade_base_best_regions
            and plot_spec.get("has_base_backtest_rule")
        ):
            base_position_df = _backtest_position_for_plot(
                df=df,
                signal_table=signals,
                instrument=str(selected_instrument),
                factor=str(plot_spec["base_background_factor"]),
                open_condition=str(plot_spec["base_open_condition"]),
                close_condition=str(plot_spec["base_close_condition"]),
            )
            base_regions = _long_regions_from_position(base_position_df, plot_df.index)

        regions = []
        if plot_spec.get("has_backtest_rule"):
            position_df = _backtest_position_for_plot(
                df=df,
                signal_table=signals,
                instrument=str(selected_instrument),
                factor=str(plot_spec["background_factor"]),
                open_condition=str(plot_spec["open_condition"]),
                close_condition=str(plot_spec["close_condition"]),
            )
            regions = _long_regions_from_position(position_df, plot_df.index)
        if not regions:
            regions = _long_regions_from_signal_points(plot_df.index, signal_points)
        for start_dt, end_dt in regions:
            ax1.axvspan(
                start_dt,
                end_dt,
                color=long_region_color,
                alpha=long_region_alpha,
                linewidth=0,
                zorder=0,
                label="全局最优持仓区间" if start_dt == regions[0][0] and end_dt == regions[0][1] else None,
            )
        if show_base_best_trade_marks and not base_position_df.empty:
            mark_axis = ax2 if ax2 is not None else ax1
            mark_values = plot_df[price_col] if price_col in plot_df.columns else pd.Series(dtype=float)
            if not mark_values.empty:
                used_trade_label = False
                trade_pairs = _completed_trade_pairs_from_position(
                    base_position_df,
                    plot_df.index,
                    major_regions=None,
                )
                trade_marks: list[tuple[pd.Timestamp, str, str]] = []
                for base_start, base_end in trade_pairs:
                    trade_marks.extend(
                        [
                            (pd.Timestamp(base_start), "B", base_buy_color),
                            (pd.Timestamp(base_end), "S", base_sell_color),
                        ]
                    )
                for buy_date in _open_buy_dates_from_position(base_position_df, plot_df.index):
                    trade_marks.append((pd.Timestamp(buy_date), "B", base_buy_color))

                for mark_date, mark_text, color in trade_marks:
                    if mark_date not in mark_values.index:
                        nearest_pos = mark_values.index.get_indexer([mark_date], method="nearest")
                        if len(nearest_pos) == 0 or nearest_pos[0] < 0:
                            continue
                        mark_date = pd.Timestamp(mark_values.index[int(nearest_pos[0])])
                    mark_value = mark_values.loc[mark_date]
                    if not np.isfinite(mark_value):
                        continue
                    mark_axis.annotate(
                        mark_text,
                        (mark_date, float(mark_value)),
                        textcoords="offset points",
                        xytext=(0, 0),
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white",
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "facecolor": color,
                            "edgecolor": "white",
                            "linewidth": 0.5,
                            "alpha": 0.88,
                        },
                        zorder=7,
                    )
                    used_trade_label = True
                if used_trade_label:
                    mark_axis.scatter([], [], marker="$B/S$", s=90, color=base_sell_color, label="原始指标小级别买卖点")

    if not signal_points.empty:
        open_points = signal_points[signal_points[SIGNAL_VALUE_COL].eq(1)]
        close_points = signal_points[signal_points[SIGNAL_VALUE_COL].eq(-1)]
        if show_open_markers and not open_points.empty:
            ax1.scatter(
                open_points[SIGNAL_DATE_COL],
                open_points["factor_value"],
                marker="^",
                s=72,
                color="#169B62",
                edgecolor="white",
                linewidth=0.6,
                zorder=6,
                label="+1 开仓",
            )
        if show_close_markers and not close_points.empty:
            ax1.scatter(
                close_points[SIGNAL_DATE_COL],
                close_points["factor_value"],
                marker="v",
                s=72,
                color="#D62728",
                edgecolor="white",
                linewidth=0.6,
                zorder=6,
                label="-1 闭仓",
            )

        if annotate_signals:
            labels_df = signal_points.sort_values(SIGNAL_DATE_COL, ascending=False)
            if not annotate_open_signals:
                labels_df = labels_df[labels_df[SIGNAL_VALUE_COL].eq(-1)]
            if max_signal_labels >= 0:
                labels_df = labels_df.head(max_signal_labels)
            for row in labels_df.itertuples(index=False):
                signal_value = int(getattr(row, SIGNAL_VALUE_COL))
                y_offset = 8 if signal_value > 0 else -10
                va = "bottom" if signal_value > 0 else "top"
                ax1.annotate(
                    "+1" if signal_value > 0 else "-1",
                    (getattr(row, SIGNAL_DATE_COL), row.factor_value),
                    textcoords="offset points",
                    xytext=(0, y_offset),
                    ha="center",
                    va=va,
                    fontsize=9,
                    color="#169B62" if signal_value > 0 else "#D62728",
                )

    legend_note = plot_spec.get("selection_note") or ""
    base_note = plot_spec.get("base_selection_note") or ""
    if base_note and base_note != legend_note:
        legend_note = f"{legend_note}\n{base_note}" if legend_note else base_note
    _refresh_legend(ax1, ax2, note=legend_note, fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax1, ax2), signal_points


def plot_factor_signal_charts(
    df: pd.DataFrame,
    factors: Iterable[str],
    instrument: str | None = None,
    signal_table: pd.DataFrame | None = None,
    signal_path: str | Path | None = "results/signals.csv",
    rule_summary: pd.DataFrame | None = None,
    rule_summary_path: str | Path | None = "results/rule_pair_summary.csv",
    select_by_backtest: bool = True,
    save_dir: str | Path | None = "results/plots",
    show: bool = False,
    backend: str | None = "Agg",
    **kwargs,
) -> list[dict[str, Any]]:
    factor_list = list(factors)
    results: list[dict[str, Any]] = []
    all_factor_cols = list(
        dict.fromkeys(col for factor in factor_list for col in _factor_plot_columns(df, factor))
    )
    signals = _read_or_build_signal_table_for_plot(
        df=df,
        factor_cols=all_factor_cols,
        signal_table=signal_table,
        signal_path=signal_path,
    )
    rule_summary_df = _load_rule_summary_for_plot(rule_summary, rule_summary_path)

    for factor in factor_list:
        save_path = None
        if save_dir is not None:
            code = instrument or df[CODE_COL].dropna().iloc[0]
            save_path = Path(save_dir) / f"{_safe_plot_filename(str(code))}_{_safe_plot_filename(str(factor))}.png"
        fig, axes, signal_points = plot_factor_signal_chart(
            df=df,
            factor=factor,
            instrument=instrument,
            signal_table=signals,
            signal_path=None,
            rule_summary=rule_summary_df,
            rule_summary_path=None,
            select_by_backtest=select_by_backtest,
            backend=backend,
            show=show,
            save_path=save_path,
            **kwargs,
        )
        if not show:
            plt = _load_pyplot(None)
            plt.close(fig)
        results.append(
            {
                "factor": factor,
                "save_path": str(save_path) if save_path is not None else None,
                "signal_count": int(len(signal_points)),
            }
        )
    return results


def _load_equity_curves_for_plot(
    equity_curves: pd.DataFrame | None = None,
    equity_curves_path: str | Path | None = "results/equity_curves.csv",
) -> pd.DataFrame:
    if equity_curves is not None:
        data = equity_curves.copy()
    elif equity_curves_path is not None:
        path = next((candidate for candidate in table_candidates(equity_curves_path) if candidate.exists()), None)
        if path is None:
            return pd.DataFrame()
        data = read_table(path)
    else:
        return pd.DataFrame()

    if not data.empty and DATE_COL in data.columns:
        data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    return data


def _best_rule_pair_for_base_factor(
    rule_summary: pd.DataFrame,
    base_factor: str,
    instrument: str | None = None,
) -> pd.Series | None:
    factor_candidates = [base_factor, f"{base_factor}_季线", f"{base_factor}_年线"]
    return _best_rule_row(rule_summary, factor_candidates, instrument=instrument)


def _trade_marks_from_position(position_df: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    if position_df.empty:
        return []

    pos = position_df[[DATE_COL, "position"]].copy()
    pos[DATE_COL] = pd.to_datetime(pos[DATE_COL], errors="coerce")
    pos = pos.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    marks: list[tuple[pd.Timestamp, str]] = []
    prev = 0.0
    for dt_value, curr_value in zip(pos[DATE_COL], pos["position"]):
        curr = float(curr_value)
        dt = pd.Timestamp(dt_value)
        if prev <= 0 and curr > 0:
            marks.append((dt, "B"))
        elif prev > 0 and curr <= 0:
            marks.append((dt, "S"))
        prev = curr
    return marks


def _rule_signal_points(
    group: pd.DataFrame,
    plot_df: pd.DataFrame,
    signals: pd.DataFrame,
    instrument: str,
    factor: str,
    open_condition: str,
    close_condition: str,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    signal_points = signals[
        signals[SIGNAL_INSTRUMENT_COL].astype(str).eq(str(instrument))
        & signals[SIGNAL_FACTOR_COL].astype(str).eq(str(factor))
        & signals[SIGNAL_PATTERN_COL].isin([open_condition, close_condition])
    ].copy()
    signal_points[SIGNAL_DATE_COL] = pd.to_datetime(signal_points[SIGNAL_DATE_COL], errors="coerce")
    signal_points = signal_points[
        signal_points[SIGNAL_DATE_COL].between(plot_df.index.min(), plot_df.index.max())
    ]
    if signal_points.empty:
        return signal_points

    values = (
        group[[DATE_COL, factor]]
        .rename(columns={DATE_COL: SIGNAL_DATE_COL, factor: "factor_value"})
    )
    signal_points = signal_points.merge(values, on=SIGNAL_DATE_COL, how="left")
    return signal_points.dropna(subset=["factor_value"]).copy()


def plot_best_rule_pair_chart(
    df: pd.DataFrame,
    base_factor: str,
    instrument: str | None = None,
    signal_table: pd.DataFrame | None = None,
    signal_path: str | Path | None = "results/signals.csv",
    rule_summary: pd.DataFrame | None = None,
    rule_summary_path: str | Path | None = "results/rule_pair_summary.csv",
    equity_curves: pd.DataFrame | None = None,
    equity_curves_path: str | Path | None = "results/equity_curves.csv",
    price_col: str = PRICE_COL,
    lookback_days: int | None = None,
    backend: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    plt = _load_pyplot(backend)
    rule_summary_df = _load_rule_summary_for_plot(rule_summary, rule_summary_path)
    equity_df = _load_equity_curves_for_plot(equity_curves, equity_curves_path)
    best_row = _best_rule_pair_for_base_factor(rule_summary_df, base_factor, instrument=instrument)
    if best_row is None:
        raise ValueError(f"No best rule_pair found for base_factor={base_factor}")

    selected_instrument = str(instrument or best_row[CODE_COL])
    factor = str(best_row["factor"])
    open_condition = str(best_row["open_condition"])
    close_condition = str(best_row["close_condition"])

    group = df[df[CODE_COL].astype(str).eq(selected_instrument)].copy()
    if group.empty:
        raise ValueError(f"Instrument not found: {selected_instrument}")
    group[DATE_COL] = pd.to_datetime(group[DATE_COL], errors="coerce")
    group = group.sort_values(DATE_COL).reset_index(drop=True)

    position_df = _backtest_position_for_plot(
        df=df,
        signal_table=_read_or_build_signal_table_for_plot(df, [factor], signal_table, signal_path),
        instrument=selected_instrument,
        factor=factor,
        open_condition=open_condition,
        close_condition=close_condition,
    )
    if position_df.empty:
        raise ValueError(f"No equity curve found for {factor} | {open_condition} -> {close_condition}")

    position_df[DATE_COL] = pd.to_datetime(position_df[DATE_COL], errors="coerce")
    position_df = position_df.sort_values(DATE_COL).reset_index(drop=True)
    if lookback_days is not None and lookback_days > 0 and len(position_df) > lookback_days:
        position_df = position_df.iloc[-lookback_days:].copy()
    start_dt = pd.Timestamp(position_df[DATE_COL].iloc[0])
    end_dt = pd.Timestamp(position_df[DATE_COL].iloc[-1])
    group = group[group[DATE_COL].between(start_dt, end_dt)].copy()

    plot_df = group[[DATE_COL, factor, price_col]].set_index(DATE_COL)
    long_regions = _long_regions_from_position(position_df, plot_df.index)
    trade_marks = _trade_marks_from_position(position_df)
    signals = _read_or_build_signal_table_for_plot(df, [factor], signal_table, signal_path)
    rule_signal_points = _rule_signal_points(
        group=group,
        plot_df=plot_df,
        signals=signals,
        instrument=selected_instrument,
        factor=factor,
        open_condition=open_condition,
        close_condition=close_condition,
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True, constrained_layout=True)
    ax_open, ax_excess, ax_factor = axes
    ax_open_price = ax_open.twinx()

    ax_open_price.plot(
        plot_df.index,
        plot_df[price_col],
        color="#444444",
        linewidth=1.6,
        alpha=0.9,
        label="收盘价",
    )
    for idx, (start_region, end_region) in enumerate(long_regions):
        ax_open.axvspan(
            start_region,
            end_region,
            color="#f6c7c7",
            alpha=0.22,
            linewidth=0,
            label="多头区间" if idx == 0 else None,
        )
    price_series = plot_df[price_col]
    used_buy = False
    used_sell = False
    for dt, mark in trade_marks:
        if dt not in price_series.index:
            continue
        value = float(price_series.loc[dt])
        if mark == "B":
            ax_open_price.scatter(
                dt,
                value,
                marker="^",
                s=70,
                color="#169B62",
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
                label="开仓点" if not used_buy else None,
            )
            used_buy = True
        else:
            ax_open_price.scatter(
                dt,
                value,
                marker="v",
                s=70,
                color="#D62728",
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
                label="平仓点" if not used_sell else None,
            )
            used_sell = True
    ax_open.set_ylabel("持仓")
    ax_open.set_yticks([])
    ax_open_price.set_ylabel(price_col)
    _refresh_legend(ax_open, ax_open_price, fontsize=10)
    ax_open.set_title(
        f"{selected_instrument} {group[NAME_COL].iloc[-1]} | {factor}\n"
        f"开仓: {open_condition} | 平仓: {close_condition}",
        fontsize=12,
    )

    ax_excess.plot(
        position_df[DATE_COL],
        position_df["strategy_equity"],
        color="#1D3557",
        linewidth=1.5,
        alpha=0.9,
        label="策略净值",
    )
    ax_excess.plot(
        position_df[DATE_COL],
        position_df["excess_equity"],
        color="#C1121F",
        linewidth=1.9,
        label="超额曲线",
    )
    ax_excess.axhline(1.0, color="#777777", linewidth=0.9, linestyle=":")
    ax_excess.set_ylabel("超额净值")
    metric_text = (
        f"超额年化: {float(best_row['excess_annual_return']):.2%}\n"
        f"夏普: {float(best_row['sharpe']):.2f}\n"
        f"最大回撤: {float(best_row['max_drawdown']):.2%}"
    )
    ax_excess.text(
        0.5,
        0.98,
        metric_text,
        transform=ax_excess.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#999999",
            "linewidth": 0.6,
            "alpha": 0.88,
        },
    )
    ax_excess.legend(loc="upper left", ncol=2, fontsize=10)

    _, ax_factor_left, ax_factor_right = plot_dual_axis(
        plot_df,
        left_col=factor,
        right_col=price_col,
        title=None,
        backend=backend,
        show=False,
        save_path=None,
        ax1=ax_factor,
    )
    ax_factor_left.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
    ax_factor_left.axhline(1.0, color="#169B62", linewidth=0.9, alpha=0.5, linestyle="--")
    ax_factor_left.axhline(-1.0, color="#D62728", linewidth=0.9, alpha=0.5, linestyle="--")
    _refresh_legend(ax_factor_left, ax_factor_right, fontsize=10)
    ax_factor_left.set_ylabel("因子值 / sigma")
    if ax_factor_right is not None:
        ax_factor_right.set_ylabel(price_col)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, best_row


def plot_best_rule_pair_charts(
    df: pd.DataFrame,
    factors: Iterable[str],
    instrument: str | None = None,
    signal_table: pd.DataFrame | None = None,
    signal_path: str | Path | None = "results/signals.csv",
    rule_summary: pd.DataFrame | None = None,
    rule_summary_path: str | Path | None = "results/rule_pair_summary.csv",
    equity_curves: pd.DataFrame | None = None,
    equity_curves_path: str | Path | None = "results/equity_curves.csv",
    save_dir: str | Path | None = "results/rule_pair_best_plots",
    show: bool = False,
    backend: str | None = "Agg",
    lookback_days: int | None = None,
) -> list[dict[str, Any]]:
    plt = _load_pyplot(backend)
    factor_list = factor_plot_roots(factors)
    results: list[dict[str, Any]] = []
    codes = [instrument] if instrument is not None else df[CODE_COL].dropna().astype(str).drop_duplicates().tolist()

    signals = _read_or_build_signal_table_for_plot(df, list(factor_list), signal_table, signal_path)
    rule_summary_df = _load_rule_summary_for_plot(rule_summary, rule_summary_path)
    equity_df = _load_equity_curves_for_plot(equity_curves, equity_curves_path)

    for code in codes:
        for base_factor in factor_list:
            save_path = None
            if save_dir is not None:
                save_path = Path(save_dir) / f"{_safe_plot_filename(str(code))}_{_safe_plot_filename(str(base_factor))}_best_rule_pair.png"
            try:
                fig, _, best_row = plot_best_rule_pair_chart(
                    df=df,
                    base_factor=base_factor,
                    instrument=str(code),
                    signal_table=signals,
                    signal_path=None,
                    rule_summary=rule_summary_df,
                    rule_summary_path=None,
                    equity_curves=equity_df,
                    equity_curves_path=None,
                    lookback_days=lookback_days,
                    backend=backend,
                    show=show,
                    save_path=save_path,
                )
                results.append(
                    {
                        "code": str(code),
                        "base_factor": base_factor,
                        "factor": str(best_row["factor"]),
                        "open_condition": str(best_row["open_condition"]),
                        "close_condition": str(best_row["close_condition"]),
                        "excess_annual_return": float(best_row["excess_annual_return"]),
                        "save_path": str(save_path) if save_path is not None else None,
                    }
                )
                if not show:
                    plt.close(fig)
            except ValueError:
                continue
    return results
