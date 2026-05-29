r"""Score event backtest with standardized z-scores (monthly_refresh mode, from pre-computed CSV).

Reads ``monthly_refresh_daily_score.csv`` and ``input_snapshot.csv`` from a
previous ``run_full_pipeline(..., score_mode="monthly_refresh")`` run.  No
re-computation of scores 鈥?only backtest against benchmark (always-long).

Key design:
- entry_score 鈫?open rules only; exit_score 鈫?close rules only
- z-score standardization with shift(1) rolling window (no look-ahead)
- MA smoothing applied after z-score for noise reduction
- signal has 1-bar delay (t signal 鈫?t+1 position)
- benchmark is always-long within the same date window
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

from io_utils import read_table, resolve_table_file, write_table
from timing_config import CODE_COL, DATE_COL, NAME_COL, PRICE_COL, TRADING_DAYS


def _resolve_existing_file(root: str | Path, candidates: list[str]) -> Path:
    return resolve_table_file(root, candidates)


def _strategy_output_dirs(root: str | Path) -> tuple[Path, Path]:
    base = Path(root)
    result_dir = base / "results" / "strategy"
    plot_dir = base / "plots" / "strategy"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, plot_dir

# ---------------------------------------------------------------------------
# Z-score helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Z-score helper
# ---------------------------------------------------------------------------
def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Compute rolling z-score (shifted to avoid look-ahead)."""
    if min_periods is None:
        min_periods = max(10, window // 4)
    s = pd.to_numeric(series, errors="coerce")
    mean = s.shift(1).rolling(window, min_periods=min_periods).mean()
    std = s.shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)


def _format_theta(value: str) -> str:
    try:
        return f"{float(value):g}"
    except Exception:
        return value


def format_rule_name_cn(rule_name: str) -> str:
    text = str(rule_name).removeprefix("open_").removeprefix("close_")

    patterns: list[tuple[str, str]] = [
        (r"^cumrise_lb(\d+)_sm(\d+)_d(\d+)_t(.+)$", "过去{2}天内，抄底/逃顶得分快速上升超过{3}，统计窗口{0}天，平滑{1}天"),
        (r"^cross_up_lb(\d+)_sm(\d+)_t(.+)$", "得分向上突破阈值{2}，统计窗口{0}天，平滑{1}天"),
        (r"^consec_up_lb(\d+)_sm(\d+)_n(\d+)$", "得分连续{2}天走强，统计窗口{0}天，平滑{1}天"),
        (r"^above_ma_lb(\d+)_sm(\d+)_w(\d+)$", "得分上穿自身{2}天均线，统计窗口{0}天，平滑{1}天"),
        (r"^bb_breakout_lb(\d+)_sm(\d+)_w(\d+)$", "得分向上突破布林上轨，布林窗口{2}天，统计窗口{0}天，平滑{1}天"),
        (r"^accel_up_lb(\d+)_sm(\d+)$", "得分增速继续走强，统计窗口{0}天，平滑{1}天"),
        (r"^roll_max_lb(\d+)_sm(\d+)_w(\d+)$", "得分创出近{2}天新高，统计窗口{0}天，平滑{1}天"),
        (r"^rank_extreme_lb(\d+)_sm(\d+)_w(\d+)_p(.+)$", "得分进入近{2}天的高分位区间（{3}分位），统计窗口{0}天，平滑{1}天"),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, text)
        if match:
            values = list(match.groups())
            if values:
                values[-1] = _format_theta(values[-1])
            return template.format(*values)
    return text


# ---------------------------------------------------------------------------
# Rule data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZRule:
    """A single signal rule (open *or* close) operating on z-scored columns."""
    name: str          # e.g. "open_cross_up_lb252_sm5_t0.5"
    side: str          # "open" or "close"
    logic: str         # discriminator for the detection function
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PairedZRule:
    key: str
    open_rule: ZRule
    close_rule: ZRule


# ---------------------------------------------------------------------------
# Rule detection helpers (all operate on a single z-scored series)
# ---------------------------------------------------------------------------
def _cross_up(s: pd.Series, threshold: pd.Series | float) -> pd.Series:
    """True on the first bar where s >= threshold."""
    left = pd.to_numeric(s, errors="coerce")
    right = threshold if isinstance(threshold, pd.Series) else float(threshold)
    cur = left.ge(right)
    prev = left.shift(1).lt(right if isinstance(right, pd.Series) else float(right))
    return cur.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _cross_down(s: pd.Series, threshold: pd.Series | float) -> pd.Series:
    """True on the first bar where s <= threshold."""
    left = pd.to_numeric(s, errors="coerce")
    right = threshold if isinstance(threshold, pd.Series) else float(threshold)
    cur = left.le(right)
    prev = left.shift(1).gt(right if isinstance(right, pd.Series) else float(right))
    return cur.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _consecutive_up(s: pd.Series, days: int) -> pd.Series:
    """True at the *end* of the first 'days'-bar up-streak."""
    rising = pd.to_numeric(s, errors="coerce").diff().gt(0)
    streak = rising.rolling(days, min_periods=days).sum().eq(days)
    s_bool = streak.fillna(False).astype(bool)
    return s_bool & ~s_bool.shift(1, fill_value=False)


def _consecutive_down(s: pd.Series, days: int) -> pd.Series:
    """True at the *end* of the first 'days'-bar down-streak."""
    falling = pd.to_numeric(s, errors="coerce").diff().lt(0)
    streak = falling.rolling(days, min_periods=days).sum().eq(days)
    s_bool = streak.fillna(False).astype(bool)
    return s_bool & ~s_bool.shift(1, fill_value=False)


def _cumrise(s: pd.Series, days: int, threshold: float) -> pd.Series:
    """True on the bar where running *d*-day change crosses >= threshold."""
    s = pd.to_numeric(s, errors="coerce")
    delta = s - s.shift(days)
    triggered = delta.ge(float(threshold))
    prev = delta.shift(1).lt(float(threshold))
    return triggered.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _cumdrop(s: pd.Series, days: int, threshold: float) -> pd.Series:
    """True on the bar where running *d*-day change crosses <= -threshold."""
    s = pd.to_numeric(s, errors="coerce")
    delta = s - s.shift(days)
    triggered = delta.le(-float(threshold))
    prev = delta.shift(1).gt(-float(threshold))
    return triggered.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _above_ma(s: pd.Series, window: int) -> pd.Series:
    ma = s.rolling(window, min_periods=max(5, window // 2)).mean()
    cur = s.ge(ma)
    prev = s.shift(1).lt(ma.shift(1))
    return cur.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _below_ma(s: pd.Series, window: int) -> pd.Series:
    ma = s.rolling(window, min_periods=max(5, window // 2)).mean()
    cur = s.le(ma)
    prev = s.shift(1).gt(ma.shift(1))
    return cur.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _bb_breakout(s: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    """Break out above upper Bollinger band."""
    ma = s.rolling(window, min_periods=max(5, window // 2)).mean()
    std = s.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)
    upper = ma + n_std * std
    return _cross_up(s, upper)


def _bb_breakdown(s: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    """Break down below lower Bollinger band."""
    ma = s.rolling(window, min_periods=max(5, window // 2)).mean()
    std = s.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)
    lower = ma - n_std * std
    return _cross_down(s, lower)


def _accel_up(s: pd.Series) -> pd.Series:
    """螖 > 0 and 螖虏 > 0."""
    d1 = pd.to_numeric(s, errors="coerce").diff()
    d2 = d1.diff()
    trigger = d1.gt(0) & d2.gt(0)
    prev = d1.shift(1).le(0) | d2.shift(1).le(0)
    return trigger.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _accel_down(s: pd.Series) -> pd.Series:
    """螖 < 0 and 螖虏 < 0."""
    d1 = pd.to_numeric(s, errors="coerce").diff()
    d2 = d1.diff()
    trigger = d1.lt(0) & d2.lt(0)
    prev = d1.shift(1).ge(0) | d2.shift(1).ge(0)
    return trigger.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _roll_max_event(s: pd.Series, window: int) -> pd.Series:
    """True when s equals the rolling max (new high)."""
    s = pd.to_numeric(s, errors="coerce")
    roll_max = s.rolling(window, min_periods=max(5, window // 2)).max()
    cur = s.ge(roll_max)
    prev = s.shift(1).lt(roll_max.shift(1))
    return cur.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _roll_min_event(s: pd.Series, window: int) -> pd.Series:
    """True when s equals the rolling min (new low)."""
    s = pd.to_numeric(s, errors="coerce")
    roll_min = s.rolling(window, min_periods=max(5, window // 2)).min()
    cur = s.le(roll_min)
    prev = s.shift(1).gt(roll_min.shift(1))
    return cur.fillna(False).astype(bool) & prev.fillna(False).astype(bool)


def _rank_extreme(s: pd.Series, window: int, pct: float, upper: bool) -> pd.Series:
    """True when percentile rank exceeds threshold."""
    s = pd.to_numeric(s, errors="coerce")
    rank_pct = s.rolling(window, min_periods=max(10, window // 2)).rank(pct=True)
    if upper:
        return rank_pct.ge(pct) & rank_pct.shift(1).lt(pct)
    else:
        return rank_pct.le(1 - pct) & rank_pct.shift(1).gt(1 - pct)


# ---------------------------------------------------------------------------
# Rule detection dispatcher
# ---------------------------------------------------------------------------
def detect_z_event(df: pd.DataFrame, rule: ZRule) -> pd.Series:
    """Detect events for a single ZRule.

    Open rules operate on ``entry_z`` only; close rules operate on ``exit_z`` only.
    This enforces strict separation: entry_score drives open signals,
    exit_score drives close signals.
    """
    s = df["entry_z"].copy() if rule.side == "open" else df["exit_z"].copy()

    logic = rule.logic
    p = rule.params

    # --- threshold cross ---
    if logic == "cross_up":
        return _cross_up(s, float(p["theta"]))
    if logic == "cross_down":
        return _cross_down(s, float(p["theta"]))

    # --- consecutive streaks ---
    if logic == "consecutive_up":
        return _consecutive_up(s, int(p["n"]))
    if logic == "consecutive_down":
        return _consecutive_down(s, int(p["n"]))

    # --- cumulative change ---
    if logic == "cumrise":
        return _cumrise(s, int(p["d"]), float(p["theta"]))
    if logic == "cumdrop":
        return _cumdrop(s, int(p["d"]), float(p["theta"]))

    # --- MA cross ---
    if logic == "above_ma":
        return _above_ma(s, int(p["w"]))
    if logic == "below_ma":
        return _below_ma(s, int(p["w"]))

    # --- Bollinger bands ---
    if logic == "bb_breakout":
        return _bb_breakout(s, int(p["w"]), float(p.get("n_std", 2.0)))
    if logic == "bb_breakdown":
        return _bb_breakdown(s, int(p["w"]), float(p.get("n_std", 2.0)))

    # --- acceleration ---
    if logic == "accel_up":
        return _accel_up(s)
    if logic == "accel_down":
        return _accel_down(s)

    # --- rolling extremes ---
    if logic == "roll_max":
        return _roll_max_event(s, int(p["w"]))
    if logic == "roll_min":
        return _roll_min_event(s, int(p["w"]))

    # --- percentile rank ---
    if logic == "rank_extreme":
        return _rank_extreme(s, int(p["w"]), float(p["p"]), upper=bool(p.get("upper", True)))

    raise ValueError(f"Unsupported logic: {logic}")


# ---------------------------------------------------------------------------
# Rule builder: generate all (open, close) paired rules
# ---------------------------------------------------------------------------
def build_all_z_rules() -> list[PairedZRule]:
    """Build paired rules. Both open and close use the same logic family,
    but open operates on entry_z and close operates on exit_z."""
    lookbacks = [60, 120, 252]
    smooths = [3, 5, 10]
    thetas = [0.0, 0.5, 1.0, 1.5]
    ws = [10, 20, 60]
    ds = [5, 10, 20]
    ns = [2, 3, 5]
    ps = [0.80, 0.90, 0.95]

    rules: list[PairedZRule] = []

    def add(key: str, logic: str, params: dict):
        rules.append(PairedZRule(
            key=key,
            open_rule=ZRule(f"open_{key}", "open", logic, dict(params)),
            close_rule=ZRule(f"close_{key}", "close", logic, dict(params)),
        ))

    # 1. cross_up: score crosses above threshold
    for lb in lookbacks:
        for sm in smooths:
            for theta in thetas:
                add(f"cross_up_lb{lb}_sm{sm}_t{theta}", "cross_up",
                    {"lookback": lb, "smooth": sm, "theta": theta})

    # 2. consecutive_up: score rising for n consecutive days
    for lb in lookbacks:
        for sm in smooths:
            for n in ns:
                add(f"consec_up_lb{lb}_sm{sm}_n{n}", "consecutive_up",
                    {"lookback": lb, "smooth": sm, "n": n})

    # 3. cumrise: d-day cumulative rise exceeds threshold
    for lb in lookbacks:
        for sm in smooths:
            for d in ds:
                for theta in thetas:
                    if theta == 0.0:
                        continue
                    add(f"cumrise_lb{lb}_sm{sm}_d{d}_t{theta}", "cumrise",
                        {"lookback": lb, "smooth": sm, "d": d, "theta": theta})

    # 4. above_ma: score crosses above w-day MA
    for lb in lookbacks:
        for sm in smooths:
            for w in ws:
                add(f"above_ma_lb{lb}_sm{sm}_w{w}", "above_ma",
                    {"lookback": lb, "smooth": sm, "w": w})

    # 5. bb_breakout: score breaks above upper Bollinger band
    for lb in lookbacks:
        for sm in smooths:
            for w in ws:
                add(f"bb_breakout_lb{lb}_sm{sm}_w{w}", "bb_breakout",
                    {"lookback": lb, "smooth": sm, "w": w})

    # 6. accel_up: score shows positive acceleration
    for lb in lookbacks:
        for sm in smooths:
            add(f"accel_up_lb{lb}_sm{sm}", "accel_up",
                {"lookback": lb, "smooth": sm})

    # 7. roll_max: score hits w-day rolling max
    for lb in lookbacks:
        for sm in smooths:
            for w in ws:
                add(f"roll_max_lb{lb}_sm{sm}_w{w}", "roll_max",
                    {"lookback": lb, "smooth": sm, "w": w})

    # 8. rank_extreme: score in upper percentile
    for lb in lookbacks:
        for sm in smooths:
            for w in ws:
                for p in ps:
                    add(f"rank_extreme_lb{lb}_sm{sm}_w{w}_p{p}", "rank_extreme",
                        {"lookback": lb, "smooth": sm, "w": w, "p": p, "upper": True})

    return rules


# ---------------------------------------------------------------------------
# Position builder
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    vol = float(returns.std(ddof=0))
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(returns.mean() / vol * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def backtest_z_rules(
    price_df: pd.DataFrame,
    daily_score: pd.DataFrame,
    paired_rules: Iterable[PairedZRule],
    output_dir: str | Path | None = None,
    score_suffix: str = "",
    file_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    best_equity = pd.DataFrame()
    best_excess_annual_return = -np.inf

    entry_col = f"entry_score{score_suffix}"
    exit_col = f"exit_score{score_suffix}"
    codes = sorted(set(price_df[CODE_COL].astype(str)) & set(daily_score[CODE_COL].astype(str)))
    for code in codes:
        # merge price and scores
        price_part = price_df[price_df[CODE_COL].astype(str).eq(str(code))].copy()
        score_part = daily_score[daily_score[CODE_COL].astype(str).eq(str(code))].copy()
        merged = price_part[[DATE_COL, CODE_COL, NAME_COL, PRICE_COL]].merge(
            score_part[[DATE_COL, CODE_COL, entry_col, exit_col]],
            on=[DATE_COL, CODE_COL],
            how="left",
        )
        merged = merged.sort_values(DATE_COL).reset_index(drop=True)
        valid_score_mask = merged[[entry_col, exit_col]].notna().any(axis=1)
        if not bool(valid_score_mask.any()):
            continue
        first_valid_idx = int(np.flatnonzero(valid_score_mask.to_numpy(dtype=bool))[0])
        merged = merged.iloc[first_valid_idx:].reset_index(drop=True)
        merged["entry_score"] = pd.to_numeric(merged[entry_col], errors="coerce").fillna(0.0)
        merged["exit_score"] = pd.to_numeric(merged[exit_col], errors="coerce").fillna(0.0)
        merged[PRICE_COL] = pd.to_numeric(merged[PRICE_COL], errors="coerce")
        if merged.empty:
            continue

        benchmark_returns = merged[PRICE_COL].pct_change().fillna(0.0)
        benchmark_equity = (1.0 + benchmark_returns).cumprod()

        for pair in paired_rules:
            # get the lookback and smooth from params
            lookback = int(pair.open_rule.params.get("lookback",
                          pair.close_rule.params.get("lookback", 252)))
            smooth = int(pair.open_rule.params.get("smooth",
                         pair.close_rule.params.get("smooth", 5)))

            # standardize entry and exit independently with this lookback, then MA-smooth
            merged["entry_z"] = rolling_zscore(merged["entry_score"], lookback)
            merged["entry_z"] = merged["entry_z"].rolling(smooth, min_periods=1).mean()
            merged["exit_z"] = rolling_zscore(merged["exit_score"], lookback)
            merged["exit_z"] = merged["exit_z"].rolling(smooth, min_periods=1).mean()

            open_events = detect_z_event(merged, pair.open_rule).fillna(False).to_numpy(dtype=bool)
            close_events = detect_z_event(merged, pair.close_rule).fillna(False).to_numpy(dtype=bool)

            position = _build_position(open_events, close_events)
            strategy_returns = benchmark_returns * position
            strategy_equity = (1.0 + strategy_returns).cumprod()
            excess_equity = strategy_equity / benchmark_equity.replace(0, np.nan)
            trade_count = int(((pd.Series(position).shift(1).fillna(0.0) <= 0) & (pd.Series(position) > 0)).sum())

            row = {
                CODE_COL: code,
                NAME_COL: merged[NAME_COL].dropna().iloc[-1] if merged[NAME_COL].notna().any() else np.nan,
                "rule_key": pair.key,
                "open_rule": pair.open_rule.name,
                "close_rule": pair.close_rule.name,
                "lookback": lookback,
                "smooth": smooth,
                "annual_return": _annual_return(strategy_equity),
                "benchmark_annual_return": _annual_return(benchmark_equity),
                "excess_annual_return": _annual_return(excess_equity),
                "max_drawdown": _max_drawdown(strategy_equity),
                "benchmark_max_drawdown": _max_drawdown(benchmark_equity),
                "excess_max_drawdown": _max_drawdown(excess_equity),
                "sharpe": _sharpe(strategy_returns),
                "benchmark_sharpe": _sharpe(benchmark_returns),
                "holding_ratio": float(position.mean()) if len(position) else np.nan,
                "trade_count": trade_count,
                "final_equity": float(strategy_equity.iloc[-1]) if not strategy_equity.empty else np.nan,
                "benchmark_final_equity": float(benchmark_equity.iloc[-1]) if not benchmark_equity.empty else np.nan,
                "excess_final_equity": float(excess_equity.dropna().iloc[-1]) if not excess_equity.dropna().empty else np.nan,
                "open_event_count": int(open_events.sum()),
                "close_event_count": int(close_events.sum()),
            }
            summary_rows.append(row)

            current_excess = float(row["excess_annual_return"]) if np.isfinite(row["excess_annual_return"]) else -np.inf
            if current_excess > best_excess_annual_return:
                best_excess_annual_return = current_excess
                equity_data = {
                    DATE_COL: merged[DATE_COL],
                    CODE_COL: code,
                    NAME_COL: merged[NAME_COL],
                    PRICE_COL: merged[PRICE_COL],
                    "entry_score": merged["entry_score"],
                    "exit_score": merged["exit_score"],
                    "entry_z": merged["entry_z"],
                    "exit_z": merged["exit_z"],
                    "open_event": open_events.astype(int),
                    "close_event": close_events.astype(int),
                    "position": position,
                    "strategy_equity": strategy_equity,
                    "benchmark_equity": benchmark_equity,
                    "excess_equity": excess_equity,
                    "open_rule": pair.open_rule.name,
                    "close_rule": pair.close_rule.name,
                }
                best_equity = pd.DataFrame(equity_data)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["excess_annual_return", "sharpe", "annual_return"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    if output_dir is not None:
        output_path, _ = _strategy_output_dirs(output_dir)
        write_table(summary, output_path / f"monthly_strategy_summary{file_suffix}.csv")
        if not best_equity.empty:
            write_table(best_equity, output_path / f"monthly_strategy_best_equity{file_suffix}.csv")

    return summary, best_equity


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly-refresh z-score rule backtest (reads pre-computed CSV)."
    )
    parser.add_argument("--input-dir", default="results_score_event_full_monthly")
    parser.add_argument("--output-dir", default="results_score_event_full_monthly")
    parser.add_argument("--score-suffix", default="default", type=str,
                        help="Score variant suffix (default=entry_score, _sum=entry_score_sum, etc.)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    daily_score = read_table(
        _resolve_existing_file(
            input_dir,
            [
                "results/score/monthly_refresh_daily_score.csv",
                "monthly_refresh_daily_score.csv",
            ],
        )
    )
    price_df = read_table(
        _resolve_existing_file(
            input_dir,
            [
                "data/input_snapshot.csv",
                "input_snapshot.csv",
            ],
        )
    )

    rules = build_all_z_rules()
    print(f"Total paired rules: {len(rules)}")

    # map "default" keyword back to "" for backward compatibility
    score_suffix = "" if args.score_suffix == "default" else args.score_suffix
    suffix_label = score_suffix if score_suffix else "_default"
    print(f"Score variant: entry_score{score_suffix} / exit_score{score_suffix}")
    summary, best_equity = backtest_z_rules(
        price_df=price_df,
        daily_score=daily_score,
        paired_rules=rules,
        output_dir=output_dir,
        score_suffix=score_suffix,
        file_suffix=suffix_label,
    )

    print(f"Summary rows: {len(summary)}")
    if not summary.empty:
        print(summary.head(20).to_string(index=False))

    if not best_equity.empty:
        plot_best_rule(best_equity, output_dir=output_dir, file_suffix=suffix_label)


def plot_best_rule(
    best_equity: pd.DataFrame,
    output_dir: str | Path | None = None,
    file_suffix: str = "_default",
) -> None:
    """Save baseline strategy plots.

    主图改成三联图：
    1. 开平仓点与持仓区间
    2. 抄底/逃顶得分 + 收盘价
    3. 策略净值与基准净值

    另外固定额外输出一张：
    - 20日 zscore + 3日 MA 的 score z 图
    """
    df = best_equity.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    output_path: Path | None = None
    if output_dir is not None:
        _, output_path = _strategy_output_dirs(output_dir)

    in_long = df["position"].gt(0).to_numpy(dtype=bool)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(18, 12), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.6, 1.2]},
    )
    open_rule_cn = format_rule_name_cn(str(df["open_rule"].iloc[-1]))
    close_rule_cn = format_rule_name_cn(str(df["close_rule"].iloc[-1]))
    fig.suptitle(
        f"{df[CODE_COL].iloc[0]} {df[NAME_COL].iloc[0] if not df[NAME_COL].isna().all() else ''}\n"
        f"开仓: {open_rule_cn} | 平仓: {close_rule_cn}",
        fontsize=13,
        fontweight="bold",
    )

    # --- subplot 1: 开平仓点与持仓区间 ---
    ax1.plot(df[DATE_COL], df[PRICE_COL], color="steelblue", linewidth=0.8, label="收盘价")
    ax1.set_ylabel("收盘价", fontsize=11)

    start_idx: int | None = None
    for i in range(len(in_long)):
        if in_long[i] and start_idx is None:
            start_idx = i
        elif not in_long[i] and start_idx is not None:
            ax1.axvspan(
                df[DATE_COL].iloc[start_idx],
                df[DATE_COL].iloc[i],
                facecolor="crimson",
                alpha=0.18,
                edgecolor="none",
            )
            start_idx = None
    if start_idx is not None:
        ax1.axvspan(
            df[DATE_COL].iloc[start_idx],
            df[DATE_COL].iloc[-1],
            facecolor="crimson",
            alpha=0.18,
            edgecolor="none",
        )

    open_dates = df[DATE_COL][df["open_event"].gt(0) & df["open_event"].shift(1, fill_value=0).eq(0)]
    close_dates = df[DATE_COL][df["close_event"].gt(0) & df["close_event"].shift(1, fill_value=0).eq(0)]
    ax1.scatter(
        open_dates,
        df.loc[open_dates.index, PRICE_COL],
        marker="^",
        color="green",
        s=40,
        zorder=5,
        label="开仓信号",
    )
    ax1.scatter(
        close_dates,
        df.loc[close_dates.index, PRICE_COL],
        marker="v",
        color="darkorange",
        s=40,
        zorder=5,
        label="平仓信号",
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- subplot 2: 得分 + 收盘价 ---
    ax2_2 = ax2.twinx()
    ax2.plot(df[DATE_COL], df["entry_z"], color="#d62728", linewidth=1.0, label="抄底得分")
    ax2.plot(df[DATE_COL], df["exit_z"], color="#2ca02c", linewidth=1.0, label="逃顶得分")
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("得分Z", fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax2_2.plot(df[DATE_COL], df[PRICE_COL], color="steelblue", linewidth=0.8, alpha=0.75, label="收盘价")
    ax2_2.set_ylabel("收盘价", fontsize=11)

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles22, labels22 = ax2_2.get_legend_handles_labels()
    ax2.legend(handles2 + handles22, labels2 + labels22, loc="upper left", fontsize=9)

    # --- subplot 3: 策略净值 ---
    excess = df["excess_equity"].dropna()
    ax3.plot(df[DATE_COL], df["strategy_equity"], color="crimson", linewidth=1.0, label="策略净值")
    ax3.plot(
        df[DATE_COL],
        df["benchmark_equity"],
        color="gray",
        linewidth=0.8,
        linestyle="--",
        label="基准净值 (Always-Long)",
    )
    ax3.set_ylabel("净值", fontsize=11)
    ax3.set_xlabel("日期", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    if not excess.empty:
        final_excess = float(excess.iloc[-1])
        ax3.text(
            0.02,
            0.92,
            f"超额净值: {final_excess:.4f}",
            transform=ax3.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path is not None:
        plot_path = output_path / f"monthly_best_rule_plot{file_suffix}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    plt.close(fig)

    # --- extra fixed chart: 20日zscore + 3日MA ---
    entry_z_20_3 = rolling_zscore(df["entry_score"], 20).rolling(3, min_periods=1).mean()
    exit_z_20_3 = rolling_zscore(df["exit_score"], 20).rolling(3, min_periods=1).mean()
    z20_df = df[[DATE_COL, PRICE_COL]].copy()
    z20_df["entry_z_20_3"] = entry_z_20_3
    z20_df["exit_z_20_3"] = exit_z_20_3

    z20_df = z20_df.tail(750).reset_index(drop=True)

    z_fig, (z_ax1, z_ax2) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    z_ax1_r = z_ax1.twinx()
    z_ax2_r = z_ax2.twinx()

    z_ax1.plot(z20_df[DATE_COL], z20_df["entry_z_20_3"], color="#d62728", linewidth=1.0, label="抄底得分 (20Z+3MA)")
    z_ax1.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    z_ax1.set_ylabel("抄底Z", fontsize=11)
    z_ax1.grid(True, alpha=0.3)
    z_ax1_r.plot(z20_df[DATE_COL], z20_df[PRICE_COL], color="steelblue", linewidth=0.8, alpha=0.75, label="收盘价")
    z_ax1_r.set_ylabel("收盘价", fontsize=11)

    z_ax2.plot(z20_df[DATE_COL], z20_df["exit_z_20_3"], color="#2ca02c", linewidth=1.0, label="逃顶得分 (20Z+3MA)")
    z_ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    z_ax2.set_ylabel("逃顶Z", fontsize=11)
    z_ax2.set_xlabel("日期", fontsize=11)
    z_ax2.grid(True, alpha=0.3)
    z_ax2_r.plot(z20_df[DATE_COL], z20_df[PRICE_COL], color="steelblue", linewidth=0.8, alpha=0.75, label="收盘价")
    z_ax2_r.set_ylabel("收盘价", fontsize=11)

    z_fig.suptitle(
        f"{df[CODE_COL].iloc[0]} {df[NAME_COL].iloc[0] if not df[NAME_COL].isna().all() else ''}\n20日 ZScore + 3日 MA 得分图（最近750交易日）",
        fontsize=13,
        fontweight="bold",
    )
    handles_top_l, labels_top_l = z_ax1.get_legend_handles_labels()
    handles_top_r, labels_top_r = z_ax1_r.get_legend_handles_labels()
    z_ax1.legend(handles_top_l + handles_top_r, labels_top_l + labels_top_r, loc="upper left", fontsize=9)
    handles_bot_l, labels_bot_l = z_ax2.get_legend_handles_labels()
    handles_bot_r, labels_bot_r = z_ax2_r.get_legend_handles_labels()
    z_ax2.legend(handles_bot_l + handles_bot_r, labels_bot_l + labels_bot_r, loc="upper left", fontsize=9)

    z_fig.autofmt_xdate()
    z_fig.tight_layout()

    if output_path is not None:
        z20_plot_path = output_path / f"monthly_score_z20_sm3{file_suffix}.png"
        z_fig.savefig(z20_plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {z20_plot_path}")
    plt.close(z_fig)


if __name__ == "__main__":
    main()
