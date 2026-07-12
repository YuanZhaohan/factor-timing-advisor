from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from io_utils import read_run_table, write_table
from timing_config import CODE_COL, DATE_COL, NAME_COL, TRADING_DAYS


DEFAULT_RUN_DIR = Path("skills/factor-timing-advisor/workspace/runs/default")
OUTPUT_SUBDIR = "composite_timing_strategies"
PRIMARY_STRATEGY_ID = "two_speed_category_strong0p25"
CHALLENGER_STRATEGY_ID = "entry_frequency_inverse_sqrt_weekly_binary"
FREQUENCY_TRAIN_END = pd.Timestamp("2020-12-31")
STRONG_EVENT_THRESHOLD = 0.25
DEFAULT_COST_BPS = 5.0
SCORE_EPSILON = 1e-12


def _write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        clean = table.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
        written[name] = write_table(clean, output_dir / f"{name}.csv")
        clean.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return written


def _load_inputs(
    run_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    dict[str, str],
    str,
    str,
]:
    positions = read_run_table(
        run_dir,
        [
            "results/selected_single_factor_rules/selected_rule_daily_positions.parquet",
            "results/selected_single_factor_rules/selected_rule_daily_positions.csv",
        ],
    ).copy()
    specs = read_run_table(
        run_dir,
        [
            "results/selected_single_factor_rules/selected_rule_specs.parquet",
            "results/selected_single_factor_rules/selected_rule_specs.csv",
        ],
    ).copy()

    required_positions = {CODE_COL, NAME_COL, DATE_COL, "rule_id", "position", "benchmark_return"}
    required_specs = {"rule_id", "factor", "category"}
    if missing := required_positions.difference(positions.columns):
        raise KeyError(f"selected-rule positions missing columns: {sorted(missing)}")
    if missing := required_specs.difference(specs.columns):
        raise KeyError(f"selected-rule specs missing columns: {sorted(missing)}")

    positions[DATE_COL] = pd.to_datetime(positions[DATE_COL])
    if positions.duplicated([DATE_COL, "rule_id"]).any():
        raise ValueError("duplicate date/rule_id rows in selected-rule positions")
    if not set(pd.to_numeric(positions["position"], errors="coerce").dropna().unique()).issubset({0.0, 1.0}):
        raise ValueError("selected-rule positions must be binary 0/1")
    instruments = positions[[CODE_COL, NAME_COL]].drop_duplicates()
    if len(instruments) != 1:
        raise ValueError("composite timing currently expects exactly one benchmark instrument")

    position_wide = (
        positions.pivot(index=DATE_COL, columns="rule_id", values="position")
        .sort_index()
        .astype(float)
    )
    if position_wide.isna().any().any():
        raise ValueError("selected-rule daily positions are not aligned across rules")
    benchmark_count = positions.groupby(DATE_COL)["benchmark_return"].nunique(dropna=False)
    if benchmark_count.gt(1).any():
        raise ValueError("benchmark returns disagree across rules on the same date")
    benchmark_return = (
        positions.groupby(DATE_COL, sort=True)["benchmark_return"]
        .first()
        .reindex(position_wide.index)
        .fillna(0.0)
        .astype(float)
    )

    specs = specs.drop_duplicates("rule_id").copy()
    active_rules = list(position_wide.columns)
    if missing := set(active_rules).difference(specs["rule_id"]):
        raise ValueError(f"active rules missing from selected_rule_specs: {sorted(missing)}")
    category_by_rule = (
        specs.set_index("rule_id")["category"].reindex(active_rules).astype(str).to_dict()
    )
    scores = position_wide.mul(2.0).sub(1.0)
    instrument_row = instruments.iloc[0]
    return (
        position_wide,
        scores,
        benchmark_return,
        specs,
        category_by_rule,
        str(instrument_row[CODE_COL]),
        str(instrument_row[NAME_COL]),
    )


def _category_equal_weights(scores: pd.DataFrame, category_by_rule: dict[str, str]) -> pd.Series:
    categories = pd.Series(category_by_rule).reindex(scores.columns)
    if categories.isna().any():
        raise ValueError("category mapping is incomplete")
    category_counts = categories.value_counts()
    category_budget = 1.0 / len(category_counts)
    weights = pd.Series(
        {rule: category_budget / category_counts[categories[rule]] for rule in scores.columns},
        dtype=float,
    ).reindex(scores.columns)
    return weights.div(weights.sum())


def _entry_frequency_statistics(
    positions: pd.DataFrame,
    train_end: pd.Timestamp = FREQUENCY_TRAIN_END,
) -> pd.DataFrame:
    entry_events = positions.diff().eq(1.0)
    entry_events.iloc[0] = positions.iloc[0].eq(1.0)
    train_positions = positions.loc[positions.index <= train_end]
    train_entries = entry_events.loc[entry_events.index <= train_end]
    if len(train_positions) < TRADING_DAYS:
        raise ValueError(
            f"frequency training window through {train_end.date()} has fewer than {TRADING_DAYS} rows"
        )
    train_years = max((train_positions.index[-1] - train_positions.index[0]).days / 365.25, 1.0 / 365.25)
    full_years = max((positions.index[-1] - positions.index[0]).days / 365.25, 1.0 / 365.25)
    entry_count_train = train_entries.sum().astype(int)
    entry_count_all = entry_events.sum().astype(int)
    if entry_count_train.le(0).any():
        missing = entry_count_train[entry_count_train.le(0)].index.tolist()
        raise ValueError(f"rules have no entry event in frequency training window: {missing}")

    stats = pd.DataFrame(
        {
            "entry_count_train": entry_count_train,
            "entries_per_year_train": entry_count_train.div(train_years),
            "holding_ratio_train": train_positions.mean(),
            "mean_holding_days_train": train_positions.sum().div(entry_count_train),
            "entry_count_all": entry_count_all,
            "entries_per_year_all": entry_count_all.div(full_years),
            "holding_ratio_all": positions.mean(),
            "mean_holding_days_all": positions.sum().div(entry_count_all.replace(0, np.nan)),
        }
    )
    stats.index.name = "rule_id"
    return stats.reset_index()


def _inverse_sqrt_frequency_weights(
    scores: pd.DataFrame,
    frequency_stats: pd.DataFrame,
) -> pd.Series:
    rates = frequency_stats.set_index("rule_id")["entries_per_year_train"].reindex(scores.columns)
    if rates.isna().any() or rates.le(0.0).any():
        raise ValueError("entry frequencies must be positive and complete")
    weights = rates.pow(-0.5)
    return weights.div(weights.sum())


def _weekly_decision_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    date_series = pd.Series(index, index=index)
    return pd.DatetimeIndex(date_series.groupby(index.to_period("W-FRI")).max().to_numpy())


def _weighted_score(scores: pd.DataFrame, weights: pd.Series) -> pd.Series:
    result = scores.mul(weights.reindex(scores.columns), axis=1).sum(axis=1).round(12)
    return result.mask(result.abs().lt(SCORE_EPSILON), 0.0)


def _applied_daily_score(raw_score: pd.Series) -> pd.Series:
    return raw_score.shift(1).fillna(0.0).rename("composite_score")


def _applied_weekly_score(raw_score: pd.Series, weekly_dates: pd.DatetimeIndex) -> pd.Series:
    decision = raw_score.loc[weekly_dates]
    held = decision.reindex(raw_score.index).ffill()
    return held.shift(1).fillna(0.0).rename("weekly_score")


def _weekly_application_dates(
    index: pd.DatetimeIndex,
    weekly_dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    decision_locs = index.get_indexer(weekly_dates)
    application_locs = decision_locs + 1
    application_locs = application_locs[application_locs < len(index)]
    return index[application_locs]


def _two_speed_exposure(
    daily_score: pd.Series,
    weekly_score: pd.Series,
    weekly_dates: pd.DatetimeIndex,
    strong_threshold: float,
) -> pd.Series:
    if strong_threshold <= 0.0:
        raise ValueError("strong event threshold must be positive")
    if not daily_score.index.equals(weekly_score.index):
        raise ValueError("daily and weekly score indexes must align")
    application_dates = set(_weekly_application_dates(daily_score.index, weekly_dates))
    daily_values = daily_score.to_numpy(dtype=float)
    weekly_values = weekly_score.to_numpy(dtype=float)
    exposure = np.zeros(len(daily_score), dtype=float)
    state = 0.0
    for location, date in enumerate(daily_score.index):
        if date in application_dates:
            state = float(weekly_values[location] > 0.0)
        if daily_values[location] >= strong_threshold:
            state = 1.0
        elif daily_values[location] <= -strong_threshold:
            state = 0.0
        exposure[location] = state
    return pd.Series(exposure, index=daily_score.index, name="exposure")


def _strategy_daily(
    strategy_id: str,
    strategy_role: str,
    score: pd.Series,
    weekly_score: pd.Series,
    exposure: pd.Series,
    benchmark_return: pd.Series,
    instrument_code: str,
    instrument_name: str,
    cost_bps: float,
) -> pd.DataFrame:
    exposure = exposure.reindex(score.index).astype(float).clip(0.0, 1.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    gross_return = benchmark_return.mul(exposure)
    cost = turnover.mul(cost_bps / 10000.0)
    net_return = gross_return.sub(cost)
    return pd.DataFrame(
        {
            DATE_COL: score.index,
            CODE_COL: instrument_code,
            NAME_COL: instrument_name,
            "strategy_id": strategy_id,
            "strategy_role": strategy_role,
            "composite_score": score.to_numpy(dtype=float),
            "weekly_anchor_score": weekly_score.reindex(score.index).to_numpy(dtype=float),
            "exposure": exposure.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "benchmark_return": benchmark_return.to_numpy(dtype=float),
            "gross_return": gross_return.to_numpy(dtype=float),
            "cost": cost.to_numpy(dtype=float),
            "net_return": net_return.to_numpy(dtype=float),
            "benchmark_equity": benchmark_return.add(1.0).cumprod().to_numpy(dtype=float),
            "net_equity": net_return.add(1.0).cumprod().to_numpy(dtype=float),
        }
    )


def _performance_summary(daily: pd.DataFrame) -> dict[str, Any]:
    returns = daily["net_return"].astype(float)
    equity = returns.add(1.0).cumprod()
    years = len(daily) / TRADING_DAYS
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(TRADING_DAYS)) if returns.std(ddof=0) > 0 else np.nan
    drawdown = equity.div(equity.cummax()).sub(1.0)
    max_drawdown = float(drawdown.min())
    exposure = daily["exposure"].astype(float)
    binary_state = exposure.gt(0.5)
    entries = binary_state & ~binary_state.shift(1, fill_value=False)
    exits = ~binary_state & binary_state.shift(1, fill_value=False)
    return {
        "strategy_id": str(daily["strategy_id"].iloc[0]),
        "strategy_role": str(daily["strategy_role"].iloc[0]),
        "start_date": pd.Timestamp(daily[DATE_COL].iloc[0]),
        "end_date": pd.Timestamp(daily[DATE_COL].iloc[-1]),
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "annual_turnover": float(daily["turnover"].sum() / years) if years > 0 else np.nan,
        "mean_exposure": float(exposure.mean()),
        "entry_count": int(entries.sum()),
        "exit_count": int(exits.sum()),
        "mean_holding_days": float(binary_state.sum() / max(int(entries.sum()), 1)),
        "final_equity": float(equity.iloc[-1]),
    }


def _trade_events(
    daily: pd.DataFrame,
    weekly_application_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    indexed = daily.set_index(DATE_COL)
    change = indexed["exposure"].diff().fillna(indexed["exposure"])
    events = indexed.loc[
        change.ne(0.0),
        [CODE_COL, NAME_COL, "strategy_id", "strategy_role", "composite_score", "weekly_anchor_score", "exposure"],
    ].copy()
    events.index.name = "execution_date"
    events = events.reset_index()
    previous_date = pd.Series(indexed.index, index=indexed.index).shift(1)
    events["signal_date"] = events["execution_date"].map(previous_date)
    events["trade_side"] = np.where(events["exposure"].gt(0.5), "entry", "exit")
    events["execution_weekday"] = events["execution_date"].dt.day_name()
    events["is_weekly_anchor_date"] = events["execution_date"].isin(weekly_application_dates)
    events["trigger_source"] = np.where(
        events["is_weekly_anchor_date"], "weekly_anchor", "intraweek_strong_event"
    )
    return events


def _latest_status(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    strong_threshold: float | None,
) -> dict[str, Any]:
    latest = daily.iloc[-1]
    strategy_id = str(latest["strategy_id"])
    strategy_trades = trades[trades["strategy_id"].eq(strategy_id)]
    entries = strategy_trades[strategy_trades["trade_side"].eq("entry")]
    exits = strategy_trades[strategy_trades["trade_side"].eq("exit")]
    if strategy_id == PRIMARY_STRATEGY_ID and abs(float(latest["composite_score"])) < float(strong_threshold):
        state_reason = "weekly anchor retained; daily score remains inside the strong-event hold band"
    else:
        state_reason = "latest weekly decision or composite threshold"
    return {
        DATE_COL: pd.Timestamp(latest[DATE_COL]),
        CODE_COL: str(latest[CODE_COL]),
        NAME_COL: str(latest[NAME_COL]),
        "strategy_id": strategy_id,
        "strategy_role": str(latest["strategy_role"]),
        "composite_score": float(latest["composite_score"]),
        "weekly_anchor_score": float(latest["weekly_anchor_score"]),
        "strong_event_threshold": strong_threshold,
        "exposure": float(latest["exposure"]),
        "state": "long" if float(latest["exposure"]) > 0.5 else "flat",
        "state_reason": state_reason,
        "latest_entry_date": pd.Timestamp(entries["execution_date"].max()) if not entries.empty else pd.NaT,
        "latest_exit_date": pd.Timestamp(exits["execution_date"].max()) if not exits.empty else pd.NaT,
    }


def _signal_markdown(latest_status: pd.DataFrame) -> str:
    lines = [
        "# 复合择时最新信号",
        "",
        "| 角色 | 策略 | 日期 | 日度/复合分数 | 周频锚分数 | 仓位 | 状态 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in latest_status.itertuples(index=False):
        lines.append(
            f"| {row.strategy_role} | `{row.strategy_id}` | {pd.Timestamp(getattr(row, DATE_COL)).date()} | "
            f"{row.composite_score:.4f} | {row.weekly_anchor_score:.4f} | {row.exposure:.0%} | {row.state} |"
        )
    lines.extend(
        [
            "",
            "- 主策略：四类因子各占 25%，周频锚定；日度分数达到 ±0.25 时在下一交易日提前调整。",
            "- 挑战者：规则权重与训练期年均开仓频率平方根倒数成正比，周频判断仓位。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_composite_timing_strategies(
    input_dir: str | Path = DEFAULT_RUN_DIR,
    output_dir: str | Path | None = None,
    frequency_train_end: str | pd.Timestamp = FREQUENCY_TRAIN_END,
    strong_event_threshold: float = STRONG_EVENT_THRESHOLD,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict[str, int]:
    run_dir = Path(input_dir)
    out_root = Path(output_dir) if output_dir is not None else run_dir
    output_path = out_root / "results" / OUTPUT_SUBDIR
    (
        positions,
        scores,
        benchmark_return,
        specs,
        category_by_rule,
        instrument_code,
        instrument_name,
    ) = _load_inputs(run_dir)
    frequency_stats = _entry_frequency_statistics(positions, pd.Timestamp(frequency_train_end))
    primary_weights = _category_equal_weights(scores, category_by_rule)
    challenger_weights = _inverse_sqrt_frequency_weights(scores, frequency_stats)

    weekly_dates = _weekly_decision_dates(scores.index)
    weekly_applications = set(_weekly_application_dates(scores.index, weekly_dates))
    primary_raw = _weighted_score(scores, primary_weights)
    primary_daily_score = _applied_daily_score(primary_raw)
    primary_weekly_score = _applied_weekly_score(primary_raw, weekly_dates)
    primary_exposure = _two_speed_exposure(
        primary_daily_score,
        primary_weekly_score,
        weekly_dates,
        strong_event_threshold,
    )
    challenger_raw = _weighted_score(scores, challenger_weights)
    challenger_weekly_score = _applied_weekly_score(challenger_raw, weekly_dates)
    challenger_exposure = challenger_weekly_score.gt(0.0).astype(float).rename("exposure")

    primary_daily = _strategy_daily(
        PRIMARY_STRATEGY_ID,
        "primary",
        primary_daily_score,
        primary_weekly_score,
        primary_exposure,
        benchmark_return,
        instrument_code,
        instrument_name,
        cost_bps,
    )
    challenger_daily = _strategy_daily(
        CHALLENGER_STRATEGY_ID,
        "challenger",
        challenger_weekly_score,
        challenger_weekly_score,
        challenger_exposure,
        benchmark_return,
        instrument_code,
        instrument_name,
        cost_bps,
    )
    daily = pd.concat([primary_daily, challenger_daily], ignore_index=True)
    summary = pd.DataFrame([_performance_summary(primary_daily), _performance_summary(challenger_daily)])
    trades = pd.concat(
        [
            _trade_events(primary_daily, weekly_applications),
            _trade_events(challenger_daily, weekly_applications),
        ],
        ignore_index=True,
    )
    latest_status = pd.DataFrame(
        [
            _latest_status(primary_daily, trades, strong_event_threshold),
            _latest_status(challenger_daily, trades, None),
        ]
    )

    rule_info = specs[["rule_id", "factor", "category"]].drop_duplicates("rule_id")
    rule_weights = rule_info.merge(frequency_stats, on="rule_id", how="right", validate="one_to_one")
    rule_weights["primary_weight"] = rule_weights["rule_id"].map(primary_weights)
    rule_weights["challenger_weight"] = rule_weights["rule_id"].map(challenger_weights)
    strategy_specs = pd.DataFrame(
        [
            {
                "strategy_id": PRIMARY_STRATEGY_ID,
                "strategy_role": "primary",
                "weighting": "four_categories_equal_25pct_then_equal_within_category",
                "rebalance": "weekly_anchor_plus_daily_strong_event",
                "open_rule": f"weekly_score>0 or daily_score>={strong_event_threshold:g}",
                "close_rule": f"weekly_score<=0 or daily_score<=-{strong_event_threshold:g}",
                "execution": "next_trading_day",
            },
            {
                "strategy_id": CHALLENGER_STRATEGY_ID,
                "strategy_role": "challenger",
                "weighting": "normalized_inverse_sqrt_training_entry_frequency",
                "rebalance": "weekly",
                "open_rule": "weekly_score>0",
                "close_rule": "weekly_score<=0",
                "execution": "next_trading_day",
            },
        ]
    )

    _write_outputs(
        output_path,
        {
            "composite_strategy_specs": strategy_specs,
            "composite_rule_weights": rule_weights,
            "composite_strategy_daily": daily,
            "composite_strategy_summary": summary,
            "composite_strategy_trades": trades,
            "composite_strategy_latest_status": latest_status,
        },
    )
    latest_json = {
        "date": str(pd.Timestamp(latest_status[DATE_COL].max()).date()),
        "primary": latest_status[latest_status["strategy_role"].eq("primary")].iloc[0].to_dict(),
        "challenger": latest_status[latest_status["strategy_role"].eq("challenger")].iloc[0].to_dict(),
    }
    for item in (latest_json["primary"], latest_json["challenger"]):
        for key, value in list(item.items()):
            if isinstance(value, pd.Timestamp):
                item[key] = str(value.date())
            elif pd.isna(value):
                item[key] = None
    (output_path / "current_composite_signal.json").write_text(
        json.dumps(latest_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "current_composite_signal.md").write_text(
        _signal_markdown(latest_status), encoding="utf-8"
    )
    return {
        "composite_strategy_count": len(strategy_specs),
        "composite_rule_weight_rows": len(rule_weights),
        "composite_daily_rows": len(daily),
        "composite_summary_rows": len(summary),
        "composite_trade_rows": len(trades),
        "composite_latest_status_rows": len(latest_status),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two retained composite timing strategies.")
    parser.add_argument("--input-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--frequency-train-end", default=str(FREQUENCY_TRAIN_END.date()))
    parser.add_argument("--strong-event-threshold", type=float, default=STRONG_EVENT_THRESHOLD)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = run_composite_timing_strategies(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frequency_train_end=args.frequency_train_end,
        strong_event_threshold=args.strong_event_threshold,
        cost_bps=args.cost_bps,
    )
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
