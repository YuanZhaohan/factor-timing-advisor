from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtest import combine_open_close_rules, run_event_study, run_rule_pair_backtest
from data_cleaning import get_factor_columns, load_data
from io_utils import read_table, table_candidates, write_table
from plotting import factor_plot_roots, plot_dual_axis, plot_factor_signal_chart, plot_factor_signal_charts
from reporting import (
    build_advisor_summary,
    build_signal_point_summary,
    run_advisor_summary_report,
    run_current_status_report,
    run_signal_point_status_report,
    score_signal_points_for_advisor,
    write_advisor_summary_report,
    write_current_signal_report,
    write_signal_point_report,
)
from role_strategy import (
    build_daily_signal_state_score,
    build_factor_signal_utility,
    build_signal_edge_decay,
    strategy_vote_summary,
)
from signal_generation import (
    _build_factor_event_cache_from_signal_table,
    build_event_conditions,
    detect_events,
    generate_signal_table,
    process_signal,
    save_signal_table,
)
from timing_config import *  # noqa: F403 - keep timing_tool.py constants available
from timing_config import _format_pct, _format_sigma, _split_factor_frequency


def run_all(
    csv_path: str | Path = "宽基得分.csv",
    output_dir: str | Path = "results",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_equity_curves: int | None = 200,
    report_top_n: int = 200,
    taxonomy_path: str | Path | None = DEFAULT_TAXONOMY_PATH,
    min_raw_open_events_per_quarter: float | None = 2.0,
    min_excess_annual_return: float | None = 0.05,
) -> dict[str, object]:
    df = load_data(csv_path)
    signal_table = save_signal_table(df, output_dir=output_dir)
    event_summary, trade_summary = run_event_study(
        df,
        output_dir=output_dir,
        horizons=horizons,
        signal_table=signal_table,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
    )
    rule_summary, equity_curves = run_rule_pair_backtest(
        df,
        output_dir=output_dir,
        signal_table=signal_table,
        max_equity_curves=max_equity_curves,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
        min_excess_annual_return=min_excess_annual_return,
    )
    factor_signal_utility = build_factor_signal_utility(event_summary, output_dir=output_dir)
    signal_edge_decay = build_signal_edge_decay(df, signal_table, output_dir=output_dir)
    daily_signal_state_score = build_daily_signal_state_score(
        df,
        signal_table,
        signal_edge_decay,
        output_dir=output_dir,
    )
    role_strategy_summary = strategy_vote_summary(daily_signal_state_score, factor_signal_utility)
    signal_status, signal_summary, current_report = run_signal_point_status_report(
        df,
        output_dir=output_dir,
        signal_table=signal_table,
        taxonomy_path=taxonomy_path,
        report_top_n=report_top_n,
    )
    advisor_summary, advisor_scored_points, advisor_report = run_advisor_summary_report(
        status_df=signal_status,
        summary_df=signal_summary,
        rule_summary=rule_summary,
        role_strategy_summary=role_strategy_summary,
        output_dir=output_dir,
        top_n=report_top_n,
    )
    return {
        "signals": signal_table,
        "event_forward_returns": event_summary,
        "open_close_trades": trade_summary,
        "rule_pair_summary": rule_summary,
        "equity_curves": equity_curves,
        "factor_signal_utility": factor_signal_utility,
        "signal_edge_decay": signal_edge_decay,
        "daily_signal_state_score": daily_signal_state_score,
        "signal_points_state": signal_status,
        "signal_points_summary": signal_summary,
        "current_signal_report": current_report,
        "advisor_summary": advisor_summary,
        "advisor_scored_points": advisor_scored_points,
        "advisor_report": advisor_report,
    }


def _parse_horizons(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _read_existing_csv(path: Path) -> pd.DataFrame:
    existing = next((candidate for candidate in table_candidates(path) if candidate.exists()), None)
    if existing is None:
        return pd.DataFrame()
    return read_table(existing)


def _safe_path_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unknown"


def _instrument_output_dir(df: pd.DataFrame, instrument: str, output_root: str | Path) -> Path:
    group = df[df[CODE_COL].astype(str).eq(str(instrument))]
    if group.empty:
        raise ValueError(f"Instrument not found: {instrument}")
    name = group[NAME_COL].dropna().astype(str).iloc[-1] if NAME_COL in group and not group.empty else ""
    dirname = f"{_safe_path_part(instrument)}_{_safe_path_part(name)}" if name else _safe_path_part(instrument)
    return Path(output_root) / dirname


def _filter_instrument(df: pd.DataFrame, instrument: str | None) -> pd.DataFrame:
    if not instrument:
        return df
    filtered = df[df[CODE_COL].astype(str).eq(str(instrument))].copy()
    if filtered.empty:
        available = ", ".join(map(str, df[CODE_COL].dropna().astype(str).drop_duplicates().head(10)))
        raise ValueError(f"Instrument not found: {instrument}. Available examples: {available}")
    return filtered.reset_index(drop=True)


def _save_input_snapshot(df: pd.DataFrame, output_path: Path, enabled: bool = True) -> None:
    if not enabled:
        return
    output_path.mkdir(parents=True, exist_ok=True)
    write_table(df, output_path / "input_snapshot.csv")


def _plot_all_factors_for_df(
    df: pd.DataFrame,
    output_path: Path,
    plot_output_dir: str | Path | None,
    instrument: str | None,
    lookback_days: int | None,
    select_by_backtest: bool,
    show_base_factor_points: bool = True,
) -> pd.DataFrame:
    output_path.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(plot_output_dir) if plot_output_dir is not None else output_path / "plots"
    factors = factor_plot_roots(get_factor_columns(df))
    signal_path = output_path / "signals.csv"
    rule_summary_path = output_path / "rule_pair_summary.csv"
    signal_table = _read_existing_csv(signal_path)
    rule_summary = _read_existing_csv(rule_summary_path)
    if signal_table.empty:
        signal_table = generate_signal_table(df)

    results = plot_factor_signal_charts(
        df=df,
        factors=factors,
        instrument=instrument,
        signal_table=signal_table,
        signal_path=None,
        rule_summary=rule_summary,
        rule_summary_path=None,
        save_dir=plot_dir,
        show=False,
        backend="Agg",
        lookback_days=lookback_days,
        select_by_backtest=select_by_backtest,
        show_base_factor_points=show_base_factor_points,
    )
    manifest = pd.DataFrame(results)
    write_table(manifest, plot_dir / "plot_manifest.csv")
    return manifest


def _run_pipeline_for_df(
    df: pd.DataFrame,
    output_path: Path,
    horizons: Iterable[int],
    max_equity_curves: int | None,
    taxonomy_path: str | Path | None,
    status_top_n: int,
    skip_signal_generation: bool,
    skip_event_study: bool,
    skip_rule_backtest: bool,
    skip_signal_point_status: bool,
    skip_advisor_summary: bool,
    run_rule_combo_status: bool,
    save_input_snapshot: bool,
    min_raw_open_events_per_quarter: float | None,
    min_excess_annual_return: float | None,
) -> dict[str, object]:
    output_path.mkdir(parents=True, exist_ok=True)
    _save_input_snapshot(df, output_path, enabled=save_input_snapshot)

    signal_table = None
    needs_signal_table = (
        not skip_event_study
        or not skip_rule_backtest
        or not skip_signal_point_status
        or run_rule_combo_status
    )

    if not skip_signal_generation:
        signal_table = save_signal_table(df, output_dir=output_path)
    elif needs_signal_table:
        signal_table = _read_existing_csv(output_path / "signals.csv")
        if signal_table.empty:
            signal_table = generate_signal_table(df)

    event_summary = pd.DataFrame()
    trade_summary = pd.DataFrame()
    rule_summary = pd.DataFrame()
    equity_curves = pd.DataFrame()
    factor_signal_utility = pd.DataFrame()
    signal_edge_decay = pd.DataFrame()
    daily_signal_state_score = pd.DataFrame()
    if not skip_event_study:
        event_summary, trade_summary = run_event_study(
            df,
            output_dir=output_path,
            horizons=horizons,
            signal_table=signal_table,
            min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
        )
    else:
        event_summary = _read_existing_csv(output_path / "event_forward_returns.csv")
        trade_summary = _read_existing_csv(output_path / "open_close_trades.csv")
    if not skip_rule_backtest:
        rule_summary, equity_curves = run_rule_pair_backtest(
            df,
            output_dir=output_path,
            signal_table=signal_table,
            max_equity_curves=max_equity_curves,
            min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
            min_excess_annual_return=min_excess_annual_return,
        )
    else:
        rule_summary = _read_existing_csv(output_path / "rule_pair_summary.csv")

    if signal_table is not None and not signal_table.empty and not event_summary.empty:
        factor_signal_utility = build_factor_signal_utility(event_summary, output_dir=output_path)
        signal_edge_decay = build_signal_edge_decay(df, signal_table, output_dir=output_path)
        daily_signal_state_score = build_daily_signal_state_score(
            df,
            signal_table,
            signal_edge_decay,
            output_dir=output_path,
        )
    else:
        factor_signal_utility = _read_existing_csv(output_path / "factor_signal_utility.csv")
        signal_edge_decay = _read_existing_csv(output_path / "signal_edge_decay.csv")
        daily_signal_state_score = _read_existing_csv(output_path / "daily_signal_state_score.csv")

    role_strategy_summary = strategy_vote_summary(daily_signal_state_score, factor_signal_utility)

    signal_status = pd.DataFrame()
    signal_summary = pd.DataFrame()
    if not skip_signal_point_status:
        signal_status, signal_summary, _ = run_signal_point_status_report(
            df,
            output_dir=output_path,
            signal_table=signal_table,
            taxonomy_path=taxonomy_path,
            report_top_n=status_top_n,
        )
        if not skip_advisor_summary:
            run_advisor_summary_report(
                status_df=signal_status,
                summary_df=signal_summary,
                rule_summary=rule_summary,
                role_strategy_summary=role_strategy_summary,
                output_dir=output_path,
                top_n=status_top_n,
            )
    if run_rule_combo_status:
        run_current_status_report(
            df,
            output_dir=output_path,
            signal_table=signal_table,
            report_top_n=status_top_n,
        )

    return {
        "output_dir": str(output_path),
        "signals": signal_table if signal_table is not None else pd.DataFrame(),
        "event_forward_returns": event_summary,
        "open_close_trades": trade_summary,
        "rule_pair_summary": rule_summary,
        "equity_curves": equity_curves,
        "factor_signal_utility": factor_signal_utility,
        "signal_edge_decay": signal_edge_decay,
        "daily_signal_state_score": daily_signal_state_score,
        "signal_points_state": signal_status,
        "signal_points_summary": signal_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-driven timing rule study.")
    parser.add_argument("--csv", default="宽基得分.csv", help="Input timing score CSV path.")
    parser.add_argument("--output", default="results", help="Output directory.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for automatic per-instrument output folders.",
    )
    parser.add_argument("--instrument", default=None, help="Run only one instrument code, for example 000985.CSI.")
    parser.add_argument(
        "--by-instrument",
        action="store_true",
        help="Run each instrument into output-root/{code}_{name}/.",
    )
    parser.add_argument(
        "--no-input-snapshot",
        action="store_true",
        help="Do not write input_snapshot.csv into the output directory.",
    )
    parser.add_argument("--horizons", default="1,5,10,20,60", help="Comma-separated forward return horizons.")
    parser.add_argument(
        "--max-equity-curves",
        type=int,
        default=200,
        help="Save equity curves for the top N rule pairs. Use -1 to save all pairs.",
    )
    parser.add_argument("--skip-signal-generation", action="store_true", help="Skip writing results/signals.csv.")
    parser.add_argument("--skip-event-study", action="store_true", help="Skip event forward return and trade-pair study.")
    parser.add_argument("--skip-rule-backtest", action="store_true", help="Skip rule-pair equity backtest.")
    parser.add_argument(
        "--skip-signal-point-status",
        action="store_true",
        help="Skip point-level current signal status report.",
    )
    parser.add_argument(
        "--skip-current-status",
        action="store_true",
        help="Deprecated alias for --skip-signal-point-status.",
    )
    parser.add_argument(
        "--run-rule-combo-status",
        action="store_true",
        help="Also write the legacy open-condition x close-condition current status report.",
    )
    parser.add_argument("--skip-advisor-summary", action="store_true", help="Skip deterministic advisor summary.")
    parser.add_argument(
        "--taxonomy",
        default=str(DEFAULT_TAXONOMY_PATH),
        help="Factor taxonomy markdown path. Use an empty string to disable category mapping.",
    )
    parser.add_argument(
        "--status-top-n",
        type=int,
        default=200,
        help="Rows to include in status report sections. Use -1 to include all rows.",
    )
    parser.add_argument(
        "--min-raw-open-events-per-quarter",
        type=float,
        default=2.0,
        help="For raw factors, keep only open rules with at least this many average open events per quarter. Use -1 to disable.",
    )
    parser.add_argument(
        "--min-excess-annual-return",
        type=float,
        default=0.05,
        help="Keep only rule-pair backtests with excess annual return above this value. Use -999 to disable.",
    )
    parser.add_argument(
        "--plot-all-factors",
        action="store_true",
        help="Only generate one factor chart per base factor, using existing signals/backtests when available.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        help="Output directory for factor charts. Default is {output}/plots or each instrument output/plots.",
    )
    parser.add_argument(
        "--plot-lookback-days",
        type=int,
        default=TRADING_DAYS * 3,
        help="Trading rows to show in factor charts. Use -1 for full history.",
    )
    parser.add_argument(
        "--plot-no-backtest-selection",
        action="store_true",
        help="Do not select display lines and holding regions by rule_pair_summary.csv.",
    )
    parser.add_argument(
        "--plot-hide-base-points",
        action="store_true",
        help="Hide raw factor daily dots in factor charts.",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    if args.by_instrument and args.instrument:
        raise ValueError("Use either --by-instrument or --instrument, not both.")

    output_path = Path(args.output)
    max_equity_curves = None if args.max_equity_curves < 0 else args.max_equity_curves
    horizons = _parse_horizons(args.horizons)
    skip_signal_point_status = args.skip_signal_point_status or args.skip_current_status
    taxonomy_path = args.taxonomy if args.taxonomy else None
    save_input_snapshot = not args.no_input_snapshot
    min_raw_open_events_per_quarter = (
        None if args.min_raw_open_events_per_quarter < 0 else args.min_raw_open_events_per_quarter
    )
    min_excess_annual_return = None if args.min_excess_annual_return <= -999 else args.min_excess_annual_return
    plot_lookback_days = None if args.plot_lookback_days < 0 else args.plot_lookback_days
    plot_select_by_backtest = not args.plot_no_backtest_selection

    if args.by_instrument:
        output_root = Path(args.output_root or args.output)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for code in df[CODE_COL].dropna().astype(str).drop_duplicates():
            instrument_df = _filter_instrument(df, code)
            instrument_output = _instrument_output_dir(df, code, output_root)
            if args.plot_all_factors:
                plot_output_dir = (
                    Path(args.plot_output_dir) / instrument_output.name
                    if args.plot_output_dir is not None
                    else None
                )
                plot_manifest = _plot_all_factors_for_df(
                    df=instrument_df,
                    output_path=instrument_output,
                    plot_output_dir=plot_output_dir,
                    instrument=code,
                    lookback_days=plot_lookback_days,
                    select_by_backtest=plot_select_by_backtest,
                    show_base_factor_points=not args.plot_hide_base_points,
                )
                name = instrument_df[NAME_COL].dropna().astype(str).iloc[-1] if NAME_COL in instrument_df else ""
                manifest_rows.append(
                    {
                        CODE_COL: code,
                        NAME_COL: name,
                        "output_dir": str(instrument_output),
                        "plot_count": int(len(plot_manifest)),
                    }
                )
                continue
            result = _run_pipeline_for_df(
                df=instrument_df,
                output_path=instrument_output,
                horizons=horizons,
                max_equity_curves=max_equity_curves,
                taxonomy_path=taxonomy_path,
                status_top_n=args.status_top_n,
                skip_signal_generation=args.skip_signal_generation,
                skip_event_study=args.skip_event_study,
                skip_rule_backtest=args.skip_rule_backtest,
                skip_signal_point_status=skip_signal_point_status,
                skip_advisor_summary=args.skip_advisor_summary,
                run_rule_combo_status=args.run_rule_combo_status,
                save_input_snapshot=save_input_snapshot,
                min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
                min_excess_annual_return=min_excess_annual_return,
            )
            name = instrument_df[NAME_COL].dropna().astype(str).iloc[-1] if NAME_COL in instrument_df else ""
            manifest_rows.append(
                {
                    CODE_COL: code,
                    NAME_COL: name,
                    "output_dir": result["output_dir"],
                    "row_count": int(len(instrument_df)),
                    "start_date": instrument_df[DATE_COL].min(),
                    "end_date": instrument_df[DATE_COL].max(),
                }
            )
        write_table(pd.DataFrame(manifest_rows), output_root / "run_manifest.csv")
        return

    if args.instrument and args.output_root:
        output_path = _instrument_output_dir(df, args.instrument, args.output_root)
    elif args.instrument and args.output == "results":
        output_path = _instrument_output_dir(df, args.instrument, args.output)
    run_df = _filter_instrument(df, args.instrument)
    if args.plot_all_factors:
        plot_manifest = _plot_all_factors_for_df(
            df=run_df,
            output_path=output_path,
            plot_output_dir=args.plot_output_dir,
            instrument=args.instrument,
            lookback_days=plot_lookback_days,
            select_by_backtest=plot_select_by_backtest,
            show_base_factor_points=not args.plot_hide_base_points,
        )
        print(f"Saved {len(plot_manifest)} factor charts to {args.plot_output_dir or output_path / 'plots'}")
        return

    _run_pipeline_for_df(
        df=run_df,
        output_path=output_path,
        horizons=horizons,
        max_equity_curves=max_equity_curves,
        taxonomy_path=taxonomy_path,
        status_top_n=args.status_top_n,
        skip_signal_generation=args.skip_signal_generation,
        skip_event_study=args.skip_event_study,
        skip_rule_backtest=args.skip_rule_backtest,
        skip_signal_point_status=skip_signal_point_status,
        skip_advisor_summary=args.skip_advisor_summary,
        run_rule_combo_status=args.run_rule_combo_status,
        save_input_snapshot=save_input_snapshot,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
        min_excess_annual_return=min_excess_annual_return,
    )


if __name__ == "__main__":
    main()
