from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from backtest import (
    run_best_rule_pair_backtest,
    run_event_study,
    run_rule_pair_backtest,
    run_rule_pair_backtest_by_year_end,
)
from data_cleaning import get_factor_columns, load_data
from generate_timing_report import build_view_data, render_html
from plotting import factor_plot_roots, plot_best_rule_pair_charts, plot_factor_signal_charts
from reporting import run_advisor_summary_report, run_signal_point_status_report
from role_strategy import (
    build_factor_signal_utility,
    run_monthly_refresh_daily_score,
    strategy_vote_summary,
    update_monthly_refresh_daily_score_incremental,
)
from signal_generation import save_signal_table
from baseline_score_strategy import backtest_z_rules, build_all_z_rules, plot_best_rule
from io_utils import read_run_table, read_table, resolve_table_file, write_table
from timing_config import CODE_COL


DEFAULT_HORIZONS = (1, 3, 5, 10, 15, 20, 60)


def _run_dirs(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "root": base,
        "data": base / "data",
        "results": base / "results",
        "plots": base / "plots",
        "signals": base / "results" / "signals",
        "events": base / "results" / "events",
        "rule_pair": base / "results" / "rule_pair",
        "score": base / "results" / "score",
        "strategy": base / "results" / "strategy",
        "report": base / "results" / "report",
        "strategy_plots": base / "plots" / "strategy",
        "factor_plots": base / "plots" / "factor",
        "rule_pair_best_plots": base / "plots" / "rule_pair_best",
    }


def _resolve_run_file(root: str | Path, candidates: list[str]) -> Path:
    return resolve_table_file(root, candidates)


def _parse_year_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))


def run_upstream_pipeline(
    csv_path: str | Path,
    output_dir: str | Path,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    warmup_years: int = 3,
    min_history_years: int = 2,
    min_raw_open_events_per_quarter: float | None = 2.0,
    min_excess_annual_return: float | None = 0.05,
    max_equity_curves: int | None = 200,
    parallel_n_jobs: int = 1,
    save_intermediates: bool = True,
    year_end_target_years: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full upstream research pipeline.

    Outputs:
    - input_snapshot.csv
    - signals.csv
    - event_forward_returns.csv
    - open_close_trades.csv
    - rule_pair_summary.csv
    - equity_curves.csv
    - rule_pair_summary_by_year_end.csv
    - factor_signal_utility.csv
    - monthly_refresh_daily_score.csv
    """
    dirs = _run_dirs(output_dir)
    for key in ["root", "data", "results", "signals", "events", "rule_pair", "score", "plots"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    write_table(df, dirs["data"] / "input_snapshot.csv")

    signal_table = save_signal_table(df, output_dir=dirs["signals"])
    event_summary, trade_summary = run_event_study(
        df,
        output_dir=dirs["events"],
        horizons=horizons,
        signal_table=signal_table,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
    )
    rule_summary, equity_curves = run_rule_pair_backtest(
        df,
        output_dir=dirs["rule_pair"],
        signal_table=signal_table,
        max_equity_curves=max_equity_curves,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
        min_excess_annual_return=min_excess_annual_return,
    )
    if year_end_target_years is None:
        max_year = int(pd.to_datetime(df["日期"]).dt.year.max())
        year_end_target_years = (max_year - 1, max_year)
    rule_summary_by_year_end = run_rule_pair_backtest_by_year_end(
        df,
        output_dir=dirs["rule_pair"],
        signal_table=signal_table,
        min_history_years=min_history_years,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
        min_excess_annual_return=min_excess_annual_return,
        target_years=year_end_target_years,
    )
    build_factor_signal_utility(event_summary, output_dir=dirs["score"])
    daily_score = run_monthly_refresh_daily_score(
        df=df,
        signal_table=signal_table,
        min_history_years=min_history_years,
        warmup_years=warmup_years,
        parallel_n_jobs=parallel_n_jobs,
        save_intermediates=save_intermediates,
        output_dir=dirs["score"],
    )
    return signal_table, event_summary, trade_summary, rule_summary, rule_summary_by_year_end, equity_curves, daily_score


def run_score_update_pipeline(
    csv_path: str | Path,
    output_dir: str | Path,
    min_history_years: int = 2,
    warmup_years: int = 3,
    save_intermediates: bool = True,
) -> pd.DataFrame:
    """Incrementally update only the trailing monthly_refresh daily score."""
    dirs = _run_dirs(output_dir)
    for key in ["root", "data", "results", "signals", "score"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    write_table(df, dirs["data"] / "input_snapshot.csv")
    signal_table = save_signal_table(df, output_dir=dirs["signals"])
    daily_score = update_monthly_refresh_daily_score_incremental(
        df=df,
        signal_table=signal_table,
        output_dir=dirs["score"],
        min_history_years=min_history_years,
        warmup_years=warmup_years,
        save_intermediates=save_intermediates,
    )
    return daily_score


def run_daily_refresh_pipeline(
    csv_path: str | Path,
    output_dir: str | Path,
    min_history_years: int = 2,
    warmup_years: int = 3,
    min_raw_open_events_per_quarter: float | None = 2.0,
    min_excess_annual_return: float | None = 0.05,
    max_equity_curves: int | None = 200,
    score_suffix: str = "default",
    report_top_n: int = 30,
    lookback_days: int = 252 * 3,
    save_intermediates: bool = True,
) -> dict[str, int]:
    """Daily refresh path:
    - refresh input snapshot
    - refresh signals
    - refresh event-study outputs
    - incrementally refresh daily score
    - refresh only the best rule-pair for each base factor
    - refresh baseline strategy outputs
    - refresh advisor report outputs
    - refresh plots
    """
    dirs = _run_dirs(output_dir)
    for key in ["root", "data", "results", "signals", "events", "rule_pair", "score", "plots"]:
        dirs[key].mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    write_table(df, dirs["data"] / "input_snapshot.csv")
    signal_table = save_signal_table(df, output_dir=dirs["signals"])
    event_summary, trade_summary = run_event_study(
        df,
        output_dir=dirs["events"],
        signal_table=signal_table,
        min_raw_open_events_per_quarter=min_raw_open_events_per_quarter,
        include_trade_summary=False,
    )

    daily_score = update_monthly_refresh_daily_score_incremental(
        df=df,
        signal_table=signal_table,
        output_dir=dirs["score"],
        min_history_years=min_history_years,
        warmup_years=warmup_years,
        save_intermediates=save_intermediates,
    )

    reference_rule_summary_path = _resolve_run_file(
        output_dir,
        [
            "results/rule_pair/rule_pair_summary.csv",
            "rule_pair_summary.csv",
        ],
    )
    reference_rule_summary = read_table(reference_rule_summary_path)
    rule_summary, equity_curves = run_best_rule_pair_backtest(
        df=df,
        reference_rule_summary=reference_rule_summary,
        output_dir=dirs["rule_pair"],
        signal_table=signal_table,
    )
    summary, best_equity = run_baseline_strategy(
        input_dir=output_dir,
        output_dir=output_dir,
        score_suffix=score_suffix,
    )
    advisor_summary, scored, report = run_reporting_pipeline(
        input_dir=output_dir,
        output_dir=None,
        report_top_n=report_top_n,
    )
    manifest = run_factor_plots(
        input_dir=output_dir,
        output_dir=None,
        lookback_days=lookback_days,
        select_by_backtest=True,
        show_base_factor_points=True,
    )
    return {
        "signals": len(signal_table),
        "event_summary_rows": len(event_summary),
        "trade_summary_rows": len(trade_summary),
        "daily_score_rows": len(daily_score),
        "best_rule_pair_rows": len(rule_summary),
        "equity_curve_rows": len(equity_curves),
        "strategy_summary_rows": len(summary),
        "best_equity_rows": len(best_equity),
        "advisor_scored_rows": len(scored),
        "plot_count": len(manifest),
    }


def run_reporting_pipeline(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    report_top_n: int = 30,
) -> tuple[dict, pd.DataFrame, str]:
    input_dirs = _run_dirs(input_dir)
    output_dirs = _run_dirs(output_dir if output_dir is not None else input_dir)
    output_dirs["report"].mkdir(parents=True, exist_ok=True)

    df = load_data(
        _resolve_run_file(
            input_dir,
            [
                "data/input_snapshot.csv",
                "input_snapshot.csv",
            ],
        )
    )
    signals = read_run_table(input_dir, ["results/signals/signals.csv", "signals.csv"])
    rule_summary = read_run_table(
        input_dir,
        [
            "results/rule_pair/rule_pair_best_base_summary.csv",
            "results/rule_pair/rule_pair_summary.csv",
            "rule_pair_best_base_summary.csv",
            "rule_pair_summary.csv",
        ],
    )
    daily_score = read_run_table(input_dir, ["results/score/monthly_refresh_daily_score.csv", "monthly_refresh_daily_score.csv"])
    utility = read_run_table(input_dir, ["results/score/factor_signal_utility.csv", "factor_signal_utility.csv"])

    status_df, summary_df, _ = run_signal_point_status_report(
        df=df,
        output_dir=output_dirs["report"],
        signal_table=signals,
        report_top_n=report_top_n,
    )
    role_summary = strategy_vote_summary(daily_score, utility)
    advisor_summary, scored, report = run_advisor_summary_report(
        status_df=status_df,
        summary_df=summary_df,
        rule_summary=rule_summary,
        role_strategy_summary=role_summary,
        output_dir=output_dirs["report"],
        top_n=report_top_n,
    )
    taxonomy_path = Path("skills") / "factor-timing-advisor" / "references" / "factor_taxonomy.md"
    taxonomy_arg = str(taxonomy_path) if taxonomy_path.exists() else None
    html_data = build_view_data(
        str(Path(input_dir)),
        taxonomy_path=taxonomy_arg,
        report_title="宽基择时信号报告",
    )
    html = render_html(html_data)
    (output_dirs["report"] / "timing_report.html").write_text(html, encoding="utf-8")
    return advisor_summary, scored, report


def run_baseline_strategy(
    input_dir: str | Path,
    output_dir: str | Path,
    score_suffix: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the retained baseline score strategy."""
    input_dirs = _run_dirs(input_dir)
    output_dirs = _run_dirs(output_dir)
    daily_score = read_run_table(input_dir, ["results/score/monthly_refresh_daily_score.csv", "monthly_refresh_daily_score.csv"])
    price_df = read_run_table(input_dir, ["data/input_snapshot.csv", "input_snapshot.csv"])

    rules = build_all_z_rules()
    score_suffix_value = "" if score_suffix == "default" else score_suffix
    suffix_label = score_suffix_value if score_suffix_value else "_default"

    summary, best_equity = backtest_z_rules(
        price_df=price_df,
        daily_score=daily_score,
        paired_rules=rules,
        output_dir=output_dir,
        score_suffix=score_suffix_value,
        file_suffix=suffix_label,
    )
    if not best_equity.empty:
        plot_best_rule(best_equity, output_dir=output_dir, file_suffix=suffix_label)
    return summary, best_equity


def run_factor_plots(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    lookback_days: int = 252 * 3,
    select_by_backtest: bool = True,
    show_base_factor_points: bool = True,
) -> pd.DataFrame:
    """Plot each factor with its best rule_pair marks and best excess charts."""
    input_dirs = _run_dirs(input_dir)
    output_dirs = _run_dirs(output_dir if output_dir is not None else input_dir)
    output_dirs["factor_plots"].mkdir(parents=True, exist_ok=True)
    output_dirs["rule_pair_best_plots"].mkdir(parents=True, exist_ok=True)

    df = read_run_table(input_dir, ["data/input_snapshot.csv", "input_snapshot.csv"])
    signals = read_run_table(input_dir, ["results/signals/signals.csv", "signals.csv"])
    rule_summary = read_run_table(
        input_dir,
        [
            "results/rule_pair/rule_pair_best_base_summary.csv",
            "results/rule_pair/rule_pair_summary.csv",
            "rule_pair_best_base_summary.csv",
            "rule_pair_summary.csv",
        ],
    )

    factors = factor_plot_roots(get_factor_columns(df))
    results: list[dict[str, object]] = []
    codes = df[CODE_COL].dropna().astype(str).drop_duplicates().tolist()
    for code in codes:
        chart_results = plot_factor_signal_charts(
            df=df,
            factors=factors,
            instrument=code,
            signal_table=signals,
            signal_path=None,
            rule_summary=rule_summary,
            rule_summary_path=None,
            select_by_backtest=select_by_backtest,
            save_dir=output_dirs["factor_plots"],
            show=False,
            backend="Agg",
            lookback_days=lookback_days,
            show_base_factor_points=show_base_factor_points,
        )
        for item in chart_results:
            item["code"] = code
        results.extend(chart_results)

    manifest = pd.DataFrame(results)
    write_table(manifest, output_dirs["factor_plots"] / "plot_manifest.csv")

    best_rule_pair_results = plot_best_rule_pair_charts(
        df=df,
        factors=factors,
        instrument=None,
        signal_table=signals,
        signal_path=None,
        rule_summary=rule_summary,
        rule_summary_path=None,
        equity_curves=read_run_table(
            input_dir,
            [
                "results/rule_pair/rule_pair_best_base_equity_curves.csv",
                "results/rule_pair/equity_curves.csv",
                "rule_pair_best_base_equity_curves.csv",
                "equity_curves.csv",
            ],
        ),
        equity_curves_path=None,
        save_dir=output_dirs["rule_pair_best_plots"],
        show=False,
        backend="Agg",
        lookback_days=None,
    )
    write_table(
        pd.DataFrame(best_rule_pair_results),
        output_dirs["rule_pair_best_plots"] / "best_rule_pair_plot_manifest.csv",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified baseline pipeline runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upstream = subparsers.add_parser("upstream", help="Generate signals, event study, rule pair, utility, and daily score.")
    upstream.add_argument("--csv", default="data/宽基得分.csv")
    upstream.add_argument("--output-dir", default="results_score_event_full_monthly")
    upstream.add_argument("--warmup-years", type=int, default=3)
    upstream.add_argument("--min-history-years", type=int, default=2)
    upstream.add_argument("--min-raw-open-events-per-quarter", type=float, default=2.0)
    upstream.add_argument("--min-excess-annual-return", type=float, default=0.05)
    upstream.add_argument("--max-equity-curves", type=int, default=200)
    upstream.add_argument("--parallel-n-jobs", type=int, default=1)
    upstream.add_argument("--save-intermediates", action="store_true", default=True)
    upstream.add_argument("--year-end-target-years", default=None, help="Comma-separated target years for rule_pair year-end snapshots. Default: latest two target years.")

    strategy = subparsers.add_parser("strategy", help="Run the retained baseline score strategy from precomputed daily score.")
    strategy.add_argument("--input-dir", default="results_score_event_full_monthly")
    strategy.add_argument("--output-dir", default="results_score_event_full_monthly")
    strategy.add_argument("--score-suffix", default="default")

    score_update = subparsers.add_parser("score-update", help="Daily refresh: update signals, events, best rule-pair per base, score strategy, report, and plots.")
    score_update.add_argument("--csv", default="data/宽基得分.csv")
    score_update.add_argument("--output-dir", default="results_score_event_full_monthly")
    score_update.add_argument("--warmup-years", type=int, default=3)
    score_update.add_argument("--min-history-years", type=int, default=2)
    score_update.add_argument("--min-raw-open-events-per-quarter", type=float, default=2.0)
    score_update.add_argument("--min-excess-annual-return", type=float, default=0.05)
    score_update.add_argument("--max-equity-curves", type=int, default=200)
    score_update.add_argument("--score-suffix", default="default")
    score_update.add_argument("--report-top-n", type=int, default=30)
    score_update.add_argument("--lookback-days", type=int, default=252 * 3)
    score_update.add_argument("--save-intermediates", action="store_true", default=True)

    plot = subparsers.add_parser("plot", help="Plot each factor with its best rule_pair buy/sell marks.")
    plot.add_argument("--input-dir", default="results_score_event_full_monthly")
    plot.add_argument("--output-dir", default=None)
    plot.add_argument("--lookback-days", type=int, default=252 * 3)
    plot.add_argument("--no-select-by-backtest", action="store_true")
    plot.add_argument("--hide-base-factor-points", action="store_true")

    report = subparsers.add_parser("report", help="Generate advisor_summary.json / md and signal-point reports.")
    report.add_argument("--input-dir", default="results_score_event_full_monthly")
    report.add_argument("--output-dir", default=None)
    report.add_argument("--report-top-n", type=int, default=30)

    full = subparsers.add_parser("all", help="Run upstream pipeline, strategy, report, and plot.")
    full.add_argument("--csv", default="data/宽基得分.csv")
    full.add_argument("--output-dir", default="results_score_event_full_monthly")
    full.add_argument("--warmup-years", type=int, default=3)
    full.add_argument("--min-history-years", type=int, default=2)
    full.add_argument("--min-raw-open-events-per-quarter", type=float, default=2.0)
    full.add_argument("--min-excess-annual-return", type=float, default=0.05)
    full.add_argument("--max-equity-curves", type=int, default=200)
    full.add_argument("--parallel-n-jobs", type=int, default=1)
    full.add_argument("--save-intermediates", action="store_true", default=True)
    full.add_argument("--score-suffix", default="default")
    full.add_argument("--lookback-days", type=int, default=252 * 3)
    full.add_argument("--year-end-target-years", default=None, help="Comma-separated target years for rule_pair year-end snapshots. Default: latest two target years.")
    full.add_argument("--report-top-n", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "upstream":
        signal_table, event_summary, trade_summary, rule_summary, rule_summary_by_year_end, equity_curves, daily_score = run_upstream_pipeline(
            csv_path=args.csv,
            output_dir=args.output_dir,
            warmup_years=args.warmup_years,
            min_history_years=args.min_history_years,
            min_raw_open_events_per_quarter=args.min_raw_open_events_per_quarter,
            min_excess_annual_return=args.min_excess_annual_return,
            max_equity_curves=None if args.max_equity_curves < 0 else args.max_equity_curves,
            parallel_n_jobs=args.parallel_n_jobs,
            save_intermediates=args.save_intermediates,
            year_end_target_years=_parse_year_list(args.year_end_target_years),
        )
        print(f"signals={len(signal_table)}")
        print(f"event_summary_rows={len(event_summary)}")
        print(f"trade_summary_rows={len(trade_summary)}")
        print(f"rule_pair_rows={len(rule_summary)}")
        print(f"rule_pair_year_end_rows={len(rule_summary_by_year_end)}")
        print(f"equity_curve_rows={len(equity_curves)}")
        print(f"daily_score_rows={len(daily_score)}")
        return

    if args.command == "strategy":
        summary, best_equity = run_baseline_strategy(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            score_suffix=args.score_suffix,
        )
        print(f"summary_rows={len(summary)}")
        print(f"best_equity_rows={len(best_equity)}")
        return

    if args.command == "score-update":
        stats = run_daily_refresh_pipeline(
            csv_path=args.csv,
            output_dir=args.output_dir,
            min_history_years=args.min_history_years,
            warmup_years=args.warmup_years,
            min_raw_open_events_per_quarter=args.min_raw_open_events_per_quarter,
            min_excess_annual_return=args.min_excess_annual_return,
            max_equity_curves=None if args.max_equity_curves < 0 else args.max_equity_curves,
            score_suffix=args.score_suffix,
            report_top_n=args.report_top_n,
            lookback_days=args.lookback_days,
            save_intermediates=args.save_intermediates,
        )
        for key, value in stats.items():
            print(f"{key}={value}")
        return

    if args.command == "plot":
        manifest = run_factor_plots(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            lookback_days=args.lookback_days,
            select_by_backtest=not args.no_select_by_backtest,
            show_base_factor_points=not args.hide_base_factor_points,
        )
        print(f"plot_count={len(manifest)}")
        return

    if args.command == "report":
        advisor_summary, scored, report = run_reporting_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            report_top_n=args.report_top_n,
        )
        print(f"advisor_summary_keys={len(advisor_summary)}")
        print(f"scored_rows={len(scored)}")
        print(f"report_chars={len(report)}")
        return

    if args.command == "all":
        signal_table, event_summary, trade_summary, rule_summary, rule_summary_by_year_end, equity_curves, daily_score = run_upstream_pipeline(
            csv_path=args.csv,
            output_dir=args.output_dir,
            warmup_years=args.warmup_years,
            min_history_years=args.min_history_years,
            min_raw_open_events_per_quarter=args.min_raw_open_events_per_quarter,
            min_excess_annual_return=args.min_excess_annual_return,
            max_equity_curves=None if args.max_equity_curves < 0 else args.max_equity_curves,
            parallel_n_jobs=args.parallel_n_jobs,
            save_intermediates=args.save_intermediates,
            year_end_target_years=_parse_year_list(args.year_end_target_years),
        )
        summary, best_equity = run_baseline_strategy(
            input_dir=args.output_dir,
            output_dir=args.output_dir,
            score_suffix=args.score_suffix,
        )
        advisor_summary, scored, report = run_reporting_pipeline(
            input_dir=args.output_dir,
            output_dir=None,
            report_top_n=args.report_top_n,
        )
        manifest = run_factor_plots(
            input_dir=args.output_dir,
            output_dir=None,
            lookback_days=args.lookback_days,
            select_by_backtest=True,
            show_base_factor_points=True,
        )
        print(f"signals={len(signal_table)}")
        print(f"event_summary_rows={len(event_summary)}")
        print(f"trade_summary_rows={len(trade_summary)}")
        print(f"rule_pair_rows={len(rule_summary)}")
        print(f"rule_pair_year_end_rows={len(rule_summary_by_year_end)}")
        print(f"equity_curve_rows={len(equity_curves)}")
        print(f"daily_score_rows={len(daily_score)}")
        print(f"strategy_summary_rows={len(summary)}")
        print(f"best_equity_rows={len(best_equity)}")
        print(f"advisor_scored_rows={len(scored)}")
        print(f"plot_count={len(manifest)}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
