"""Two frozen channel-rule variants; never included in core score weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import _annual_return, _max_drawdown, _sharpe
from io_utils import read_run_table, write_table
from timing_config import CODE_COL, DATE_COL, NAME_COL, PRICE_COL, TRADING_DAYS

FACTOR = "轨道偏离度"
FACTOR_COLUMNS = (FACTOR, FACTOR + "_季线", FACTOR + "_年线")
NON_CORE_INPUT_COLUMNS = FACTOR_COLUMNS + ("季线乖离率", "季线乖离率_季线", "季线乖离率_年线")
RULE_ID = "channel_deviation_auxiliary"
CATEGORY = "辅助/技术状态"
INPUT_PATH = "data/auxiliary_input_snapshot.parquet"
OUTPUT_SUBDIR = "auxiliary_signal_rules"
OPEN_LABEL = "季线10日修复首次达0.5且5日变化>=0，OR月/季<=-2后回升并在0-5日内价格同涨确认"
CLOSE_LABEL = "价格5日收益>0且年线5日变化<0的状态首次出现"


def extract_auxiliary_input(raw: pd.DataFrame) -> pd.DataFrame | None:
    present = [c for c in FACTOR_COLUMNS if c in raw]
    if not present:
        return None
    if len(present) != len(FACTOR_COLUMNS):
        raise ValueError("轨道偏离度辅助输入缺列：" + str(sorted(set(FACTOR_COLUMNS)-set(present))))
    frame = raw[[CODE_COL, DATE_COL, PRICE_COL, *FACTOR_COLUMNS]].copy()
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="raise")
    frame = frame.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    numeric = frame[[PRICE_COL, *FACTOR_COLUMNS]].apply(pd.to_numeric, errors="coerce")
    if frame[[CODE_COL, DATE_COL]].isna().any().any() or frame.duplicated([CODE_COL, DATE_COL]).any() or not np.isfinite(numeric.to_numpy()).all() or numeric[PRICE_COL].le(0).any():
        raise ValueError("辅助输入存在重复日期、缺失/非有限值或非正价格，拒绝生成信号")
    frame[numeric.columns] = numeric
    return frame


def validate_alignment(core: pd.DataFrame, auxiliary: pd.DataFrame) -> None:
    keys = [CODE_COL, DATE_COL]
    a = core.sort_values(keys).reset_index(drop=True)
    b = auxiliary.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False)
    np.testing.assert_allclose(a[PRICE_COL], b[PRICE_COL], rtol=0, atol=1e-10,
                               err_msg="辅助与主策略价格必须相同")


def load_pipeline_input(path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    from data_cleaning import load_data
    raw = load_data(path)
    auxiliary = extract_auxiliary_input(raw)
    if auxiliary is not None:
        write_table(auxiliary, Path(output_dir) / INPUT_PATH)
    return raw.drop(columns=list(NON_CORE_INPUT_COLUMNS), errors="ignore")


def channel_events(group: pd.DataFrame) -> pd.DataFrame:
    group = group.reset_index(drop=True)
    p = group[PRICE_COL].astype(float)
    m, q, y = [group[c].astype(float) for c in FACTOR_COLUMNS]
    state = p.pct_change(5, fill_method=None).gt(0) & y.diff(5).lt(0)
    close = (state & ~state.shift(fill_value=False)).to_numpy(bool)
    change = q.diff(10)
    main = (change.ge(.5) & change.shift().lt(.5) & q.diff(5).ge(0)).to_numpy(bool)
    low = p.rolling(5, min_periods=5).min().to_numpy()
    price = p.to_numpy()
    supplemental = np.zeros(len(group), bool)
    for f in (m, q):
        delta = f.diff()
        seeds = (f.shift().le(-2) & delta.gt(0) & delta.shift().le(0) & f.lt(0) & pd.Series(low).notna()).to_numpy()
        confirm = (p.pct_change(fill_method=None).gt(0) & delta.gt(0)).to_numpy()
        values, pending = f.to_numpy(), None
        for i in range(len(group)):
            if seeds[i] and pending is None:
                pending = (i, low[i], values[i-1])
            if pending is None:
                continue
            origin, fixed_price_low, fixed_factor_low = pending
            cancelled = close[i] or values[i] >= 0 or (i > origin and (price[i] < fixed_price_low or values[i] < fixed_factor_low))
            if cancelled:
                pending = None
            elif confirm[i]:
                supplemental[i], pending = True, None
            elif i-origin == 5:
                pending = None
    ma = p.rolling(60, min_periods=60).mean()
    up = p.gt(ma) & ma.gt(ma.shift(20))
    return pd.DataFrame({"main_event": main, "oversold_event": supplemental,
                         "entry_event": main | supplemental, "exit_event": close,
                         "up_environment": up, "ma60": ma, "ma60_change20": ma-ma.shift(20)})


def simulate_channel(group: pd.DataFrame, events: pd.DataFrame, up_only: bool = False):
    """Next-close fills, open final marks, and entry-day close eligibility match research."""
    group = group.reset_index(drop=True)
    prices, dates, n = group[PRICE_COL].to_numpy(float), group[DATE_COL].to_numpy(), len(group)
    entry = events.entry_event.to_numpy(bool) & (events.up_environment.to_numpy(bool) if up_only else True)
    closes = events.exit_event.to_numpy(bool)
    fillable = np.flatnonzero(closes & (np.arange(n) < n-1))
    position, post, costs = np.zeros(n), np.zeros(n), np.zeros(n)
    signal_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    records, i = [], 0
    while i < n-1:
        if not entry[i]:
            i += 1
            continue
        start = i+1
        k = np.searchsorted(fillable, start)
        closed = k < len(fillable)
        exit_signal = int(fillable[k]) if closed else None
        end = exit_signal+1 if closed else n-1
        path = prices[start:end+1]
        gross = prices[end]/prices[start]-1
        net = (1+gross)*.9995**(2 if closed else 1)-1
        records.append({"entry_signal_idx": i, "entry_idx": start, "exit_signal_idx": exit_signal,
                        "exit_idx": end, "entry_signal_date": dates[i], "entry_date": dates[start],
                        "exit_signal_date": dates[exit_signal] if closed else pd.NaT,
                        "exit_date": dates[end], "entry_price": prices[start], "exit_price": prices[end],
                        "trade_return": net, "gross_trade_return": gross, "holding_days": end-start,
                        "annualized_trade_return": (1+net)**(TRADING_DAYS/(end-start))-1 if end > start else np.nan,
                        "max_drawdown": float(np.min(path/np.maximum.accumulate(path)-1)), "forced_exit": not closed,
                        "close_reason": CLOSE_LABEL if closed else "end", "rule_version": "channel_auxiliary_v1",
                        "exit_execution_date": dates[end] if closed else pd.NaT, "mark_date": dates[end],
                        "trade_status": "closed" if closed else "open", "exit_reason": CLOSE_LABEL if closed else "期末持仓估值"})
        position[start+1:end+1] = 1
        signal_dates[start+1:end+1] = dates[i]
        post[start:end if closed else end+1] = 1
        costs[start] += .0005
        if closed:
            costs[end] += .0005
        i = end
    bm = group[PRICE_COL].pct_change(fill_method=None).fillna(0).to_numpy()
    net_returns = (1+bm*position)*(1-costs)-1
    daily = pd.DataFrame({DATE_COL: dates, CODE_COL: group[CODE_COL].to_numpy(), PRICE_COL: prices,
                          "return_position": position, "post_close_position": post, "benchmark_return": bm,
                          "net_return": net_returns, "net_equity": np.cumprod(1+net_returns),
                          "entry_event": entry, "exit_event": closes, "source_signal_date": signal_dates})
    return records, daily


def build_auxiliary_tables(core: pd.DataFrame, run_dir: Path):
    from selected_single_factor_rules import SelectedRuleSpec, _summarize_rule, _position_rows
    path = run_dir / INPUT_PATH
    if not path.exists():
        return {}, {}
    source = extract_auxiliary_input(pd.read_parquet(path))
    validate_alignment(core, source)
    spec = SelectedRuleSpec(rule_id=RULE_ID, factor=FACTOR, category=CATEGORY,
        strategy_label="轨道偏离度_完整短线辅助_V1", base_open_condition=OPEN_LABEL, close_condition=CLOSE_LABEL,
        notes="完整短线版；单边5bp。辅助角色不参与原两大策略打分；复合另用上涨环境子账户。")
    summaries, statuses, trades, positions, variants = [], [], [], [], []
    for code, g in source.groupby(CODE_COL, sort=False):
        g = g.reset_index(drop=True)
        name = core.loc[core[CODE_COL].eq(code), NAME_COL].iloc[-1] if NAME_COL in core else ""
        g[NAME_COL] = name
        events = channel_events(g)
        for variant, up_only in (("standalone", False), ("composite_up", True)):
            records, d = simulate_channel(g, events, up_only)
            variants.append(d.assign(variant=variant))
            if up_only:
                continue
            item = {CODE_COL: code, "group": g, "prices": g[PRICE_COL].to_numpy(), "dates": g[DATE_COL].to_numpy()}
            summary = _summarize_rule(spec, item, records, d.return_position.to_numpy(), d.entry_event.to_numpy())
            bm_equity = (1+d.benchmark_return).cumprod()
            summary.update(annual_return=_annual_return(d.net_equity), excess_annual_return=_annual_return(d.net_equity/bm_equity),
                           max_drawdown=_max_drawdown(d.net_equity), excess_max_drawdown=_max_drawdown(d.net_equity/bm_equity),
                           sharpe=_sharpe(d.net_return), final_equity=d.net_equity.iloc[-1], excess_final_equity=(d.net_equity/bm_equity).iloc[-1])
            summaries.append(summary)
            pos = _position_rows(spec, item, d.return_position.to_numpy())
            pos["strategy_return"] = d.net_return.to_numpy()
            positions.append(pos)
            metadata = {CODE_COL: code, NAME_COL: name, "rule_id": RULE_ID, "factor": FACTOR, "category": CATEGORY,
                        "strategy_label": spec.strategy_label, "open_condition": OPEN_LABEL, "close_condition": CLOSE_LABEL}
            trades.extend([dict(r, **metadata) for r in records])
            live = bool(d.post_close_position.iloc[-1])
            last = records[-1] if records else {}
            pending = "待闭仓" if live and d.exit_event.iloc[-1] else "待开仓" if not live and d.entry_event.iloc[-1] else ""
            closed_records = [r for r in records if not r["forced_exit"]]
            statuses.append(dict(metadata, latest_date=g[DATE_COL].iloc[-1], latest_price=g[PRICE_COL].iloc[-1],
                latest_factor_value=g[FACTOR].iloc[-1], current_state="多" if live else "空", pending_signal=pending,
                pending_signal_date=g[DATE_COL].iloc[-1] if pending else pd.NaT,
                entry_signal_date=last.get("entry_signal_date", pd.NaT) if live else pd.NaT,
                entry_date=last.get("entry_date", pd.NaT) if live else pd.NaT,
                current_holding_days=last.get("holding_days", np.nan) if live else np.nan,
                current_return=last.get("trade_return", np.nan) if live else np.nan,
                last_open_signal_date=g[DATE_COL].iloc[-1] if pending == "待开仓" else last.get("entry_signal_date", pd.NaT),
                last_close_signal_date=g[DATE_COL].iloc[-1] if pending == "待闭仓" else closed_records[-1]["exit_signal_date"] if closed_records else pd.NaT,
                latest_open_signal=bool(d.entry_event.iloc[-1]), latest_close_signal=bool(d.exit_event.iloc[-1]),
                open_event_count=int(d.entry_event.sum()), close_event_count=int(d.exit_event.sum()),
                base_open_condition=OPEN_LABEL, base_close_condition=CLOSE_LABEL, notes=spec.notes))
    spec_row = {key: getattr(spec, key) for key in ("rule_id", "factor", "category", "strategy_label", "base_open_condition",
                "close_condition", "open_transform", "close_transform", "min_hold_days_for_close", "stop_loss_return", "stop_loss_window", "notes")}
    spec_row.update(open_condition=OPEN_LABEL, base_close_condition=CLOSE_LABEL)
    selected = {"selected_rule_specs": pd.DataFrame([spec_row]), "selected_rule_summary": pd.DataFrame(summaries),
                "selected_rule_latest_status": pd.DataFrame(statuses), "selected_rule_trades": pd.DataFrame(trades),
                "selected_rule_daily_positions": pd.concat(positions, ignore_index=True)}
    auxiliary = {"auxiliary_rule_daily": pd.concat(variants, ignore_index=True)}
    return selected, auxiliary
