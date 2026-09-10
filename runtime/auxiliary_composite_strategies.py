"""Full-size, fresh-signal overlays; original composite accounts stay unchanged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from auxiliary_signal_rules import INPUT_PATH, OUTPUT_SUBDIR as SIGNAL_SUBDIR
from timing_config import CODE_COL, DATE_COL, NAME_COL, TRADING_DAYS

OUTPUT_SUBDIR = "auxiliary_composite_strategies"
SUFFIX = "__channel_auxiliary_v1"


def supplement_path(core, child):
    b, c = np.asarray(core, float), np.asarray(child, float)
    if len(b) != len(c) or not np.isin(b, [0., 1.]).all() or not np.isin(c, [0., 1.]).all():
        raise ValueError("辅助接入只支持日期对齐的0/1主仓位和子仓位")
    leg = np.zeros(len(b))
    active, previous_child = False, False
    for i in range(len(b)):
        on = bool(c[i])
        if b[i] or not on:
            active = False
        elif on and not previous_child:
            active = True
        leg[i], previous_child = float(active), on
    return leg


def run_auxiliary_composites(run_dir: Path, core_daily: pd.DataFrame, cost_bps: float):
    from composite_timing_strategies import _strategy_daily, _performance_summary, _trade_events, _latest_status, _write_outputs
    if not (run_dir / INPUT_PATH).exists():
        return {"auxiliary_strategy_count": 0}
    children = pd.read_parquet(run_dir / "results" / SIGNAL_SUBDIR / "auxiliary_rule_daily.parquet")
    children = children.loc[children.variant.eq("composite_up")].copy()
    daily_tables, events_tables, summaries, statuses, ownership_tables = [], [], [], [], []
    for sid, base in core_daily.groupby("strategy_id", sort=False):
        base = base.reset_index(drop=True)
        code = str(base[CODE_COL].iloc[0])
        child = children.loc[children[CODE_COL].astype(str).eq(code)].sort_values(DATE_COL).reset_index(drop=True)
        pd.testing.assert_series_equal(base[DATE_COL], child[DATE_COL], check_names=False)
        applied = child.return_position.shift(1).fillna(0)
        source = child.source_signal_date.shift(1)
        leg = supplement_path(base.exposure, applied)
        target = base.exposure.to_numpy()+leg
        index = pd.DatetimeIndex(base[DATE_COL])
        enhanced_id = sid + SUFFIX
        d = _strategy_daily(enhanced_id, "auxiliary_observation", pd.Series(base.composite_score.to_numpy(), index=index),
             pd.Series(base.weekly_anchor_score.to_numpy(), index=index), pd.Series(target, index=index),
             pd.Series(base.benchmark_return.to_numpy(), index=index), code, str(base[NAME_COL].iloc[0]), cost_bps)
        d["base_strategy_id"] = sid
        d["core_exposure"], d["auxiliary_exposure"] = base.exposure.to_numpy(), leg
        d["base_net_equity"] = base.net_equity.to_numpy()
        d["owner"] = np.select([base.exposure.gt(0), leg > 0], ["原组合", "辅助信号"], default="空仓")
        d["auxiliary_signal_date"] = source.where(leg > 0).to_numpy()
        e = _trade_events(d, set())
        indexed = d.set_index(DATE_COL)
        e["trigger_source"] = np.where(e.trade_side.eq("entry") & e.execution_date.map(indexed.auxiliary_exposure).gt(0),
                                         "auxiliary_entry", "core_transition")
        prev_leg = pd.Series(leg, index=index).shift(1, fill_value=0)
        e.loc[e.trade_side.eq("exit") & e.execution_date.map(prev_leg).gt(0), "trigger_source"] = "auxiliary_exit"
        source_map = pd.Series(source.to_numpy(), index=index)
        mask = e.trigger_source.eq("auxiliary_entry")
        e.loc[mask, "signal_date"] = e.loc[mask, "execution_date"].map(source_map)
        handoff = base.exposure.gt(0).to_numpy() & np.r_[False, leg[:-1] > 0]
        ownership = d.loc[pd.Series(handoff)].copy()
        ownership["event"] = "原组合接管（不交易）"
        ownership["source_signal_date"] = source.shift(1).loc[handoff].to_numpy()
        ownership_tables.append(ownership)
        summary = _performance_summary(d)
        original = _performance_summary(base)
        years = len(d)/TRADING_DAYS
        excess = (d.net_equity.iloc[-1]/d.benchmark_equity.iloc[-1])**(1/years)-1
        base_excess = (base.net_equity.iloc[-1]/base.benchmark_equity.iloc[-1])**(1/years)-1
        summary.update(base_strategy_id=sid, excess_annual_return=excess, base_excess_annual_return=base_excess,
                       excess_delta_pp=100*(excess-base_excess), base_sharpe=original["sharpe"], base_max_drawdown=original["max_drawdown"],
                       extra_exposure_days=int((leg > 0).sum()), auxiliary_entries=int(((leg > 0) & ~np.r_[False, leg[:-1] > 0]).sum()))
        status = _latest_status(d, e, None)
        status.update(base_strategy_id=sid, owner=d.owner.iloc[-1], state_reason="辅助增强观察版；"+d.owner.iloc[-1])
        daily_tables.append(d)
        events_tables.append(e)
        summaries.append(summary)
        statuses.append(status)
    tables = {"auxiliary_composite_daily": pd.concat(daily_tables, ignore_index=True),
              "auxiliary_composite_summary": pd.DataFrame(summaries),
              "auxiliary_composite_trades": pd.concat(events_tables, ignore_index=True),
              "auxiliary_composite_latest_status": pd.DataFrame(statuses),
              "auxiliary_ownership_events": pd.concat(ownership_tables, ignore_index=True)}
    output = run_dir / "results" / OUTPUT_SUBDIR
    _write_outputs(output, tables)
    payload = tables["auxiliary_composite_latest_status"].to_json(orient="records", date_format="iso", force_ascii=False)
    (output / "current_auxiliary_signal.json").write_text(payload, encoding="utf-8")
    return {"auxiliary_strategy_count": len(summaries), "auxiliary_daily_rows": len(tables["auxiliary_composite_daily"])}
