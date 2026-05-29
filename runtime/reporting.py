from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtest import _rule_status_from_cache
from data_cleaning import get_factor_columns
from io_utils import write_table
from signal_generation import (
    _build_factor_event_cache_from_signal_table,
    build_event_conditions,
    generate_signal_table,
)
from timing_config import (
    AUXILIARY_CATEGORIES,
    CATEGORY_SCORE_WEIGHT,
    CODE_COL,
    CORE_CATEGORIES,
    DEFAULT_TAXONOMY_PATH,
    FREQUENCY_SCORE_WEIGHT,
    NAME_COL,
    SIGNAL_DATE_COL,
    SIGNAL_FACTOR_COL,
    SIGNAL_INSTRUMENT_COL,
    SIGNAL_NAME_COL,
    SIGNAL_PATTERN_COL,
    SIGNAL_VALUE_COL,
    STATE_FLAT,
    STATE_LONG,
    STATE_ORDER,
    STATE_WAIT,
    EventCondition,
)


def _format_report_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_无_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_format_report_value(value) for value in row) + " |")
    return "\n".join(lines)


def _split_factor_frequency(factor: str) -> tuple[str, str]:
    """把字段名拆成基础因子和周期标签。"""
    if factor.endswith("_季线"):
        return factor.removesuffix("_季线"), "季线"
    if factor.endswith("_年线"):
        return factor.removesuffix("_年线"), "年线"
    return factor, "原始"


def _load_factor_taxonomy(taxonomy_path: str | Path | None = DEFAULT_TAXONOMY_PATH) -> dict[str, dict[str, str]]:
    """
    从字段说明 markdown 读取 `基础因子 -> 字段信息` 映射。

    这个函数只解析 `### 字段名`、`- 类别：...`、`- 默认方向：...`，
    因此后续手工修改字段含义时，不需要同步维护一份 Python 字典。
    """
    if taxonomy_path is None:
        return {}

    path = Path(taxonomy_path)
    if not path.exists():
        return {}

    taxonomy: dict[str, dict[str, str]] = {}
    current_factor = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            current_factor = line.removeprefix("### ").strip()
            taxonomy.setdefault(current_factor, {})
            continue
        if not current_factor:
            continue
        if line.startswith("- 类别：") or line.startswith("- 类别:"):
            sep = "：" if "：" in line else ":"
            taxonomy[current_factor]["category"] = line.split(sep, 1)[1].strip()
        elif line.startswith("- 默认方向：") or line.startswith("- 默认方向:"):
            sep = "：" if "：" in line else ":"
            taxonomy[current_factor]["default_direction"] = line.split(sep, 1)[1].strip().rstrip("。")
    return taxonomy


def _load_factor_category_map(taxonomy_path: str | Path | None = DEFAULT_TAXONOMY_PATH) -> dict[str, str]:
    taxonomy = _load_factor_taxonomy(taxonomy_path)
    return {factor: info.get("category", "未分类") for factor, info in taxonomy.items()}


def _factor_category(factor: str, category_map: dict[str, str]) -> str:
    base_factor, _ = _split_factor_frequency(factor)
    return category_map.get(base_factor) or category_map.get(factor) or "未分类"


def _factor_default_direction(factor: str, taxonomy: dict[str, dict[str, str]]) -> str:
    base_factor, _ = _split_factor_frequency(factor)
    return (
        taxonomy.get(base_factor, {}).get("default_direction")
        or taxonomy.get(factor, {}).get("default_direction")
        or "待确认"
    )


def _direction_bucket(default_direction: str) -> str:
    text = str(default_direction)
    if "均值回复" in text:
        return "均值回复"
    if "正向" in text and not any(token in text for token in ("风险", "拥挤", "待确认")):
        return "正向"
    if "正向" in text and any(token in text for token in ("风险", "拥挤")):
        return "正向但有拥挤风险"
    if any(token in text for token in ("反向", "风险", "压力")):
        return "反向/风险"
    return "待确认"


def _open_pattern_family(pattern: str) -> str:
    """把具体开仓规则归到更粗的规则类型，方便汇总。"""
    name = pattern.removeprefix("开仓_")
    if "日振幅放大向上" in name:
        return "日振幅放大向上"
    for family in (
        "下方拐点",
        "上穿均线",
        "低位均值回复",
        "连续上升",
        "累计上升",
        "快速上升",
        "日振幅放大向上",
        "低位持续",
        "价格因子底背离",
        "价格新低因子未确认",
        "价格下跌因子上升",
        "价因子同步上升",
        "价格突破因子确认",
        "原始季线年线多头共振",
        "加速度转正",
        "指标上穿",
        "斜率转正",
        "上穿",
        "process_signal买点",
    ):
        if name.startswith(family):
            return family
    return name.split("_", 1)[0]


def _cross_up_level_from_pattern(pattern: str) -> float | None:
    name = pattern.removeprefix("开仓_上穿_").removesuffix("sigma")
    try:
        return float(name)
    except ValueError:
        return None


def _core_open_rule_style(pattern: str) -> tuple[str, str, str]:
    """赔率/胜率类指标的开仓规则风格。"""
    family = _open_pattern_family(pattern)
    if family == "下方拐点":
        return "抄底/拐点", "低位右侧拐点确认", "容易过早，最好和胜率/资金信号共振"
    if family == "低位均值回复":
        return "抄底/均值回复", "低分位区域开始回升", "适合赔率修复，不宜单独追涨"
    if family == "低位持续":
        return "抄底/低位脱离", "低位持续后重新回升", "偏左侧修复，最好观察价格是否同步止跌"
    if family == "价格因子底背离":
        return "抄底/底背离", "价格新低但因子未同步新低", "底背离不等于立即反转，需要价格确认"
    if family == "价格新低因子未确认":
        return "抄底/底背离", "价格创新低但因子未同步恶化", "比严格底背离更宽松，需要价格确认"
    if family == "价格下跌因子上升":
        return "抄底/内部修复", "价格下跌但因子已经回升", "偏左侧，需要观察指数是否止跌"
    if family == "process_signal买点":
        return "抄底/拐点", "局部谷值右侧确认", "信号滞后但避免把未来拐点放到过去"
    if family == "上穿":
        level = _cross_up_level_from_pattern(pattern)
        if level is None:
            return "趋势确认", "阈值上穿", "需结合字段方向解释"
        if level < 0:
            return "低位修复", f"上穿 {level:g} sigma，低位修复确认", "偏修复而非强趋势"
        if level == 0:
            return "趋势确认", "上穿 0 轴，状态由弱转强", "确认性强于低位拐点，但可能滞后"
        return "追高/强势突破", f"上穿 {level:g} sigma，强势区继续突破", "可能有动量延续，也更容易拥挤"
    if family == "连续上升":
        return "追涨/动量延续", "短期连续改善", "对噪声较敏感，适合结合中期趋势"
    if family in ("累计上升", "快速上升"):
        return "动量冲击", "N 日内因子累计抬升超过阈值", "适合平滑因子，最好结合大级别趋势"
    if family == "日振幅放大向上":
        return "动量冲击", "N 日振幅放大且向上选择", "波动放大后可能延续，也可能短期过热"
    if family == "价因子同步上升":
        return "趋势确认", "价格和因子同步改善", "更偏右侧确认，可能滞后"
    if family == "价格突破因子确认":
        return "趋势确认", "价格突破且因子同步确认", "趋势确认较强，但需防突破后回落"
    if family == "加速度转正":
        return "动量改善", "N 日动量的改善速度转正", "比趋势确认更早，也更容易受噪声影响"
    if family == "指标上穿":
        return "多周期确认", "原始指标上穿季线或年线", "用于确认短周期重新强于大级别趋势"
    if family == "原始季线年线多头共振":
        return "多周期确认", "原始、季线、年线同步向上", "确认性较强，但一般更偏右侧"
    if family == "斜率转正":
        return "趋势反转/修复", "中短期斜率由负转正", "比连续上升更宽松，需防假反转"
    if family == "上穿均线":
        return "趋势确认", "上穿自身均线", "适合确认状态改善，不代表低估"
    return "其他开仓", family, "需要人工补充规则解释"


def _auxiliary_signal_style(factor_category: str, pattern: str) -> tuple[str, str, str]:
    """
    辅助类指标不直接标成抄底/追高。

    它们只能说明风险环境、筹码结构或资金分歧状态，
    不能单独作为开仓依据。
    """
    family = _open_pattern_family(pattern)
    if factor_category == "辅助/风险状态":
        return "风险过滤", f"{family}：风险状态变化", "只用于仓位和风险过滤，不单独作为开仓信号"
    if factor_category == "辅助/资金分歧":
        return "分歧观察", f"{family}：资金分歧或轮动状态变化", "用于判断信号可信度，不单独决定多空"
    if factor_category == "辅助/筹码结构":
        return "结构确认", f"{family}：筹码结构状态变化", "用于确认压力/支撑结构，不单独决定开仓"
    return "辅助观察", f"{family}：辅助状态变化", "辅助指标不单独作为开仓依据"


def _open_rule_metadata(pattern: str, factor_category: str) -> dict[str, Any]:
    is_auxiliary = factor_category in AUXILIARY_CATEGORIES or factor_category.startswith("辅助/")
    if is_auxiliary:
        style, stage, risk = _auxiliary_signal_style(factor_category, pattern)
        return {
            "signal_role": "辅助观察",
            "open_rule_style": style,
            "open_rule_stage": stage,
            "open_rule_risk": risk,
            "is_auxiliary_signal": True,
        }

    style, stage, risk = _core_open_rule_style(pattern)
    return {
        "signal_role": "核心开仓",
        "open_rule_style": style,
        "open_rule_stage": stage,
        "open_rule_risk": risk,
        "is_auxiliary_signal": False,
    }


def _close_pattern_hits_by_date(events: dict[str, np.ndarray], close_conditions: list[EventCondition]) -> list[str]:
    """把同一天触发的静态闭仓事件合并成可读字符串。"""
    first_event = next(iter(events.values()), np.array([], dtype=bool))
    hits: list[list[str]] = [[] for _ in range(len(first_event))]
    for condition in close_conditions:
        if condition.requires_position:
            continue
        condition_events = events[condition.name]
        for idx in np.flatnonzero(condition_events):
            hits[int(idx)].append(condition.name)
    return ["；".join(items) for items in hits]


def _signal_point_state(
    dates: np.ndarray,
    open_events: np.ndarray,
    close_events: np.ndarray,
    open_pattern: str,
    close_pattern_hits: list[str],
) -> dict[str, Any]:
    """
    计算单个 `指数 + 因子 + 开仓规则` 点位的当前状态。

    规则：
    - 初始是 `观望`。
    - 该开仓规则触发后变成 `多`。
    - 任一静态闭仓/反向事件触发后变成 `空`。
    - 没有新事件时延续上一状态。
    - 同一天既触发该开仓规则又触发闭仓事件时，开仓规则优先，同时记录冲突标记。
    """
    current_state = STATE_WAIT
    state_start_idx: int | None = None
    last_signal_idx: int | None = None
    last_signal_type = ""
    last_signal_pattern = ""
    last_open_idx: int | None = None
    last_close_idx: int | None = None
    same_day_conflict_count = 0

    long_durations: list[int] = []
    flat_durations: list[int] = []

    for idx in range(len(dates)):
        open_hit = bool(open_events[idx])
        close_hit = bool(close_events[idx])
        if open_hit and close_hit:
            same_day_conflict_count += 1

        next_state = ""
        signal_type = ""
        signal_pattern = ""

        if open_hit:
            next_state = STATE_LONG
            signal_type = "开仓"
            signal_pattern = open_pattern
            last_open_idx = idx
        elif close_hit:
            next_state = STATE_FLAT
            signal_type = "闭仓"
            signal_pattern = close_pattern_hits[idx]
            last_close_idx = idx

        if not next_state:
            continue

        if next_state != current_state:
            if current_state == STATE_LONG and state_start_idx is not None:
                long_durations.append(idx - state_start_idx)
            elif current_state == STATE_FLAT and state_start_idx is not None:
                flat_durations.append(idx - state_start_idx)
            current_state = next_state
            state_start_idx = idx

        last_signal_idx = idx
        last_signal_type = signal_type
        last_signal_pattern = signal_pattern

    latest_idx = len(dates) - 1 if len(dates) else None
    state_age_days = (
        latest_idx - state_start_idx
        if latest_idx is not None and state_start_idx is not None
        else np.nan
    )
    mean_long_days = float(np.mean(long_durations)) if long_durations else np.nan
    median_long_days = float(np.median(long_durations)) if long_durations else np.nan
    mean_flat_days = float(np.mean(flat_durations)) if flat_durations else np.nan
    median_flat_days = float(np.median(flat_durations)) if flat_durations else np.nan
    expected_mean_days = (
        mean_long_days
        if current_state == STATE_LONG
        else mean_flat_days
        if current_state == STATE_FLAT
        else np.nan
    )
    expected_remaining_days = (
        max(expected_mean_days - state_age_days, 0)
        if np.isfinite(expected_mean_days) and np.isfinite(state_age_days)
        else np.nan
    )

    return {
        "current_state": current_state,
        "state_start_date": dates[state_start_idx] if state_start_idx is not None else pd.NaT,
        "state_age_days": state_age_days,
        "expected_remaining_days": expected_remaining_days,
        "last_signal_date": dates[last_signal_idx] if last_signal_idx is not None else pd.NaT,
        "last_signal_type": last_signal_type,
        "last_signal_pattern": last_signal_pattern,
        "last_open_date": dates[last_open_idx] if last_open_idx is not None else pd.NaT,
        "last_close_date": dates[last_close_idx] if last_close_idx is not None else pd.NaT,
        "open_event_count": int(np.sum(open_events)),
        "close_event_count": int(np.sum(close_events)),
        "long_episode_count": len(long_durations),
        "flat_episode_count": len(flat_durations),
        "mean_long_days": mean_long_days,
        "median_long_days": median_long_days,
        "mean_flat_days": mean_flat_days,
        "median_flat_days": median_flat_days,
        "same_day_conflict_count": same_day_conflict_count,
    }


def _signal_point_rows_from_cache(
    factor: str,
    factor_cache: list[dict[str, Any]],
    open_conditions: list[EventCondition],
    close_conditions: list[EventCondition],
    category_map: dict[str, str],
    taxonomy: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factor_base, frequency = _split_factor_frequency(factor)
    category = _factor_category(factor, category_map)
    default_direction = _factor_default_direction(factor, taxonomy)
    direction_bucket = _direction_bucket(default_direction)
    static_close_conditions = [condition for condition in close_conditions if not condition.requires_position]

    for item in factor_cache:
        group = item["group"]
        dates = item["dates"]
        events = item["events"]
        close_events = np.zeros(len(group), dtype=bool)
        for condition in static_close_conditions:
            close_events |= events[condition.name]
        close_pattern_hits = _close_pattern_hits_by_date(events, static_close_conditions)

        for open_condition in open_conditions:
            open_events = events[open_condition.name]
            rule_metadata = _open_rule_metadata(open_condition.name, category)
            state = _signal_point_state(
                dates=dates,
                open_events=open_events,
                close_events=close_events,
                open_pattern=open_condition.name,
                close_pattern_hits=close_pattern_hits,
            )
            rows.append(
                {
                    "latest_date": dates[-1] if len(dates) else pd.NaT,
                    SIGNAL_INSTRUMENT_COL: item[CODE_COL],
                    SIGNAL_NAME_COL: group[NAME_COL].iloc[-1] if len(group) else "",
                    SIGNAL_FACTOR_COL: factor,
                    "factor_base": factor_base,
                    "frequency": frequency,
                    "factor_category": category,
                    "default_direction": default_direction,
                    "direction_bucket": direction_bucket,
                    "open_pattern": open_condition.name,
                    "open_pattern_family": _open_pattern_family(open_condition.name),
                    **rule_metadata,
                    **state,
                }
            )
    return rows


def _count_summary(
    status_df: pd.DataFrame,
    group_cols: list[str],
    group_type: str,
) -> pd.DataFrame:
    if status_df.empty:
        return pd.DataFrame()

    grouped = status_df.groupby(group_cols + ["current_state"], dropna=False).size().unstack(fill_value=0)
    for state in STATE_ORDER:
        if state not in grouped.columns:
            grouped[state] = 0
    grouped = grouped[STATE_ORDER].reset_index()
    grouped["total"] = grouped[STATE_ORDER].sum(axis=1)
    for state in STATE_ORDER:
        grouped[f"{state}占比"] = np.where(grouped["total"] > 0, grouped[state] / grouped["total"], np.nan)
    state_counts = grouped[STATE_ORDER]
    max_count = state_counts.max(axis=1)
    winner_count = state_counts.eq(max_count, axis=0).sum(axis=1)
    grouped["主状态"] = np.where(winner_count.eq(1), state_counts.idxmax(axis=1), "均衡")
    grouped.insert(0, "group_type", group_type)
    grouped["group_value"] = grouped[group_cols].astype(str).agg(" / ".join, axis=1)
    return grouped


def build_signal_point_summary(status_df: pd.DataFrame) -> pd.DataFrame:
    summary_parts = [
        _count_summary(status_df.assign(全部="全部"), ["全部"], "全部"),
        _count_summary(status_df, ["factor_category"], "因子类别"),
        _count_summary(status_df, ["frequency"], "周期"),
        _count_summary(status_df, ["factor_category", "frequency"], "因子类别+周期"),
        _count_summary(status_df, ["signal_role"], "信号角色"),
        _count_summary(status_df, ["open_rule_style"], "开仓/辅助风格"),
        _count_summary(status_df, ["open_pattern_family"], "开仓规则类型"),
        _count_summary(status_df, ["factor_category", "open_rule_style"], "因子类别+开仓/辅助风格"),
    ]
    summary_parts = [part for part in summary_parts if not part.empty]
    if not summary_parts:
        return pd.DataFrame()
    return pd.concat(summary_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _summary_slice(summary_df: pd.DataFrame, group_type: str, display_cols: list[str]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=display_cols)
    sliced = summary_df[summary_df["group_type"].eq(group_type)].copy()
    return sliced[display_cols] if not sliced.empty else pd.DataFrame(columns=display_cols)


def write_signal_point_report(
    status_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 100,
) -> str:
    latest_date = status_df["latest_date"].max() if not status_df.empty else pd.NaT
    total = len(status_df)
    state_counts = status_df["current_state"].value_counts() if not status_df.empty else pd.Series(dtype=int)
    long_count = int(state_counts.get(STATE_LONG, 0))
    flat_count = int(state_counts.get(STATE_FLAT, 0))
    wait_count = int(state_counts.get(STATE_WAIT, 0))

    display_cols = ["group_value", STATE_LONG, STATE_FLAT, STATE_WAIT, "total", "多占比", "空占比", "主状态"]
    category_summary = _summary_slice(summary_df, "因子类别", display_cols)
    frequency_summary = _summary_slice(summary_df, "周期", display_cols)
    category_frequency_summary = _summary_slice(summary_df, "因子类别+周期", display_cols)
    role_summary = _summary_slice(summary_df, "信号角色", display_cols)
    style_summary = _summary_slice(summary_df, "开仓/辅助风格", display_cols)
    pattern_summary = _summary_slice(summary_df, "开仓规则类型", display_cols)

    recent_cols = [
        SIGNAL_INSTRUMENT_COL,
        SIGNAL_FACTOR_COL,
        "frequency",
        "factor_category",
        "signal_role",
        "open_rule_style",
        "open_rule_stage",
        "open_pattern",
        "current_state",
        "state_start_date",
        "state_age_days",
        "expected_remaining_days",
        "last_signal_date",
        "last_signal_type",
        "last_signal_pattern",
    ]
    recent = status_df.copy()
    if not recent.empty:
        recent = recent.sort_values(["last_signal_date", "factor_category", SIGNAL_FACTOR_COL], ascending=[False, True, True])
        if top_n >= 0:
            recent = recent.head(top_n)

    report = "\n".join(
        [
            "# 当前信号点状态报告",
            "",
            f"- 最新日期：{_format_report_value(latest_date)}",
            "- 信号点口径：指数 + 因子 + 开仓规则",
            f"- 信号点总数：{total}",
            f"- 当前多：{long_count}",
            f"- 当前空：{flat_count}",
            f"- 当前观望：{wait_count}",
            "",
            "## 按因子类别",
            "",
            _markdown_table(category_summary, display_cols),
            "",
            "## 按周期",
            "",
            _markdown_table(frequency_summary, display_cols),
            "",
            "## 按因子类别和周期",
            "",
            _markdown_table(category_frequency_summary, display_cols),
            "",
            "## 按信号角色",
            "",
            _markdown_table(role_summary, display_cols),
            "",
            "## 按开仓/辅助风格",
            "",
            _markdown_table(style_summary, display_cols),
            "",
            "## 按开仓规则类型",
            "",
            _markdown_table(pattern_summary, display_cols),
            "",
            "## 最近状态变化",
            "",
            _markdown_table(recent, recent_cols),
            "",
            "## 说明",
            "",
            "- `多` 表示该信号点最近一次有效状态切换来自它自己的开仓规则。",
            "- `空` 表示该因子最近一次有效状态切换来自任一静态闭仓/反向事件；这里不是做空，只是 long/cash 里的空仓或偏谨慎。",
            "- `观望` 表示该信号点尚未出现过有效开仓或闭仓状态切换。",
            "- `signal_role=核心开仓` 才能按抄底、趋势确认、追高等方式解释。",
            "- `signal_role=辅助观察` 只能作为风险过滤、结构确认或分歧观察，不能单独作为开仓依据。",
            "- 没有新事件的日期会延续上一状态，因此当前统计不会只看最新一天触发了什么。",
            "- 止损、止盈、持仓满 N 日依赖入场价和持仓状态，不进入这个点位状态表，只留在规则组合回测里。",
            "- 全量明细见 `signal_points_state.csv`；汇总明细见 `signal_points_summary.csv`。",
            "",
        ]
    )
    Path(output_path).write_text(report, encoding="utf-8")
    return report


def run_signal_point_status_report(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    factors: Iterable[str] | None = None,
    signal_table: pd.DataFrame | None = None,
    taxonomy_path: str | Path | None = DEFAULT_TAXONOMY_PATH,
    report_top_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_factors = list(factors) if factors is not None else get_factor_columns(df)
    conditions = build_event_conditions()
    if signal_table is None:
        signal_table = generate_signal_table(df, factors=selected_factors, conditions=conditions)

    taxonomy = _load_factor_taxonomy(taxonomy_path)
    category_map = {factor: info.get("category", "未分类") for factor, info in taxonomy.items()}
    rows: list[dict[str, Any]] = []
    for factor in selected_factors:
        factor_cache = _build_factor_event_cache_from_signal_table(df, factor, conditions, signal_table)
        rows.extend(
            _signal_point_rows_from_cache(
                factor=factor,
                factor_cache=factor_cache,
                open_conditions=conditions["open"],
                close_conditions=conditions["close"],
                category_map=category_map,
                taxonomy=taxonomy,
            )
        )

    status_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    if not status_df.empty:
        status_df = status_df.sort_values(
            ["current_state", "signal_role", "factor_category", "frequency", SIGNAL_FACTOR_COL, "open_pattern"],
            ascending=[True, True, True, True, True, True],
        ).reset_index(drop=True)

    summary_df = build_signal_point_summary(status_df)
    write_table(status_df, output_path / "signal_points_state.csv")
    write_table(summary_df, output_path / "signal_points_summary.csv")
    report = write_signal_point_report(
        status_df,
        summary_df,
        output_path / "current_signal_report.md",
        top_n=report_top_n,
    )
    return status_df, summary_df, report


def _state_direction_score(current_state: str, direction_bucket: str) -> tuple[str, float]:
    """把点位状态折算为确定性的看多/看空证据。"""
    if current_state == STATE_WAIT:
        return "中性", 0.0

    if direction_bucket == "正向":
        return ("看多", 1.0) if current_state == STATE_LONG else ("看空", -1.0)
    if direction_bucket == "正向但有拥挤风险":
        return ("看多", 0.6) if current_state == STATE_LONG else ("看空", -0.6)
    if direction_bucket == "均值回复":
        return ("看多", 0.6) if current_state == STATE_LONG else ("看空", -0.5)
    if direction_bucket == "反向/风险":
        return ("看空", -1.0) if current_state == STATE_LONG else ("风险缓和", 0.5)
    return "待确认", 0.0


def _aggregate_open_rule_backtest(rule_summary: pd.DataFrame | None) -> pd.DataFrame:
    if rule_summary is None or rule_summary.empty:
        return pd.DataFrame()

    required = {"factor", "open_condition", "excess_annual_return"}
    if not required.issubset(rule_summary.columns):
        return pd.DataFrame()

    data = rule_summary.copy()
    for col in ("excess_annual_return", "sharpe", "max_drawdown", "trade_count"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    valid = data[data["excess_annual_return"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    grouped = valid.groupby(["factor", "open_condition"], dropna=False)
    evidence = grouped.agg(
        rule_pair_count=("excess_annual_return", "size"),
        positive_excess_rule_pair_count=("excess_annual_return", lambda x: int((x > 0).sum())),
        median_excess_annual_return=("excess_annual_return", "median"),
        best_excess_annual_return=("excess_annual_return", "max"),
    ).reset_index()
    evidence["positive_excess_ratio"] = (
        evidence["positive_excess_rule_pair_count"] / evidence["rule_pair_count"]
    )

    if "sharpe" in valid.columns:
        sharpe = grouped["sharpe"].median().rename("median_sharpe").reset_index()
        evidence = evidence.merge(sharpe, on=["factor", "open_condition"], how="left")
    if "max_drawdown" in valid.columns:
        drawdown = grouped["max_drawdown"].median().rename("median_max_drawdown").reset_index()
        evidence = evidence.merge(drawdown, on=["factor", "open_condition"], how="left")
    if "trade_count" in valid.columns:
        trades = grouped["trade_count"].median().rename("median_trade_count").reset_index()
        evidence = evidence.merge(trades, on=["factor", "open_condition"], how="left")

    return evidence


def _history_multiplier(row: pd.Series) -> float:
    median_excess = row.get("median_excess_annual_return", np.nan)
    positive_ratio = row.get("positive_excess_ratio", np.nan)
    if not np.isfinite(median_excess) or not np.isfinite(positive_ratio):
        return 1.0
    if median_excess > 0 and positive_ratio >= 0.5:
        return 1.2
    if median_excess < 0 and positive_ratio < 0.4:
        return 0.8
    return 1.0


def score_signal_points_for_advisor(
    status_df: pd.DataFrame,
    rule_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scored = status_df.copy()
    if scored.empty:
        return scored

    evidence = _aggregate_open_rule_backtest(rule_summary)
    if not evidence.empty:
        scored = scored.merge(
            evidence,
            left_on=[SIGNAL_FACTOR_COL, "open_pattern"],
            right_on=["factor", "open_condition"],
            how="left",
            suffixes=("", "_history"),
        )
        scored = scored.drop(columns=[col for col in ("factor_history", "open_condition") if col in scored.columns])

    interpreted = scored.apply(
        lambda row: _state_direction_score(row["current_state"], row.get("direction_bucket", "待确认")),
        axis=1,
        result_type="expand",
    )
    scored["evidence_view"] = interpreted[0]
    scored["base_signal_score"] = interpreted[1].astype(float)
    scored["category_weight"] = scored["factor_category"].map(CATEGORY_SCORE_WEIGHT).fillna(0.5)
    scored["frequency_weight"] = scored["frequency"].map(FREQUENCY_SCORE_WEIGHT).fillna(1.0)
    scored["history_multiplier"] = scored.apply(_history_multiplier, axis=1)
    scored["point_score"] = (
        scored["base_signal_score"]
        * scored["category_weight"]
        * scored["frequency_weight"]
        * scored["history_multiplier"]
    )
    scored["score_denominator"] = np.where(
        scored["base_signal_score"].ne(0),
        scored["category_weight"] * scored["frequency_weight"] * scored["history_multiplier"],
        0.0,
    )
    return scored.replace([np.inf, -np.inf], np.nan)


def _normalized_score(scored: pd.DataFrame) -> float:
    if scored.empty or "score_denominator" not in scored:
        return np.nan
    denominator = float(scored["score_denominator"].abs().sum())
    if denominator <= 0:
        return np.nan
    return float(scored["point_score"].sum() / denominator)


def _advisor_conclusion(total_score: float, core_score: float, interpretable_ratio: float) -> str:
    if not np.isfinite(total_score) or not np.isfinite(core_score) or interpretable_ratio < 0.35:
        return "观望"
    if total_score >= 0.20 and core_score >= 0.10:
        return "偏多"
    if total_score <= -0.20 or core_score <= -0.20:
        return "降仓"
    if abs(total_score) <= 0.05:
        return "观望"
    return "中性"


def _evidence_summary(scored: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()

    counts = scored.groupby(group_cols + ["evidence_view"], dropna=False).size().unstack(fill_value=0)
    for col in ("看多", "看空", "风险缓和", "中性", "待确认"):
        if col not in counts.columns:
            counts[col] = 0

    sums = scored.groupby(group_cols, dropna=False).agg(
        point_score=("point_score", "sum"),
        score_denominator=("score_denominator", lambda x: float(np.abs(x).sum())),
        total=("evidence_view", "size"),
    )
    result = counts.join(sums).reset_index()
    result["net_score"] = np.where(
        result["score_denominator"] > 0,
        result["point_score"] / result["score_denominator"],
        np.nan,
    )
    result["主证据"] = result[["看多", "看空", "风险缓和", "中性", "待确认"]].idxmax(axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def _bullish_style_bucket(open_rule_style: Any) -> str:
    """把看多开仓风格收敛成最终 JSON 里更稳定的少数几类。"""
    text = str(open_rule_style)
    if any(token in text for token in ("抄底", "低位", "均值回复")):
        return "抄底/低位修复"
    if any(token in text for token in ("追高", "追涨", "动量", "强势突破")):
        return "追高/动量"
    if any(token in text for token in ("趋势确认", "多周期确认", "趋势反转", "修复")):
        return "趋势确认/修复"
    if any(token in text for token in ("风险过滤", "结构确认", "分歧观察", "辅助观察")):
        return "辅助确认"
    return "其他看多"


def _bearish_reason_bucket(row: pd.Series) -> str:
    """把看空点位归因到趋势转弱或拥挤逃顶等可解释类别。"""
    category = str(row.get("factor_category", ""))
    direction = str(row.get("direction_bucket", ""))
    last_pattern = str(row.get("last_signal_pattern", ""))
    style = str(row.get("open_rule_style", ""))

    if "拥挤" in direction or "风险" in direction or category.startswith("辅助/风险"):
        return "拥挤逃顶/风险预警"
    if any(token in last_pattern for token in ("上方拐点", "高位均值回复", "高位钝化转弱", "高位持续", "价格因子顶背离", "价格新高因子未确认", "价格上涨因子下降", "process_signal卖点")):
        return "拥挤逃顶/高位回落"
    if any(token in last_pattern for token in ("连续下降", "累计下降", "快速下降", "振幅放大向下", "加速度转负", "斜率转负", "下穿", "下穿均线", "空头共振", "价因子同步下降", "价格破位因子确认")):
        return "止损/趋势转弱"
    if any(token in style for token in ("追高", "追涨", "动量", "强势突破")):
        return "追高失败/动量转弱"
    if str(row.get("current_state", "")) == STATE_FLAT:
        return "空仓/规则未恢复"
    return "其他看空"


def _signal_structure_summary(scored: pd.DataFrame) -> dict[str, Any]:
    """生成最终 JSON 使用的确定性多空结构拆分。"""
    if scored.empty:
        return {
            "bullish": {"total": 0, "breakdown": []},
            "bearish": {"total": 0, "breakdown": []},
            "notes": [
                "没有可用的 signal_points_state 数据，无法拆分看多/看空信号结构。",
            ],
        }

    data = scored.copy()
    bullish = data[data["evidence_view"].isin(["看多", "风险缓和"])].copy()
    bearish = data[data["evidence_view"].eq("看空")].copy()

    if not bullish.empty:
        bullish["signal_style_bucket"] = bullish["open_rule_style"].map(_bullish_style_bucket)
        bullish_summary = _bucket_summary(bullish, "signal_style_bucket", score_abs=False)
    else:
        bullish_summary = pd.DataFrame()

    if not bearish.empty:
        bearish["bearish_reason_bucket"] = bearish.apply(_bearish_reason_bucket, axis=1)
        bearish_summary = _bucket_summary(bearish, "bearish_reason_bucket", score_abs=True)
    else:
        bearish_summary = pd.DataFrame()

    return {
        "bullish": {
            "total": int(len(bullish)),
            "breakdown": _records(bullish_summary, bullish_summary.columns.tolist(), -1),
        },
        "bearish": {
            "total": int(len(bearish)),
            "breakdown": _records(bearish_summary, bearish_summary.columns.tolist(), -1),
        },
        "notes": [
            "看多结构按开仓规则风格归类：抄底/低位修复、追高/动量、趋势确认/修复、辅助确认。",
            "看空结构按最后闭仓/风险信号归因；点位状态不包含动态止损价，因此止损与下行趋势统一记为“止损/趋势转弱”。",
        ],
    }


def _bucket_summary(data: pd.DataFrame, bucket_col: str, score_abs: bool) -> pd.DataFrame:
    if data.empty or bucket_col not in data:
        return pd.DataFrame()

    score = data["point_score"].abs() if score_abs else data["point_score"]
    working = data.assign(_bucket_score=score)
    summary = working.groupby(bucket_col, dropna=False).agg(
        count=("evidence_view", "size"),
        score_sum=("_bucket_score", "sum"),
        core_count=("signal_role", lambda x: int((x == "核心开仓").sum())),
        auxiliary_count=("signal_role", lambda x: int((x == "辅助观察").sum())),
    ).reset_index()
    total = float(summary["count"].sum())
    summary["count_share"] = np.where(total > 0, summary["count"] / total, np.nan)
    return summary.sort_values(["count", "score_sum"], ascending=[False, False]).replace([np.inf, -np.inf], np.nan)


def _records(df: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    safe_cols = [col for col in columns if col in df.columns]
    data = df[safe_cols]
    if limit >= 0:
        data = data.head(limit)
    return data.to_dict(orient="records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else str(value.date())
    if isinstance(value, np.datetime64):
        return None if pd.isna(value) else str(pd.Timestamp(value).date())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def build_advisor_summary(
    status_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rule_summary: pd.DataFrame | None = None,
    role_strategy_summary: dict[str, Any] | None = None,
    top_n: int = 30,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scored = score_signal_points_for_advisor(status_df, rule_summary=rule_summary)
    latest_date = status_df["latest_date"].max() if not status_df.empty else pd.NaT
    total_points = int(len(status_df))
    state_counts = status_df["current_state"].value_counts().to_dict() if not status_df.empty else {}
    evidence_counts = scored["evidence_view"].value_counts().to_dict() if not scored.empty else {}

    interpretable = scored[scored["base_signal_score"].ne(0)] if not scored.empty else scored
    core_scored = scored[scored["factor_category"].isin(CORE_CATEGORIES)] if not scored.empty else scored
    total_score = _normalized_score(scored)
    core_score = _normalized_score(core_scored)
    interpretable_ratio = float(len(interpretable) / total_points) if total_points else np.nan
    conclusion = _advisor_conclusion(total_score, core_score, interpretable_ratio)

    category_summary = _evidence_summary(scored, ["factor_category"])
    frequency_summary = _evidence_summary(scored, ["frequency"])
    category_frequency_summary = _evidence_summary(scored, ["factor_category", "frequency"])
    role_summary = _evidence_summary(scored, ["signal_role"])
    style_summary = _evidence_summary(scored, ["open_rule_style"])
    category_style_summary = _evidence_summary(scored, ["factor_category", "open_rule_style"])
    signal_structure = _signal_structure_summary(scored)

    top_cols = [
        SIGNAL_INSTRUMENT_COL,
        SIGNAL_FACTOR_COL,
        "frequency",
        "factor_category",
        "signal_role",
        "open_rule_style",
        "open_rule_stage",
        "default_direction",
        "open_pattern",
        "current_state",
        "evidence_view",
        "last_signal_type",
        "last_signal_pattern",
        "state_age_days",
        "expected_remaining_days",
        "net_score",
        "median_excess_annual_return",
        "positive_excess_ratio",
        "median_sharpe",
    ]
    bullish = scored[scored["evidence_view"].isin(["看多", "风险缓和"])].copy()
    if not bullish.empty:
        bullish["net_score"] = bullish["point_score"]
        bullish = bullish.sort_values(
            ["point_score", "median_excess_annual_return", "last_signal_date"],
            ascending=[False, False, False],
            na_position="last",
        )
    bearish = scored[scored["evidence_view"].eq("看空")].copy()
    if not bearish.empty:
        bearish["net_score"] = bearish["point_score"]
        bearish = bearish.sort_values(
            ["point_score", "median_excess_annual_return", "last_signal_date"],
            ascending=[True, True, False],
            na_position="last",
        )

    rule_overview: dict[str, Any] = {"available": False}
    if rule_summary is not None and not rule_summary.empty and "excess_annual_return" in rule_summary.columns:
        excess = pd.to_numeric(rule_summary["excess_annual_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rule_overview = {
            "available": True,
            "rule_pair_count": int(len(rule_summary)),
            "valid_excess_count": int(len(excess)),
            "positive_excess_count": int((excess > 0).sum()) if not excess.empty else 0,
            "positive_excess_ratio": float((excess > 0).mean()) if not excess.empty else np.nan,
            "median_excess_annual_return": float(excess.median()) if not excess.empty else np.nan,
        }

    decision_rules = {
        "score_formula": "sum(point_score) / sum(abs(score_denominator))",
        "point_score": "state_direction_score * category_weight * frequency_weight * history_multiplier",
        "conclusion_thresholds": {
            "偏多": "total_score >= 0.20 and core_score >= 0.10",
            "降仓": "total_score <= -0.20 or core_score <= -0.20",
            "观望": "interpretable_ratio < 0.35 or abs(total_score) <= 0.05",
            "中性": "其他有方向但强度不足或内部冲突的情形",
        },
        "category_score_weight": CATEGORY_SCORE_WEIGHT,
        "frequency_score_weight": FREQUENCY_SCORE_WEIGHT,
    }

    summary = {
        "latest_date": latest_date,
        "conclusion": conclusion,
        "scores": {
            "total_score": total_score,
            "core_score": core_score,
            "interpretable_ratio": interpretable_ratio,
        },
        "state_counts": {state: int(state_counts.get(state, 0)) for state in STATE_ORDER},
        "evidence_counts": {
            key: int(evidence_counts.get(key, 0))
            for key in ("看多", "看空", "风险缓和", "中性", "待确认")
        },
        "category_evidence": _records(category_summary, category_summary.columns.tolist(), -1),
        "frequency_evidence": _records(frequency_summary, frequency_summary.columns.tolist(), -1),
        "category_frequency_evidence": _records(category_frequency_summary, category_frequency_summary.columns.tolist(), -1),
        "signal_role_evidence": _records(role_summary, role_summary.columns.tolist(), -1),
        "open_rule_style_evidence": _records(style_summary, style_summary.columns.tolist(), -1),
        "category_style_evidence": _records(category_style_summary, category_style_summary.columns.tolist(), -1),
        "signal_structure": signal_structure,
        "top_bullish_points": _records(bullish, top_cols, top_n),
        "top_bearish_points": _records(bearish, top_cols, top_n),
        "rule_backtest_overview": rule_overview,
        "role_strategy": role_strategy_summary or {},
        "decision_rules": decision_rules,
    }
    return _json_safe(summary), scored


def write_advisor_summary_report(
    advisor_summary: dict[str, Any],
    scored: pd.DataFrame,
    output_dir: str | Path = "results",
    top_n: int = 30,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "advisor_summary.json"
    md_path = output_path / "advisor_summary.md"
    json_path.write_text(
        json.dumps(advisor_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    category_df = pd.DataFrame(advisor_summary.get("category_evidence", []))
    frequency_df = pd.DataFrame(advisor_summary.get("frequency_evidence", []))
    role_df = pd.DataFrame(advisor_summary.get("signal_role_evidence", []))
    style_df = pd.DataFrame(advisor_summary.get("open_rule_style_evidence", []))
    bullish_df = pd.DataFrame(advisor_summary.get("top_bullish_points", []))
    bearish_df = pd.DataFrame(advisor_summary.get("top_bearish_points", []))
    signal_structure = advisor_summary.get("signal_structure", {})
    bullish_structure_df = pd.DataFrame(signal_structure.get("bullish", {}).get("breakdown", []))
    bearish_structure_df = pd.DataFrame(signal_structure.get("bearish", {}).get("breakdown", []))

    summary_cols = [
        "factor_category",
        "frequency",
        "signal_role",
        "open_rule_style",
        "看多",
        "看空",
        "风险缓和",
        "中性",
        "待确认",
        "total",
        "net_score",
        "主证据",
    ]
    category_cols = [col for col in summary_cols if col in category_df.columns]
    frequency_cols = [col for col in summary_cols if col in frequency_df.columns]
    role_cols = [col for col in summary_cols if col in role_df.columns]
    style_cols = [col for col in summary_cols if col in style_df.columns]
    structure_cols = [
        col
        for col in (
            "signal_style_bucket",
            "bearish_reason_bucket",
            "count",
            "count_share",
            "score_sum",
            "core_count",
            "auxiliary_count",
        )
        if col in bullish_structure_df.columns or col in bearish_structure_df.columns
    ]
    bullish_structure_cols = [col for col in structure_cols if col in bullish_structure_df.columns]
    bearish_structure_cols = [col for col in structure_cols if col in bearish_structure_df.columns]
    point_cols = [
        SIGNAL_FACTOR_COL,
        "frequency",
        "factor_category",
        "signal_role",
        "open_rule_style",
        "open_rule_stage",
        "default_direction",
        "open_pattern",
        "current_state",
        "evidence_view",
        "state_age_days",
        "expected_remaining_days",
        "net_score",
        "median_excess_annual_return",
        "positive_excess_ratio",
    ]
    point_cols = [col for col in point_cols if col in bullish_df.columns or col in bearish_df.columns]
    bullish_show = bullish_df.head(top_n) if top_n >= 0 else bullish_df
    bearish_show = bearish_df.head(top_n) if top_n >= 0 else bearish_df

    scores = advisor_summary["scores"]
    states = advisor_summary["state_counts"]
    evidence = advisor_summary["evidence_counts"]
    backtest = advisor_summary["rule_backtest_overview"]
    role_strategy = advisor_summary.get("role_strategy", {})
    role_counts_df = pd.DataFrame(
        [
            {"usage_role": role, "count": count}
            for role, count in role_strategy.get("role_counts", {}).items()
        ]
    )
    term_counts_df = pd.DataFrame(
        [
            {"term_structure_label": label, "count": count}
            for label, count in role_strategy.get("term_structure_counts", {}).items()
        ]
    )
    best_template = role_strategy.get("best_template", {})
    report = "\n".join(
        [
            "# 确定性择时摘要",
            "",
            f"- 最新日期：{_format_report_value(advisor_summary.get('latest_date'))}",
            f"- 当前结论：{advisor_summary['conclusion']}",
            f"- total_score：{_format_report_value(scores.get('total_score'))}",
            f"- core_score：{_format_report_value(scores.get('core_score'))}",
            f"- 可解释信号占比：{_format_report_value(scores.get('interpretable_ratio'))}",
            f"- 状态计数：多 {states.get(STATE_LONG, 0)}，空 {states.get(STATE_FLAT, 0)}，观望 {states.get(STATE_WAIT, 0)}",
            f"- 证据计数：看多 {evidence.get('看多', 0)}，看空 {evidence.get('看空', 0)}，风险缓和 {evidence.get('风险缓和', 0)}，待确认 {evidence.get('待确认', 0)}",
            "",
            "## 最终信号结构",
            "",
            f"- 看多/风险缓和信号总数：{signal_structure.get('bullish', {}).get('total', 0)}",
            f"- 看空信号总数：{signal_structure.get('bearish', {}).get('total', 0)}",
            "",
            "### 看多结构",
            "",
            _markdown_table(bullish_structure_df, bullish_structure_cols),
            "",
            "### 看空结构",
            "",
            _markdown_table(bearish_structure_df, bearish_structure_cols),
            "",
            "## 按因子类别折算证据",
            "",
            _markdown_table(category_df, category_cols),
            "",
            "## 按周期折算证据",
            "",
            _markdown_table(frequency_df, frequency_cols),
            "",
            "## 按信号角色折算证据",
            "",
            _markdown_table(role_df, role_cols),
            "",
            "## 按开仓/辅助风格折算证据",
            "",
            _markdown_table(style_df, style_cols),
            "",
            "## 看多或风险缓和点位",
            "",
            _markdown_table(bullish_show, point_cols),
            "",
            "## 看空点位",
            "",
            _markdown_table(bearish_show, point_cols),
            "",
            "## 回测证据概览",
            "",
            f"- 是否可用：{backtest.get('available')}",
            f"- 规则组合数：{backtest.get('rule_pair_count', '')}",
            f"- 正超额组合占比：{_format_report_value(backtest.get('positive_excess_ratio'))}",
            f"- 中位超额年化收益：{_format_report_value(backtest.get('median_excess_annual_return'))}",
            "",
            "## 角色化策略证据",
            "",
            _markdown_table(role_counts_df, ["usage_role", "count"] if not role_counts_df.empty else []),
            "",
            "## 期限结构信号识别",
            "",
            _markdown_table(term_counts_df, ["term_structure_label", "count"] if not term_counts_df.empty else []),
            "",
            f"- 规则名角色与期限结构角色不一致数量：{_format_report_value(role_strategy.get('term_structure_role_changed_count'))}",
            "",
            f"- 状态分数模型：{_format_report_value(best_template.get('template_name'))}",
            f"- 当前状态信号：{_format_report_value(best_template.get('current_position'))}",
            f"- 最新开仓分：{_format_report_value(best_template.get('latest_entry_score'))}",
            f"- 最新闭仓分：{_format_report_value(best_template.get('latest_exit_score'))}",
            f"- 最新净分：{_format_report_value(best_template.get('latest_net_score'))}",
            f"- 最新风险分：{_format_report_value(best_template.get('latest_risk_score'))}",
            f"- 最新开仓占比：{_format_report_value(best_template.get('latest_entry_ratio'))}",
            f"- 最新闭仓占比：{_format_report_value(best_template.get('latest_exit_ratio'))}",
            f"- 最新方向票数：{_format_report_value(best_template.get('latest_directional_count'))}",
            f"- 最新开仓滚动阈值：{_format_report_value(best_template.get('latest_entry_ratio_rolling_threshold'))}",
            f"- 最新闭仓滚动阈值：{_format_report_value(best_template.get('latest_exit_ratio_rolling_threshold'))}",
            "",
            "## 固定判断规则",
            "",
            "- 点位先按字段默认方向折算为看多、看空、风险缓和、待确认。",
            "- 核心开仓信号才按抄底、趋势确认、追高等风格解释。",
            "- 辅助指标统一按风险过滤、结构确认、分歧观察解释，不能单独作为开仓依据。",
            "- 辅助类权重低于赔率和胜率类；季线和年线权重略高于原始因子。",
            "- 历史规则组合表现较好的开仓规则会小幅提高该点位权重，历史表现较弱则小幅降低。",
            "- 结论由 `total_score`、`core_score` 和可解释信号占比按固定阈值给出。",
            "- 这个文件是确定性事实摘要；大模型只能基于它解释原因，不应改写结论口径。",
            "",
        ]
    )
    md_path.write_text(report, encoding="utf-8")
    return report


def run_advisor_summary_report(
    status_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rule_summary: pd.DataFrame | None = None,
    role_strategy_summary: dict[str, Any] | None = None,
    output_dir: str | Path = "results",
    top_n: int = 30,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    advisor_summary, scored = build_advisor_summary(
        status_df=status_df,
        summary_df=summary_df,
        rule_summary=rule_summary,
        role_strategy_summary=role_strategy_summary,
        top_n=top_n,
    )
    report = write_advisor_summary_report(
        advisor_summary=advisor_summary,
        scored=scored,
        output_dir=output_dir,
        top_n=top_n,
    )
    return advisor_summary, scored, report




def write_current_signal_report(
    status_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 200,
) -> str:
    latest_date = pd.to_datetime(status_df["latest_date"]).max() if not status_df.empty else pd.NaT
    total = len(status_df)
    long_count = int(status_df["current_state"].eq("多").sum()) if not status_df.empty else 0
    flat_count = int(status_df["current_state"].eq("空").sum()) if not status_df.empty else 0
    pending_open = int(status_df["pending_signal"].eq("待开仓").sum()) if not status_df.empty else 0
    pending_close = int(status_df["pending_signal"].eq("待闭仓").sum()) if not status_df.empty else 0

    active = status_df[status_df["current_state"].eq("多")].copy()
    if not active.empty:
        active = active.sort_values(
            ["median_annualized_trade_return", "trade_count", "expected_remaining_days"],
            ascending=[False, False, True],
            na_position="last",
        )
    pending = status_df[status_df["pending_signal"].ne("")].copy()
    if not pending.empty:
        pending = pending.sort_values(["pending_signal", "factor", "open_condition", "close_condition"])

    if top_n >= 0:
        active_show = active.head(top_n)
        pending_show = pending.head(top_n)
    else:
        active_show = active
        pending_show = pending

    columns = [
        "factor",
        "open_condition",
        "close_condition",
        "current_state",
        "pending_signal",
        "entry_date",
        "current_holding_days",
        "historical_mean_holding_days",
        "expected_remaining_days",
        "trade_count",
        "median_trade_return",
        "median_annualized_trade_return",
        "max_drawdown",
    ]

    report = "\n".join(
        [
            "# 当前择时规则状态报告",
            "",
            f"- 最新日期：{_format_report_value(latest_date)}",
            f"- 全部规则组合：{total}",
            f"- 当前多头：{long_count}",
            f"- 当前空仓：{flat_count}",
            f"- 待开仓：{pending_open}",
            f"- 待闭仓：{pending_close}",
            "",
            "## 当前多头规则",
            "",
            _markdown_table(active_show, columns),
            "",
            "## 待执行信号",
            "",
            _markdown_table(pending_show, columns),
            "",
            "## 说明",
            "",
            "- `current_state=多` 表示按该开仓/闭仓规则组合，当前仍处于持仓状态。",
            "- `current_state=空` 表示当前没有持仓。",
            "- `pending_signal=待开仓` 表示最新交易日触发开仓信号，但按下一交易日执行口径尚未入场。",
            "- `pending_signal=待闭仓` 表示最新交易日触发闭仓信号，但按下一交易日执行口径尚未离场。",
            "- `historical_mean_holding_days` 使用已完成交易统计，不包含当前尚未结束的持仓。",
            "- `expected_remaining_days = historical_mean_holding_days - current_holding_days`，小于 0 时记为 0。",
            "- 全量明细见 `current_rule_status.csv`。",
            "",
        ]
    )
    Path(output_path).write_text(report, encoding="utf-8")
    return report


def run_current_status_report(
    df: pd.DataFrame,
    output_dir: str | Path = "results",
    factors: Iterable[str] | None = None,
    signal_table: pd.DataFrame | None = None,
    report_top_n: int = 200,
) -> tuple[pd.DataFrame, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_factors = list(factors) if factors is not None else get_factor_columns(df)
    conditions = build_event_conditions()
    if signal_table is None:
        signal_table = generate_signal_table(df, factors=selected_factors, conditions=conditions)

    rows = []
    for factor in selected_factors:
        factor_cache = _build_factor_event_cache_from_signal_table(df, factor, conditions, signal_table)
        for open_rule in conditions["open"]:
            for close_rule in conditions["close"]:
                rows.extend(_rule_status_from_cache(factor, open_rule, close_rule, factor_cache))

    status_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    status_df = status_df.sort_values(
        ["current_state", "factor", "open_condition", "close_condition"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    write_table(status_df, output_path / "current_rule_status.csv")
    report = write_current_signal_report(
        status_df,
        output_path / "current_rule_combo_report.md",
        top_n=report_top_n,
    )
    return status_df, report


