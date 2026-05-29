from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATE_COL = "\u65e5\u671f"
CODE_COL = "\u4ee3\u7801"
NAME_COL = "\u6307\u6570\u540d\u79f0"
PRICE_COL = "\u6536\u76d8\u4ef7"
BASE_COLS = {DATE_COL, CODE_COL, NAME_COL, PRICE_COL}
DEFAULT_HORIZONS = (1, 3, 5, 10, 15, 20, 60)
TRADING_DAYS = 252
SIGNAL_DATE_COL = "date"
SIGNAL_INSTRUMENT_COL = "instrument"
SIGNAL_NAME_COL = "instrument_name"
SIGNAL_FACTOR_COL = "factor"
SIGNAL_PATTERN_COL = "pattern"
SIGNAL_VALUE_COL = "signal"
DEFAULT_TAXONOMY_PATH = Path("skills/factor-timing-advisor/references/factor_taxonomy.md")
STATE_LONG = "\u591a"
STATE_FLAT = "\u7a7a"
STATE_WAIT = "\u89c2\u671b"
STATE_ORDER = [STATE_LONG, STATE_FLAT, STATE_WAIT]
CORE_CATEGORIES = {
    "\u8d54\u7387/\u4f30\u503c",
    "\u8d54\u7387/\u7b79\u7801",
    "\u80dc\u7387/\u91cf",
    "\u80dc\u7387/\u8d44\u91d1",
}
AUXILIARY_CATEGORIES = {
    "\u8f85\u52a9/\u7b79\u7801\u7ed3\u6784",
    "\u8f85\u52a9/\u8d44\u91d1\u5206\u6b67",
    "\u8f85\u52a9/\u98ce\u9669\u72b6\u6001",
}
CATEGORY_SCORE_WEIGHT = {
    "\u8d54\u7387/\u4f30\u503c": 1.15,
    "\u8d54\u7387/\u7b79\u7801": 1.05,
    "\u80dc\u7387/\u91cf": 1.15,
    "\u80dc\u7387/\u8d44\u91d1": 1.20,
    "\u8f85\u52a9/\u7b79\u7801\u7ed3\u6784": 0.60,
    "\u8f85\u52a9/\u8d44\u91d1\u5206\u6b67": 0.70,
    "\u8f85\u52a9/\u98ce\u9669\u72b6\u6001": 0.80,
}
FREQUENCY_SCORE_WEIGHT = {
    "\u539f\u59cb": 0.80,
    "\u5b63\u7ebf": 1.00,
    "\u5e74\u7ebf": 1.10,
}


@dataclass(frozen=True)
class EventCondition:
    name: str
    side: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    requires_position: bool = False


def _format_sigma(value: float | int) -> str:
    return f"{value:g}sigma"


def _format_pct(value: float | int) -> str:
    return f"{value * 100:g}%"


def _split_factor_frequency(factor: str) -> tuple[str, str]:
    """\u628a\u5b57\u6bb5\u540d\u62c6\u6210\u57fa\u7840\u56e0\u5b50\u548c\u5468\u671f\u6807\u7b7e\u3002"""
    if factor.endswith("_\u5b63\u7ebf"):
        return factor.removesuffix("_\u5b63\u7ebf"), "\u5b63\u7ebf"
    if factor.endswith("_\u5e74\u7ebf"):
        return factor.removesuffix("_\u5e74\u7ebf"), "\u5e74\u7ebf"
    return factor, "\u539f\u59cb"
