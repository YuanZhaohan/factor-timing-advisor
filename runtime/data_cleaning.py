from __future__ import annotations

from pathlib import Path

import pandas as pd

from io_utils import read_table
from timing_config import BASE_COLS, CODE_COL, DATE_COL, NAME_COL


def load_data(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    missing = BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    numeric_cols = [c for c in df.columns if c not in {DATE_COL, CODE_COL, NAME_COL}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values([CODE_COL, DATE_COL]).reset_index(drop=True)
    duplicate_count = df.duplicated([CODE_COL, DATE_COL]).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicated code/date rows")
    return df


def get_factor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLS]


