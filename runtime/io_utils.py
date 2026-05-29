from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def table_artifact_path(path: str | Path) -> Path:
    """Use parquet for dataframe artifacts while accepting legacy .csv paths."""
    p = Path(path)
    if p.suffix.lower() in {".csv", ".pkl"}:
        return p.with_suffix(".parquet")
    return p


def table_candidates(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return [p.with_suffix(".parquet"), p.with_suffix(".pkl"), p]
    if p.suffix.lower() == ".parquet":
        return [p, p.with_suffix(".pkl"), p.with_suffix(".csv")]
    if p.suffix.lower() == ".pkl":
        return [p.with_suffix(".parquet"), p, p.with_suffix(".csv")]
    return [p]


def resolve_table_file(root: str | Path, candidates: Iterable[str | Path]) -> Path:
    base = Path(root)
    checked: list[str] = []
    for rel in candidates:
        for candidate in table_candidates(base / rel):
            checked.append(str(candidate))
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not find any table artifact under {base}: {checked}")


def read_table(path: str | Path, **csv_kwargs) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".pkl":
        return pd.read_pickle(p)
    kwargs = {"encoding": "utf-8-sig"}
    kwargs.update(csv_kwargs)
    return pd.read_csv(p, **kwargs)


def read_run_table(root: str | Path, candidates: Iterable[str | Path], **csv_kwargs) -> pd.DataFrame:
    return read_table(resolve_table_file(root, candidates), **csv_kwargs)


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    target = table_artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        try:
            df.to_parquet(target, index=index, compression="zstd")
        except Exception:
            df.to_parquet(target, index=index)
        return target
    if target.suffix.lower() == ".pkl":
        df.to_pickle(target)
        return target
    df.to_csv(target, index=index, encoding="utf-8-sig")
    return target
