from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from data_cleaning import load_data
from io_utils import read_run_table, resolve_table_file, write_table
from timing_config import CODE_COL, DATE_COL, NAME_COL


class SafeUpdateError(RuntimeError):
    """Raised before production results are promoted."""


CRITICAL_TABLES: tuple[dict[str, Any], ...] = (
    {
        "name": "input_snapshot",
        "candidates": ["data/input_snapshot.parquet", "data/input_snapshot.csv"],
        "date_col": DATE_COL,
    },
    {
        "name": "signals",
        "candidates": ["results/signals/signals.parquet", "results/signals/signals.csv"],
        "date_col": "date",
    },
    {
        "name": "score",
        "candidates": [
            "results/score/monthly_refresh_daily_score.parquet",
            "results/score/monthly_refresh_daily_score.csv",
        ],
        "date_col": DATE_COL,
    },
    {
        "name": "selected_rule_positions",
        "candidates": [
            "results/selected_single_factor_rules/selected_rule_daily_positions.parquet",
            "results/selected_single_factor_rules/selected_rule_daily_positions.csv",
        ],
        "date_col": DATE_COL,
    },
    {
        "name": "composite_strategy_daily",
        "candidates": [
            "results/composite_timing_strategies/composite_strategy_daily.parquet",
            "results/composite_timing_strategies/composite_strategy_daily.csv",
        ],
        "date_col": DATE_COL,
    },
    {
        "name": "baseline_strategy_equity",
        "candidates": [
            "results/strategy/monthly_strategy_best_equity_default.parquet",
            "results/strategy/monthly_strategy_best_equity_default.csv",
        ],
        "date_col": DATE_COL,
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("|".join(map(str, normalized.columns)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_value) + "\n")


def _write_audit(run_dir: Path, audit: dict[str, Any]) -> None:
    report_dir = run_dir / "results" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "update_status.json"
    temporary_path = report_dir / "update_status.json.tmp"
    temporary_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, status_path)
    _append_json_line(report_dir / "update_history.jsonl", audit)


class _UpdateLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "_UpdateLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError as exc:
                detail = self.path.read_text(encoding="utf-8", errors="replace") if self.path.exists() else ""
                match = re.search(r"pid=(\d+)", detail)
                pid = int(match.group(1)) if match else None
                process_alive = True
                if pid is not None:
                    try:
                        os.kill(pid, 0)
                    except (OSError, OverflowError):
                        process_alive = False
                if attempt == 0 and pid is not None and not process_alive:
                    self.path.unlink(missing_ok=True)
                    continue
                raise SafeUpdateError(f"另一个更新任务正在运行：{self.path}\n{detail}") from exc
        if self.fd is None:
            raise SafeUpdateError(f"无法创建更新锁：{self.path}")
        os.write(self.fd, f"pid={os.getpid()} started_at={_utc_now()}\n".encode("utf-8"))
        os.close(self.fd)
        self.fd = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def build_append_only_input(
    raw: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    numeric_atol: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze the baseline history and append only strictly newer rows."""
    raw = raw.copy()
    baseline = baseline.copy()
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="raise")
    baseline[DATE_COL] = pd.to_datetime(baseline[DATE_COL], errors="raise")
    raw_columns = list(raw.columns)
    baseline_columns = list(baseline.columns)
    if set(raw_columns) != set(baseline_columns):
        added_columns = sorted(set(raw_columns).difference(baseline_columns))
        missing_columns = sorted(set(baseline_columns).difference(raw_columns))
        raise SafeUpdateError(
            "输入字段发生变化，不能做追加式更新。"
            f" added_columns={added_columns}, missing_columns={missing_columns}"
        )
    raw = raw[baseline_columns]
    keys = [CODE_COL, DATE_COL]
    if raw.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise SafeUpdateError("输入或历史快照存在重复 code/date，拒绝更新。")
    raw = raw.sort_values(keys).reset_index(drop=True)
    baseline = baseline.sort_values(keys).reset_index(drop=True)

    raw_indexed = raw.set_index(keys, drop=False)
    baseline_indexed = baseline.set_index(keys, drop=False)
    baseline_keys = baseline_indexed.index
    present_mask = baseline_keys.isin(raw_indexed.index)
    aligned_raw = raw_indexed.reindex(baseline_keys)

    value_columns = [column for column in baseline_columns if column not in keys]
    changed_masks: dict[str, pd.Series] = {}
    for column in value_columns:
        old_values = baseline_indexed[column]
        new_values = aligned_raw[column]
        if pd.api.types.is_numeric_dtype(old_values) or pd.api.types.is_numeric_dtype(new_values):
            old_numeric = pd.to_numeric(old_values, errors="coerce")
            new_numeric = pd.to_numeric(new_values, errors="coerce")
            both_missing = old_numeric.isna() & new_numeric.isna()
            equal = pd.Series(
                np.isclose(
                    old_numeric.to_numpy(dtype=float),
                    new_numeric.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=numeric_atol,
                    equal_nan=True,
                ),
                index=baseline_keys,
            )
            changed_masks[column] = (~equal & ~both_missing) & present_mask
        else:
            old_text = old_values.astype("string").fillna("<NA>")
            new_text = new_values.astype("string").fillna("<NA>")
            changed_masks[column] = old_text.ne(new_text) & present_mask

    changed_frame = pd.DataFrame(changed_masks, index=baseline_keys)
    changed_cells = int(changed_frame.to_numpy().sum()) if not changed_frame.empty else 0
    changed_rows_mask = changed_frame.any(axis=1) if not changed_frame.empty else pd.Series(False, index=baseline_keys)
    changed_rows = int(changed_rows_mask.sum())
    changed_examples: list[dict[str, Any]] = []
    if changed_cells:
        locations = np.argwhere(changed_frame.to_numpy())[:20]
        for row_pos, col_pos in locations:
            key = baseline_keys[row_pos]
            column = changed_frame.columns[col_pos]
            changed_examples.append(
                {
                    CODE_COL: _json_value(key[0]),
                    DATE_COL: _json_value(pd.Timestamp(key[1])),
                    "column": str(column),
                    "baseline": _json_value(baseline_indexed.iloc[row_pos][column]),
                    "raw": _json_value(aligned_raw.iloc[row_pos][column]),
                }
            )

    max_date_by_code = baseline.groupby(CODE_COL, sort=False)[DATE_COL].max()
    old_key_set = baseline_keys
    raw_key_index = raw_indexed.index
    not_existing = ~raw_key_index.isin(old_key_set)
    code_max = raw[CODE_COL].map(max_date_by_code)
    strictly_new = code_max.isna() | raw[DATE_COL].gt(code_max)
    added = raw.loc[not_existing & strictly_new].copy()
    ignored_backfill = raw.loc[not_existing & ~strictly_new].copy()

    merged = pd.concat([baseline, added], ignore_index=True)
    merged = merged.sort_values(keys).reset_index(drop=True)
    if merged.duplicated(keys).any():
        raise SafeUpdateError("追加合并后出现重复 code/date，拒绝更新。")

    changed_dates = baseline_indexed.index[changed_rows_mask]
    report = {
        "baseline_rows": len(baseline),
        "raw_rows": len(raw),
        "safe_rows": len(merged),
        "added_rows": len(added),
        "added_dates": sorted({str(pd.Timestamp(value).date()) for value in added[DATE_COL]}),
        "ignored_history_changed_rows": changed_rows,
        "ignored_history_changed_cells": changed_cells,
        "ignored_history_missing_rows": int((~present_mask).sum()),
        "ignored_backfill_rows": len(ignored_backfill),
        "first_changed_date": (
            str(pd.Timestamp(min(key[1] for key in changed_dates)).date()) if len(changed_dates) else None
        ),
        "last_changed_date": (
            str(pd.Timestamp(max(key[1] for key in changed_dates)).date()) if len(changed_dates) else None
        ),
        "changed_examples": changed_examples,
        "baseline_latest_date": str(pd.Timestamp(baseline[DATE_COL].max()).date()),
        "raw_latest_date": str(pd.Timestamp(raw[DATE_COL].max()).date()),
        "safe_latest_date": str(pd.Timestamp(merged[DATE_COL].max()).date()),
        "baseline_fingerprint": _frame_fingerprint(baseline),
        "safe_fingerprint": _frame_fingerprint(merged),
    }
    return merged, report


def _table(root: Path, spec: dict[str, Any]) -> pd.DataFrame:
    return read_run_table(root, spec["candidates"])


def _historical_prefix(frame: pd.DataFrame, date_col: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if date_col not in frame.columns:
        raise SafeUpdateError(f"关键表缺少日期列 {date_col}: columns={list(frame.columns)}")
    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result[result[date_col].le(cutoff)].reset_index(drop=True)
    return result


def _assert_historical_prefix_unchanged(
    production_dir: Path,
    staging_dir: Path,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for spec in CRITICAL_TABLES:
        old = _historical_prefix(_table(production_dir, spec), spec["date_col"], cutoff)
        new = _historical_prefix(_table(staging_dir, spec), spec["date_col"], cutoff)
        try:
            pd.testing.assert_frame_equal(
                old,
                new,
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
                check_like=False,
            )
        except AssertionError as exc:
            raise SafeUpdateError(f"历史前缀发生变化：{spec['name']}\n{exc}") from exc
        checks[spec["name"]] = {
            "rows": len(old),
            "fingerprint": _frame_fingerprint(old),
            "unchanged": True,
        }
    return checks


def _verify_latest_dates(staging_dir: Path, expected_latest: pd.Timestamp) -> dict[str, str]:
    latest_dates: dict[str, str] = {}
    for spec in CRITICAL_TABLES:
        frame = _table(staging_dir, spec)
        if spec["date_col"] not in frame.columns:
            raise SafeUpdateError(f"关键表缺少日期列：{spec['name']} / {spec['date_col']}")
        latest = pd.to_datetime(frame[spec["date_col"]], errors="coerce").max()
        if pd.isna(latest) or pd.Timestamp(latest) != expected_latest:
            raise SafeUpdateError(
                f"关键表日期未对齐：{spec['name']} latest={latest}, expected={expected_latest.date()}"
            )
        latest_dates[spec["name"]] = str(pd.Timestamp(latest).date())

    html_path = staging_dir / "results" / "report" / "timing_report.html"
    if not html_path.exists():
        raise SafeUpdateError(f"缺少最终 HTML：{html_path}")
    expected_text = str(expected_latest.date())
    if expected_text not in html_path.read_text(encoding="utf-8", errors="replace"):
        raise SafeUpdateError(f"HTML 未包含最新日期 {expected_text}")
    latest_dates["timing_report"] = expected_text
    return latest_dates


def _output_fingerprints(run_dir: Path) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for spec in CRITICAL_TABLES:
        path = resolve_table_file(run_dir, spec["candidates"])
        fingerprints[spec["name"]] = {
            "path": str(path.relative_to(run_dir)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    html_path = run_dir / "results" / "report" / "timing_report.html"
    fingerprints["timing_report"] = {
        "path": str(html_path.relative_to(run_dir)),
        "size": html_path.stat().st_size,
        "sha256": _sha256(html_path),
    }
    return fingerprints


def _code_fingerprints(skill_root: Path) -> dict[str, str]:
    paths = [
        skill_root / "SKILL.md",
        skill_root / "scripts" / "run_pipeline.py",
        skill_root / "runtime" / "safe_incremental_update.py",
        skill_root / "runtime" / "run_baseline_pipeline.py",
        skill_root / "runtime" / "signal_generation.py",
        skill_root / "runtime" / "role_strategy.py",
        skill_root / "runtime" / "selected_single_factor_rules.py",
        skill_root / "runtime" / "composite_timing_strategies.py",
        skill_root / "runtime" / "generate_timing_report.py",
    ]
    return {str(path.relative_to(skill_root)): _sha256(path) for path in paths if path.exists()}


def _safe_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    if child_resolved.parent != parent_resolved:
        raise SafeUpdateError(f"拒绝操作 runs 根目录之外的路径：{child_resolved}")


def _promote_staging(staging_dir: Path, production_dir: Path, session: str) -> None:
    runs_root = production_dir.parent
    previous_dir = runs_root / f".{production_dir.name}_previous_{session}"
    _safe_child(runs_root, staging_dir)
    _safe_child(runs_root, production_dir)
    _safe_child(runs_root, previous_dir)
    if previous_dir.exists():
        raise SafeUpdateError(f"临时备份目录已存在：{previous_dir}")
    os.replace(production_dir, previous_dir)
    try:
        os.replace(staging_dir, production_dir)
    except Exception:
        os.replace(previous_dir, production_dir)
        raise
    shutil.rmtree(previous_dir)


def run_safe_one_click_update(
    *,
    csv_path: str | Path,
    run_dir: str | Path,
    skill_root: str | Path,
    daily_runner: Callable[[Path, Path], dict[str, Any]],
    full_runner: Callable[[Path, Path], dict[str, Any]],
    numeric_atol: float = 1e-12,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run an append-only update in staging and promote only after verification."""
    csv_path = Path(csv_path).resolve()
    run_dir = Path(run_dir).resolve()
    skill_root = Path(skill_root).resolve()
    runs_root = run_dir.parent
    session = _session_id()
    lock_path = runs_root / ".factor_timing_update.lock"
    audit: dict[str, Any] = {
        "session_id": session,
        "started_at_utc": _utc_now(),
        "status": "running",
        "policy": "append_only_freeze_history",
        "csv_path": str(csv_path),
        "run_dir": str(run_dir),
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "code_fingerprints": _code_fingerprints(skill_root),
    }
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV：{csv_path}")

    with _UpdateLock(lock_path):
        staging_dir = runs_root / f".{run_dir.name}_staging_{session}"
        failed_dir = runs_root / f".{run_dir.name}_failed_{session}"
        _safe_child(runs_root, staging_dir)
        _safe_child(runs_root, failed_dir)
        try:
            raw = load_data(csv_path)
            snapshot_candidates = [run_dir / "data" / "input_snapshot.parquet", run_dir / "data" / "input_snapshot.csv"]
            snapshot_exists = run_dir.exists() and any(path.exists() for path in snapshot_candidates)
            if snapshot_exists:
                baseline = read_run_table(run_dir, ["data/input_snapshot.parquet", "data/input_snapshot.csv"])
                safe_input, input_diff = build_append_only_input(raw, baseline, numeric_atol=numeric_atol)
                audit["mode"] = "append_only_incremental"
                audit["input_diff"] = input_diff
            else:
                safe_input = raw.copy()
                baseline = pd.DataFrame()
                audit["mode"] = "initial_full_build"
                audit["input_diff"] = {
                    "baseline_rows": 0,
                    "raw_rows": len(raw),
                    "safe_rows": len(raw),
                    "added_rows": len(raw),
                    "added_dates": sorted({str(pd.Timestamp(value).date()) for value in raw[DATE_COL]}),
                    "ignored_history_changed_rows": 0,
                    "ignored_history_changed_cells": 0,
                    "safe_latest_date": str(pd.Timestamp(raw[DATE_COL].max()).date()),
                    "safe_fingerprint": _frame_fingerprint(raw),
                }

            audit["raw_file"] = {"size": csv_path.stat().st_size, "sha256": _sha256(csv_path)}
            if dry_run:
                audit["status"] = "dry_run_ok"
                audit["finished_at_utc"] = _utc_now()
                print(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_value))
                return audit

            if snapshot_exists and int(audit["input_diff"]["added_rows"]) == 0:
                expected_latest = pd.Timestamp(safe_input[DATE_COL].max())
                audit["latest_dates"] = _verify_latest_dates(run_dir, expected_latest)
                audit["output_fingerprints"] = _output_fingerprints(run_dir)
                audit["status"] = "no_new_rows"
                audit["finished_at_utc"] = _utc_now()
                _write_audit(run_dir, audit)
                return audit

            if staging_dir.exists():
                raise SafeUpdateError(f"暂存目录已存在：{staging_dir}")
            if snapshot_exists:
                shutil.copytree(run_dir, staging_dir, copy_function=shutil.copy2)
                history_dir = staging_dir / "data" / "history"
                history_dir.mkdir(parents=True, exist_ok=True)
                write_table(baseline, history_dir / f"input_snapshot_before_{session}.parquet")
            else:
                staging_dir.mkdir(parents=True)

            safe_input_path = staging_dir / "data" / f"_append_only_input_{session}.parquet"
            write_table(safe_input, safe_input_path)
            if snapshot_exists:
                pipeline_stats = daily_runner(safe_input_path, staging_dir)
            else:
                pipeline_stats = full_runner(safe_input_path, staging_dir)
            safe_input_path.unlink(missing_ok=True)

            expected_latest = pd.Timestamp(safe_input[DATE_COL].max())
            audit["pipeline_stats"] = pipeline_stats
            audit["latest_dates"] = _verify_latest_dates(staging_dir, expected_latest)
            audit["output_fingerprints"] = _output_fingerprints(staging_dir)
            if snapshot_exists:
                cutoff = pd.Timestamp(baseline[DATE_COL].max())
                audit["historical_prefix_checks"] = _assert_historical_prefix_unchanged(
                    run_dir,
                    staging_dir,
                    cutoff,
                )
            else:
                audit["historical_prefix_checks"] = {}
            audit["status"] = "success"
            audit["finished_at_utc"] = _utc_now()
            _write_audit(staging_dir, audit)
            if run_dir.exists():
                _promote_staging(staging_dir, run_dir, session)
            else:
                os.replace(staging_dir, run_dir)
            return audit
        except Exception as exc:
            audit["status"] = "failed"
            audit["finished_at_utc"] = _utc_now()
            audit["error"] = str(exc)
            audit["traceback"] = traceback.format_exc()
            if staging_dir.exists():
                if failed_dir.exists():
                    shutil.rmtree(failed_dir)
                os.replace(staging_dir, failed_dir)
                audit["failed_staging_dir"] = str(failed_dir)
            if run_dir.exists():
                _write_audit(run_dir, audit)
            raise
