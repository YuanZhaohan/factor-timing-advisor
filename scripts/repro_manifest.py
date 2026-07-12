#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_FILES = [
    "workspace/data/宽基得分.csv",
    "workspace/runs/default/data/input_snapshot.csv",
    "workspace/runs/default/data/input_snapshot.parquet",
    "workspace/runs/default/results/signals/signals.parquet",
    "workspace/runs/default/results/score/monthly_refresh_daily_score.parquet",
    "workspace/runs/default/results/selected_single_factor_rules/selected_rule_specs.csv",
    "workspace/runs/default/results/selected_single_factor_rules/selected_rule_summary.csv",
    "workspace/runs/default/results/selected_single_factor_rules/selected_rule_latest_status.csv",
    "workspace/runs/default/results/selected_single_factor_rules/selected_rule_trades.csv",
    "workspace/runs/default/results/report/current_signal_report.md",
    "workspace/runs/default/results/report/timing_report.html",
]

PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "scipy",
    "openpyxl",
]


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        return {"path": relative, "exists": False}
    stat = path.stat()
    return {
        "path": relative,
        "exists": True,
        "size": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": file_sha256(path),
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = sum(1 for _ in reader)
    return max(rows - 1, 0)


def selected_rule_metrics(root: Path) -> dict[str, Any]:
    base = root / "workspace/runs/default/results/selected_single_factor_rules"
    summary = base / "selected_rule_summary.csv"
    trades = base / "selected_rule_trades.csv"
    latest = base / "selected_rule_latest_status.csv"
    return {
        "summary_rows": csv_row_count(summary),
        "trade_rows": csv_row_count(trades),
        "latest_status_rows": csv_row_count(latest),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "cache_tag": sys.implementation.cache_tag,
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "files": [file_entry(root, relative) for relative in DEFAULT_FILES],
        "selected_rule_metrics": selected_rule_metrics(root),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(args: argparse.Namespace) -> int:
    root = args.skill_root.resolve()
    manifest = build_manifest(root)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote={output}")
    return 0


def check_manifest(args: argparse.Namespace) -> int:
    root = args.skill_root.resolve()
    expected = load_manifest(args.manifest.resolve())
    current = build_manifest(root)
    expected_files = {item["path"]: item for item in expected.get("files", [])}
    current_files = {item["path"]: item for item in current.get("files", [])}

    mismatches: list[str] = []
    for relative, expected_item in expected_files.items():
        current_item = current_files.get(relative, {"path": relative, "exists": False})
        if expected_item.get("exists") != current_item.get("exists"):
            mismatches.append(f"{relative}: exists expected={expected_item.get('exists')} current={current_item.get('exists')}")
            continue
        if not expected_item.get("exists"):
            continue
        if expected_item.get("sha256") != current_item.get("sha256"):
            mismatches.append(f"{relative}: sha256 mismatch")
        if expected_item.get("size") != current_item.get("size"):
            mismatches.append(f"{relative}: size expected={expected_item.get('size')} current={current_item.get('size')}")

    expected_metrics = expected.get("selected_rule_metrics", {})
    current_metrics = current.get("selected_rule_metrics", {})
    for key, expected_value in expected_metrics.items():
        current_value = current_metrics.get(key)
        if expected_value != current_value:
            mismatches.append(f"selected_rule_metrics.{key}: expected={expected_value} current={current_value}")

    if mismatches:
        print("repro_check=failed")
        for item in mismatches:
            print(item)
        return 1

    print("repro_check=passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write or check reproducibility fingerprints for factor timing runs.")
    parser.add_argument("--skill-root", type=Path, default=skill_root(), help="Path to skills/factor-timing-advisor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="Write the current reproducibility manifest.")
    write_parser.add_argument(
        "--output",
        type=Path,
        default=skill_root() / "workspace/runs/default/results/report/repro_manifest.json",
    )
    write_parser.set_defaults(func=write_manifest)

    check_parser = subparsers.add_parser("check", help="Check the current run against a previously written manifest.")
    check_parser.add_argument("--manifest", type=Path, required=True)
    check_parser.set_defaults(func=check_manifest)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
