#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("references") / "reproducibility_manifest.json"
DEFAULT_FILES = [
    Path("workspace") / "data" / "\u5bbd\u57fa\u5f97\u5206.csv",
    Path("workspace") / "runs" / "default" / "data" / "input_snapshot.parquet",
    Path("workspace") / "runs" / "default" / "results" / "signals" / "signals.parquet",
    Path("workspace")
    / "runs"
    / "default"
    / "results"
    / "selected_single_factor_rules"
    / "selected_rule_summary.csv",
    Path("workspace")
    / "runs"
    / "default"
    / "results"
    / "selected_single_factor_rules"
    / "selected_rule_latest_status.csv",
    Path("workspace")
    / "runs"
    / "default"
    / "results"
    / "report"
    / "current_signal_report.md",
]
PACKAGE_NAMES = ["numpy", "pandas", "pyarrow", "matplotlib", "plotly"]


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not-installed"
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def collect_manifest(root: Path, files: list[Path]) -> dict[str, Any]:
    records = []
    for relative_path in files:
        path = root / relative_path
        if path.exists():
            records.append(
                {
                    "path": relative_path.as_posix(),
                    "exists": True,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            records.append(
                {
                    "path": relative_path.as_posix(),
                    "exists": False,
                    "size": None,
                    "sha256": None,
                }
            )

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "files": records,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def verify_manifest(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_packages = expected.get("packages", {})
    actual_packages = actual.get("packages", {})
    for name, expected_version in expected_packages.items():
        actual_version = actual_packages.get(name)
        if actual_version != expected_version:
            errors.append(f"package mismatch: {name}: expected {expected_version}, got {actual_version}")

    actual_files = {item["path"]: item for item in actual.get("files", [])}
    for expected_file in expected.get("files", []):
        path = expected_file["path"]
        actual_file = actual_files.get(path)
        if actual_file is None:
            errors.append(f"missing file check record: {path}")
            continue
        if actual_file["exists"] != expected_file["exists"]:
            errors.append(f"file existence mismatch: {path}")
            continue
        if not expected_file["exists"]:
            continue
        if actual_file["size"] != expected_file["size"]:
            errors.append(
                f"file size mismatch: {path}: expected {expected_file['size']}, got {actual_file['size']}"
            )
        if actual_file["sha256"] != expected_file["sha256"]:
            errors.append(f"file hash mismatch: {path}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local reproducibility inputs and outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("write", "Write a reproducibility manifest from the current local files."),
        ("verify", "Verify current local files against an existing manifest."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--manifest",
            type=Path,
            default=DEFAULT_MANIFEST,
            help="Manifest path relative to the skill root unless absolute.",
        )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = skill_root()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    actual = collect_manifest(root, DEFAULT_FILES)

    if args.command == "write":
        write_manifest(manifest_path, actual)
        print(f"wrote={manifest_path}")
        return 0

    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = verify_manifest(expected, actual)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("reproducibility_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
