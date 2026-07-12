#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_entry() -> tuple[Path, Path]:
    skill_root = Path(__file__).resolve().parents[1]
    local_entry = skill_root / "runtime" / "run_baseline_pipeline.py"
    if local_entry.exists():
        return skill_root, local_entry

    seen: set[Path] = set()
    roots: list[Path] = []
    for start in [Path.cwd(), skill_root]:
        for path in [start, *start.parents]:
            if path in seen:
                continue
            seen.add(path)
            roots.append(path)
    for root in roots:
        entry = root / "run_baseline_pipeline.py"
        if entry.exists():
            return root, entry
    raise SystemExit(
        "Could not find run_baseline_pipeline.py. "
        "Expected either skills/factor-timing-advisor/runtime/run_baseline_pipeline.py "
        "or a repository root containing run_baseline_pipeline.py."
    )


def main() -> int:
    work_root, entry = find_entry()
    skill_root = Path(__file__).resolve().parents[1]
    local_entry = skill_root / "runtime" / "run_baseline_pipeline.py"
    if entry.resolve() == local_entry.resolve() and skill_root.parent.name == "skills":
        work_root = skill_root.parents[1]
    python_exe = os.environ.get("OPENCLAW_PYTHON") or sys.executable
    cmd = [python_exe, str(entry), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(work_root))


if __name__ == "__main__":
    raise SystemExit(main())
