#!/usr/bin/env python
"""Run all six per-notebook validators and report an aggregate verdict.

Used by the reproducible run (see code/run). Runs each validate_nbN.py as a
subprocess so one failure cannot stop the others -- you get every report, and
this script exits non-zero if ANY notebook failed validation.

For focused work, run a single validator directly instead, e.g.:
    python validation/validate_nb4.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATORS = [f"validate_nb{i}.py" for i in range(1, 7)]


def main() -> int:
    results: dict[str, int] = {}
    for name in VALIDATORS:
        print(f"\n{'=' * 78}\nRunning {name}\n{'=' * 78}")
        proc = subprocess.run([sys.executable, str(HERE / name)])
        results[name] = proc.returncode

    print(f"\n{'=' * 78}\nVALIDATION SUMMARY\n{'=' * 78}")
    for name, code in results.items():
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {name}")
    n_fail = sum(c != 0 for c in results.values())
    print(f"\n{n_fail} of {len(results)} notebook(s) failed validation.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
