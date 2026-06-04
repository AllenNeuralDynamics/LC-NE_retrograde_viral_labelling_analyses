"""Shared helpers for the per-notebook validation scripts.

TEMP (validation harness): this whole `code/validation/` folder is scaffolding
for the open-data refactor and is removed in the final PR (playbook §5).

Each `validate_nbN.py` compares the files its notebook writes into `/results/`
against the frozen reference asset, and prints + writes a markdown report.

Design (playbook §5):
  * Lead with a one-line verdict, then per-file detail only where it matters.
  * CSVs are parsed and compared as data (tolerant float compare), with
    per-column drilldowns when something differs.
  * PDF/SVG/HTML are NOT byte-compared (timestamps, fonts, anti-aliasing make it
    useless) -- only presence + non-trivial size are checked.
  * CCF download artifacts (annotation/, manifest.json, structures.json) and the
    run log (`output`) are skipped by design -- they are allensdk byproducts /
    bookkeeping, not scientific results.

Paths are overridable via env vars so the scripts can be pointed at local
fixtures for testing:
    FROZEN_REF_DIR   default /data/LC-NE_retrograde_viral_labelling_analyses_frozen_v1_release_result
    RESULTS_DIR      default /results
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = Path(
    os.environ.get(
        "FROZEN_REF_DIR",
        "/data/LC-NE_retrograde_viral_labelling_analyses_frozen_v1_release_result",
    )
)
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
REPORT_DIR = RESULTS_DIR / "validation"

# Default tolerances for numeric CSV comparison. rtol carries values that span
# many orders of magnitude (e.g. p-values down to 1e-27); atol is tiny so that
# small-but-nonzero values are still compared relatively.
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-12

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_SYMBOL = {PASS: "✅", FAIL: "❌", WARN: "⚠️", SKIP: "➖"}


class Check:
    def __init__(self, name: str, status: str, detail: str = ""):
        self.name = name
        self.status = status
        self.detail = detail


class Reporter:
    """Collects Checks for one notebook, prints + writes a markdown report."""

    def __init__(self, notebook: str):
        self.notebook = notebook
        self.checks: list[Check] = []

    def add(self, name: str, status: str, detail: str = "") -> Check:
        c = Check(name, status, detail)
        self.checks.append(c)
        return c

    @property
    def ok(self) -> bool:
        # WARN and SKIP do not fail the run; only FAIL does.
        return all(c.status != FAIL for c in self.checks)

    def render(self) -> str:
        n_fail = sum(c.status == FAIL for c in self.checks)
        n_warn = sum(c.status == WARN for c in self.checks)
        verdict = (
            "Behavior preserved" if n_fail == 0 else f"{n_fail} check(s) FAILED"
        )
        lines = [
            f"# Validation report -- {self.notebook}",
            "",
            f"**Verdict: {verdict}**"
            + (f" ({n_warn} warning(s))" if n_warn else ""),
            "",
            f"Reference: `{REF_DIR}`",
            f"Results:   `{RESULTS_DIR}`",
            "",
            "| Status | Check | Detail |",
            "| --- | --- | --- |",
        ]
        for c in self.checks:
            detail = (c.detail or "").replace("\n", "<br>")
            lines.append(f"| {_SYMBOL[c.status]} {c.status} | {c.name} | {detail} |")
        return "\n".join(lines) + "\n"

    def finish(self) -> int:
        """Print the report, write it to REPORT_DIR, return an exit code."""
        report = self.render()
        print(report)
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / f"{self.notebook}_report.md").write_text(report)
        except OSError as exc:  # never let report-writing mask the result
            print(f"(could not write report file: {exc})")
        return 0 if self.ok else 1


# --------------------------------------------------------------------------- #
# CSV comparison
# --------------------------------------------------------------------------- #
def _read_csv(path: Path, index_col):
    return pd.read_csv(path, index_col=index_col)


def compare_csv(
    rel_path: str,
    reporter: Reporter,
    *,
    index_col=None,
    key: str | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Compare RESULTS_DIR/rel_path against REF_DIR/rel_path as data frames.

    index_col : passed to pandas (use 0 for matrices whose first column is row
                labels, e.g. confusion matrices).
    key       : if given, both frames are sorted by this column before
                comparison so row order does not matter.
    """
    name = rel_path
    ref_path = REF_DIR / rel_path
    out_path = RESULTS_DIR / rel_path

    if not out_path.exists():
        reporter.add(name, FAIL, f"output missing: {out_path}")
        return
    if not ref_path.exists():
        reporter.add(name, SKIP, f"no reference file at {ref_path}")
        return

    try:
        ref = _read_csv(ref_path, index_col)
        out = _read_csv(out_path, index_col)
    except Exception as exc:  # noqa: BLE001 - report any parse error
        reporter.add(name, FAIL, f"could not parse: {exc}")
        return

    # Column set / order
    if list(ref.columns) != list(out.columns):
        missing = [c for c in ref.columns if c not in out.columns]
        extra = [c for c in out.columns if c not in ref.columns]
        detail = []
        if missing:
            detail.append(f"missing cols: {missing}")
        if extra:
            detail.append(f"extra cols: {extra}")
        if not detail:
            detail.append(
                f"column ORDER differs: ref={list(ref.columns)} out={list(out.columns)}"
            )
        reporter.add(name, FAIL, "; ".join(detail))
        return

    # Row count
    if len(ref) != len(out):
        reporter.add(name, FAIL, f"row count differs: ref={len(ref)} out={len(out)}")
        return

    # Optional key-based alignment (row order independence)
    if key is not None and key in ref.columns:
        ref = ref.sort_values(key).reset_index(drop=True)
        out = out.sort_values(key).reset_index(drop=True)
    elif index_col is not None:
        # align matrices by their row/col labels
        if sorted(map(str, ref.index)) != sorted(map(str, out.index)):
            reporter.add(name, FAIL, f"index labels differ: ref={list(ref.index)} out={list(out.index)}")
            return
        out = out.reindex(index=ref.index)

    # Per-column comparison
    col_problems: list[str] = []
    for col in ref.columns:
        a, b = ref[col], out[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            close = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
            n_bad = int((~close).sum())
            if n_bad:
                diff = np.abs(av - bv)
                diff[np.isnan(diff)] = 0.0
                imax = int(np.nanargmax(diff))
                col_problems.append(
                    f"`{col}`: {n_bad}/{len(av)} differ "
                    f"(max |Δ|={diff.max():.3g} at row {imax}: "
                    f"ref={av[imax]:.6g} out={bv[imax]:.6g})"
                )
        else:
            eq = (a.astype("object") == b.astype("object")) | (a.isna() & b.isna())
            n_bad = int((~eq).sum())
            if n_bad:
                first = int(np.argmax((~eq).to_numpy()))
                col_problems.append(
                    f"`{col}`: {n_bad}/{len(a)} differ "
                    f"(first row {first}: ref={a.iloc[first]!r} out={b.iloc[first]!r})"
                )

    if col_problems:
        reporter.add(name, FAIL, "; ".join(col_problems))
    else:
        reporter.add(
            name, PASS, f"{len(ref)} rows x {len(ref.columns)} cols match (rtol={rtol:g})"
        )


# --------------------------------------------------------------------------- #
# File-presence checks (plots / html)
# --------------------------------------------------------------------------- #
def check_present(rel_path: str, reporter: Reporter, *, min_bytes: int = 1024) -> None:
    """Confirm RESULTS_DIR/rel_path exists and is non-trivial; compare size to ref."""
    out_path = RESULTS_DIR / rel_path
    ref_path = REF_DIR / rel_path
    if not out_path.exists():
        reporter.add(rel_path, FAIL, f"output missing: {out_path}")
        return
    size = out_path.stat().st_size
    if size < min_bytes:
        reporter.add(rel_path, FAIL, f"suspiciously small: {size} B (< {min_bytes} B)")
        return
    detail = f"present, {size:,} B"
    status = PASS
    if ref_path.exists():
        ref_size = ref_path.stat().st_size
        # Soft signal only: a wildly different size hints at a broken figure.
        if ref_size and not (0.5 <= size / ref_size <= 2.0):
            status = WARN
            detail += f" (ref {ref_size:,} B -- size ratio {size / ref_size:.2f})"
    reporter.add(rel_path, status, detail)


def check_glob_present(
    patterns: list[str], reporter: Reporter, *, subdir: str = "plots", min_bytes: int = 1024
) -> None:
    """For each glob pattern, find files in the reference's `subdir` and confirm
    each also exists in RESULTS_DIR/`subdir`. Flags missing and extra files.
    """
    ref_sub = REF_DIR / subdir
    out_sub = RESULTS_DIR / subdir
    expected: set[str] = set()
    for pat in patterns:
        expected.update(p.name for p in ref_sub.glob(pat))
    if not expected:
        reporter.add(
            f"{subdir}/ [{', '.join(patterns)}]",
            WARN,
            f"no reference files matched in {ref_sub}",
        )
        return
    for fname in sorted(expected):
        check_present(f"{subdir}/{fname}", reporter, min_bytes=min_bytes)
    # Extra files this notebook produced that the reference lacks (informational).
    if out_sub.exists():
        produced = set()
        for pat in patterns:
            produced.update(p.name for p in out_sub.glob(pat))
        extra = sorted(produced - expected)
        if extra:
            reporter.add(f"{subdir}/ extra files", WARN, ", ".join(extra))
