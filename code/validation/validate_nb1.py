#!/usr/bin/env python
"""Validate outputs of NB1: FINAL_1_csv_compile_df_manually_proofed_Ai65.ipynb

NB1 compiles the per-brain proofread CCF coordinate CSVs into one table.
Only data output: /results/FINAL_manual_proofread_ccf_37brains.csv
"""
import sys

import pandas as pd

import _validation_utils as v

CSV = "FINAL_manual_proofread_ccf_37brains.csv"


def main() -> int:
    r = v.Reporter("nb1")
    v.compare_csv(CSV, r)  # full-table tolerant compare

    # Domain spot-check (informational): row count + #brains + region spread.
    out = v.RESULTS_DIR / CSV
    if out.exists():
        try:
            df = pd.read_csv(out)
            bits = [f"{len(df)} rows"]
            if "brain" in df.columns:
                bits.append(f"{df['brain'].nunique()} brains")
            if "injection_region" in df.columns:
                bits.append(f"regions={sorted(df['injection_region'].unique())}")
            r.add("spot-check: compiled table", v.PASS, ", ".join(bits))
        except Exception as exc:  # noqa: BLE001
            r.add("spot-check: compiled table", v.WARN, str(exc))
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
