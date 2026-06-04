#!/usr/bin/env python
"""Validate outputs of NB6: H2B-LC_meshes_MAPseq_data.ipynb

Figure-only: MAPseq / BARseq soma plots + per-gene (Dbh/Th/Slc18a2) expression
overlays, written to /results/plots/. Presence + non-trivial size only.
"""
import sys

import _validation_utils as v

PLOT_PATTERNS = [
    "BARseq_*",
    "MAPseq_*",
    "*_expression_*",
]


def main() -> int:
    r = v.Reporter("nb6")
    v.check_glob_present(PLOT_PATTERNS, r, subdir="plots")
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
