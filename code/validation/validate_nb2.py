#!/usr/bin/env python
"""Validate outputs of NB2: FINAL_2_manually_proofed_Ai65_plot_counts_densities.ipynb

NB2 is figure-only (no data files). We confirm its plots are present and
non-trivial in /results/plots; byte/pixel comparison of PDF/SVG is meaningless
(playbook §5).
"""
import sys

import _validation_utils as v

PLOT_PATTERNS = [
    "ipsi_contra_cell_counts_per_region_barplot.*",
    "density_ML-DV_*",
]


def main() -> int:
    r = v.Reporter("nb2")
    v.check_glob_present(PLOT_PATTERNS, r, subdir="plots")
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
