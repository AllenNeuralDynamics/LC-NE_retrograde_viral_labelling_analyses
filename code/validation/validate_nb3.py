#!/usr/bin/env python
"""Validate outputs of NB3: FINAL_3_manually_proofed_Ai65_3Dplot_somata_positions_byROI.ipynb

NB3 is figure-only: interactive k3d HTML plots written to /results/plots/.
Presence + non-trivial size only.
"""
import sys

import _validation_utils as v

PLOT_PATTERNS = ["somata_3d_*.html"]


def main() -> int:
    r = v.Reporter("nb3")
    v.check_glob_present(PLOT_PATTERNS, r, subdir="plots")
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
