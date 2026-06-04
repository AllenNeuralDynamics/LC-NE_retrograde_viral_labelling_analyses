#!/usr/bin/env python
"""Validate outputs of NB5: FINAL_5_manually_proofed_AI65_meshes_plots_H2B-LC.ipynb

Figure-only: coronal/sagittal mesh-overlay plots for retro + ExA-SPIM somata,
written to /results/plots/. Presence + non-trivial size only.
"""
import sys

import _validation_utils as v

PLOT_PATTERNS = [
    "ExA-SPIM_*",
    "all_retro_*",
    "retro_eXa_*",
    "ctx_sp_*",
    "individual_retro_*",
]


def main() -> int:
    r = v.Reporter("nb5")
    v.check_glob_present(PLOT_PATTERNS, r, subdir="plots")
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
