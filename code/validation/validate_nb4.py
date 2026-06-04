#!/usr/bin/env python
"""Validate outputs of NB4: FINAL_4_manually_proofed_Ai65_stats_regression.ipynb

NB4 is fully seeded (random_state=42 throughout), so its CSVs are deterministic
and worth a strict (tolerant) data comparison:
  * pairwise_posthoc_results.csv                       (pair, p)  -- key='pair'
  * normalized_confusion_matrix_logistic_regression.csv
  * normalized_confusion_matrix_logistic_regression_with_scaling.csv
  * normalized_confusion_matrix_random_forest.csv      (matrices: index_col=0)

NOTE: the confusion matrices are recomputed live by sklearn, so they are
sensitive to the scikit-learn version. If these FAIL after an env change, check
the pinned sklearn version before suspecting a real regression.
"""
import sys

import pandas as pd

import _validation_utils as v

POSTHOC = "pairwise_posthoc_results.csv"
CONFUSION = [
    "normalized_confusion_matrix_logistic_regression.csv",
    "normalized_confusion_matrix_logistic_regression_with_scaling.csv",
    "normalized_confusion_matrix_random_forest.csv",
]


def main() -> int:
    r = v.Reporter("nb4")

    # pairwise post-hoc: row order is not meaningful -> align on the 'pair' key.
    v.compare_csv(POSTHOC, r, key="pair")

    # confusion matrices: first column is the row label.
    for name in CONFUSION:
        v.compare_csv(name, r, index_col=0)

    # Domain spot-check (informational): echo a couple of post-hoc p-values.
    out = v.RESULTS_DIR / POSTHOC
    if out.exists():
        try:
            df = pd.read_csv(out)
            if {"pair", "p"} <= set(df.columns):
                sample = df.sort_values("p").head(3)
                bits = ", ".join(f"{row.pair}: p={row.p:.3g}" for row in sample.itertuples())
                r.add("spot-check: smallest post-hoc p-values", v.PASS, bits)
        except Exception as exc:  # noqa: BLE001
            r.add("spot-check: post-hoc", v.WARN, str(exc))
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
