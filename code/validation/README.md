# Validation harness (TEMP)

Scaffolding for the open-data refactor: it proves that refactoring changes do
**not** alter the capsule's scientific output, by comparing `/results/` against a
frozen reference of the released capsule's output (playbook §4–5).

**This whole folder is temporary** and is removed in the final PR, along with the
`validate_*` lines in [`code/run`](../run) and the frozen reference asset.

## Frozen reference

The released `results/` is attached as a data asset and mounted at:

```
/data/LC-NE_retrograde_viral_labelling_analyses_frozen_v1_release_result
```

Override paths for local testing:

```bash
FROZEN_REF_DIR=/some/ref  RESULTS_DIR=/some/results  python validation/validate_nb1.py
```

## One validator per notebook

| Notebook | Validator | Files checked | Method |
|---|---|---|---|
| FINAL_1 (compile) | `validate_nb1.py` | `FINAL_manual_proofread_ccf_37brains.csv` | **CSV: strict** (tolerant value compare) |
| FINAL_2 (counts/densities) | `validate_nb2.py` | `plots/ipsi_contra_*`, `plots/density_ML-DV_*` | presence + size |
| FINAL_3 (3D k3d) | `validate_nb3.py` | `plots/somata_3d_*.html` | presence + size |
| FINAL_4 (stats/regression) | `validate_nb4.py` | `pairwise_posthoc_results.csv`, 3× `normalized_confusion_matrix_*.csv` | **CSV: strict** |
| FINAL_5 (meshes) | `validate_nb5.py` | `plots/ExA-SPIM_*`, `all_retro_*`, `retro_eXa_*`, `ctx_sp_*`, `individual_retro_*` | presence + size |
| H2B-LC (MAPseq/BARseq) | `validate_nb6.py` | `plots/BARseq_*`, `MAPseq_*`, `*_expression_*` | presence + size |

### What is and isn't validated

- **CSVs (NB1, NB4) — strict.** Parsed as data frames and compared with a float
  tolerance (`rtol=1e-5`), with per-column diagnostics on any mismatch. NB4 is
  fully seeded (`random_state=42`), so its CSVs are deterministic — but the
  confusion matrices are recomputed live by sklearn, so they are sensitive to
  the **scikit-learn version**. If NB4 FAILs after an env change, check the
  pinned sklearn version before suspecting a real regression.
- **Plots / HTML (NB2, NB3, NB5, NB6) — presence only.** Byte/pixel comparison
  of PDF/SVG/HTML is meaningless (timestamps, fonts, anti-aliasing), so we only
  confirm the expected files exist and are non-trivially sized, with a soft
  warning if a file's size differs wildly from the reference. **Only NB1 and NB4
  emit machine-checkable data.** To get true data-level validation on the figure
  notebooks, they would need to also write the plotted numbers as CSVs — a worthwhile
  future enhancement (it also helps publication-standards / figure-source provenance).
- **Skipped by design:** `output` (the run log) and the six `nbconvert` HTML
  renders (huge, embedded images, non-deterministic).

## Workflows

**Full reproducible run.** [`code/run`](../run) runs each notebook and then its
validator. A failing validator does not abort the run (you get every report);
the run exits non-zero at the end if anything diverged. Reports land in
`/results/validation/*_report.md`.

**Focused single-notebook iteration** (the fast path for short work windows).
Run from the code dir (`/root/capsule/code` in a workstation terminal):

```bash
# run just the notebook you're working on, then validate only its outputs.
# No install step: the notebooks import `ccf_utils`, which sits beside them in
# notebooks/, so a plain import resolves under nbconvert.
jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=/results notebooks/FINAL_4_...ipynb
python validation/validate_nb4.py
```

**Re-validate everything without re-running notebooks** (e.g. against an existing
`/results`):

```bash
python validation/validate_all.py
```

### Note for NB2–NB6 standalone runs

NB2–NB6 read `/results/FINAL_manual_proofread_ccf_37brains.csv` (produced by
NB1). To run one of them without first running NB1, seed it from the frozen
reference:

```bash
mkdir -p /results
cp "/data/LC-NE_retrograde_viral_labelling_analyses_frozen_v1_release_result/FINAL_manual_proofread_ccf_37brains.csv" /results/
```

Also note: NB2–NB6 instantiate `ccf_utils.CCF`, which reads the CCF atlas from
the mounted `.brainglobe` data asset — no network access or download at run time.

And remember the Code Ocean gotcha: launching an interactive terminal **wipes
`/results/`** — re-seed the NB1 CSV after doing so.
