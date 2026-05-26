# LC-NE retrograde viral labelling analyses (Dbh-Cre;Ai65)

Code Ocean capsule for the retrograde labelling arm of the LC-NE manuscript.
Brains are *Dbh-Cre;Ai65* mice that received retrograde, Cre-dependent FlpO
injections into one of several target regions and were imaged through the
SmartSPIM pipeline. This capsule starts from per-brain CCF-registered soma
coordinates that have been manually proofread, and ends with the count,
density, 3D-position, statistical, and mesh-overlay figures used in the paper.

---

## 1. Pipeline overview

The capsule runs six notebooks in order (see [run](run)), each rendered to
HTML in `/results`:

| # | Notebook | Purpose |Additonal notes
|---|----------|---------|
| 1 | [FINAL_1_csv_compile_df_manually_proofed_Ai65.ipynb](FINAL_1_csv_compile_df_manually_proofed_Ai65.ipynb) | Compile per-brain manually-proofread CCF coordinate CSVs into a single dataframe | Input data is extracted from outputs of the LC-NE_Register_Annotations_retrograde_cells capsule |
| 2 | [FINAL_2_manually_proofed_Ai65_plot_counts_densities.ipynb](FINAL_2_manually_proofed_Ai65_plot_counts_densities.ipynb) | Per-region / per-hemisphere LC-NE cell counts and density plots |
| 3 | [FINAL_3_manually_proofed_Ai65_3Dplot_somata_positions_byROI.ipynb](FINAL_3_manually_proofed_Ai65_3Dplot_somata_positions_byROI.ipynb) | Interactive 3D `k3d` plots of soma positions in CCF, coloured by injection ROI | Currently NOT USED for the figures included in the manuscript|
| 4 | [FINAL_4_manually_proofed_Ai65_stats_regression.ipynb](FINAL_4_manually_proofed_Ai65_stats_regression.ipynb) | PCA, LDA, logistic-regression / random-forest / Gaussian-process classifiers, pairwise post-hoc statistics | Currently NOT USED for the figures included in the manuscript|
| 5 | [FINAL_5_manually_proofed_AI65_meshes_plots_H2B-LC.ipynb](FINAL_5_manually_proofed_AI65_meshes_plots_H2B-LC.ipynb) | Coronal/sagittal projections of retro somata and ExA-SPIM-reconstructed somata over LC percentile-density meshes | Input data is extracted from outputs of the ExA-SPIM processing capsule |
| 6 | [H2B-LC_meshes_MAPseq_data.ipynb](H2B-LC_meshes_MAPseq_data.ipynb) | Same mesh framework applied to MAPseq and BARseq soma coordinates, plus per-gene (Dbh / Th / Slc18a2) expression overlays | Input data is extracted from outputs of the LC-NE_BARseq_MAPseq_analyses capsule |

The reproducible run is driven by [run](run):

```bash
cd /code && pip install -e .
jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=../results notebooks/<each notebook>.ipynb
```

## 2. Input data assets

| Mount path | Used by | Contents |
|------------|---------|----------|
| `/data/Dbh-CreAi65_retrograde_soma_manual_proofread_10-14-25/final_results/{brain_id}_registered_pts.csv` | NB 1 | Per-brain manually-proofread, CCF-registered soma coordinates. Columns are loaded as `x, y, z, Location`; only rows with `Location == "inside"` are kept which denotes somata segmented within the PONS CCF mesh, rather than spurios cells detected elsewhere in the specimen. |
| `/data/ccf_2017_obj/` | NB 2–6 (via `vta.utils.CCF` / `CCFMesh`) | Allen CCF 2017 region meshes |
| `/data/LC_percentile_meshes_1/percentile_{10..90}.obj`, `new_core_mesh.obj` | NB 5, 6 | LC density-percentile shell meshes generated from Dbh-Cre;H2B-GFP animals and defining anatomically and functionally correct LC location within the CCF|
| `/data/LC_reconstruction_somas/LC reconstruction soma location and top projection.csv` | NB 5 | ExA-SPIM single-neuron reconstruction soma locations organized by top projection target |
| `/data/PK_MAPseqBARseq_LC-NEpaper_files/MAPseq_combined_cell_top_projections_with_coords.csv` | NB 6 | MAPseq cell coordinates + top projection labels |
| `/data/PK_MAPseqBARseq_LC-NEpaper_files/fromLCNE_combined_LCcluster_neurons_BARseq.csv` | NB 6 | BARseq cell coordinates + cluster labels |
| `/data/PK_MAPseqBARseq_LC-NEpaper_files/Dbh_Th_Slc18a2_logcounts_adj_Dbh.csv` | NB 6 | Per-cell log-counts for `Dbh`, `Th`, `Slc18a2` used for gene-expression overlays |
| `/data/manually_proofed_Ai65_classifier/gp_classification_results.pkl` | NB 4 | Pre-trained Gaussian-process classifier results; if present it is loaded, otherwise NB 4 retrains and writes a new one |

## 3. Outputs (`/results/`)

Written by the notebooks themselves:

- `FINAL_manual_proofread_ccf_37brains.csv` — unified soma table (NB 1).
- `pairwise_posthoc_results.csv` — pairwise post-hoc statistical comparisons (NB 4).
- `plots/` — PDF + SVG figures from NB 2, 5, 6 (counts/density bar plots,
  coronal/sagittal mesh overlays for retro, ExA-SPIM, MAPseq, BARseq, and
  per-gene expression plots) and the random-forest confusion-matrix figures
  from NB 4.
- One HTML rendering per notebook (produced by `jupyter nbconvert` in
  [run](run)).

## 4. Coordinate and laterality conventions

- Per-brain CSV columns `x, y, z` are renamed to `ML, DV, RC` (CCF voxel
  indices) in notebooks 2–6.
- Notebook 4 converts voxel indices to microns by multiplying `ML, DV, RC` by
  `ccf.resolution` from `vta.utils.CCF`.
- Hemisphere assignment uses an approximate CCF midline of `ML = 230` voxels:
  `ipsi = ML > 230`. The cut-off is hard-coded.

## 5 Analyses performed in notebook 4, which is not used for current LC-NE manuscript version

- PCA on ipsilateral soma `RC, DV, ML` coordinates.
- Linear Discriminant Analysis on injection-region labels.
- Classifiers compared on the same feature set:
  - Logistic regression
  - Random forest
  - Gaussian-process classifier (RBF kernel) — loaded from the
    `gp_classification_results.pkl` asset if available, otherwise retrained
    and re-pickled.
- Stratified k-fold cross-validation, classification reports, and normalized
  confusion-matrix figures.
- Pairwise post-hoc statistical tests, saved to `pairwise_posthoc_results.csv`.


## 6. Environment

Built from [Dockerfile](Dockerfile) on
`codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04`. Pinned scientific
stack: `allensdk==2.16.2`, `antspyx==0.4.2`, `dask==2023.5.0`,
`opencv-python-headless==4.8.0.74`, `zarr==2.14.2`. Additional (unpinned):
`pandas`, `scikit-learn`, `seaborn`, `rich`, `tqdm`, `timebudget`, `ruff`,
`k3d`, `trimesh`, `rtree`, `openpyxl`, `matplotlib-venn`.

The local package **`vta`** (providing `vta.utils.CCF` and `vta.utils.CCFMesh`)
is installed in editable mode by [run](run):

```bash
cd /code
pip install -e .
```

VS Code / code-server set-up for the cloud workstation (extensions, Copilot
VSIX, etc.) is handled by [postInstall](postInstall).



### Notes

 - [Tutorial for ccf interaction](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/reference_space.html)

 ### Links
 GitHub: https://github.com/AllenNeuralDynamics/LC-NE_retrograde_viral_labelling_analyses
 CodeOcean: TBD