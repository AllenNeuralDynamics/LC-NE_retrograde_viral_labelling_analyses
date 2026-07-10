"""Organize the run's figures into results/paper_figures/ and results/other_figures/.

After the notebooks run (each saving to results/plots/ via their own savefig calls, which are
left untouched), this copies the published manuscript panels into results/paper_figures/ named
by figure/panel, and copies everything else into results/other_figures/. results/plots/ (the raw
notebook output) is left in place, so nothing is lost and the README's plots/ references stay
valid.

The panel -> source mapping mirrors the "Manuscript figure panels" tables in the README. A panel
whose source file is missing is warned about, not fatal (e.g. if a notebook was skipped).
"""
import shutil
from pathlib import Path

# results root: /results on Code Ocean, else the repo's results/ (this script lives in code/).
RESULTS = Path("/results") if Path("/results").is_dir() else Path(__file__).resolve().parent.parent / "results"
PLOTS = RESULTS / "plots"
PAPER = RESULTS / "paper_figures"
OTHER = RESULTS / "other_figures"

# Published manuscript panels: source filename in plots/ -> name in paper_figures/.
# Mirrors the README "Manuscript figure panels" tables (Fig 2f, S5c/d/h, S7c/d/f/g/h).
PANELS = {
    # Figure 2f (NB 5)
    "ctx_sp_coronal_contours_bitmap_mesh.pdf": "fig2f_coronal.pdf",
    "ctx_sp_sagittal_right_contours_bitmap_mesh.pdf": "fig2f_sagittal_right.pdf",
    # Figure S5d (NB 6, BARseq)
    "BARseq_batch_coronal_plot_bitmap_mesh.pdf": "figS5d_BARseq_coronal.pdf",
    "BARseq_batch_sagittal_left_plot_bitmap_mesh.pdf": "figS5d_BARseq_sagittal_left.pdf",
    "BARseq_batch_sagittal_right_plot_bitmap_mesh.pdf": "figS5d_BARseq_sagittal_right.pdf",
    # Figure S5h (NB 6, MAPseq)
    "MAPseq_coronal_plot_bitmap_mesh.pdf": "figS5h_MAPseq_coronal.pdf",
    "MAPseq_sagittal_left_plot_bitmap_mesh.pdf": "figS5h_MAPseq_sagittal_left.pdf",
    "MAPseq_sagittal_right_plot_bitmap_mesh.pdf": "figS5h_MAPseq_sagittal_right.pdf",
    # Figure S7c/d (NB 2)
    "ipsi_contra_cell_counts_per_region_barplot.pdf": "figS7c_counts.pdf",
    "density_ML-DV_viridis_ipsi.pdf": "figS7d_density_ipsi.pdf",
    "density_ML-DV_viridis_contra.pdf": "figS7d_density_contra.pdf",
    # Figure S7f/g/h (NB 5)
    "all_retro_coronal_plot_with_histograms_bitmap_mesh.pdf": "figS7f_coronal.pdf",
    "all_retro_sagittal_left_plot_with_histograms_bitmap_mesh.pdf": "figS7g_sagittal_left.pdf",
    "all_retro_sagittal_right_plot_with_histograms_bitmap_mesh.pdf": "figS7h_sagittal_right.pdf",
}
# Figure S5c is per-gene (Dbh, Th, Slc18a2), left + right sagittal each (NB 6).
for gene in ("Dbh", "Th", "Slc18a2"):
    for side in ("left", "right"):
        PANELS[f"{gene}_expression_sagittal_{side}_bitmap_mesh.pdf"] = f"figS5c_{gene}_sagittal_{side}.pdf"


def main() -> None:
    if not PLOTS.is_dir():
        print(f"[organize_paper_figures] no {PLOTS}; nothing to organize.")
        return
    PAPER.mkdir(parents=True, exist_ok=True)
    OTHER.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, []
    for src_name, dst_name in PANELS.items():
        src = PLOTS / src_name
        if src.is_file():
            shutil.copy2(src, PAPER / dst_name)
            copied += 1
        else:
            missing.append(src_name)

    panel_sources = set(PANELS)
    other = 0
    for item in sorted(PLOTS.iterdir()):
        if item.name in panel_sources:
            continue  # panels are curated into paper_figures/ (and remain in plots/)
        dest = OTHER / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
        other += 1

    print(f"[organize_paper_figures] paper_figures/: {copied}/{len(PANELS)} panels copied; "
          f"other_figures/: {other} item(s); plots/ left as the raw output.")
    if missing:
        print(f"[organize_paper_figures] WARNING: {len(missing)} expected panel file(s) not found "
              f"in plots/ (notebook skipped?): {', '.join(missing)}")


if __name__ == "__main__":
    main()
