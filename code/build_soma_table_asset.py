"""Package notebook 1's compiled soma table as a standalone AIND data asset (issue #26).

After notebook 1 writes results/FINAL_manual_proofread_ccf_37brains.csv, this assembles a
self-contained results/<ASSET_NAME>/ subfolder that can be saved as a Code Ocean data asset: the
CSV (as manual_proofread_ccf_37brains.csv), AIND data_description.json + processing.json, and a
README.md. It lets Zhixiao Su's capsule consume the 37-brain table with proper metadata,
superseding the older 18-brain LC_retro asset.

The data_description inherits its project-owned fields (project_name, investigators, funding,
license, institution, modalities) from the parent retrograde asset's mounted data_description.json
-- this table is derived from that asset -- and overrides name, data_level (derived), source_data,
creation_time, and data_summary. processing.json records this capsule's git commit as provenance,
so no Code Ocean API credentials are needed.

Requires aind-data-schema, which is installed in the capsule environment (see environment/Dockerfile).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema_models.process_names import ProcessName

UTC = timezone.utc
HERE = Path(__file__).resolve().parent

# Parent asset (mounted) this table is compiled from; its metadata is inherited.
PARENT_ASSET = "LC-NE-Register-Annotations-retrograde-cells_2026-07-08_18-58-48"
SOURCE_CSV = "FINAL_manual_proofread_ccf_37brains.csv"   # what notebook 1 writes to results/
ASSET_CSV = "manual_proofread_ccf_37brains.csv"          # name inside the standalone asset
REPO_URL = "https://github.com/AllenNeuralDynamics/LC-NE_retrograde_viral_labelling_analyses"

# The saved asset's name (folder + data_description.name). Dated by default so each run mints a
# distinct asset; pin a stable name by setting ASSET_NAME before the run.
ASSET_NAME = os.environ.get("ASSET_NAME") or f"LC-NE_retrograde_soma_table_{datetime.now(UTC):%Y-%m-%d_%H-%M-%S}"

# /results and /data on Code Ocean; the repo-local folders otherwise.
RESULTS = (Path("/results") if Path("/results").is_dir() else HERE.parent / "results")
DATA = (Path("/data") if Path("/data").is_dir() else HERE.parent / "data")

DATA_SUMMARY = (
    "Manually-proofread, CCF-registered locus coeruleus soma coordinates for 37 Dbh-Cre;Ai65 "
    "retrograde-labelled brains, compiled by notebook 1 of this capsule from the per-brain files "
    f"in the {PARENT_ASSET} asset. One row per soma; columns: brain, channel, injection_region, "
    "and x/y/z (CCF voxel coordinates). Only somata inside the pons LC CCF mesh are kept."
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def readme_text(asset: str) -> str:
    return (
        f"# {asset}\n\n"
        "Manually-proofread, CCF-registered locus coeruleus (LC) soma coordinates for 37\n"
        "Dbh-Cre;Ai65 retrograde-labelled brains.\n\n"
        "## Contents\n"
        f"- `{ASSET_CSV}`: one row per soma; columns `brain`, `channel`, `injection_region`,\n"
        "  `x`, `y`, `z` (CCF voxel coordinates). Only somata inside the pons LC CCF mesh are kept.\n"
        "- `data_description.json`, `processing.json`: AIND metadata.\n\n"
        "## Provenance\n"
        "Compiled by notebook 1 (`FINAL_1_csv_compile_df_manually_proofed_Ai65.ipynb`) of the\n"
        f"[LC-NE_retrograde_viral_labelling_analyses]({REPO_URL}) capsule, from the per-brain\n"
        f"`final_results/*_registered_pts.csv` files in the `{PARENT_ASSET}` asset (aind-open-data).\n"
    )


def main() -> None:
    out = RESULTS / ASSET_NAME
    out.mkdir(parents=True, exist_ok=True)

    # 1. the CSV, under its asset-facing name
    src = RESULTS / SOURCE_CSV
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found -- run notebook 1 first.")
    shutil.copy2(src, out / ASSET_CSV)

    # 2. data_description.json: inherit the parent asset's project-owned fields, override the rest
    now = datetime.now(UTC)
    parent = json.loads((DATA / PARENT_ASSET / "data_description.json").read_text())
    parent.update(
        name=ASSET_NAME,
        data_level="derived",
        source_data=[PARENT_ASSET],
        creation_time=now.isoformat(),
        data_summary=DATA_SUMMARY,
    )
    parent.pop("schema_version", None)  # let the installed aind-data-schema stamp its version
    dd = DataDescription.model_validate(parent)
    dd.write_standard_file(output_directory=str(out))

    # 3. processing.json: this capsule's compilation step, with git-commit provenance
    code = Code(
        url=REPO_URL,
        name="LC-NE_retrograde_viral_labelling_analyses",
        commit_hash=git_commit(),
        run_script=Path("code/run"),
        language="Python",
        input_data=[DataAsset(name=PARENT_ASSET)],
    )
    process = DataProcess(
        process_type=ProcessName.ANALYSIS,
        name="Compile manually-proofread CCF soma coordinates",
        stage=ProcessStage.ANALYSIS,
        code=code,
        experimenters=["Polina Kosillo"],
        start_date_time=now,
        end_date_time=now,
        notes="Notebook 1 compiles the per-brain manually-proofread CCF soma CSVs into a single "
              "table (inside-LC somata only).",
    )
    Processing(data_processes=[process]).write_standard_file(output_directory=str(out))

    # 4. README
    (out / "README.md").write_text(readme_text(ASSET_NAME))

    print(f"[build_soma_table_asset] wrote {out}: {ASSET_CSV} + data_description.json + "
          f"processing.json + README.md (name: {ASSET_NAME})")


if __name__ == "__main__":
    main()
