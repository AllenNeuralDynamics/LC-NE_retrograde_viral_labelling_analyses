"""Package notebook 1's compiled soma table as a standalone AIND data asset (issue #26).

After notebook 1 writes results/FINAL_manual_proofread_ccf_37brains.csv, this assembles a
self-contained results/<ASSET_NAME>/ subfolder that can be saved as a Code Ocean data asset: the
CSV (as manual_proofread_ccf_37brains.csv), AIND data_description.json + processing.json, and a
README.md. It lets Zhixiao Su's capsule consume the 37-brain table with proper metadata,
superseding the older 18-brain LC_retro asset.

The data_description inherits its project-owned fields (project_name, investigators, funding,
license, institution, modalities) from the parent retrograde asset's mounted data_description.json
-- this table is derived from that asset -- and overrides name, data_level (derived), source_data,
creation_time, and data_summary. processing.json's Code block (capsule web URL + release version)
is introspected from the Code Ocean REST API at run time; that needs the "Code Ocean API
Credentials" secret attached (exposing API_KEY, with CO_CAPSULE_ID/CO_COMPUTATION_ID set
automatically during a run). Without it, processing.json is skipped with a warning.

Run in two phases so a credentials problem surfaces early rather than after the full ~1.5h run:
``--metadata`` writes data_description.json + processing.json + README (neither needs notebook
output, so code/run calls it FIRST); ``--csv`` copies notebook 1's compiled CSV into the asset
folder (called LAST, after notebook 1). With neither flag both phases run, for standalone/local
use. The two invocations share results/<ASSET_NAME>/ via ASSET_NAME, which code/run exports.

Requires aind-data-schema, which is installed in the capsule environment (see environment/Dockerfile).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import urllib.error
import urllib.request
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

# Provenance sources for processing.json's Code block (introspected from the Code Ocean REST API).
CO_API_BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
CO_WEB_BASE = "https://codeocean.allenneuraldynamics.org/capsule"
# Env vars the provenance introspection needs: API_KEY from the attached "Code Ocean API
# Credentials" secret; CO_CAPSULE_ID/CO_COMPUTATION_ID are set automatically during a run.
CO_API_ENV = ("API_KEY", "CO_CAPSULE_ID", "CO_COMPUTATION_ID")

# The saved asset's name (folder + data_description.name). code/run exports ASSET_NAME (dated,
# AIND convention) so its two invocations of this script -- --metadata first, --csv last -- write
# to the same results/<ASSET_NAME>/. The dated default here is only for a standalone single run;
# pin a stable name by setting ASSET_NAME before the run.
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


def fetch_co_provenance() -> tuple[str, str]:
    """Return (capsule_url, version) for the running Code Ocean capsule via the CO REST API.

    ``version`` is like ``"v3.0"`` for a released capsule, or ``"from non-release editable
    capsule"`` for an editable run. Requires API_KEY (from the attached "Code Ocean API
    Credentials" secret) plus the auto-set CO_CAPSULE_ID and CO_COMPUTATION_ID. Raises
    RuntimeError if credentials/env are missing or the API call fails.
    """
    missing = [v for v in CO_API_ENV if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing Code Ocean env vars ({' / '.join(missing)}). "
            "Attach the 'Code Ocean API Credentials' secret (Capsule Settings -> Credentials)."
        )
    auth = base64.b64encode(f"{os.environ['API_KEY']}:".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}

    def _get(path: str) -> dict:
        req = urllib.request.Request(f"{CO_API_BASE}{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    try:
        capsule = _get(f"/capsules/{os.environ['CO_CAPSULE_ID']}")
        computation = _get(f"/computations/{os.environ['CO_COMPUTATION_ID']}")
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Code Ocean API call failed: {e}") from e

    capsule_url = f"{CO_WEB_BASE}/{capsule['slug']}/tree"
    version = f"v{computation['version']}.0" if "version" in computation else "from non-release editable capsule"
    return capsule_url, version


def readme_text(asset: str) -> str:
    return (
        f"# {asset}\n\n"
        "Manually-proofread, CCF-registered locus coeruleus (LC) soma coordinates for 37\n"
        "Dbh-Cre;Ai65 retrograde-labelled brains.\n\n"
        "## Contents\n"
        f"- `{ASSET_CSV}`: one row per soma; columns `brain`, `channel`, `injection_region`,\n"
        "  `x`, `y`, `z` (CCF voxel coordinates). Only somata inside the LC CCF mesh are kept.\n"
        "- `data_description.json`, `processing.json`: AIND metadata.\n\n"
        "## Provenance\n"
        "Compiled by notebook 1 (`FINAL_1_csv_compile_df_manually_proofed_Ai65.ipynb`) of the\n"
        f"[LC-NE_retrograde_viral_labelling_analyses]({REPO_URL}) capsule, from the per-brain\n"
        f"`final_results/*_registered_pts.csv` files in the `{PARENT_ASSET}` asset (aind-open-data).\n"
    )


def write_metadata(out: Path, now: datetime) -> None:
    """Write data_description.json + processing.json + README into the asset folder.

    None of these read notebook 1's CSV, so code/run calls this FIRST: a missing "Code Ocean API
    Credentials" secret then surfaces in the first minute (processing.json skipped with a warning)
    instead of after the full run.
    """
    # data_description.json: inherit the parent asset's project-owned fields, override the rest
    parent = json.loads((DATA / PARENT_ASSET / "data_description.json").read_text())
    parent.update(
        name=ASSET_NAME,
        data_level="derived",
        source_data=[PARENT_ASSET],
        creation_time=now.isoformat(),
        data_summary=DATA_SUMMARY,
    )
    parent.pop("schema_version", None)  # let the installed aind-data-schema stamp its version
    DataDescription.model_validate(parent).write_standard_file(output_directory=str(out))

    # README (static text describing the asset)
    (out / "README.md").write_text(readme_text(ASSET_NAME))
    print(f"[build_soma_table_asset] wrote data_description.json + README.md (name: {ASSET_NAME})")

    # processing.json: capsule URL + release version from the Code Ocean API; skipped (with a
    # warning) if the "Code Ocean API Credentials" secret is not attached.
    try:
        capsule_url, version = fetch_co_provenance()
    except RuntimeError as e:
        print(f"  WARNING: skipping processing.json -- {e}")
        return
    code = Code(
        url=capsule_url,
        name="LC-NE_retrograde_viral_labelling_analyses",
        version=version,
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
    print(f"  wrote processing.json (capsule {capsule_url}, {version})")


def copy_csv(out: Path) -> None:
    """Copy notebook 1's compiled CSV into the asset folder under its asset-facing name.

    code/run calls this LAST, after notebook 1 has written results/<SOURCE_CSV>.
    """
    src = RESULTS / SOURCE_CSV
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found -- run notebook 1 first.")
    shutil.copy2(src, out / ASSET_CSV)
    print(f"[build_soma_table_asset] copied {ASSET_CSV} into {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", action="store_true",
        help="write data_description.json + processing.json + README only, and run FIRST so a "
             "missing Code Ocean credentials secret surfaces early (no notebook output needed)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="copy notebook 1's compiled CSV into the asset folder only, run LAST (after notebook "
             "1). With neither flag, both phases run (standalone/local use).",
    )
    args = parser.parse_args()
    do_meta = args.metadata or not args.csv
    do_csv = args.csv or not args.metadata

    out = RESULTS / ASSET_NAME
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    if do_meta:
        write_metadata(out, now)
    if do_csv:
        copy_csv(out)


if __name__ == "__main__":
    main()
