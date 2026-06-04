#!/usr/bin/env python
"""Inspect the GP-classifier pickle and extract its predictions to a flat CSV.

Background (see RECON.md §2): NB4 currently loads a large, version-fragile
pickle (`gp_classification_results.pkl`) from a Code Ocean data asset. The model
is never re-invoked downstream -- only its precomputed outputs are used. This
one-time tool reads that pickle and writes a lightweight, transparent CSV of
those outputs, which is then committed into the repo and read by NB4 instead of
the pickle. The pickle/asset can then be retired.

Run this ON THE CAPSULE (the pickle is mounted under /data). It prints the
pickle's structure first (so we can confirm what's in it), then writes the CSV.

    python code/scripts/extract_gp_predictions.py            # inspect + write
    python code/scripts/extract_gp_predictions.py --inspect-only

Defaults assume capsule paths; override with --pickle / --out.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PICKLE = "/data/manually_proofed_Ai65_classifier/gp_classification_results.pkl"
# Default output is the committed location, resolved relative to this script
# (code/scripts/ -> code/data/) so it works regardless of cwd / capsule mount.
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "data" / "gp_classification_predictions.csv")

# Keys we expect in the results dict (from NB4's retrain branch that creates it).
EXPECTED_KEYS = {
    "gp_model", "label_encoder", "y_pred_classes", "y_pred_proba", "y_pred",
    "sigma", "coordinates", "target", "category_mapping",
}


def describe(obj, name: str = "results") -> None:
    """Print a structural summary of the unpickled object."""
    print(f"\n{'=' * 70}\nPICKLE STRUCTURE\n{'=' * 70}")
    print(f"top-level type: {type(obj)}")
    if not isinstance(obj, dict):
        print("(not a dict -- inspect manually)")
        return
    print(f"keys ({len(obj)}): {sorted(obj)}")
    missing = EXPECTED_KEYS - set(obj)
    extra = set(obj) - EXPECTED_KEYS
    if missing:
        print(f"  !! expected keys MISSING: {sorted(missing)}")
    if extra:
        print(f"  -- unexpected extra keys: {sorted(extra)}")
    print("-" * 70)
    for k, val in obj.items():
        line = f"{k:>18} : {type(val).__name__}"
        if isinstance(val, np.ndarray):
            line += f"  shape={val.shape} dtype={val.dtype}"
            if val.size:
                sample = val.ravel()[:3]
                line += f"  e.g. {sample}"
        elif isinstance(val, (list, tuple)):
            line += f"  len={len(val)}  e.g. {list(val)[:3]}"
        elif isinstance(val, dict):
            line += f"  len={len(val)}  e.g. {dict(list(val.items())[:3])}"
        elif hasattr(val, "classes_"):
            line += f"  classes_={list(getattr(val, 'classes_'))}"
        else:
            line += f"  value={val!r}"[:80]
        print(line)


def class_order(results: dict, n_proba_cols: int) -> list[str]:
    """Determine the class label order for predict_proba columns.

    sklearn orders predict_proba columns by the estimator's classes_, so prefer
    the model; fall back through label_encoder, category_mapping, then target.
    """
    for src in ("gp_model", "label_encoder"):
        est = results.get(src)
        if est is not None and hasattr(est, "classes_"):
            order = [str(c) for c in est.classes_]
            if len(order) == n_proba_cols:
                return order
    cm = results.get("category_mapping")
    if isinstance(cm, dict) and len(cm) == n_proba_cols:
        return [str(cm[i]) for i in sorted(cm)]
    target = results.get("target")
    if target is not None:
        order = sorted({str(t) for t in np.asarray(target)})
        if len(order) == n_proba_cols:
            return order
    raise ValueError(
        f"could not determine {n_proba_cols} class labels from the pickle"
    )


def build_table(results: dict) -> pd.DataFrame:
    y_pred_classes = np.asarray(results["y_pred_classes"])
    y_pred_proba = np.asarray(results["y_pred_proba"], dtype=float)
    sigma = np.asarray(results["sigma"], dtype=float)
    coords = np.asarray(results["coordinates"], dtype=float)
    target = np.asarray(results["target"])

    n = len(y_pred_classes)
    if not (y_pred_proba.shape[0] == sigma.shape[0] == coords.shape[0] == len(target) == n):
        raise ValueError(
            f"length mismatch: y_pred_classes={n} proba={y_pred_proba.shape} "
            f"sigma={sigma.shape} coords={coords.shape} target={len(target)}"
        )
    if coords.shape[1] != 3:
        raise ValueError(f"expected 3 coordinate columns (RC,DV,ML), got {coords.shape[1]}")

    labels = class_order(results, y_pred_proba.shape[1])

    df = pd.DataFrame(
        {
            "RC": coords[:, 0],
            "DV": coords[:, 1],
            "ML": coords[:, 2],
            "true_region": target.astype(str),
            "predicted_region": y_pred_classes.astype(str),
            "entropy": sigma,
        }
    )
    for j, lab in enumerate(labels):
        df[f"p_{lab}"] = y_pred_proba[:, j]
    return df, labels


def verify(df: pd.DataFrame, labels: list[str]) -> None:
    """Sanity checks; print PASS/WARN, never abort the extraction."""
    print(f"\n{'=' * 70}\nVERIFICATION\n{'=' * 70}")
    proba = df[[f"p_{l}" for l in labels]].to_numpy()

    sums = proba.sum(axis=1)
    ok_sum = np.allclose(sums, 1.0, atol=1e-6)
    print(f"  proba rows sum to 1: {'PASS' if ok_sum else 'WARN'} "
          f"(min={sums.min():.6f} max={sums.max():.6f})")

    recomputed = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    ok_ent = np.allclose(recomputed, df["entropy"].to_numpy(), atol=1e-6)
    print(f"  entropy == -Σ p·log(p+1e-10): {'PASS' if ok_ent else 'WARN'} "
          f"(max|Δ|={np.abs(recomputed - df['entropy']).max():.3g})")

    argmax_lab = np.array(labels)[proba.argmax(axis=1)]
    agree = (argmax_lab == df["predicted_region"].to_numpy()).mean()
    print(f"  argmax(proba) == predicted_region: {agree:.4%} "
          f"({'PASS' if agree > 0.999 else 'note: GPC predict can differ from proba-argmax near ties'})")

    print(f"  rows={len(df)}  classes={labels}")
    print(f"  predicted_region counts:\n{df['predicted_region'].value_counts().to_string()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pickle", default=DEFAULT_PICKLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args(argv)

    pkl = Path(args.pickle)
    if not pkl.exists():
        print(f"ERROR: pickle not found at {pkl}", file=sys.stderr)
        return 2

    # The released pickle was made under a different sklearn; show the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        with open(pkl, "rb") as f:
            results = pickle.load(f)

    describe(results)

    if not isinstance(results, dict):
        print("\nCannot auto-extract: top-level object is not a dict.", file=sys.stderr)
        return 1

    try:
        df, labels = build_table(results)
    except Exception as exc:  # noqa: BLE001 - inspection above already printed
        print(f"\nEXTRACTION FAILED: {exc}", file=sys.stderr)
        return 1

    verify(df, labels)
    print(f"\nfirst rows:\n{df.head().to_string()}")

    if args.inspect_only:
        print("\n--inspect-only: not writing CSV.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWROTE {len(df)} rows x {len(df.columns)} cols -> {out} "
          f"({out.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
