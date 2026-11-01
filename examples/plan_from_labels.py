#!/usr/bin/env python3
"""Measure STCL on a folder of training labels and print fixed-budget patches.

Example
-------
python examples/plan_from_labels.py \
    --labels /data/Airway-TNN/labelsTr \
    --order xyz \
    --spacing-mm 0.78125 0.78125 0.5 \
    --divisibility 32 16 32 \
    --budgets 96 128 160
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toposas.patch import fit_patches
from toposas.stcl import aggregate_measurements, measure_nifti


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Folder of .nii/.nii.gz labels")
    parser.add_argument("--order", required=True, choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    parser.add_argument("--spacing-mm", required=True, nargs=3, type=float)
    parser.add_argument("--divisibility", required=True, nargs=3, type=int)
    parser.add_argument("--budgets", nargs="+", type=int, default=[96, 128, 160])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.labels.glob("*.nii.gz")) + sorted(args.labels.glob("*.nii"))
    if not files:
        raise SystemExit(f"No NIfTI files found in {args.labels}")
    cases = [measure_nifti(path) for path in files]
    aggregate = aggregate_measurements(cases)
    patches = fit_patches(
        aggregate["stcl_xyz_mm"],
        args.order,
        args.spacing_mm,
        args.divisibility,
        budgets=args.budgets,
    )
    report = {
        "labels": str(args.labels.resolve()),
        "aggregate": aggregate,
        "patches": {str(budget): allocation.to_dict() for budget, allocation in patches.items()},
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
