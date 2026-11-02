#!/usr/bin/env python3
"""Apply a TopoSAS patch size to an nnU-Net plan and start training.

This is a thin operational wrapper, not a library API.  It changes only
``configurations[<config>]["patch_size"]``, keeps a baseline copy of the plan,
and then invokes ``nnUNetv2_train``.  Use ``--dry-run`` to update the plan and
print the training command without starting a run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_plans(dataset: str, plans: str | None) -> Path:
    if plans is not None:
        return Path(plans)
    preprocessed = os.environ.get("nnUNet_preprocessed")
    if not preprocessed:
        try:
            from nnunetv2.paths import nnUNet_preprocessed as preprocessed  # type: ignore
        except Exception as error:  # pragma: no cover - depends on nnU-Net install
            raise SystemExit(
                "Cannot locate nnUNet_preprocessed; set the environment variable "
                "or pass --plans /path/to/nnUNetPlans.json"
            ) from error
    try:
        from nnunetv2.utilities.dataset_name_id_conversion import (  # type: ignore
            maybe_convert_to_dataset_name,
        )

        dataset_name = maybe_convert_to_dataset_name(dataset)
    except Exception as error:  # pragma: no cover - depends on nnU-Net install
        if str(dataset).isdigit():
            raise SystemExit(
                "Install nnU-Net or pass --plans so the dataset name can be resolved"
            ) from error
        dataset_name = str(dataset)
    return Path(preprocessed) / dataset_name / "nnUNetPlans.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _backup_path(plans_path: Path) -> Path:
    stem = plans_path.name[:-5] if plans_path.name.endswith(".json") else plans_path.name
    return plans_path.with_name(stem + ".baseline.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="nnU-Net dataset id or name")
    parser.add_argument("--config", default="3d_fullres", help="nnU-Net configuration")
    parser.add_argument(
        "--patch-size",
        required=True,
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        help="Patch size in the plan's axis order",
    )
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", type=Path, help="Explicit path to nnUNetPlans.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Update the plan and print the command without starting training",
    )
    args = parser.parse_args(argv)

    patch_size = [int(value) for value in args.patch_size]
    if any(value <= 0 for value in patch_size):
        raise SystemExit("--patch-size values must be positive")

    plans_path = _resolve_plans(args.dataset, args.plans)
    if not plans_path.is_file():
        raise SystemExit(f"Plan not found: {plans_path}")
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    configurations = plans.get("configurations", {})
    if args.config not in configurations:
        raise SystemExit(
            f"Configuration {args.config!r} not found in {plans_path}; "
            f"available: {sorted(configurations)}"
        )

    backup = _backup_path(plans_path)
    if backup.exists():
        print(f"Backup already exists, keeping it: {backup}")
    else:
        shutil.copy2(plans_path, backup)
        print(f"Backup written: {backup}")

    previous = configurations[args.config].get("patch_size")
    configurations[args.config]["patch_size"] = patch_size
    _write_json_atomic(plans_path, plans)
    print(f"{args.config}: patch_size {previous} -> {patch_size}")

    plans_name = str(plans.get("plans_name", "nnUNetPlans"))
    command = [
        "nnUNetv2_train",
        str(args.dataset),
        args.config,
        str(args.fold),
        "-tr",
        args.trainer,
    ]
    if plans_name != "nnUNetPlans":
        command.extend(["-p", plans_name])
    print("Training command:", " ".join(command))
    if args.dry_run:
        return 0

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise SystemExit(
            "nnUNetv2_train was not found; activate an nnU-Net environment first"
        ) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    return 0


if __name__ == "__main__":
    sys.exit(main())
