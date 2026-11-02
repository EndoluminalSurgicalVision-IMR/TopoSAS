"""Command-line interface for the TopoSAS STCL patch planner."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Sequence

from . import __version__
from .evaluation import evaluate_directories, write_report
from .patch import fit_patch
from .presets import PRESETS, reproduce_all, reproduce_preset
from .stcl import aggregate_measurements, measure_nifti


def _parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/labelsTr")
    return name, Path(path)


def _write_json(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _cmd_stcl(args: argparse.Namespace) -> int:
    report: dict = {
        "schema_version": 1,
        "descriptor": "semantic_tubular_context_length",
        "coverage": args.coverage,
        "uses_skeleton": False,
        "datasets": {},
    }
    for name, folder in args.dataset:
        files = sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))
        if not files:
            raise FileNotFoundError(f"No NIfTI label maps found in {folder}")
        measure = partial(measure_nifti, coverage=args.coverage)
        if args.workers == 1:
            cases = [measure(path) for path in files]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                cases = list(pool.map(measure, files))
        aggregate = aggregate_measurements(cases, coverage=args.coverage)
        report["datasets"][name] = {
            "label_folder": str(folder.resolve()),
            "aggregate": aggregate,
            "cases": [case.to_dict() for case in cases],
        }
        print(
            f"{name}: n={aggregate['n_cases']} "
            f"STCL(mm)={[round(v, 3) for v in aggregate['stcl_xyz_mm']]} "
            f"anisotropy={aggregate['stcl_anisotropy']:.3f}"
        )
    _write_json(args.output, report)
    if args.output is None:
        print(json.dumps(report, indent=2))
    return 0


def _cmd_patch(args: argparse.Namespace) -> int:
    if args.preset:
        dataset = PRESETS[args.preset]
        stcl = dataset.stcl_xyz_mm
        order = "xyz"
        spacing = dataset.spacing_xyz_mm
        divisibility = dataset.divisibility_xyz
    else:
        missing = [
            name
            for name, value in (
                ("--stcl-mm", args.stcl_mm),
                ("--order", args.order),
                ("--spacing-mm", args.spacing_mm),
                ("--divisibility", args.divisibility),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
        stcl = args.stcl_mm
        order = args.order
        spacing = args.spacing_mm
        divisibility = args.divisibility

    budgets = args.budget or [128]
    result = {}
    for budget in budgets:
        allocation = fit_patch(
            stcl,
            order,
            spacing,
            divisibility,
            int(budget),
            window=args.window,
        )
        result[str(budget)] = allocation.to_dict()
        print(
            f"N={budget}: patch={allocation.patch} "
            f"volume_ratio={allocation.volume_ratio:.4f} "
            f"objective={allocation.objective:.4f}"
        )
    _write_json(args.output, result)
    if args.output is None:
        print(json.dumps(result, indent=2))
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    if args.preset:
        report = {"datasets": {args.preset: reproduce_preset(args.preset)}}
        datasets = report["datasets"]
        report["n_cells"] = sum(item["n_cells"] for item in datasets.values())
        report["n_match"] = sum(item["n_match"] for item in datasets.values())
        report["all_match"] = report["n_match"] == report["n_cells"]
    else:
        report = reproduce_all()

    for name, item in report["datasets"].items():
        print(f"\n{name}")
        for budget, cell in item["cells"].items():
            status = "OK" if cell["match"] else "MISMATCH"
            print(
                f"  N={budget:>3} patch={cell['patch']} "
                f"published={cell['published']}  {status}"
            )
    print(
        f"\n{report['n_match']}/{report['n_cells']} "
        "published N=128 patches exact"
    )
    _write_json(args.output, report)
    return 0 if report["all_match"] else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_directories(
        args.pred_dir,
        args.gt_dir,
        n_labels=args.labels,
        workers=args.workers,
        check_affine=not args.ignore_affine,
    )
    summary = report["summary"]
    print(
        f"cases={report['num_cases']} "
        f"Dice={summary['dice']:.4f}±{summary['dice_std']:.4f} "
        f"clDice={summary['clDice']:.4f}±{summary['clDice_std']:.4f} "
        f"beta0={summary['betti0']:.4f}±{summary['betti0_std']:.4f}"
    )
    if args.output is not None:
        write_report(report, args.output)
    else:
        print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toposas",
        description="STCL measurement, fixed-budget patch planning and evaluation for TopoSAS.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stcl = subparsers.add_parser(
        "stcl", help="Measure STCL from semantic NIfTI label maps."
    )
    stcl.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=_parse_dataset,
        metavar="NAME=DIR",
        help="Named folder of .nii/.nii.gz label maps; repeatable.",
    )
    stcl.add_argument("--output", type=Path, help="Write the JSON report here.")
    stcl.add_argument("--coverage", type=float, default=1.0 / 3.0)
    stcl.add_argument("--workers", type=int, default=1)
    stcl.set_defaults(func=_cmd_stcl)

    patch = subparsers.add_parser(
        "patch", help="Compute a fixed-budget patch shape from STCL."
    )
    patch.add_argument("--preset", choices=sorted(PRESETS), help="Use a paper preset.")
    patch.add_argument("--stcl-mm", nargs=3, type=float, metavar=("X", "Y", "Z"))
    patch.add_argument("--order", choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    patch.add_argument("--spacing-mm", nargs=3, type=float, metavar=("A", "B", "C"))
    patch.add_argument("--divisibility", nargs=3, type=int, metavar=("A", "B", "C"))
    patch.add_argument("--budget", nargs="+", type=int, help="Nominal N (default: 128).")
    patch.add_argument("--window", type=float, default=0.05)
    patch.add_argument("--output", type=Path)
    patch.set_defaults(func=_cmd_patch)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate prediction masks with Dice, clDice and beta0."
    )
    evaluate.add_argument("--pred-dir", required=True, type=Path)
    evaluate.add_argument("--gt-dir", required=True, type=Path)
    evaluate.add_argument(
        "--labels",
        type=int,
        default=None,
        help="Number of foreground classes (default: infer from the masks).",
    )
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument(
        "--ignore-affine",
        action="store_true",
        help="Do not require prediction and ground truth affines to match.",
    )
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(func=_cmd_evaluate)

    reproduce = subparsers.add_parser(
        "reproduce", help="Check the published N=128 patches."
    )
    reproduce.add_argument("--preset", choices=sorted(PRESETS))
    reproduce.add_argument("--output", type=Path)
    reproduce.set_defaults(func=_cmd_reproduce)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"toposas: error: {error}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
