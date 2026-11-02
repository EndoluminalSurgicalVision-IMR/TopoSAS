"""Directory-level evaluation for multi-class tubular segmentation."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from .metrics import PROTOCOL, evaluate_pair

_NIFTI_SUFFIXES = (".nii.gz", ".nii")


def _strip_nifti_suffix(name: str) -> str:
    for suffix in _NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _nifti_files(folder: str | Path) -> dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    files: dict[str, Path] = {}
    for suffix in _NIFTI_SUFFIXES:
        for path in sorted(folder.glob(f"*{suffix}")):
            files.setdefault(_strip_nifti_suffix(path.name), path)
    return files


def _load_label(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    if np.issubdtype(values.dtype, np.integer):
        label = np.asarray(values, dtype=np.int32)
    else:
        rounded = np.rint(values)
        if not np.all(np.isfinite(values)) or not np.all(np.equal(values, rounded)):
            raise ValueError(f"Non-integer labels in {path}")
        label = np.asarray(rounded, dtype=np.int32)
    if np.any(label < 0):
        raise ValueError(f"Negative labels in {path}")
    return label, np.asarray(image.affine, dtype=float)


def evaluate_case(
    pred_path: str | Path,
    gt_path: str | Path,
    n_labels: int | None = None,
    check_affine: bool = True,
) -> dict:
    """Evaluate one prediction / ground-truth NIfTI pair."""
    pred, pred_affine = _load_label(pred_path)
    gt, gt_affine = _load_label(gt_path)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch for {_strip_nifti_suffix(Path(pred_path).name)}: "
            f"{pred.shape} vs {gt.shape}"
        )
    if check_affine and not np.allclose(pred_affine, gt_affine, atol=1e-4):
        raise ValueError(
            f"Affine mismatch for {_strip_nifti_suffix(Path(pred_path).name)}; "
            "pass check_affine=False if this is intentional"
        )
    # Nibabel exposes arrays as x,y,z; the reference evaluation scripts used
    # SimpleITK, which exposes the same volumes as z,y,x.  Dice and beta0 are
    # invariant to this permutation, but Lee skeletonization is not exactly
    # rotation-invariant, so clDice must use the same array orientation.
    pred = np.transpose(pred, (2, 1, 0))
    gt = np.transpose(gt, (2, 1, 0))
    result = evaluate_pair(pred, gt, n_labels=n_labels)
    result["case"] = _strip_nifti_suffix(Path(pred_path).name)
    return result


def evaluate_directories(
    pred_dir: str | Path,
    gt_dir: str | Path,
    n_labels: int | None = None,
    workers: int = 1,
    check_affine: bool = True,
) -> dict:
    """Evaluate every matching case in two NIfTI directories."""
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    pred_files = _nifti_files(pred_dir)
    gt_files = _nifti_files(gt_dir)
    if not pred_files:
        raise FileNotFoundError(f"No NIfTI predictions in {pred_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No NIfTI ground truth in {gt_dir}")
    missing_pred = sorted(set(gt_files) - set(pred_files))
    missing_gt = sorted(set(pred_files) - set(gt_files))
    if missing_pred or missing_gt:
        details = []
        if missing_pred:
            details.append(f"missing predictions: {missing_pred[:5]}")
        if missing_gt:
            details.append(f"missing ground truth: {missing_gt[:5]}")
        raise ValueError("Case set mismatch (" + "; ".join(details) + ")")

    cases = sorted(set(pred_files) & set(gt_files))
    evaluator = partial(
        evaluate_case, n_labels=n_labels, check_affine=check_affine
    )
    if int(workers) == 1:
        records = [evaluator(pred_files[case], gt_files[case]) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            records = list(
                pool.map(
                    evaluator,
                    [pred_files[case] for case in cases],
                    [gt_files[case] for case in cases],
                )
            )

    summary = {}
    for key in ("dice", "clDice", "betti0"):
        values = np.asarray([record[key] for record in records], dtype=float)
        summary[key] = float(values.mean())
        summary[key + "_std"] = float(values.std(ddof=0))
    return {
        "protocol": PROTOCOL,
        "pred_dir": str(Path(pred_dir).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "labels": int(n_labels) if n_labels is not None else None,
        "num_cases": len(records),
        "summary": summary,
        "cases": records,
    }


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
