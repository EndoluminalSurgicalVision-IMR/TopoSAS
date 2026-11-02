"""Evaluation metrics for multi-class tubular segmentation.

The definitions follow the paper:

* Dice is averaged over classes present in the ground truth **or** the
  prediction (classes absent from both are excluded; a class present only in
  the prediction scores zero).
* clDice is computed on the binarized foreground mask, i.e. globally over the
  union of GT and prediction, using 3-D Lee skeletons.
* beta0 error is the mean absolute difference between the numbers of
  26-connected components of each class in the prediction and the ground truth.

All returned scores are percentages except ``beta0``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PROTOCOL = "toposas-eval-v1"
_DEFAULT_MARGIN = 1


def _prepare_pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
    if pred.ndim != 3:
        raise ValueError("Evaluation expects 3-D label maps")
    for name, array in (("prediction", pred), ("ground truth", gt)):
        if np.issubdtype(array.dtype, np.integer):
            if np.any(array < 0):
                raise ValueError(f"{name} must contain non-negative labels")
            continue
        if np.any(~np.isfinite(array)) or not np.all(np.equal(array, np.rint(array))):
            raise ValueError(f"{name} must contain finite integer labels")
        if np.any(array < 0):
            raise ValueError(f"{name} must contain non-negative labels")
    return pred.astype(np.int32), gt.astype(np.int32)


def _resolve_n_labels(
    pred: np.ndarray, gt: np.ndarray, n_labels: int | None
) -> int:
    maximum = int(max(pred.max(initial=0), gt.max(initial=0)))
    if n_labels is None:
        return maximum
    n_labels = int(n_labels)
    if n_labels < 0:
        raise ValueError("n_labels must be non-negative")
    if maximum > n_labels:
        raise ValueError(
            f"n_labels={n_labels} is smaller than the largest label present ({maximum})"
        )
    return n_labels


def _counts(
    pred: np.ndarray, gt: np.ndarray, n_labels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = n_labels + 1
    pred_counts = np.bincount(pred.ravel(), minlength=length).astype(np.int64)
    gt_counts = np.bincount(gt.ravel(), minlength=length).astype(np.int64)
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    intersection = np.bincount(
        pred_flat[pred_flat == gt_flat], minlength=length
    ).astype(np.int64)
    present = np.flatnonzero((pred_counts + gt_counts) > 0)
    present = present[present > 0]
    return present, pred_counts, gt_counts, intersection


def _union_bbox(pred_binary: np.ndarray, gt_binary: np.ndarray, margin: int = _DEFAULT_MARGIN):
    mask = pred_binary | gt_binary
    if not mask.any():
        return None
    slices = []
    for axis in range(3):
        other_axes = tuple(a for a in range(3) if a != axis)
        along_axis = mask.any(axis=other_axes)
        indices = np.flatnonzero(along_axis)
        slices.append(
            slice(
                max(0, int(indices[0]) - margin),
                min(mask.shape[axis], int(indices[-1]) + margin + 1),
            )
        )
    return tuple(slices)


def _binary_skeleton(binary: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Evaluation requires scikit-image; install with `pip install -e '.[eval]'`"
        ) from error
    return skeletonize(binary, method="lee")


def _cl_score(skeleton: np.ndarray, volume: np.ndarray) -> float:
    total = int(np.count_nonzero(skeleton))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(skeleton & volume) / total)


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary-foreground clDice [%] using 3-D Lee skeletons."""
    pred, gt = _prepare_pair(pred, gt)
    pred_binary = pred > 0
    gt_binary = gt > 0
    box = _union_bbox(pred_binary, gt_binary)
    if box is not None:
        pred_binary = pred_binary[box]
        gt_binary = gt_binary[box]
    pred_skeleton = _binary_skeleton(pred_binary)
    gt_skeleton = _binary_skeleton(gt_binary)
    precision = _cl_score(pred_skeleton, gt_binary)
    recall = _cl_score(gt_skeleton, pred_binary)
    if precision + recall == 0:
        return 0.0
    return float(200.0 * precision * recall / (precision + recall))


def _betti0_per_class(
    pred: np.ndarray, gt: np.ndarray, present: Sequence[int]
) -> dict[int, int]:
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Evaluation requires scipy; install with `pip install -e '.[eval]'`"
        ) from error
    pred_binary = pred > 0
    gt_binary = gt > 0
    box = _union_bbox(pred_binary, gt_binary)
    if box is not None:
        pred = pred[box]
        gt = gt[box]
    structure = np.ones((3, 3, 3), dtype=bool)
    errors: dict[int, int] = {}
    for class_id in present:
        pred_components = int(ndimage.label(pred == class_id, structure=structure)[1])
        gt_components = int(ndimage.label(gt == class_id, structure=structure)[1])
        errors[int(class_id)] = abs(pred_components - gt_components)
    return errors


def dice(pred: np.ndarray, gt: np.ndarray, n_labels: int | None = None) -> float:
    """Class-average Dice [%] over classes present in ``GT union Pred``."""
    pred, gt = _prepare_pair(pred, gt)
    n_labels = _resolve_n_labels(pred, gt, n_labels)
    present, pred_counts, gt_counts, intersection = _counts(pred, gt, n_labels)
    if present.size == 0:
        return 0.0
    scores = 200.0 * intersection[present] / (pred_counts[present] + gt_counts[present])
    return float(np.mean(scores))


def betti0(pred: np.ndarray, gt: np.ndarray, n_labels: int | None = None) -> float:
    """Mean per-class 26-connectivity error over classes in ``GT union Pred``."""
    pred, gt = _prepare_pair(pred, gt)
    n_labels = _resolve_n_labels(pred, gt, n_labels)
    present, _, _, _ = _counts(pred, gt, n_labels)
    if present.size == 0:
        return 0.0
    errors = _betti0_per_class(pred, gt, present)
    return float(np.mean(list(errors.values())))


def evaluate_pair(
    pred: np.ndarray, gt: np.ndarray, n_labels: int | None = None
) -> dict:
    """Return Dice, clDice, beta0 and per-class details for one case."""
    pred, gt = _prepare_pair(pred, gt)
    n_labels = _resolve_n_labels(pred, gt, n_labels)
    present, pred_counts, gt_counts, intersection = _counts(pred, gt, n_labels)
    errors = _betti0_per_class(pred, gt, present)
    per_class = {}
    dice_scores = []
    for class_id in present:
        class_id = int(class_id)
        score = float(
            200.0 * intersection[class_id] / (pred_counts[class_id] + gt_counts[class_id])
        )
        dice_scores.append(score)
        per_class[str(class_id)] = {
            "dice": score,
            "betti0": int(errors.get(class_id, 0)),
            "gt_voxels": int(gt_counts[class_id]),
            "pred_voxels": int(pred_counts[class_id]),
        }
    return {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "clDice": cldice(pred, gt),
        "betti0": float(np.mean(list(errors.values()))) if errors else 0.0,
        "n_union": int(present.size),
        "per_class": per_class,
    }
