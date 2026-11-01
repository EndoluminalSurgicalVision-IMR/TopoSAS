"""Skeleton-free Semantic Tubular Context Length (STCL).

STCL is a directional, physically interpretable descriptor of how much context a
training patch must cover for a multi-class tubular structure.  It is measured
directly on a semantic label map; no centreline, skeleton, or learned component
is involved.

For each physical axis ``i`` (x, y, z in millimetres)::

    M_i = V * Delta_i / R_i                    (network mean intercept)
    Sigma_i = std_{c in C}(mu_{c,i})           (equal-class centroid spread)
    S_i = M_i + (1/3) * Sigma_i                (STCL)

where ``V`` is the number of foreground voxels, ``Delta_i`` the voxel spacing,
``R_i`` the number of maximal foreground runs along axis ``i``, ``C`` the set of
semantic classes, and ``mu_{c,i}`` the centroid of class ``c`` along axis ``i``.

``M`` measures a local tubular calibre (Cavalieri-style mean intercept);
``Sigma`` measures the semantic layout of the classes.  ``S`` therefore asks a
patch to cover at least one local calibre plus one third of the class spread.
If only one class is present, ``Sigma`` reduces to the weighted standard
deviation of the foreground coordinates.  ``Sigma`` is floored at half a voxel
so that a degenerate axis remains numerically well defined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_COVERAGE = 1.0 / 3.0


@dataclass(frozen=True)
class STCLMeasurement:
    """STCL components measured on one semantic label map (physical x/y/z)."""

    case_id: str
    shape_xyz: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    foreground_voxels: int
    mean_intercept_xyz_mm: tuple[float, float, float]
    centroid_spread_xyz_mm: tuple[float, float, float]
    stcl_xyz_mm: tuple[float, float, float]
    coverage: float

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    if label.ndim != 3:
        raise ValueError(f"Expected a 3-D label map, got shape {label.shape}")
    if not np.issubdtype(label.dtype, np.integer):
        raise ValueError("Label map must have an integer dtype")
    if np.any(label < 0):
        raise ValueError("Label map must contain non-negative integer labels")
    return label


def _validate_spacing(spacing_xyz_mm: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz_mm must contain three positive finite values")
    return spacing


def _run_counts_per_axis(mask: np.ndarray) -> np.ndarray:
    """Number of maximal foreground runs along each axis."""
    counts = np.empty(3, dtype=np.int64)
    for axis in range(3):
        first = np.take(mask, 0, axis=axis)
        current = [slice(None)] * 3
        previous = [slice(None)] * 3
        current[axis] = slice(1, None)
        previous[axis] = slice(None, -1)
        starts = mask[tuple(current)] & ~mask[tuple(previous)]
        counts[axis] = int(first.sum()) + int(starts.sum())
    return counts


def _slice_counts(mask: np.ndarray, axis: int) -> np.ndarray:
    others = tuple(a for a in range(3) if a != axis)
    return np.asarray(mask.sum(axis=others), dtype=np.float64)


def _class_centroids_mm(
    label: np.ndarray, spacing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroids_mm, present_labels, class_voxel_counts)."""
    n_labels = int(label.max()) + 1
    class_voxels = np.bincount(label.ravel(), minlength=n_labels).astype(np.int64)
    present = np.flatnonzero(class_voxels[1:] > 0) + 1
    coordinate_sums = np.zeros((n_labels, 3), dtype=np.float64)
    for axis in range(3):
        n_axis = label.shape[axis]
        flat = np.moveaxis(label, axis, 0).reshape(n_axis, -1)
        counts = np.stack(
            [np.bincount(flat[t], minlength=n_labels) for t in range(n_axis)],
            axis=0,
        ).astype(np.float64)
        coordinate_sums[:, axis] = (np.arange(n_axis, dtype=np.float64)[:, None] * counts).sum(0)
    centroids_mm = coordinate_sums[present] / class_voxels[present, None] * spacing
    return centroids_mm, present, class_voxels


def _single_class_spread_mm(mask: np.ndarray, spacing: np.ndarray, n_voxels: int) -> np.ndarray:
    spread = np.empty(3, dtype=np.float64)
    for axis in range(3):
        counts = _slice_counts(mask, axis)
        coordinates = np.arange(counts.size, dtype=np.float64)
        mean = float((coordinates * counts).sum() / n_voxels)
        variance = float(((coordinates - mean) ** 2 * counts).sum() / n_voxels)
        spread[axis] = np.sqrt(max(variance, 0.0)) * spacing[axis]
    return spread


def measure_label(
    label: np.ndarray,
    spacing_xyz_mm: Sequence[float],
    case_id: str = "case",
    coverage: float = DEFAULT_COVERAGE,
) -> STCLMeasurement:
    """Measure STCL on a 3-D x/y/z integer label array."""
    label = _validate_label(label)
    spacing = _validate_spacing(spacing_xyz_mm)
    if not np.isfinite(coverage) or coverage < 0:
        raise ValueError("coverage must be a finite non-negative value")

    mask = label > 0
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise ValueError(f"{case_id!r} has no foreground voxels")

    run_counts = _run_counts_per_axis(mask)
    mean_intercept = n_voxels * spacing / run_counts

    centroids_mm, present, _ = _class_centroids_mm(label, spacing)
    if present.size > 1:
        centroid_spread = np.std(centroids_mm, axis=0)
    else:
        centroid_spread = _single_class_spread_mm(mask, spacing, n_voxels)
    centroid_spread = np.maximum(centroid_spread, spacing / 2.0)

    stcl = mean_intercept + float(coverage) * centroid_spread
    return STCLMeasurement(
        case_id=str(case_id),
        shape_xyz=tuple(int(v) for v in label.shape),
        spacing_xyz_mm=tuple(float(v) for v in spacing),
        foreground_voxels=n_voxels,
        mean_intercept_xyz_mm=tuple(float(v) for v in mean_intercept),
        centroid_spread_xyz_mm=tuple(float(v) for v in centroid_spread),
        stcl_xyz_mm=tuple(float(v) for v in stcl),
        coverage=float(coverage),
    )


def measure_nifti(path: str | Path, coverage: float = DEFAULT_COVERAGE) -> STCLMeasurement:
    """Measure STCL on a semantic NIfTI label map (``.nii`` or ``.nii.gz``)."""
    import nibabel as nib

    path = Path(path)
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    rounded = np.rint(values)
    if not np.all(np.isfinite(values)) or not np.all(np.equal(values, rounded)):
        raise ValueError(f"Non-integer labels in {path}")
    label = np.asarray(rounded, dtype=np.int32)
    case_id = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    spacing = image.header.get_zooms()[:3]
    return measure_label(label, spacing, case_id=case_id, coverage=coverage)


def aggregate_measurements(
    cases: Iterable[STCLMeasurement], coverage: float = DEFAULT_COVERAGE
) -> dict:
    """Aggregate a dataset by the per-axis median of ``M`` and ``Sigma``.

    The median is taken over cases (not over classes), after each case has
    already weighted its classes equally in ``Sigma``.  The aggregate STCL is
    then formed from the aggregate components.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("At least one case is required")
    mean_intercept = np.median(
        np.asarray([case.mean_intercept_xyz_mm for case in cases], dtype=float), axis=0
    )
    centroid_spread = np.median(
        np.asarray([case.centroid_spread_xyz_mm for case in cases], dtype=float), axis=0
    )
    stcl = mean_intercept + float(coverage) * centroid_spread

    def anisotropy(values: np.ndarray) -> float:
        return float(values.max() / values.min())

    return {
        "n_cases": len(cases),
        "coverage": float(coverage),
        "mean_intercept_xyz_mm": mean_intercept.tolist(),
        "centroid_spread_xyz_mm": centroid_spread.tolist(),
        "stcl_xyz_mm": stcl.tolist(),
        "mean_intercept_anisotropy": anisotropy(mean_intercept),
        "centroid_spread_anisotropy": anisotropy(centroid_spread),
        "stcl_anisotropy": anisotropy(stcl),
        "median_spacing_xyz_mm": np.median(
            np.asarray([case.spacing_xyz_mm for case in cases], dtype=float), axis=0
        ).tolist(),
        "median_shape_xyz": np.median(
            np.asarray([case.shape_xyz for case in cases], dtype=float), axis=0
        ).tolist(),
    }


def stcl_xyz_mm(
    label: np.ndarray,
    spacing_xyz_mm: Sequence[float],
    coverage: float = DEFAULT_COVERAGE,
) -> np.ndarray:
    """Convenience wrapper returning only the STCL vector."""
    return np.asarray(
        measure_label(label, spacing_xyz_mm, coverage=coverage).stcl_xyz_mm, dtype=float
    )
