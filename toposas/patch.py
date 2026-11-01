"""Fixed-budget patch-shape allocation from a directional STCL indicator.

The patch law keeps the nominal voxel budget ``N**3`` and the plan spacing fixed,
and only redistributes that budget across axes::

    B_N = N**3 * prod_i Delta_i                     (physical volume budget)
    G   = (S_x S_y S_z) ** (1/3)                    (geometric mean STCL)
    v_i = B_N**(1/3) / Delta_i * (S_i / G)

The continuous target is discretised on the stride-compatible lattice
``p_i = m_i * d_i``::

    | prod_i p_i / N**3 - 1 | <= window

by minimising the pairwise log-ratio objective used in the paper::

    sum_{i<j} [ log(p_i / p_j) - log(v_i / v_j) ] ** 2

which is equivalent (up to a constant factor) to the centred-log-ratio form.
Ties are broken by the volume error and then by the lexicographic multiplier
order in plan-axis order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
DEFAULT_WINDOW = 0.05
DEFAULT_MAX_MULTIPLIER = 40


@dataclass(frozen=True)
class PatchAllocation:
    """One fixed-budget patch allocated from an STCL indicator."""

    budget: int
    order: str
    stcl_xyz_mm: tuple[float, float, float]
    stcl_plan_mm: tuple[float, float, float]
    continuous_target_plan_mm: tuple[float, float, float]
    continuous_target_plan_voxels: tuple[float, float, float]
    patch: tuple[int, int, int]
    physical_size_plan_mm: tuple[float, float, float]
    volume_ratio: float
    objective: float

    def to_dict(self) -> dict:
        return asdict(self)


def plan_permutation(order: str) -> tuple[int, int, int]:
    """Return the physical-axis indices in the given plan order.

    ``order`` names the physical axes in the order used by the plan, e.g.
    ``"zyx"`` means the plan tuple is ``(z, y, x)``.
    """
    if not isinstance(order, str) or len(order) != 3 or sorted(order) != ["x", "y", "z"]:
        raise ValueError("order must be a permutation of 'x', 'y', 'z'")
    return tuple(_AXIS_INDEX[axis] for axis in order)


def reorder_xyz_to_plan(values_xyz: Sequence[float], order: str) -> np.ndarray:
    """Reorder physical x/y/z values into plan order."""
    values = np.asarray(values_xyz, dtype=float)
    if values.shape != (3,):
        raise ValueError("values_xyz must contain exactly three values")
    return values[list(plan_permutation(order))]


def reorder_plan_to_xyz(values_plan: Sequence[float], order: str) -> np.ndarray:
    """Inverse of :func:`reorder_xyz_to_plan`."""
    values = np.asarray(values_plan, dtype=float)
    if values.shape != (3,):
        raise ValueError("values_plan must contain exactly three values")
    permutation = plan_permutation(order)
    result = np.empty(3, dtype=float)
    result[list(permutation)] = values
    return result


def pairwise_log_ratio_error(candidate: Sequence[float], target: Sequence[float]) -> float:
    """Pairwise log-ratio objective from the paper (lower is better)."""
    candidate = np.asarray(candidate, dtype=float)
    target = np.asarray(target, dtype=float)
    if candidate.shape != (3,) or target.shape != (3,):
        raise ValueError("candidate and target must each contain three values")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(target)):
        raise ValueError("candidate and target must be finite")
    if np.any(candidate <= 0) or np.any(target <= 0):
        raise ValueError("candidate and target must be positive")
    candidate_ratios = np.log(candidate[:, None] / candidate[None, :])
    target_ratios = np.log(target[:, None] / target[None, :])
    upper = np.triu_indices(3, k=1)
    return float(np.sum((candidate_ratios[upper] - target_ratios[upper]) ** 2))


def _validate_common(
    stcl_xyz_mm: Sequence[float],
    spacing_plan_mm: Sequence[float],
    divisibility: Sequence[int],
    budget: int,
    window: float,
    max_multiplier: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stcl = np.asarray(stcl_xyz_mm, dtype=float)
    spacing = np.asarray(spacing_plan_mm, dtype=float)
    raw_divisible = np.asarray(divisibility)
    divisible = np.asarray(divisibility, dtype=int)
    if stcl.shape != (3,) or spacing.shape != (3,) or divisible.shape != (3,):
        raise ValueError("stcl, spacing, and divisibility must each contain three values")
    if np.any(~np.isfinite(stcl)) or np.any(stcl <= 0):
        raise ValueError("STCL values must be positive finite numbers")
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must be positive finite numbers")
    if np.any(divisible <= 0) or not np.all(raw_divisible == divisible):
        raise ValueError("divisibility must contain positive integers")
    if int(budget) != budget or int(budget) <= 0:
        raise ValueError("budget must be a positive integer")
    if not 0 < float(window) < 1:
        raise ValueError("window must lie strictly between 0 and 1")
    if int(max_multiplier) != max_multiplier or int(max_multiplier) < 1:
        raise ValueError("max_multiplier must be a positive integer")
    return stcl, spacing, divisible


def continuous_target(
    stcl_plan_mm: Sequence[float],
    spacing_plan_mm: Sequence[float],
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the continuous physical target (mm) and voxel target."""
    stcl = np.asarray(stcl_plan_mm, dtype=float)
    spacing = np.asarray(spacing_plan_mm, dtype=float)
    if stcl.shape != (3,) or spacing.shape != (3,):
        raise ValueError("stcl and spacing must each contain three values")
    if np.any(~np.isfinite(stcl)) or np.any(stcl <= 0):
        raise ValueError("STCL values must be positive finite numbers")
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must be positive finite numbers")
    if int(budget) != budget or int(budget) <= 0:
        raise ValueError("budget must be a positive integer")
    geometric_mean = float(np.exp(np.mean(np.log(stcl))))
    physical_volume = float(budget) ** 3 * float(np.prod(spacing))
    target_mm = physical_volume ** (1.0 / 3.0) * (stcl / geometric_mean)
    return target_mm, target_mm / spacing


def fit_strict_window(
    target_voxels_plan: Sequence[float],
    divisibility: Sequence[int],
    budget: int,
    window: float = DEFAULT_WINDOW,
    max_multiplier: int = DEFAULT_MAX_MULTIPLIER,
) -> tuple[tuple[int, int, int], float, float]:
    """Discretise the continuous target under the strict volume window.

    Returns ``(patch, objective, volume_ratio)``.
    """
    target = np.asarray(target_voxels_plan, dtype=float)
    raw_divisible = np.asarray(divisibility)
    divisible = np.asarray(divisibility, dtype=int)
    if target.shape != (3,) or np.any(~np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("target_voxels_plan must contain three positive finite values")
    if (
        divisible.shape != (3,)
        or np.any(divisible <= 0)
        or not np.all(raw_divisible == divisible)
    ):
        raise ValueError("divisibility must contain three positive integers")
    if int(budget) != budget or int(budget) <= 0:
        raise ValueError("budget must be a positive integer")
    if not 0 < float(window) < 1:
        raise ValueError("window must lie strictly between 0 and 1")
    if int(max_multiplier) != max_multiplier or int(max_multiplier) < 1:
        raise ValueError("max_multiplier must be a positive integer")
    best_key = None
    best_patch = None
    for multipliers in product(range(1, int(max_multiplier) + 1), repeat=3):
        patch = np.asarray(multipliers, dtype=int) * divisible
        volume_ratio = float(np.prod(patch, dtype=np.float64)) / float(budget) ** 3
        if abs(volume_ratio - 1.0) > float(window):
            continue
        objective = pairwise_log_ratio_error(patch, target)
        key = (objective, abs(volume_ratio - 1.0), multipliers)
        if best_key is None or key < best_key:
            best_key, best_patch = key, patch
    if best_patch is None:
        raise RuntimeError(
            f"No feasible patch within {float(window):.1%} of the N={budget} budget"
        )
    assert best_key is not None
    return (
        tuple(int(v) for v in best_patch),
        float(best_key[0]),
        float(np.prod(best_patch, dtype=np.float64)) / float(budget) ** 3,
    )


def fit_patch(
    stcl_xyz_mm: Sequence[float],
    order: str,
    spacing_plan_mm: Sequence[float],
    divisibility: Sequence[int],
    budget: int,
    window: float = DEFAULT_WINDOW,
    max_multiplier: int = DEFAULT_MAX_MULTIPLIER,
) -> PatchAllocation:
    """Allocate one fixed-budget patch from an STCL indicator.

    Parameters
    ----------
    stcl_xyz_mm:
        Directional STCL in physical x/y/z order (mm).
    order:
        Physical axes in plan order, e.g. ``"zyx"``.
    spacing_plan_mm:
        Voxel spacing in plan order (mm).
    divisibility:
        Required stride divisibility in plan order (voxels).
    budget:
        Nominal cube edge ``N``; the actual voxel budget is ``N**3``.
    """
    stcl, spacing, divisible = _validate_common(
        stcl_xyz_mm, spacing_plan_mm, divisibility, budget, window, max_multiplier
    )
    stcl_plan = reorder_xyz_to_plan(stcl, order)
    target_mm, target_voxels = continuous_target(stcl_plan, spacing, budget)
    patch, objective, volume_ratio = fit_strict_window(
        target_voxels, divisible, budget, window=window, max_multiplier=max_multiplier
    )
    return PatchAllocation(
        budget=int(budget),
        order=order,
        stcl_xyz_mm=tuple(float(v) for v in stcl),
        stcl_plan_mm=tuple(float(v) for v in stcl_plan),
        continuous_target_plan_mm=tuple(float(v) for v in target_mm),
        continuous_target_plan_voxels=tuple(float(v) for v in target_voxels),
        patch=patch,
        physical_size_plan_mm=tuple(float(v) for v in np.asarray(patch, dtype=float) * spacing),
        volume_ratio=float(volume_ratio),
        objective=float(objective),
    )


def fit_patches(
    stcl_xyz_mm: Sequence[float],
    order: str,
    spacing_plan_mm: Sequence[float],
    divisibility: Sequence[int],
    budgets: Sequence[int] = (96, 128, 160),
    **kwargs,
) -> dict[int, PatchAllocation]:
    """Convenience wrapper applying :func:`fit_patch` at several budgets."""
    return {
        int(budget): fit_patch(
            stcl_xyz_mm,
            order,
            spacing_plan_mm,
            divisibility,
            int(budget),
            **kwargs,
        )
        for budget in budgets
    }
