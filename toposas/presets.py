"""Released dataset presets and reproduction checks.

Each preset stores the per-dataset medians of ``M`` and ``Sigma`` measured a
priori on the training label maps; STCL is derived as ``S = M + Sigma / 3``.
Only the published ``N = 128`` patch sizes are stored here.  The ``N = 96`` and
``N = 160`` patches of the budget sweep are not listed in the paper; they are
generated on demand by :func:`toposas.patch.fit_patch` with the same rule.

All tuples are in physical x/y/z order, matching the notation in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .patch import PatchAllocation, fit_patch
from .stcl import DEFAULT_COVERAGE


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    name: str
    spacing_xyz_mm: tuple[float, float, float]
    divisibility_xyz: tuple[int, int, int]
    mean_intercept_xyz_mm: tuple[float, float, float]
    centroid_spread_xyz_mm: tuple[float, float, float]
    published_patches: Mapping[int, tuple[int, int, int]]
    coverage: float = DEFAULT_COVERAGE

    @property
    def stcl_xyz_mm(self) -> tuple[float, float, float]:
        """STCL implied by the measured ``M`` and ``Sigma`` components."""
        return tuple(
            float(mean + self.coverage * spread)
            for mean, spread in zip(self.mean_intercept_xyz_mm, self.centroid_spread_xyz_mm)
        )


PRESETS: dict[str, DatasetPreset] = {
    "aortaseg24": DatasetPreset(
        key="aortaseg24",
        name="AortaSeg24",
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        divisibility_xyz=(16, 16, 16),
        mean_intercept_xyz_mm=(20.983, 21.892, 30.519),
        centroid_spread_xyz_mm=(24.486, 19.180, 164.206),
        published_patches={128: (96, 80, 272)},
    ),
    "airway-tnn": DatasetPreset(
        key="airway-tnn",
        name="Airway-TNN",
        spacing_xyz_mm=(0.78125, 0.78125, 0.5),
        divisibility_xyz=(32, 16, 32),
        mean_intercept_xyz_mm=(5.149, 4.850, 5.588),
        centroid_spread_xyz_mm=(55.460, 19.820, 36.746),
        published_patches={128: (160, 80, 160)},
    ),
    "topbrain25": DatasetPreset(
        key="topbrain25",
        name="TopBrain25",
        spacing_xyz_mm=(0.482421875, 0.482421875, 0.70001220703125),
        divisibility_xyz=(32, 32, 16),
        mean_intercept_xyz_mm=(3.038, 3.169, 2.973),
        centroid_spread_xyz_mm=(19.065, 23.780, 22.006),
        published_patches={128: (160, 160, 80)},
    ),
}


def preset(key: str) -> DatasetPreset:
    try:
        return PRESETS[key]
    except KeyError as error:
        raise KeyError(f"Unknown preset {key!r}; choose from {sorted(PRESETS)}") from error


def reproduce_preset(
    preset_or_key: DatasetPreset | str,
    budgets: Sequence[int] | None = None,
) -> dict:
    """Fit every published budget and report matches."""
    dataset = preset(preset_or_key) if isinstance(preset_or_key, str) else preset_or_key
    selected = list(budgets) if budgets is not None else sorted(dataset.published_patches)
    cells: dict[str, dict] = {}
    matches = 0
    for budget in selected:
        if budget not in dataset.published_patches:
            raise KeyError(f"{dataset.name} has no published patch for N={budget}")
        allocation: PatchAllocation = fit_patch(
            dataset.stcl_xyz_mm,
            "xyz",
            dataset.spacing_xyz_mm,
            dataset.divisibility_xyz,
            int(budget),
        )
        published = tuple(int(v) for v in dataset.published_patches[budget])
        match = allocation.patch == published
        matches += int(match)
        cells[str(budget)] = {
            "patch": list(allocation.patch),
            "published": list(published),
            "match": match,
            "volume_ratio": allocation.volume_ratio,
        }
    return {
        "dataset": dataset.name,
        "n_cells": len(selected),
        "n_match": matches,
        "all_match": matches == len(selected),
        "cells": cells,
    }


def reproduce_all() -> dict:
    """Check the three published ``N = 128`` patches."""
    datasets = {key: reproduce_preset(key) for key in PRESETS}
    n_cells = sum(item["n_cells"] for item in datasets.values())
    n_match = sum(item["n_match"] for item in datasets.values())
    return {
        "n_cells": n_cells,
        "n_match": n_match,
        "all_match": n_match == n_cells,
        "datasets": datasets,
    }
