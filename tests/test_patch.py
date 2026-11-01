from __future__ import annotations

import numpy as np
import pytest

from toposas.patch import (
    fit_patch,
    pairwise_log_ratio_error,
    reorder_plan_to_xyz,
    reorder_xyz_to_plan,
)
from toposas.presets import PRESETS, reproduce_all, reproduce_preset

PUBLISHED_STCL = {
    "aortaseg24": (29.15, 28.29, 85.25),
    "airway-tnn": (23.64, 11.46, 17.84),
    "topbrain25": (9.39, 11.10, 10.31),
}

PUBLISHED_N128 = {
    "aortaseg24": (96, 80, 272),
    "airway-tnn": (160, 80, 160),
    "topbrain25": (160, 160, 80),
}


def test_published_patches_reproduce_exactly() -> None:
    report = reproduce_all()

    assert report["n_cells"] == 3
    assert report["n_match"] == 3
    for dataset in report["datasets"].values():
        for cell in dataset["cells"].values():
            assert cell["match"], cell


def test_published_stcl_values() -> None:
    for key, expected in PUBLISHED_STCL.items():
        assert PRESETS[key].stcl_xyz_mm == pytest.approx(expected, abs=0.01)


def test_published_n128_patches() -> None:
    for key, expected in PUBLISHED_N128.items():
        dataset = PRESETS[key]
        result = reproduce_preset(dataset, [128])
        cell = result["cells"]["128"]
        assert tuple(cell["published"]) == expected
        assert tuple(cell["patch"]) == expected


def test_budget_patches_stay_inside_five_percent() -> None:
    for preset in PRESETS.values():
        for budget in (96, 128, 160):
            allocation = fit_patch(
                preset.stcl_xyz_mm,
                "xyz",
                preset.spacing_xyz_mm,
                preset.divisibility_xyz,
                budget,
            )
            assert abs(allocation.volume_ratio - 1.0) <= 0.05


def test_budget_patches_obey_divisibility() -> None:
    for preset in PRESETS.values():
        for budget in (96, 128, 160):
            allocation = fit_patch(
                preset.stcl_xyz_mm,
                "xyz",
                preset.spacing_xyz_mm,
                preset.divisibility_xyz,
                budget,
            )
            for value, divisor in zip(allocation.patch, preset.divisibility_xyz):
                assert value % divisor == 0, (preset.name, budget, allocation.patch)


def test_reorder_is_inverse() -> None:
    values = np.array([1.0, 2.0, 3.0])
    for order in ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"):
        assert reorder_plan_to_xyz(reorder_xyz_to_plan(values, order), order) == pytest.approx(
            values
        )


def test_pairwise_objective_is_scale_invariant() -> None:
    target = np.array([64.0, 96.0, 160.0])
    assert pairwise_log_ratio_error(target, target) == pytest.approx(0.0)
    assert pairwise_log_ratio_error(target * 3.0, target) == pytest.approx(0.0)
    assert pairwise_log_ratio_error((80.0, 96.0, 160.0), target) > 0.0


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        fit_patch((1.0, 2.0), "xyz", (1.0, 1.0, 1.0), (16, 16, 16), 128)
    with pytest.raises(ValueError):
        fit_patch((1.0, 2.0, 3.0), "abc", (1.0, 1.0, 1.0), (16, 16, 16), 128)
    with pytest.raises(ValueError):
        fit_patch((1.0, 2.0, 3.0), "xyz", (1.0, 1.0, 1.0), (16, 16, 16), 128, window=1.5)
    with pytest.raises(ValueError):
        fit_patch((1.0, 2.0, 3.0), "xyz", (1.0, 1.0, 1.0), (16.5, 16, 16), 128)
    with pytest.raises(ValueError):
        fit_patch((1.0, 2.0, 3.0), "xyz", (1.0, 1.0, 1.0), (16, 16, 16), 128.5)
    with pytest.raises(ValueError):
        fit_patch((1.0, 2.0, 3.0), "xyz", (1.0, 1.0, 1.0), (16, 16, 16), 128,
                  max_multiplier=0)
