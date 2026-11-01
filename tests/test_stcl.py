from __future__ import annotations

import numpy as np
import pytest

from toposas.stcl import (
    aggregate_measurements,
    measure_label,
    measure_nifti,
)


def test_bar_geometry_is_exact() -> None:
    label = np.zeros((100, 11, 11), dtype=np.int16)
    label[:, 4:7, 4:7] = 1

    measurement = measure_label(label, (1.0, 1.0, 1.0))

    # V = 900, one run per probe line: R_x = 9, R_y = R_z = 300.
    assert measurement.foreground_voxels == 900
    assert measurement.mean_intercept_xyz_mm == pytest.approx((100.0, 3.0, 3.0))
    # Single class: weighted coordinate spread of a 100 x 3 x 3 bar.
    assert measurement.centroid_spread_xyz_mm[0] == pytest.approx(
        np.sqrt(np.mean((np.arange(100) - 49.5) ** 2))
    )
    assert measurement.centroid_spread_xyz_mm[1] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert measurement.stcl_xyz_mm[0] == pytest.approx(
        measurement.mean_intercept_xyz_mm[0] + measurement.centroid_spread_xyz_mm[0] / 3.0
    )


def test_multiclass_centroid_spread() -> None:
    label = np.zeros((10, 10, 10), dtype=np.int16)
    label[:5] = 1
    label[5:] = 2

    measurement = measure_label(label, (1.0, 1.0, 1.0))

    # Class centroids at x = 2 and x = 7; y/z centroids coincide and are
    # floored at half a voxel.
    assert measurement.mean_intercept_xyz_mm == pytest.approx((10.0, 10.0, 10.0))
    assert measurement.centroid_spread_xyz_mm == pytest.approx((2.5, 0.5, 0.5))
    assert measurement.stcl_xyz_mm == pytest.approx(
        (10.0 + 2.5 / 3.0, 10.0 + 0.5 / 3.0, 10.0 + 0.5 / 3.0)
    )


def test_physical_spacing_scales_lengths() -> None:
    label = np.zeros((100, 11, 11), dtype=np.int16)
    label[:, 4:7, 4:7] = 1

    isotropic = measure_label(label, (1.0, 1.0, 1.0))
    anisotropic = measure_label(label, (2.0, 0.5, 0.5))

    assert anisotropic.mean_intercept_xyz_mm[0] == pytest.approx(
        2.0 * isotropic.mean_intercept_xyz_mm[0]
    )
    assert anisotropic.mean_intercept_xyz_mm[1] == pytest.approx(
        0.5 * isotropic.mean_intercept_xyz_mm[1]
    )


def test_aggregate_uses_marginal_medians() -> None:
    first = measure_label(np.ones((2, 2, 2), dtype=np.int16), (1.0, 1.0, 1.0), "a")
    second = measure_label(np.ones((4, 4, 4), dtype=np.int16), (1.0, 1.0, 1.0), "b")
    third = measure_label(np.ones((6, 6, 6), dtype=np.int16), (1.0, 1.0, 1.0), "c")

    aggregate = aggregate_measurements([first, second, third])

    median_spread = float(np.median([first.centroid_spread_xyz_mm[0],
                                     second.centroid_spread_xyz_mm[0],
                                     third.centroid_spread_xyz_mm[0]]))
    assert aggregate["n_cases"] == 3
    assert aggregate["mean_intercept_xyz_mm"] == pytest.approx([4.0, 4.0, 4.0])
    assert aggregate["stcl_xyz_mm"] == pytest.approx([4.0 + median_spread / 3.0] * 3)


def test_nifti_roundtrip(tmp_path) -> None:
    nibabel = pytest.importorskip("nibabel")
    label = np.zeros((8, 9, 10), dtype=np.int16)
    label[2:6, 3:7, 4:8] = 3
    affine = np.diag([0.8, 0.9, 1.1, 1.0])
    image = nibabel.Nifti1Image(label, affine)
    path = tmp_path / "case.nii.gz"
    nibabel.save(image, path)

    from_array = measure_label(label, (0.8, 0.9, 1.1), case_id="case")
    from_file = measure_nifti(path)

    assert from_file.stcl_xyz_mm == pytest.approx(from_array.stcl_xyz_mm)
    assert from_file.case_id == "case"


def test_single_voxel() -> None:
    measurement = measure_label(np.ones((1, 1, 1), dtype=np.int16), (1.0, 2.0, 3.0))

    assert measurement.mean_intercept_xyz_mm == pytest.approx((1.0, 2.0, 3.0))
    assert measurement.centroid_spread_xyz_mm == pytest.approx((0.5, 1.0, 1.5))
    assert measurement.stcl_xyz_mm == pytest.approx((1.0 + 0.5 / 3.0,
                                                     2.0 + 1.0 / 3.0,
                                                     3.0 + 1.5 / 3.0))


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        measure_label(np.zeros((3, 3, 3), dtype=np.int16), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        measure_label(np.zeros((3, 3, 3), dtype=np.float32) + 1, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        measure_label(np.ones((3, 3, 3), dtype=np.int16), (1.0, 1.0))
