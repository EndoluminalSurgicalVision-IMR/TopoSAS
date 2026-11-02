from __future__ import annotations

import numpy as np
import pytest

from toposas.metrics import betti0, cldice, dice, evaluate_pair


def two_class_volume() -> np.ndarray:
    label = np.zeros((14, 14, 14), dtype=np.int16)
    label[1:4, 1:4, 1:4] = 1
    label[7:11, 7:11, 7:11] = 2
    return label


def test_perfect_multiclass_prediction() -> None:
    gt = two_class_volume()

    result = evaluate_pair(gt, gt)

    assert result["dice"] == pytest.approx(100.0)
    assert result["clDice"] == pytest.approx(100.0)
    assert result["betti0"] == pytest.approx(0.0)
    assert result["n_union"] == 2
    assert set(result["per_class"]) == {"1", "2"}


def test_empty_prediction_scores_zero_dice_and_reports_gt_components() -> None:
    gt = two_class_volume()
    pred = np.zeros_like(gt)

    result = evaluate_pair(pred, gt)

    assert result["dice"] == pytest.approx(0.0)
    assert result["clDice"] == pytest.approx(0.0)
    assert result["betti0"] == pytest.approx(1.0)
    assert result["n_union"] == 2


def test_prediction_only_class_is_counted_as_zero_dice() -> None:
    gt = two_class_volume()
    pred = np.zeros_like(gt)
    pred[0:2, 0:2, 0:2] = 3

    result = evaluate_pair(pred, gt)

    assert result["n_union"] == 3
    assert result["dice"] == pytest.approx(0.0)
    assert result["clDice"] == pytest.approx(0.0)
    assert result["betti0"] == pytest.approx(1.0)
    assert result["per_class"]["3"]["gt_voxels"] == 0


def test_disconnected_component_is_penalised() -> None:
    gt = np.zeros((12, 12, 12), dtype=np.int16)
    gt[1:4, 1:4, 1:4] = 1
    pred = np.zeros_like(gt)
    pred[1:4, 1:4, 1:4] = 1
    pred[8:10, 8:10, 8:10] = 1

    result = evaluate_pair(pred, gt)

    assert result["betti0"] == pytest.approx(1.0)
    assert result["dice"] == pytest.approx(200.0 * 27.0 / (27.0 + 35.0))


def test_betti0_uses_26_connectivity() -> None:
    gt = np.zeros((5, 5, 5), dtype=np.int16)
    gt[1, 1, 1] = 1
    gt[2, 2, 2] = 1
    pred = np.zeros_like(gt)
    pred[1, 1, 1] = 1

    # The two diagonal GT voxels are one 26-connected component.
    assert betti0(pred, gt) == pytest.approx(0.0)


def test_binary_cldice_on_a_tube() -> None:
    gt = np.zeros((30, 15, 15), dtype=np.int16)
    gt[3:27, 6:9, 6:9] = 1
    pred = gt.copy()
    pred[15:] = 0

    assert cldice(gt, gt) == pytest.approx(100.0)
    assert 0.0 < cldice(pred, gt) < 100.0
    assert cldice(np.zeros_like(gt), gt) == pytest.approx(0.0)


def test_empty_pair_is_zero() -> None:
    empty = np.zeros((4, 4, 4), dtype=np.int16)

    result = evaluate_pair(empty, empty)

    assert result == {
        "dice": 0.0,
        "clDice": 0.0,
        "betti0": 0.0,
        "n_union": 0,
        "per_class": {},
    }


def test_n_labels_and_shape_validation() -> None:
    gt = two_class_volume()
    assert dice(gt, gt, n_labels=2) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        dice(gt, gt, n_labels=1)
    with pytest.raises(ValueError):
        dice(gt, gt[:, :, :7])
    with pytest.raises(ValueError):
        dice(gt[0], gt[0])
    with pytest.raises(ValueError):
        dice(gt.astype(float) + 0.5, gt)
