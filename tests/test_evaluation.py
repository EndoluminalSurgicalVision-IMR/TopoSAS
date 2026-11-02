from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from toposas.evaluation import evaluate_case, evaluate_directories, write_report


def _volume(offset: int = 0) -> np.ndarray:
    label = np.zeros((10, 10, 10), dtype=np.int16)
    label[1 + offset : 5 + offset, 2:6, 2:6] = 1
    label[6:9, 6:9, 6:9] = 2
    return label


def _save(path, array, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(array, affine), str(path))


def test_evaluate_case_matches_reference_orientation(tmp_path) -> None:
    gt = _volume()
    pred = gt.copy()
    pred[8:, 8:, 8:] = 0
    pred_path = tmp_path / "case_001.nii.gz"
    gt_path = tmp_path / "case_001_gt.nii.gz"
    _save(pred_path, pred)
    _save(gt_path, gt)

    result = evaluate_case(pred_path, gt_path, n_labels=2)

    # evaluate_case transposes nibabel x,y,z to the reference z,y,x order.
    from toposas.metrics import evaluate_pair

    expected = evaluate_pair(np.transpose(pred, (2, 1, 0)), np.transpose(gt, (2, 1, 0)), 2)
    for key in ("dice", "clDice", "betti0", "n_union"):
        assert result[key] == pytest.approx(expected[key])
    assert result["case"] == "case_001"


def test_evaluate_directories_summary_and_json(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    for index, name in enumerate(("case_a", "case_b")):
        gt = _volume()
        pred = gt.copy()
        if index == 1:
            pred[:] = 0
        _save(pred_dir / f"{name}.nii.gz", pred)
        _save(gt_dir / f"{name}.nii.gz", gt)

    report = evaluate_directories(pred_dir, gt_dir, n_labels=2)
    output = tmp_path / "metrics.json"
    write_report(report, output)
    loaded = json.loads(output.read_text())

    assert report["num_cases"] == 2
    assert loaded["protocol"] == "toposas-eval-v1"
    assert loaded["summary"]["dice"] == pytest.approx(50.0)
    assert loaded["summary"]["betti0"] == pytest.approx(0.5)
    assert len(loaded["cases"]) == 2


def test_evaluate_directories_rejects_case_mismatch(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _save(pred_dir / "only_pred.nii.gz", _volume())
    _save(gt_dir / "only_gt.nii.gz", _volume())

    with pytest.raises(ValueError, match="Case set mismatch"):
        evaluate_directories(pred_dir, gt_dir, n_labels=2)


def test_evaluate_case_rejects_affine_mismatch(tmp_path) -> None:
    gt = _volume()
    pred_path = tmp_path / "case.nii.gz"
    gt_path = tmp_path / "case_gt.nii.gz"
    _save(pred_path, gt)
    _save(gt_path, gt, affine=np.diag([2.0, 1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="Affine mismatch"):
        evaluate_case(pred_path, gt_path, n_labels=2)
    # Explicitly allowing the mismatch should work.
    evaluate_case(pred_path, gt_path, n_labels=2, check_affine=False)
