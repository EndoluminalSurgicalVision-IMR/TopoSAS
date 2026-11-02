from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from toposas.cli import main


def test_reproduce_cli(capsys) -> None:
    assert main(["reproduce"]) == 0
    output = capsys.readouterr().out
    assert "3/3 published N=128 patches exact" in output
    assert "published N=128 patches" in output


def test_patch_cli_with_preset(capsys) -> None:
    assert main(["patch", "--preset", "aortaseg24", "--budget", "128"]) == 0
    output = capsys.readouterr().out
    assert "patch=(96, 80, 272)" in output
    assert "volume_ratio=0.9961" in output


def test_evaluate_cli(capsys, tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    gt = np.zeros((8, 8, 8), dtype=np.int16)
    gt[1:4, 1:4, 1:4] = 1
    nib.save(nib.Nifti1Image(gt, np.eye(4)), str(pred_dir / "case.nii.gz"))
    nib.save(nib.Nifti1Image(gt, np.eye(4)), str(gt_dir / "case.nii.gz"))
    output = tmp_path / "metrics.json"

    assert main([
        "evaluate",
        "--pred-dir", str(pred_dir),
        "--gt-dir", str(gt_dir),
        "--labels", "1",
        "--output", str(output),
    ]) == 0

    report = json.loads(output.read_text())
    assert report["num_cases"] == 1
    assert report["summary"]["dice"] == 100.0
    assert "cases=1" in capsys.readouterr().out
