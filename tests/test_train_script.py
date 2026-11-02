from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_with_patch.py"


def _fake_plans(path: Path, patch_size=(176, 112, 112)) -> dict:
    plans = {
        "plans_name": "nnUNetPlans",
        "configurations": {
            "3d_fullres": {"patch_size": list(patch_size), "data_identifier": "nnUNetPlans_3d_fullres"}
        },
    }
    path.write_text(json.dumps(plans, indent=4) + "\n")
    return plans


def test_dry_run_updates_plan_and_backs_up(tmp_path) -> None:
    plans_path = tmp_path / "nnUNetPlans.json"
    original = _fake_plans(plans_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plans",
            str(plans_path),
            "--dataset",
            "39",
            "--config",
            "3d_fullres",
            "--patch-size",
            "96",
            "80",
            "272",
            "--fold",
            "0",
            "--trainer",
            "nnUNetTrainer",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    updated = json.loads(plans_path.read_text())
    backup = tmp_path / "nnUNetPlans.baseline.json"
    assert backup.exists()
    assert json.loads(backup.read_text()) == original
    assert updated["configurations"]["3d_fullres"]["patch_size"] == [96, 80, 272]
    assert "nnUNetv2_train 39 3d_fullres 0 -tr nnUNetTrainer" in result.stdout


def test_backup_is_not_overwritten(tmp_path) -> None:
    plans_path = tmp_path / "nnUNetPlans.json"
    original = _fake_plans(plans_path)
    first = [
        sys.executable,
        str(SCRIPT),
        "--plans",
        str(plans_path),
        "--dataset",
        "39",
        "--patch-size",
        "96",
        "80",
        "272",
        "--dry-run",
    ]
    subprocess.run(first, check=True, capture_output=True, text=True)
    subprocess.run(first, check=True, capture_output=True, text=True)

    backup = tmp_path / "nnUNetPlans.baseline.json"
    assert json.loads(backup.read_text()) == original


def test_unknown_configuration_fails_without_writing(tmp_path) -> None:
    plans_path = tmp_path / "nnUNetPlans.json"
    original = _fake_plans(plans_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plans",
            str(plans_path),
            "--dataset",
            "39",
            "--config",
            "missing",
            "--patch-size",
            "96",
            "80",
            "272",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert json.loads(plans_path.read_text()) == original
    assert not (tmp_path / "nnUNetPlans.baseline.json").exists()


def test_training_command_is_invoked(tmp_path) -> None:
    plans_path = tmp_path / "nnUNetPlans.json"
    _fake_plans(plans_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "called.txt"
    wrapper = bin_dir / "nnUNetv2_train"
    wrapper.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > " + str(marker) + "\n"
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plans",
            str(plans_path),
            "--dataset",
            "39",
            "--config",
            "3d_fullres",
            "--patch-size",
            "96",
            "80",
            "272",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert marker.read_text().splitlines() == [
        "39",
        "3d_fullres",
        "0",
        "-tr",
        "nnUNetTrainer",
    ]
