from __future__ import annotations

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
