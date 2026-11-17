"""PYRX_REQUIRE_CAPTURES: a gate run must fail, not skip, the tests that need the recording,
and must still run the ones that do not."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

FPGA = Path(__file__).resolve().parents[2]
ON_RECORDING = [
    "model/tests/test_compare.py::test_lo_loss_on_recording",
    "model/tests/test_loop.py::test_model_loop_tracks_like_pyrx",
    "tb/test_channel.py::test_channel[recording_200ms]",
]


def _pytest(selection, **env):
    """Run pytest on `selection` with only the given capture variables set."""
    keep = {k: v for k, v in os.environ.items() if not k.startswith("PYRX_")}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", selection]
    return subprocess.run(
        command, cwd=FPGA, env=keep | env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("selection", ON_RECORDING)
def test_a_missing_recording_skips_without_the_gate(selection):
    run = _pytest(selection)
    assert run.returncode == 0 and "1 skipped" in run.stdout, run.stdout


@pytest.mark.parametrize("selection", ON_RECORDING)
@pytest.mark.parametrize("captures", [None, "empty"])
def test_a_missing_recording_fails_under_the_gate(selection, captures, tmp_path):
    env = {"PYRX_REQUIRE_CAPTURES": "1"}
    if captures:  # a directory that does not hold the recording
        env["PYRX_CAPTURES"] = str(tmp_path)
    run = _pytest(selection, **env)
    assert run.returncode == 1 and "PYRX_REQUIRE_CAPTURES is set" in run.stdout, run.stdout
    assert "skipped" not in run.stdout


def test_the_gate_leaves_tests_that_need_no_recording_running():
    run = _pytest("tb/test_regmap.py", PYRX_REQUIRE_CAPTURES="1")
    assert run.returncode == 0 and "passed" in run.stdout, run.stdout + run.stderr
