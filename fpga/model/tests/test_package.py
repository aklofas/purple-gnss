import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import fpgamodel


def test_version():
    assert fpgamodel.__version__ == "0.1.0"


def test_ruff_is_clean():
    ruff = shutil.which("ruff", path=str(Path(sys.executable).parent))
    if ruff is None:
        pytest.skip("ruff not installed")
    root = Path(__file__).resolve().parents[2]  # fpga/
    run = subprocess.run(
        [ruff, "check", "--no-cache", str(root)], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout
