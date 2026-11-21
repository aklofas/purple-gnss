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


def test_regmap_names_the_link_and_the_dump_burst():
    from fpgamodel import regmap as R

    assert R.VERSION_QUAD_LINK | R.VERSION_EVENT_STAMP == 0x000C0000
    assert R.ch(0, R.EPOCH_TAIL) - R.ch(0, R.EPOCH_HEAD) == R.DUMP_BURST_WORDS - 1
    assert R.DUMP_BURST == R.EPOCH_HEAD and R.IE - R.EPOCH_HEAD == 1 and R.EPOCH_TAIL - R.QL == 1
    assert R.CMD_WRITE | R.CMD_INCR | R.CMD_QUAD == 0xF6
    assert R.CMD_READ & R.CMD_SIGNATURE_MASK == R.CMD_SIGNATURE == 0b10110
    assert (R.LINK_STATUS, R.EVT_CTRL, R.EVT_MISSED) == (0x0F, 0x200, 0x204)
    assert (R.CH_INTEG, R.CODE_STEP_HI) == (0x0D, 0x0E)
