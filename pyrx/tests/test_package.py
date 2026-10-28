import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import pyrx

# modules of the signal chain; positioning (orbits, pvt, rtk) must load none of them
SIGNAL_CHAIN = {
    "acquisition",
    "tracking",
    "codes",
    "correlator",
    "samples",
    "bitsync",
    "navdecode",
    "observables",
    "lnav",
}


def test_version():
    assert pyrx.__version__ == "0.1.0"


def test_ruff_is_clean():
    ruff = shutil.which("ruff", path=str(Path(sys.executable).parent))
    if ruff is None:
        pytest.skip("ruff not installed")
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [ruff, "check", "--no-cache", str(root)], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout


@pytest.mark.parametrize("module", ["pyrx.orbits", "pyrx.pvt", "pyrx.rtk"])
def test_positioning_does_not_load_the_signal_chain(module):
    code = f"import sys, {module}; print(*sorted(m for m in sys.modules if m.startswith('pyrx.')))"
    run = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    loaded = {m.removeprefix("pyrx.") for m in run.stdout.split()}
    assert not loaded & SIGNAL_CHAIN, sorted(loaded & SIGNAL_CHAIN)
