"""The notebooks' support module: data-source detection and the synthetic inputs."""

import importlib.util
from pathlib import Path

import numpy as np

from pyrx.pvt import solve
from pyrx.rtk import solve_rtk

# loaded by path: the tests have their own `support` module
_SPEC = importlib.util.spec_from_file_location(
    "nbsupport", Path(__file__).resolve().parents[1] / "notebooks" / "support.py"
)
nbsupport = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nbsupport)


def test_data_source_is_none_without_complete_captures(monkeypatch, tmp_path):
    monkeypatch.delenv("PYRX_CAPTURES", raising=False)
    assert nbsupport.data_source() is None
    monkeypatch.setenv("PYRX_CAPTURES", str(tmp_path))  # a directory without the files
    assert nbsupport.data_source() is None


def test_synthetic_pvt_epochs_solve_to_the_truth():
    epochs, ephs, iono, truth = nbsupport.synthetic_pvt_epochs(n=6)
    sols = solve(epochs, ephs, iono, 15.0)
    assert len(sols) == 6
    assert all(np.linalg.norm(s.position - truth) < 1.5 for s in sols)
    assert sols[-1].clock_bias > sols[0].clock_bias  # the clock drifts


def test_synthetic_rtk_epochs_fix_and_carry_the_slip():
    rover, base, ephs, station, delta, truth = nbsupport.synthetic_rtk_epochs(n=8, slip_at=4)
    fixes = solve_rtk(rover, base, ephs, station, delta)
    assert fixes[3].status == "fix" and np.linalg.norm(fixes[3].position - truth) < 0.02
    assert fixes[-1].status == "fix" and np.linalg.norm(fixes[-1].position - truth) < 0.02
    i, j = rover[0].sats.index("G03"), rover[0].signals.index("G1C")
    r, jj = rover[0].sats.index("G01"), j
    before = (rover[3].carrier_phase[i, j] - base[3].carrier_phase[i, j]) - (
        rover[3].carrier_phase[r, jj] - base[3].carrier_phase[r, jj]
    )
    after = (rover[4].carrier_phase[i, j] - base[4].carrier_phase[i, j]) - (
        rover[4].carrier_phase[r, jj] - base[4].carrier_phase[r, jj]
    )
    assert abs((after - before) - 3.0) < 0.5  # the double difference jumps by the slip


def test_synthetic_recording_is_cached_and_sized(tmp_path):
    path, fs, f_if, iq, prns = nbsupport.synthetic_recording(tmp_path, seconds=0.2)
    assert iq and f_if == 0.0 and prns == [4, 7, 12, 25]
    assert path.stat().st_size == 2 * int(fs * 0.2)
    mtime = path.stat().st_mtime
    nbsupport.synthetic_recording(tmp_path, seconds=0.2)
    assert path.stat().st_mtime == mtime


def test_reference_runners_return_none_without_rtklib(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert nbsupport.rnx2rtkp_single("a.obs", "a.nav", tmp_path, nbsupport.PVT_SPAN) is None
