"""Tracking compared with Pocket SDR's pocket_trk.py on two 30 s recordings.

Each PRN is acquired on the first 10 ms, tracked through the file, and compared
with Pocket SDR's log at its 0.1 s cadence over t >= 2 s (after pull-in).
"""

import numpy as np
import pytest
from conftest import capture_file
from pocketsdr_log import load_pocketsdr_trk

from pyrx.acquisition import acquire_l1ca
from pyrx.codes import CA_CHIP_RATE
from pyrx.samples import read_samples
from pyrx.tracking import track_channels

CASES = [
    ("L1_20211226_082401_24MHz_I.bin", 24e6, 6.0e6, [4, 16, 18, 26, 27, 29, 31]),
    ("L1_20220114_004440_8MHz_I.bin", 8e6, 2.0e6, [10, 12, 15, 23, 24, 32]),
]
ABSENT_PRN = 1  # about 33.5 dB-Hz (noise) in both recordings


def _at(r, times):
    return np.clip(np.searchsorted(r.t, times), 0, len(r.t) - 1)


@pytest.mark.parametrize("name,fs,f_if,prns", CASES, ids=[c[0] for c in CASES])
def test_tracking_matches_pocketsdr(captures, name, fs, f_if, prns):
    path = capture_file(captures, name)
    ref = load_pocketsdr_trk(capture_file(captures, f"pocketsdr-trk/{name[:-4]}.trk.log"))
    head = read_samples(path, fs, iq=False, duration=0.010)
    seeds = []
    for prn in prns + [ABSENT_PRN]:
        acq = acquire_l1ca(head, fs, f_if, prn)
        assert acq.detected == (prn != ABSENT_PRN), prn
        seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
    results = track_channels(path, fs, False, f_if, seeds)
    for r in results[:-1]:
        prn = r.prn
        assert r.locked, prn
        theirs = ref[prn]
        m = theirs[:, 0] >= 2.0
        tt = theirs[m, 0]
        idx = _at(r, tt)
        assert r.t[-1] >= tt[-1] - 0.2, prn  # tracked to the end
        assert abs(np.mean(r.cn0[idx]) - np.mean(theirs[m, 1])) <= 2.0, prn
        d_dop = r.doppler[idx] - theirs[m, 3]
        assert abs(np.mean(d_dop)) <= 5.0, prn
        assert np.max(np.abs(d_dop)) <= 10.0, prn
        d_coff = (r.code_offset[idx] - theirs[m, 2] + 0.5e-3) % 1e-3 - 0.5e-3
        assert np.max(np.abs(d_coff)) * CA_CHIP_RATE <= 0.1, prn
    # No false lock: acquisition rejected the absent PRN (asserted above), and a channel
    # forced onto it never reaches the 35 dB-Hz lock threshold that would admit it.
    # Cross-correlation from the strong satellites keeps its estimate near 30 dB-Hz,
    # which is why a tracker alone cannot be the gate; the acquisition threshold is.
    absent = results[-1]
    assert not absent.locked or np.max(absent.cn0[absent.t >= 2.0]) < 35.0
