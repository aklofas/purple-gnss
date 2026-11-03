"""Re-acquisition on the car-roof cold-start capture, against Pocket SDR's C tracker.

The capture (2026-11-03, antenna on the car roof, started while driving) passes under
an obstruction between 40 and 50 s in: Pocket SDR's lock detector drops every channel
and its `$CH` lines give the instant each PRN locked again. Our tracker's C/N0 dips to
31 to 35 dB-Hz on the six strong satellites and holds lock through it (its Doppler after
the event agrees with Pocket SDR's), so for those the check is that the one segment
is still on the signal; the two satellites it did not have before the event (G22,
G25) must be acquired within one probe interval of Pocket SDR's lock. Minutes long:
marked slow, run with `-m slow`.
"""

import numpy as np
import pytest
from conftest import capture_file

from pyrx.channels import REACQ_INTERVAL, track_prns

CAPTURE = "sky/20261103_032110_carroof-coldstart"
HELD = [6, 11, 12, 13, 19, 24]  # tracked from 30 s on by both receivers
JOINED = [22, 25]  # first locked by Pocket SDR after the event
T_CHECK = 80.0  # s into the file, both trackers settled after the event


def pocket_channels(log) -> dict[int, np.ndarray]:
    """PRN -> rows (t_s, lock_s, cn0_dbhz, doppler_hz) of Pocket SDR's `$CH` lines."""
    rows: dict[int, list] = {}
    with open(log) as f:
        for line in f:
            if line.startswith("$CH,"):
                p = line.split(",")
                rows.setdefault(int(p[6]), []).append(
                    (float(p[1]), float(p[7]), float(p[8]), float(p[10]))
                )
    return {prn: np.array(r) for prn, r in rows.items()}


def relock_time(ch: np.ndarray, after: float) -> float:
    """Start of the first lock of a channel begun after `after` s."""
    starts = ch[:, 0] - ch[:, 1]
    return float(starts[(ch[:, 0] > after) & (starts > after)][0])


@pytest.mark.slow
def test_cold_start_obstruction_against_pocket_sdr(captures):
    path = capture_file(captures, f"{CAPTURE}/ch1.bin")
    pocket = pocket_channels(capture_file(captures, f"{CAPTURE}/trk_l1ca.log"))
    segs = track_prns(path, 24e6, False, 6e6, HELD + JOINED, t_start=30.0, duration=60.0)
    by_prn = {p: [s for s in segs if s.prn == p] for p in HELD + JOINED}
    for prn in HELD:
        (seg,) = by_prn[prn]
        assert seg.locked and seg.t[0] < 30.1, prn
        dip = seg.cn0[(seg.t >= 40.0) & (seg.t <= 52.0)].min()
        assert dip < 36.0, (prn, dip)  # the event is real for our tracker too
        ours = seg.doppler[np.searchsorted(seg.t, T_CHECK)]
        theirs = pocket[prn][np.argmin(abs(pocket[prn][:, 0] - T_CHECK)), 3]
        assert abs(ours - theirs) < 2.0, (prn, ours, theirs)
    for prn in JOINED:
        assert by_prn[prn], prn
        first = by_prn[prn][0]
        theirs = relock_time(pocket[prn], after=40.0)
        assert first.t[0] <= theirs + REACQ_INTERVAL, (prn, first.t[0], theirs)
        assert first.locked
