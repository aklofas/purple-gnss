"""Re-acquisition and the phase-lock rule on the car-roof cold-start capture, against Pocket SDR.

The capture (2026-11-03, antenna on the car roof, started while driving) passes under
an obstruction between 40 and 50 s in: Pocket SDR's lock detector drops every channel
and its `$CH` lines give the instant each PRN locked again. Our tracker's C/N0 dips to
31 to 35 dB-Hz on the six strong satellites and its code loop holds, but the Costas
loop comes out of the event unlocked at full C/N0, so the phase-lock rule ends each
first segment in the event and a later probe brings the satellite back; its Doppler
then agrees with Pocket SDR's. G22 holds phase lock from the start for 17 s (a
second-order Costas loop lost it at 6 s under the car's Doppler rate), its second
segment from the 20 s probe ends in the event; G22 and G25 are then acquired within
one probe interval of Pocket SDR's lock (G25, at 36 dB-Hz, unlocks again later). The observations carry
no G06 carrier phase between its segments and flag the first phase of the new
segment as a slip. Minutes long: marked slow, run with `-m slow`.
"""

import numpy as np
import pytest
from conftest import capture_file

from pyrx.channels import REACQ_INTERVAL, track_prns
from pyrx.navdecode import decode_lnav
from pyrx.observables import LLI_SLIP, make_observables
from pyrx.tracking import PLI_LOCKED

CAPTURE = "sky/20261103_032110_carroof-coldstart"
HELD = [6, 11, 12, 13, 19, 24]  # tracked from the start by both receivers
JOINED = [22, 25]  # first locked by Pocket SDR after the event
T_START = 0.0
DURATION = 90.0
T_CHECK = 85.0  # s into the file, both trackers settled after the event


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
    segs = track_prns(path, 24e6, False, 6e6, HELD + JOINED, t_start=T_START, duration=DURATION)
    by_prn = {p: [s for s in segs if s.prn == p] for p in HELD + JOINED}
    for prn in HELD:
        first = by_prn[prn][0]
        assert first.t[0] < 0.1 and not first.locked and 36.0 <= first.lost_at <= 52.0, prn
        assert first.phase_lock[-1] < PLI_LOCKED, prn  # ended by the phase rule, not C/N0
        (back,) = [s for s in by_prn[prn][1:] if s.t[0] <= T_CHECK <= s.t[-1]]
        assert back.locked and back.t[-1] >= T_START + DURATION - 0.1, prn
        # at the log line's own time: the car's Doppler rate reaches several Hz/s
        t_log, _, _, theirs = pocket[prn][np.argmin(abs(pocket[prn][:, 0] - T_CHECK))]
        ours = back.doppler[np.searchsorted(back.t, t_log)]
        assert abs(ours - theirs) < 2.0, (prn, ours, theirs)
    g22 = by_prn[22]
    assert g22[0].t[0] < 0.1 and not g22[0].locked and 15.0 <= g22[0].lost_at <= 2 * REACQ_INTERVAL
    # its second segment, from the 20 s probe, ends in the event
    assert g22[1].t[0] <= 2 * REACQ_INTERVAL + 0.1 and 36.0 <= g22[1].lost_at <= 52.0
    relock = [s for s in g22[1:] if 50.0 <= s.t[0] <= 70.1 and s.t[-1] - s.t[0] >= 10.0]
    assert relock, [(s.t[0], s.lost_at) for s in g22]
    back25 = [s for s in by_prn[25] if s.t[0] > 40.0]  # its 10 s segment dies in the event
    theirs = relock_time(pocket[25], after=40.0)
    assert back25 and back25[0].t[0] <= theirs + REACQ_INTERVAL
    assert back25[0].t[-1] - back25[0].t[0] >= 10.0  # at 36 dB-Hz it unlocks again later
    # the observations: G06 has no phase from its first unlocked window to its next segment's
    # first observation, which is a slip; its pseudoranges continue through the blank ones
    lnavs = [decode_lnav(s) for s in segs]
    epochs = make_observables(segs, lnavs)
    g06 = [(e.tow, o) for e in epochs for o in e.obs if o.prn == 6]
    i_blank = next(i for i, (_, o) in enumerate(g06) if np.isnan(o.carrier_phase))
    i_next = next(i for i, (_, o) in enumerate(g06) if i > i_blank and np.isfinite(o.carrier_phase))
    assert all(
        np.isfinite(o.pseudorange) and np.isnan(o.carrier_phase) for _, o in g06[i_blank:i_next]
    )
    assert g06[i_next][1].lli & LLI_SLIP
    second = by_prn[6][1]
    assert g06[i_next][0] - g06[0][0] >= second.t[0] - by_prn[6][0].t[0] - 1.0  # next segment


STATIC = "sky/20261103_030551_carroof-static"
PILOTS = {"L5": ("L5Q", [6, 11, 13, 24]), "E5A": ("E5AQ", [15, 27])}  # 46 to 50 dB-Hz


@pytest.mark.slow
def test_car_roof_pilots(captures):
    # the pilots of the car-roof static capture's L5 band (CH2), first 60 s: one segment per
    # satellite from the first probe, secondary code found, locked to the end, at the C/N0
    # of the reference's pilot channel (Pocket SDR with corrected L5 codes)
    path = capture_file(captures, f"{STATIC}/ch2.bin")
    log = capture_file(captures, f"{STATIC}/trk_l5_l5fix.log")
    for signal, (their_name, prns) in PILOTS.items():
        theirs: dict[int, list] = {}
        with open(log) as f:
            for line in f:
                p = line.split(",")
                if p[0] == "$CH" and p[5] == their_name and 10.0 <= float(p[1]) < 60.0:
                    theirs.setdefault(int(p[6]), []).append(float(p[8]))
        segs = track_prns(path, 24e6, True, 0.0, prns, duration=60.0, signal=signal)
        assert [s.prn for s in segs] == prns, signal
        for seg in segs:
            who = (signal, seg.prn)
            assert seg.locked and seg.t[0] < 0.002 and seg.t[-1] > 59.9, who
            assert seg.sec_sync is not None and seg.t[seg.sec_sync] < 2.0, who
            ends = np.arange(999, len(seg.t), 1000)
            assert seg.phase_lock[ends[ends - 999 > seg.sec_sync]].min() > 0.9, who
            ours = np.mean(seg.cn0[seg.t >= 10.0])
            assert abs(ours - np.mean(theirs[seg.prn])) < 2.0, (who, ours)
