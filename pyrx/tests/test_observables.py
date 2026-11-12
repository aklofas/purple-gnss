import dataclasses
from functools import cache

import numpy as np
import pytest

from pyrx.bitsync import BitStream
from pyrx.codes import CA_PERIOD
from pyrx.gpstime import C_LIGHT, WEEK_S, full_week
from pyrx.lnav import PREAMBLE, Subframe
from pyrx.navdecode import LnavResult, decode_lnav
from pyrx.observables import (
    L5Q_ALIGN,
    LLI_HALF_CYCLE,
    LLI_SLIP,
    make_joint_observables,
    make_observables,
    pilot_transmit_times,
    receiver_time0,
    transmit_times,
)
from pyrx.samples import ArraySamples
from pyrx.signals import L1_FREQ
from pyrx.synth import frame, synth_l1ca
from pyrx.tracking import TrackResult, track_l1ca

T0 = 5044 * 6 - 6 - 100 * 0.020  # transmit time (s of week) of bit 0, 100 bits before the frame


@cache  # a 20 s synthetic channel takes tens of seconds to track; several tests share one
def _channel(prn, code_offset, doppler, fs=2.5e6, tow0=5044, seconds=20.1, wn=142, lead=None):
    lead = np.zeros(100, dtype=np.uint8) if lead is None else np.array(lead, dtype=np.uint8)
    seq = np.concatenate([lead, frame(tow0, wn)])
    x = synth_l1ca(prn, fs, 0.0, doppler, code_offset, int(fs * seconds), 45.0, True, bits=seq)
    track = track_l1ca(ArraySamples(x, fs), 0.0, prn, code_offset / fs, doppler, 45.0)
    return track, decode_lnav(track)


def test_transmit_times_follow_epoch_arithmetic():
    track, lnav = _channel(4, 500, 800.0)
    tt = transmit_times(track, lnav)
    sf = lnav.subframes[0]
    assert tt[sf.epoch] == sf.tow_s and tt[0] == pytest.approx(sf.tow_s - sf.epoch * CA_PERIOD)
    # the epoch at sample 500 is epoch 0 of this track: its code start left at T0
    assert tt[0] == pytest.approx(T0, abs=1e-9)


def test_transmit_times_reject_inconsistent_subframes():
    track, lnav = _channel(4, 500, 800.0)
    bad = dataclasses.replace(lnav.subframes[1], epoch=lnav.subframes[1].epoch + 20)
    lnav = dataclasses.replace(lnav, subframes=[lnav.subframes[0], bad])
    with pytest.raises(ValueError, match="inconsistent"):
        transmit_times(track, lnav)
    with pytest.raises(ValueError, match="no subframe"):
        transmit_times(track, dataclasses.replace(lnav, subframes=[]))


def test_transmit_times_continue_across_the_week_rollover():
    # subframes tagged 604794 then 0 belong to one continuous channel: the mapping from
    # epoch to transmit time keeps the first subframe's week
    track, lnav = _channel(4, 500, 800.0, tow0=100800 - 1)
    assert [sf.tow_s for sf in lnav.subframes] == [604788, 604794, 0]
    tt = transmit_times(track, lnav)
    sf = lnav.subframes[-1]
    assert tt[sf.epoch] == pytest.approx(604800.0, abs=1e-9)
    assert tt[0] == pytest.approx(604788.0 - lnav.subframes[0].epoch * CA_PERIOD, abs=1e-9)


def test_channels_either_side_of_the_week_rollover_share_one_clock():
    # channel 4 decodes subframes on both sides of the week boundary, channel 7 only the
    # one after it; the receiver epochs run past 604800 in the week of the first epoch
    # and both channels give the same pseudorange geometry as inside a week
    fs = 2.5e6
    a = _channel(4, 500, 800.0, fs=fs, tow0=100800 - 1)
    b = _channel(7, 1700, -1200.0, fs=fs, tow0=100800 - 1)
    b = (b[0], dataclasses.replace(b[1], subframes=b[1].subframes[2:]))
    assert [sf.tow_s for sf in b[1].subframes] == [0]
    epochs = make_observables([a[0], b[0]], [a[1], b[1]])
    assert epochs and all(e.week == 2190 for e in epochs)
    tows = [e.tow for e in epochs]
    assert tows == sorted(tows) and tows[0] < 604800.0 < tows[-1]
    rx0 = receiver_time0([a[0], b[0]], [a[1], b[1]])
    for e in epochs:
        assert [o.prn for o in e.obs] == [4, 7]
        n = (e.tow - rx0) * fs
        expect = C_LIGHT * (
            (n - 500) / fs * (1 + 800.0 / L1_FREQ) - (n - 1700) / fs * (1 - 1200.0 / L1_FREQ)
        )
        assert e.obs[1].pseudorange - e.obs[0].pseudorange == pytest.approx(expect, abs=3.0)
    # a recording whose every observation epoch lies past the boundary is dated in the
    # new week, whichever side sample 0 was on, and its pseudoranges are the plain
    # geometry: bit 0 of this channel left at 604798.0 of the old week
    c = _channel(4, 500, 800.0, fs=fs, tow0=100800 + 1)
    later = make_observables([c[0]], [c[1]])
    assert later[0].week == 2190 and 0.0 <= later[0].tow < 100.0
    rx0 = receiver_time0([c[0]], [c[1]])
    assert 604798.0 < rx0 < 604800.0
    for e in later:
        t_rx = e.tow + WEEK_S  # receiver time counted in the old week, as rx0 is
        n = (t_rx - rx0) * fs
        t_tx = 604798.0 + (n - 500) / fs * (1 + 800.0 / L1_FREQ)
        assert e.obs[0].pseudorange == pytest.approx(C_LIGHT * (t_rx - t_tx), abs=3.0)


def test_subframe_1_on_either_side_of_the_rollover_dates_the_run_the_same():
    # channel 4 decodes its subframe 1 before the boundary (week number 142), channel 7
    # 12 s later, after it (week number 143), 12 s into a channel whose bit 0 left at
    # 604788: both place the run in week 2190 whichever channel comes first, and the
    # epochs are identical in either order
    fs = 2.5e6
    a = _channel(4, 500, 800.0, fs=fs, tow0=100800 - 1)
    # 12 s of lead: random bits to sync on, then the zeros the frame's parity continues from
    rng = np.random.default_rng(5)
    lead = tuple(int(b) for b in np.concatenate([rng.integers(0, 2, 500), np.zeros(100)]))
    b = _channel(7, 1700, -1200.0, fs=fs, tow0=100800 + 1, wn=143, lead=lead)
    assert b[1].subframes[0].sf_id == 1 and b[1].subframes[0].tow_s == 0
    epochs = make_observables([a[0], b[0]], [a[1], b[1]])
    assert epochs == make_observables([b[0], a[0]], [b[1], a[1]])
    assert epochs and all(e.week == 2190 for e in epochs)
    tows = [e.tow for e in epochs]
    assert tows == sorted(tows) and tows[0] < 604800.0 < tows[-1]
    rx0 = receiver_time0([a[0], b[0]], [a[1], b[1]])
    for e in epochs:
        assert [o.prn for o in e.obs] == [4, 7]
        # PRN 7's bit 0 left 2 s after PRN 4's (604788 against 604786), on top of the
        # sample-offset and Doppler-rate geometry of the in-week case
        n = (e.tow - rx0) * fs
        expect = C_LIGHT * (
            (n - 500) / fs * (1 + 800.0 / L1_FREQ) - (n - 1700) / fs * (1 - 1200.0 / L1_FREQ) - 2.0
        )
        assert e.obs[1].pseudorange - e.obs[0].pseudorange == pytest.approx(expect, abs=3.0)


def test_pseudorange_matches_synthetic_truth():
    fs, c0, fd = 2.5e6, 500, 800.0
    track, lnav = _channel(4, c0, fd, fs=fs)
    epochs = make_observables([track], [lnav], interval=1.0, week=2190)
    assert 15 <= len(epochs) <= 19 and all(e.week == 2190 for e in epochs)
    rx0 = receiver_time0([track], [lnav])
    for e in epochs:
        assert (e.tow % 1.0) == pytest.approx(0.0, abs=1e-9)
        n = (e.tow - rx0) * fs  # receiver epoch as a (fractional) sample index
        t_tx = T0 + (n - c0) / fs * (1 + fd / L1_FREQ)
        assert e.obs[0].pseudorange == pytest.approx(C_LIGHT * (e.tow - t_tx), abs=3.0)
        assert e.obs[0].doppler == pytest.approx(fd, abs=3.0)
        assert 40.0 < e.obs[0].cn0 < 50.0
    phases = np.array([e.obs[0].carrier_phase for e in epochs])
    rate = np.diff(phases) / np.diff([e.tow for e in epochs])
    np.testing.assert_allclose(rate, -fd, atol=0.5, rtol=0)


def test_two_channels_pseudorange_difference():
    fs = 2.5e6
    a = _channel(4, 500, 800.0, fs=fs)
    b = _channel(7, 1700, -1200.0, fs=fs)
    epochs = make_observables([a[0], b[0]], [a[1], b[1]], week=2190)
    rx0 = receiver_time0([a[0], b[0]], [a[1], b[1]])
    for e in epochs:
        assert [o.prn for o in e.obs] == [4, 7]
        # PRN 7's code start came 1200 samples later for the same transmit time (144 km
        # further), and its Doppler-scaled code rate drifts its transmit time 2000 Hz / L1
        # slower than PRN 4's over the elapsed time (380 m per second)
        n = (e.tow - rx0) * fs
        expect = C_LIGHT * (
            (n - 500) / fs * (1 + 800.0 / L1_FREQ) - (n - 1700) / fs * (1 - 1200.0 / L1_FREQ)
        )
        assert e.obs[1].pseudorange - e.obs[0].pseudorange == pytest.approx(expect, abs=3.0)


def test_channel_without_subframes_is_skipped_and_week_defaults():
    track, lnav = _channel(4, 500, 800.0)
    silent = dataclasses.replace(lnav, subframes=[], ephemeris=None)
    epochs = make_observables([track, track], [lnav, silent])
    assert all(len(e.obs) == 1 for e in epochs) and epochs[0].week == 2048 + 142
    assert full_week(142) == 2190


def test_channel_with_inconsistent_tags_is_skipped():
    track, lnav = _channel(4, 500, 800.0)
    bad = dataclasses.replace(lnav.subframes[1], epoch=lnav.subframes[1].epoch + 20)
    broken = dataclasses.replace(lnav, subframes=[lnav.subframes[0], bad])
    epochs = make_observables([track, track], [lnav, broken])
    assert epochs and all([o.prn for o in e.obs] == [4] for e in epochs)
    assert receiver_time0([track, track], [lnav, broken]) == receiver_time0([track], [lnav])


def test_week_comes_from_subframe_1_without_an_ephemeris():
    track, lnav = _channel(4, 500, 800.0)
    partial = dataclasses.replace(lnav, ephemeris=None)
    assert make_observables([track], [partial])[0].week == 2190


def _still_channel(subframes, inverted=False, n=30000, k0=1500, phase_lock=None):
    """A 30 s channel at zero Doppler with a zero carrier phase and the given subframes.

    `subframes` is (first-bit epoch, seconds of week, polarity flipped) per decoded
    subframe; a flipped subframe's bits came complemented relative to the polarity
    frame sync settled on. `phase_lock` is the per-epoch indicator, locked throughout
    by default.
    """
    zeros = np.zeros(n)
    if phase_lock is None:
        phase_lock = np.ones(n)
    track = TrackResult(
        4, np.arange(n) * CA_PERIOD, np.zeros((n, 4), np.complex64), zeros, zeros, zeros, zeros,
        np.full(n, 45.0), phase_lock, True, None, zeros, zeros,
    )  # fmt: skip
    sfs = []
    for epoch, tow, flipped in subframes:
        bits = np.zeros(300, dtype=np.uint8)
        bits[:8] = PREAMBLE ^ int(flipped)
        sfs.append(Subframe(4, 1, tow, {"wn": 142}, epoch, bits))
    stream = BitStream(np.zeros(1, np.uint8), np.array([k0]), k0 % 20, 1.0, inverted)
    return track, LnavResult(4, stream, sfs, None, None)


@pytest.mark.parametrize("inverted", [False, True])
def test_polarity_change_sets_the_half_cycle_and_slip_indicators(inverted):
    # subframes at 2 s and 14 s, the second received with the opposite data polarity: the
    # transition happened after the first ended (8 s) and before the second began, so the
    # phases in between may be half a cycle off; the first observation on the new polarity
    # is a slip; before the first subframe the polarity rests on frame sync alone
    track, lnav = _still_channel(
        [(2000, 10.0, False), (14000, 22.0, True), (20000, 28.0, True)], inverted
    )
    epochs = make_observables([track], [lnav], week=2190)
    half = 0.5 if inverted else 0.0
    by_tow = {e.tow: (e.obs[0].lli, e.obs[0].carrier_phase) for e in epochs}
    assert epochs[0].tow == 10.0 and epochs[-1].tow == 38.0
    assert by_tow[10.0] == (3, -half)  # within 2 s of the channel start, polarity unverified
    assert by_tow[11.0] == (0, -half) and by_tow[16.0] == (0, -half)
    assert by_tow[17.0] == (2, -half) and by_tow[22.0] == (2, -half)
    assert by_tow[23.0] == (1, half - 0.5) and by_tow[24.0] == (0, half - 0.5)
    assert by_tow[38.0] == (0, half - 0.5)


def test_steady_polarity_marks_only_the_lock_in():
    track, lnav = _still_channel([(2000, 10.0, False), (8000, 16.0, False)])
    epochs = make_observables([track], [lnav], week=2190)
    assert [e.obs[0].lli for e in epochs] == [3] + [0] * (len(epochs) - 1)
    assert all(e.obs[0].carrier_phase == 0.0 for e in epochs)


def test_polarity_change_between_sparse_observations_is_a_slip():
    # with 30 s epochs an observation can sit several subframes after the previous one;
    # a change at any subframe in between is a slip on the first observation after it
    track, lnav = _still_channel(
        [(k, 10.0 + k / 1000 - 2.0, k >= 12000) for k in range(0, 30000, 6000)], n=40000
    )
    epochs = make_observables([track], [lnav], interval=30.0, week=2190)
    assert [e.tow for e in epochs] == [30.0]
    assert epochs[0].obs[0].lli == 1  # a channel's first observation; polarity settled by then
    epochs = make_observables([track], [lnav], interval=15.0, week=2190)
    assert [(e.tow, e.obs[0].lli, e.obs[0].carrier_phase) for e in epochs] == [
        (15.0, 1, -0.0), (30.0, 1, -0.5), (45.0, 0, -0.5)
    ]  # fmt: skip


def test_a_channels_first_observation_is_a_slip_whenever_it_falls():
    # observations start at the first bit-synced epoch; when that is past the 2 s lock-in
    # window the first one still begins a new carrier-phase ambiguity
    track, lnav = _still_channel([(2000, 10.0, False), (8000, 16.0, False)], k0=4000)
    epochs = make_observables([track], [lnav], week=2190)
    assert epochs[0].tow == 13.0
    assert [e.obs[0].lli for e in epochs] == [1] + [0] * (len(epochs) - 1)


def _unlocked(k_lo, k_hi):
    pl = np.ones(30000)
    pl[k_lo:k_hi] = 0.0
    return pl


def test_unlocked_epochs_have_no_phase_and_the_next_phase_is_a_slip():
    sfs = [(1500, 5044, False), (7500, 5050, False), (13500, 5056, False), (19500, 5062, False)]
    base = make_observables(*zip(*[_still_channel(sfs)], strict=True))
    track, ln = _still_channel(sfs, phase_lock=_unlocked(10000, 13000))
    epochs = make_observables([track], [ln])
    assert [e.tow for e in epochs] == [e.tow for e in base]
    blank = [i for i, e in enumerate(epochs) if np.isnan(e.obs[0].carrier_phase)]
    assert len(blank) == 3 and blank == list(range(blank[0], blank[0] + 3))
    for i in blank:
        assert epochs[i].obs[0].pseudorange == base[i].obs[0].pseudorange
        assert np.isfinite(epochs[i].obs[0].doppler) and np.isfinite(epochs[i].obs[0].cn0)
    after = blank[-1] + 1
    assert epochs[after].obs[0].lli & LLI_SLIP and not base[after].obs[0].lli & LLI_SLIP
    assert epochs[after].obs[0].carrier_phase == base[after].obs[0].carrier_phase
    for i, (e, b) in enumerate(zip(epochs, base, strict=True)):
        if i not in blank and i != after:
            assert e == b, i


def test_unlocked_window_between_sparse_observations_flags_the_next_phase():
    # at a 2 s interval an unlocked window can lie wholly between two observations: the
    # second still begins a new ambiguity
    sfs = [(1500, 5044, False), (7500, 5050, False)]
    track, ln = _still_channel(sfs)
    base = make_observables([track], [ln], interval=2.0)
    rx0 = receiver_time0([track], [ln])
    k1 = int(np.searchsorted(track.t, base[5].tow - rx0, side="right")) - 1
    pl = np.ones(30000)
    pl[k1 + 300 : k1 + 700] = 0.0  # strictly between the observations at base[5] and base[6]
    track, ln = _still_channel(sfs, phase_lock=pl)
    epochs = make_observables([track], [ln], interval=2.0)
    assert all(np.isfinite(e.obs[0].carrier_phase) for e in epochs)
    assert epochs[6].obs[0].lli & LLI_SLIP and not base[6].obs[0].lli & LLI_SLIP
    assert [e.obs[0].lli for e in epochs[7:]] == [e.obs[0].lli for e in base[7:]]


def test_nan_indicator_blanks_the_phase():
    pl = np.ones(30000)
    pl[-1500:] = np.nan
    track, ln = _still_channel([(1500, 5044, False)], phase_lock=pl)
    epochs = make_observables([track], [ln])
    assert np.isnan(epochs[-1].obs[0].carrier_phase)
    assert np.isfinite(epochs[-2].obs[0].carrier_phase)


def test_an_unlocked_first_observation_still_counts_as_the_lock_in():
    track, ln = _still_channel([(1500, 5044, False)], phase_lock=_unlocked(0, 2500))
    epochs = make_observables([track], [ln])
    first = epochs[0].obs[0]
    assert np.isnan(first.carrier_phase) and first.lli & LLI_SLIP
    with_phase = next(e.obs[0] for e in epochs if np.isfinite(e.obs[0].carrier_phase))
    assert with_phase.lli & LLI_SLIP


@pytest.mark.parametrize("interval", [0.001, 0.0005])
def test_observation_intervals_down_to_the_epoch_rate_and_below_work(interval):
    # several observations can fall in one tracking epoch: the lock scan since the previous
    # phase then has nothing new to look at
    track, ln = _still_channel([(1500, 5044, False)], n=4000)
    epochs = make_observables([track], [ln], interval=interval, week=2190)
    assert len(epochs) > 2000 / (interval / CA_PERIOD) - 10
    phases = [o.carrier_phase for e in epochs for o in e.obs]
    assert all(np.isfinite(phases))
    assert sum(o.lli & LLI_SLIP for e in epochs for o in e.obs if (e.tow - epochs[0].tow) > 2.1) == 0


SFS = [(1500, 5044, False), (7500, 5050, False), (13500, 5056, False), (19500, 5062, False)]


def _still_pilot(k_start=0, n=30000, delay=30e-9, sec_sync=1500, polarity=1, phase_lock=None,
                 doppler=-420.0, prn=4):  # fmt: skip
    """An L5 pilot segment of PRN 4 beside `_still_channel`: its code starts arrive `delay`
    after the L1 code starts from file epoch `k_start` on, the secondary code is found at its
    epoch `sec_sync`, and the carrier phase advances with a constant Doppler."""
    t = (k_start + np.arange(n)) * CA_PERIOD
    return TrackResult(
        prn, t, np.zeros((n, 4), np.complex64), (t + delay) % CA_PERIOD, np.full(n, doppler),
        doppler * CA_PERIOD * np.arange(1, n + 1), np.zeros(n), np.full(n, 47.0),
        np.ones(n) if phase_lock is None else phase_lock, True, None, np.zeros(n), np.zeros(n),
        signal="L5", sec_sync=sec_sync, sec_origin=0, sec_polarity=polarity,
        data_prompt=np.zeros(n, np.complex64),
    )  # fmt: skip


def _g5q(epoch):
    """(pseudorange, phase, doppler, cn0, lli) of G04's L5Q at an epoch, and its L1 row."""
    i, j = epoch.sats.index("G04"), epoch.signals.index("G5Q")
    row = [getattr(epoch, f)[i] for f in ("pseudorange", "carrier_phase", "doppler", "cn0", "lli")]
    return [r[j] for r in row], [r[0] for r in row]


def test_pilot_time_from_l1():
    track, ln = _still_channel(SFS)
    l1_times = transmit_times(track, ln)
    times = pilot_transmit_times(_still_pilot(k_start=200, n=5000), [(track, ln)])
    np.testing.assert_allclose(times, l1_times[200:5200], rtol=0, atol=1e-12)
    # code starts 0.3 ms from L1's are not this satellite's signal on the same clock
    assert pilot_transmit_times(_still_pilot(delay=0.3e-3), [(track, ln)]) is None
    assert pilot_transmit_times(_still_pilot(delay=-80e-9), [(track, ln)]) is not None


def test_joint_l1_columns_equal_single_signal():
    track, ln = _still_channel(SFS)
    single = make_observables([track], [ln])
    joint = make_joint_observables([track], [ln], [_still_pilot()])
    assert len(joint) == len(single) > 20
    for e, o in zip(single, joint, strict=True):
        assert (o.week, o.tow, o.sats, o.signals) == (e.week, e.tow, ("G04",), ("G1C", "G5Q"))
        _, l1 = _g5q(o)
        want = e.obs[0]
        assert l1[0] == want.pseudorange and l1[2] == want.doppler and l1[3] == want.cn0
        assert l1[4] == want.lli
        assert l1[1] == want.carrier_phase or (np.isnan(l1[1]) and np.isnan(want.carrier_phase))


@pytest.mark.parametrize("polarity", [1, -1])
def test_joint_5q_values(polarity):
    track, ln = _still_channel(SFS)
    pilot = _still_pilot(polarity=polarity)
    joint = make_joint_observables([track], [ln], [pilot])
    rx0 = receiver_time0([track], [ln])
    seen = 0
    for o in joint:
        t = o.tow - rx0  # s into the recording
        (pr, phase, doppler, cn0, lli), l1 = _g5q(o)
        if t <= pilot.t[pilot.sec_sync + 1]:
            assert np.isnan(pr) and np.isnan(phase) and lli == 0  # nothing before the sync
            continue
        seen += 1
        assert pr - l1[0] == pytest.approx(C_LIGHT * 30e-9, abs=1e-3)
        assert doppler == -420.0 and cn0 == 47.0
        # the accumulated Doppler phase at the epoch, negated, half a cycle when the loop
        # stands on the inverted pilot, and the quarter cycle that aligns L5Q with L5I
        adr = -420.0 * (t - 30e-9)
        half = 0.5 if polarity < 0 else 0.0
        assert phase == pytest.approx(-(adr + half) + L5Q_ALIGN, abs=1e-6)
        assert lli == (LLI_SLIP if seen == 1 or t <= 2.0 else 0)
        assert not lli & LLI_HALF_CYCLE
    assert seen > 20 and L5Q_ALIGN == -0.25


def test_pilot_reacquired_gets_a_new_anchor():
    # two pilot segments under one L1 segment: blanks in the 5Q columns across the gap, a
    # slip on the second segment's first observations, the L1 columns unbroken
    track, ln = _still_channel(SFS)
    first, second = _still_pilot(n=9000), _still_pilot(k_start=15000, n=12000, delay=31e-9)
    joint = make_joint_observables([track], [ln], [first, second])
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0, *_g5q(o)) for o in joint]
    gap = [t for t, l5, l1 in rows if 9.0 < t < 16.5]
    assert len(gap) >= 6
    for t, l5, l1 in rows:
        assert np.isfinite(l1[0])
        if 9.0 < t <= 15.0 + 1.502:
            assert np.isnan(l5[0]) and np.isnan(l5[1]), t
    after = [(t, l5) for t, l5, _ in rows if t > 16.502]
    assert after[0][1][4] == LLI_SLIP and after[-1][1][4] == 0
    l1_then = next(l1 for t, _, l1 in rows if t == after[0][0])
    assert after[0][1][0] - l1_then[0] == pytest.approx(C_LIGHT * 31e-9, abs=1e-3)


def test_pilot_without_anchor_is_left_out():
    track, ln = _still_channel(SFS)
    base = make_joint_observables([track], [ln], [])
    assert base and all(o.signals == ("G1C", "G5Q") for o in base)
    for pilot in (
        _still_pilot(k_start=40000, n=5000),  # after the L1 segment's end
        _still_pilot(prn=9),  # no L1 channel of its PRN
        _still_pilot(delay=0.3e-3),  # not on the L1 code grid
    ):
        joint = make_joint_observables([track], [ln], [pilot])
        assert len(joint) == len(base)
        for o, b in zip(joint, base, strict=True):
            assert o.sats == ("G04",) and np.isnan(o.pseudorange[0, 1])
            assert o.pseudorange[0, 0] == b.pseudorange[0, 0]
    # overlap instants that disagree: the code starts step by a whole period mid-segment
    pilot = _still_pilot()
    stepped = dataclasses.replace(pilot, t=np.concatenate([pilot.t[:9000], pilot.t[9000:] + 1e-3]))
    assert pilot_transmit_times(stepped, [(track, ln)]) is None


def test_pilot_phase_blank_while_unlocked():
    track, ln = _still_channel(SFS)
    locked = make_joint_observables([track], [ln], [_still_pilot()])
    pilot = _still_pilot(phase_lock=_unlocked(10000, 13000))
    joint = make_joint_observables([track], [ln], [pilot])
    blank = [i for i, o in enumerate(joint) if np.isnan(_g5q(o)[0][1]) and np.isfinite(_g5q(o)[0][0])]
    assert len(blank) == 3
    for i in blank:
        assert _g5q(joint[i])[0][0] == _g5q(locked[i])[0][0]  # the pseudorange stays
    nxt = _g5q(joint[blank[-1] + 1])[0]
    assert nxt[4] == LLI_SLIP and nxt[1] == _g5q(locked[blank[-1] + 1])[0][1]


def test_joint_observables_need_the_week():
    track, ln = _still_channel(SFS)
    no_wn = dataclasses.replace(
        ln, subframes=[dataclasses.replace(sf, sf_id=2, fields={}) for sf in ln.subframes]
    )
    with pytest.raises(ValueError, match="week unknown"):
        make_joint_observables([track], [no_wn], [_still_pilot()])
    assert make_joint_observables([track], [no_wn], [_still_pilot()], week=142)[0].week == 142


def test_pilot_phase_after_a_held_stretch_is_a_slip():
    # while the pilot loop coasted the signal was gone or too faded to use and the phase is
    # an extrapolation: it may be whole cycles away though every window still reads locked,
    # so the first phase after such a stretch begins a new ambiguity
    track, ln = _still_channel(SFS)
    base = make_joint_observables([track], [ln], [_still_pilot()])
    rx0 = receiver_time0([track], [ln])
    times = [o.tow - rx0 for o in base]  # s into the recording, one per second
    i = next(n for n, t in enumerate(times) if t > 10.0)
    k = round(times[i] * 1000)  # the epoch of that observation
    held = np.zeros(30000, dtype=bool)
    held[k + 100 : k + 500] = True  # wholly between observations i and i + 1
    held[k + 2900 : k + 3100] = True  # around observation i + 3
    pilot = dataclasses.replace(_still_pilot(), held=held)
    joint = make_joint_observables([track], [ln], [pilot])
    pairs = list(zip(joint, base, strict=True))
    flagged = [n for n, (o, b) in enumerate(pairs) if _g5q(o)[0][4] != _g5q(b)[0][4]]
    assert flagged == [i + 1, i + 3, i + 4]
    assert all(_g5q(joint[n])[0][4] == LLI_SLIP for n in flagged)
    for o, b in pairs:
        # the values themselves are untouched, and so is L1's indicator
        np.testing.assert_array_equal(_g5q(o)[0][:4], _g5q(b)[0][:4])
        assert _g5q(o)[1][4] == _g5q(b)[1][4]


def test_pilot_observations_in_the_first_two_seconds_are_flagged():
    track, ln = _still_channel(SFS)
    joint = make_joint_observables([track], [ln], [_still_pilot(k_start=3000)], interval=0.25)
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0 - 3.0, _g5q(o)[0]) for o in joint]  # time into the pilot segment
    early = [l5[4] for t, l5 in rows if 1.502 < t <= 2.0]
    later = [l5[4] for t, l5 in rows if t > 2.25 and np.isfinite(l5[0])]
    assert len(early) == 2 and set(early) == {LLI_SLIP} and later and not any(later)
