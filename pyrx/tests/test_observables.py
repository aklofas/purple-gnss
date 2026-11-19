import dataclasses
from functools import cache

import numpy as np
import pytest
from support import synth_l1ca_faded

from pyrx.bitsync import BitStream
from pyrx.codes import CA_PERIOD
from pyrx.gpstime import C_LIGHT, WEEK_S, full_week
from pyrx.lnav import PREAMBLE, Subframe
from pyrx.navdecode import LnavResult, decode_lnav
from pyrx.observables import (
    FADE_SLIP,
    FADE_SPAN,
    LLI_HALF_CYCLE,
    LLI_SLIP,
    NEAR_WRAP,
    PILOT_AGREEMENT,
    _near_wrap,
    code_starts,
    make_joint_observables,
    make_observables,
    pilot_transmit_times,
    receiver_time0,
    transmit_times,
)
from pyrx.samples import ArraySamples
from pyrx.signals import L1_FREQ
from pyrx.signalspec import L5, SIGNALS
from pyrx.synth import frame, synth_l1ca
from pyrx.tracking import T_PILOT_COH, TrackResult, track_l1ca

L5Q_ALIGN = L5.phase_align  # RINEX's quarter cycle between L5Q and L5I
PILOT_SUM = round(T_PILOT_COH / CA_PERIOD)  # epochs in a pilot loop's coherent sum
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


def _faded_channel(held_epochs, interval=1.0, phase_lock=None):
    """`_still_channel` with the integrator hold marked at `held_epochs` (offsets from the
    tracking epoch of the sixth observation at a 1 s interval), its observations, those of
    the same channel without a hold, and that epoch."""
    sfs = [(1500, 5044, False), (7500, 5050, False)]
    track, ln = _still_channel(sfs, phase_lock=phase_lock)
    rx0 = receiver_time0([track], [ln])
    t5 = make_observables([track], [ln])[5].tow
    k1 = int(np.searchsorted(track.t, t5 - rx0, side="right")) - 1
    base = make_observables([track], [ln], interval=interval)
    held = np.zeros(30000, dtype=bool)
    held[k1 + np.array(held_epochs, dtype=int)] = True
    epochs = make_observables([dataclasses.replace(track, held=held)], [ln], interval=interval)
    assert [e.tow for e in epochs] == [e.tow for e in base]
    return epochs, base, k1


def _newly_flagged(epochs, base):
    return [
        i
        for i, (e, b) in enumerate(zip(epochs, base, strict=True))
        if e.obs[0].lli & LLI_SLIP and not b.obs[0].lli & LLI_SLIP
    ]


def test_a_fade_of_fifty_ms_flags_the_next_phase():
    # the loop's integrator hold marks a fade; in one the carrier can come back half a cycle
    # or more away with every window reading phase-locked
    epochs, base, _ = _faded_channel(range(100, 100 + round(FADE_SLIP / CA_PERIOD)))
    assert _newly_flagged(epochs, base) == [6]
    assert epochs[6].obs[0].carrier_phase == base[6].obs[0].carrier_phase
    assert [e for i, e in enumerate(epochs) if i != 6] == [b for i, b in enumerate(base) if i != 6]
    epochs, base, _ = _faded_channel(range(100, 100 + round(FADE_SLIP / CA_PERIOD) - 1))
    assert epochs == base


def test_the_flag_outlasts_a_fade_by_the_rest_of_the_span():
    # the fiftieth held epoch is the one before observation 6's: a span ending up to 450
    # epochs later still holds all fifty, so observation 7 is flagged too (the loop is still
    # settling)
    epochs, base, _ = _faded_channel(range(950, 1000))
    assert _newly_flagged(epochs, base) == [6, 7]
    # a fade over well before the observation flags that one alone
    epochs, base, _ = _faded_channel(range(500, 550))
    assert _newly_flagged(epochs, base) == [6]


def test_fade_time_is_counted_within_half_a_second():
    span = round(FADE_SPAN / CA_PERIOD)
    # two 25 ms holds, the second ending 499 epochs after the first began: one span holds both
    epochs, base, _ = _faded_channel([*range(100, 125), *range(100 + span - 25, 100 + span)])
    assert _newly_flagged(epochs, base) == [6]
    epochs, base, _ = _faded_channel([*range(100, 125), *range(101 + span, 126 + span)])
    assert epochs == base


def test_a_fade_flags_the_observations_while_it_is_within_the_span():
    # at 10 Hz: the 50th held epoch is offset 149, and a span ending up to offset 599 still
    # holds all fifty, so the five observations at offsets 200 to 600 are flagged
    epochs, base, _ = _faded_channel(range(100, 150), interval=0.1)
    track, ln = _still_channel([(1500, 5044, False), (7500, 5050, False)])
    t5 = make_observables([track], [ln])[5].tow  # the observation the offsets count from
    at = [round((base[i].tow - t5) / 0.1) for i in _newly_flagged(epochs, base)]
    assert at == [2, 3, 4, 5, 6]


def test_a_fade_with_several_observations_per_epoch():
    # two observations per tracking epoch: the scan since the previous phase is empty for the
    # second one, so each of the 451 faded epochs flags one observation
    epochs, base, _ = _faded_channel(range(100, 150), interval=0.0005)
    assert len(_newly_flagged(epochs, base)) == 451


def test_a_track_without_a_hold_record_and_a_fade_at_its_start():
    track, ln = _still_channel([(1500, 5044, False), (7500, 5050, False)])
    assert track.held is None
    base = make_observables([track], [ln])
    held = np.zeros(30000, dtype=bool)
    # inside the first span after the pull-in, before the channel's first observation
    held[1000:1060] = True
    assert make_observables([dataclasses.replace(track, held=held)], [ln]) == base
    held[:] = False
    assert make_observables([dataclasses.replace(track, held=held)], [ln]) == base


def test_faded_signal_flags_its_observations():
    # a tracked 40 dB-Hz signal 10 dB down from 8 s for 2 s: the phases in and just after the
    # fade carry the flag, the ones before it (past the lock-in) do not
    fs = 2.5e6
    x = synth_l1ca_faded(4, fs, 0.0, 500, int(fs * 14.0), 40.0, 8.0, 2.0, 10.0, seed=0)
    track = track_l1ca(ArraySamples(x, fs), 0.0, 4, 500 / fs, 3.0, 40.0)
    assert track.locked and track.held.any()
    sfs = []
    for epoch, tow in ((1500, 5044), (7500, 5050)):
        bits = np.zeros(300, dtype=np.uint8)
        bits[:8] = PREAMBLE
        sfs.append(Subframe(4, 1, tow, {"wn": 142}, epoch, bits))
    stream = BitStream(np.zeros(1, np.uint8), np.array([1500]), 1500 % 20, 1.0, False)
    ln = LnavResult(4, stream, sfs, None, None)
    epochs = make_observables([track], [ln])
    rx0 = receiver_time0([track], [ln])
    slips = [e.tow - rx0 for e in epochs if e.obs[0].lli & LLI_SLIP]
    plain = make_observables([dataclasses.replace(track, held=None)], [ln])
    before = [e.tow - rx0 for e in plain if e.obs[0].lli & LLI_SLIP]
    assert any(8.0 < t <= 10.5 for t in slips) and not any(8.0 < t <= 10.5 for t in before)
    assert not any(3.5 < t <= 8.0 for t in slips)
    assert [e.obs[0].carrier_phase for e in epochs] == [e.obs[0].carrier_phase for e in plain]


def _held_channel(subframes, *holds):
    """`_still_channel` with the integrator hold marked over each (first, last) epoch range of
    `holds`: its observations, those of the same channel without a hold, and the tracking
    epoch of each observation."""
    track, ln = _still_channel(subframes)
    base = make_observables([track], [ln])
    rx0 = receiver_time0([track], [ln])
    ks = [int(np.searchsorted(track.t, e.tow - rx0, side="right")) - 1 for e in base]
    held = np.zeros(30000, dtype=bool)
    for first, last in holds:
        held[first : last + 1] = True
    epochs = make_observables([dataclasses.replace(track, held=held)], [ln])
    assert [e.tow for e in epochs] == [e.tow for e in base]
    return epochs, base, ks


def _half_cycle(epochs, base, ks):
    """The tracking epochs of the observations that gained the half-cycle bit."""
    gained = [e.obs[0].lli & LLI_HALF_CYCLE and not b.obs[0].lli & LLI_HALF_CYCLE
              for e, b in zip(epochs, base, strict=True)]  # fmt: skip
    return [k for k, g in zip(ks, gained, strict=True) if g]


GAP = [(1500, 5044, False), (19500, 5062, False)]  # nothing decoded from 7.5 to 19.5 s


def test_two_fades_between_subframes_of_one_polarity_leave_the_half_cycle_open():
    # a slip of half a cycle in each of two fades leaves the data polarity as it was, so the
    # subframes on either side agree and say nothing about the phases between the fades:
    # those may be half a cycle off, from the first held epoch of the first fade to the end
    # of the last one's span
    epochs, base, ks = _held_channel(GAP, (9000, 9059), (15000, 15059))
    end = 15059 + round(FADE_SPAN / CA_PERIOD) - 50  # the last epoch whose span holds 50 ms
    assert _half_cycle(epochs, base, ks) == [k for k in ks if 9000 < k <= end]
    assert len(_half_cycle(epochs, base, ks)) == 7
    # the first phase after it is a slip with the half cycle settled again
    after = next(i for i, k in enumerate(ks) if k > end)
    assert epochs[after].obs[0].lli == LLI_SLIP and epochs[after + 1].obs[0].lli == 0
    # the phases themselves are untouched
    assert [e.obs[0].carrier_phase for e in epochs] == [b.obs[0].carrier_phase for b in base]


def test_one_fade_between_subframes_leaves_the_half_cycle_open_inside_it():
    # the polarity came back: the fade held no slip or two, and the phases after it are whole
    # cycles from those before. With two, the phases between them, inside the fade, are half
    # a cycle off (a long flickering obstruction is one fade)
    epochs, base, ks = _held_channel(GAP, (9000, 9059))
    end = 9059 + round(FADE_SPAN / CA_PERIOD) - 50
    assert _half_cycle(epochs, base, ks) == [k for k in ks if 9000 < k <= end] == [9431]
    assert _newly_flagged(epochs, base) == [7, 8]
    # a fade of several seconds
    epochs, base, ks = _held_channel(GAP, (9000, 12059))
    assert _half_cycle(epochs, base, ks) == [9431, 10431, 11431, 12431]


def test_a_fade_belongs_where_its_held_epochs_are():
    # the flag outlasts the hold by up to 0.45 s, but the slip is where the loop was held: a
    # hold wholly under a decoded subframe opens nothing, however close to its end
    sfs = [(1500, 5044, False), (7500, 5050, False)]  # the last one ends at 13.5 s
    epochs, base, ks = _held_channel(sfs, (13300, 13359))
    assert _half_cycle(epochs, base, ks) == [] and _newly_flagged(epochs, base) == [11, 12]
    epochs, base, ks = _held_channel(GAP, (7400, 7459))  # under the first, the gap after it
    assert _half_cycle(epochs, base, ks) == []
    # nor does one that runs across the boundary of two subframes decoded back to back
    epochs, base, ks = _held_channel(SFS, (13400, 14059))
    assert _half_cycle(epochs, base, ks) == [] and _newly_flagged(epochs, base) == [12, 13]
    epochs, base, ks = _held_channel(sfs, (13499, 13558))  # its last epoch is the first after
    assert _half_cycle(epochs, base, ks) == [k for k in ks if k > 13499]


def test_fades_under_decoded_subframes_are_slips_only():
    # subframes decoded back to back show the polarity held through them
    epochs, base, ks = _held_channel(SFS, (9000, 9059), (15000, 15059))
    assert _half_cycle(epochs, base, ks) == [] and _newly_flagged(epochs, base) == [7, 8, 13, 14]


def test_a_fade_after_the_last_subframe_leaves_the_half_cycle_open_to_the_end():
    # nothing decoded after it says which half cycle the loop settled on
    sfs = [(1500, 5044, False), (7500, 5050, False)]  # the last one ends at 13.5 s
    epochs, base, ks = _held_channel(sfs, (20000, 20059))
    assert _half_cycle(epochs, base, ks) == [k for k in ks if k > 20000]
    assert len(_half_cycle(epochs, base, ks)) >= 8
    # and one that ended under the last subframe does not
    epochs, base, ks = _held_channel(sfs, (12000, 12059))
    assert _half_cycle(epochs, base, ks) == []


def _unlocked_channel(subframes, window, *holds):
    """`_held_channel` with the windows from `window[0]` to `window[1]` not phase-locked, in
    the channel and in the one it is compared with: what the half-cycle rule adds to the
    blank phases and the slip that an unlocked window always is."""
    track, ln = _still_channel(subframes, phase_lock=_unlocked(*window))
    base = make_observables([track], [ln])  # no hold record: the half-cycle rule is off
    rx0 = receiver_time0([track], [ln])
    ks = [int(np.searchsorted(track.t, e.tow - rx0, side="right")) - 1 for e in base]
    held = np.zeros(30000, dtype=bool)
    for first, last in holds:
        held[first : last + 1] = True
    epochs = make_observables([dataclasses.replace(track, held=held)], [ln])
    if not holds:
        assert [e.obs[0].lli & LLI_SLIP for e in epochs] == [e.obs[0].lli & LLI_SLIP for e in base]
    return epochs, base, ks


def test_a_window_without_phase_lock_leaves_the_half_cycle_open_as_a_fade_does():
    # the Costas loop that locks again has settled on either half cycle, with no hold of the
    # integrator to show for it. After the last subframe nothing decoded says which: the
    # phases from the window on carry the half-cycle bit, where they once had the slip alone
    sfs = [(1500, 5044, False), (7500, 5050, False)]  # the last one ends at 13.5 s
    epochs, base, ks = _unlocked_channel(sfs, (16000, 17000))
    assert _half_cycle(epochs, base, ks) == [k for k in ks if k > 16000]
    after = next(i for i, k in enumerate(ks) if k >= 17000)
    assert epochs[after].obs[0].lli == LLI_SLIP | LLI_HALF_CYCLE
    assert epochs[after + 1].obs[0].lli == LLI_HALF_CYCLE
    assert np.isfinite(epochs[after].obs[0].carrier_phase)
    # between two subframes of one polarity the half cycle is open inside the window, whose
    # phases are withheld, and settled after it
    epochs, base, ks = _unlocked_channel(GAP, (10000, 11000))
    assert _half_cycle(epochs, base, ks) == [k for k in ks if 10000 < k < 11000]
    assert all(np.isnan(epochs[ks.index(k)].obs[0].carrier_phase) for k in ks if 10000 < k < 11000)
    # with a fade later in the same gap, from the window to the end of the fade's flag
    epochs, base, ks = _unlocked_channel(GAP, (10000, 11000), (15000, 15059))
    end = 15059 + round(FADE_SPAN / CA_PERIOD) - 50
    assert _half_cycle(epochs, base, ks) == [k for k in ks if 10000 < k <= end]
    # and a window under subframes decoded back to back opens nothing
    epochs, base, ks = _unlocked_channel(SFS, (10000, 11000))
    assert _half_cycle(epochs, base, ks) == []


def test_a_held_epoch_counts_for_the_phases_after_it():
    # the carrier phase of an epoch is the loop's state before that epoch's update: the hold
    # that completes a fade at an observation's own epoch is the next phase's business
    track, ln = _still_channel(SFS)
    rx0 = receiver_time0([track], [ln])
    k = int(np.searchsorted(track.t, make_observables([track], [ln])[6].tow - rx0, side="right")) - 1
    epochs, base, _ = _held_channel(SFS, (k - 49, k))  # the fiftieth held epoch is k itself
    assert _newly_flagged(epochs, base) == [7]
    epochs, base, _ = _held_channel(SFS, (k - 50, k - 1))
    assert _newly_flagged(epochs, base) == [6, 7]  # and 7 by the rest of the span


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


# the same channel with the week boundary 13.5 s into it, and one that begins 2 s before it; near
# 604800 s a double resolves 0.12 ns, 3.5 cm of range, so the bounds there are 5 cm
SFS_ACROSS = [(1500, 604788, False), (7500, 604794, False), (13500, 0, False), (19500, 6, False)]
SFS_AFTER = [(2000, 0, False), (8000, 6, False), (14000, 12, False), (20000, 18, False)]


@pytest.mark.parametrize("pilot_start", [0, 14000])
def test_joint_observables_across_the_week_rollover(pilot_start):
    # the pilot takes L1's time on either side of the boundary, also a segment that begins
    # after it; the epochs run on past 604800 in the week of the first one
    track, ln = _still_channel(SFS_ACROSS)
    pilot = _still_pilot(k_start=pilot_start, n=30000 - pilot_start)
    single = make_observables([track], [ln])
    joint = make_joint_observables([track], [ln], [pilot])
    assert [(o.week, o.tow) for o in joint] == [(e.week, e.tow) for e in single]
    assert joint[0].week == 2190 and joint[0].tow < 604800.0 < joint[-1].tow
    rx0 = receiver_time0([track], [ln])
    seen = []
    for o in joint:
        (pr, phase, _, _, _), l1 = _g5q(o)
        if o.tow - rx0 <= pilot.t[pilot.sec_sync + 1]:
            assert np.isnan(pr)
            continue
        seen.append(o.tow)
        assert pr - l1[0] == pytest.approx(C_LIGHT * 30e-9, abs=0.05)
        since = o.tow - rx0 - pilot.t[0] - 30e-9  # the segment's phase starts with it
        assert phase == pytest.approx(420.0 * since + L5Q_ALIGN, abs=1e-4)
    assert seen[0] < 604800.0 + 16.0 and seen[-1] > 604800.0 and len(seen) > 10


@pytest.mark.parametrize("week", [None, 2190, 2191])
def test_joint_observables_of_a_run_wholly_past_the_rollover(week):
    # sample 0 lies 2 s before the boundary, every observation after it: the epochs are
    # dated in the new week, given or derived, and the pilot is on the same clock
    track, ln = _still_channel(SFS_AFTER, k0=2500)
    pilot = _still_pilot()
    assert 604798.0 < receiver_time0([track], [ln]) < 604800.0
    single = make_observables([track], [ln], week=week)
    joint = make_joint_observables([track], [ln], [pilot], week=week)
    assert [(o.week, o.tow) for o in joint] == [(e.week, e.tow) for e in single]
    # a week that is given is the one written, also against the decoded one
    assert joint[0].week == (week or 2190) and 0.0 < joint[0].tow < 2.0 and len(joint) > 20
    for o in joint:
        (pr, _, _, _, _), l1 = _g5q(o)
        assert pr - l1[0] == pytest.approx(C_LIGHT * 30e-9, abs=0.05)
        assert l1[0] == pytest.approx(C_LIGHT * 0.068, abs=0.05)


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


def test_joint_builder_takes_the_pilot_identity_from_its_description(monkeypatch):
    # the signal a pilot is reported as and the alignment of its phase are its description's
    track, ln = _still_channel(SFS)
    usual = make_joint_observables([track], [ln], [_still_pilot()])
    other = dataclasses.replace(SIGNALS["L5"], rinex="G5X", phase_align=0.125)
    monkeypatch.setitem(SIGNALS, "L5", other)
    joint = make_joint_observables([track], [ln], [_still_pilot()])
    assert all(o.signals == ("G1C", "G5X") for o in joint)
    pairs = zip(joint, usual, strict=True)
    shift = [o.carrier_phase[0, 1] - u.carrier_phase[0, 1] for o, u in pairs]
    shift = [d for d in shift if np.isfinite(d)]
    assert len(shift) > 20 and np.allclose(shift, 0.125 - L5Q_ALIGN, rtol=0, atol=1e-9)


def test_pilot_contributes_from_the_epoch_after_its_secondary_code_was_found():
    # at 0.5 ms the grid has an instant in every tracking epoch: the first one with an L5
    # row lies in epoch sec_sync + 1
    track, ln = _still_channel(SFS)
    pilot = _still_pilot(sec_sync=1600)
    joint = make_joint_observables([track], [ln], [pilot], interval=0.0005)
    rx0 = receiver_time0([track], [ln])
    first = next(o for o in joint if np.isfinite(_g5q(o)[0][0]))
    starts = code_starts(pilot)
    assert starts[1601] <= first.tow - rx0 < starts[1601] + 0.0005


def _still_e5a(travel=0.080, prn=11, n=30000, k_start=0, sec_sync=1500, polarity=1, wraps=14,
               phase_lock=None, held=None, doppler=-420.0):  # fmt: skip
    """A Galileo E5a pilot segment beside `_still_channel(SFS)`: code starts 1 ms apart, the
    one of its epoch 0 transmitted on a whole millisecond about `travel` s before it arrives
    by the receiver clock (the true travel time is returned: the millisecond rounds it).
    Epoch k carries the chip of the 100-chip secondary code that the millisecond of its
    transmit time names; `wraps` moves `sec_origin` by whole code lengths."""
    track, ln = _still_channel(SFS)
    rx0 = receiver_time0([track], [ln])
    t = (k_start + np.arange(n)) * CA_PERIOD
    delay = 0.2e-3  # the code starts' place in the receiver's millisecond
    ms0 = round((rx0 + t[0] + delay - travel) / CA_PERIOD)  # millisecond of week of epoch 0
    true_travel = rx0 + t[0] + delay - ms0 * CA_PERIOD
    pilot = TrackResult(
        prn, t, np.zeros((n, 4), np.complex64), (t + delay) % CA_PERIOD, np.full(n, doppler),
        doppler * CA_PERIOD * np.arange(1, n + 1), np.zeros(n), np.full(n, 44.0),
        np.ones(n) if phase_lock is None else phase_lock, True, None, np.zeros(n), np.zeros(n),
        signal="E5A", sec_sync=sec_sync, sec_origin=(-ms0) % 100 + 100 * wraps,
        sec_polarity=polarity, data_prompt=np.zeros(n, np.complex64), held=held,
    )  # fmt: skip
    return pilot, true_travel


def _e5q(epoch, sat="E11"):
    """(pseudorange, phase, doppler, cn0, lli) of a satellite's E5Q at an epoch."""
    i, j = epoch.sats.index(sat), epoch.signals.index("E5Q")
    return [getattr(epoch, f)[i, j] for f in ("pseudorange", "carrier_phase", "doppler", "cn0", "lli")]


GALILEO = ("L5", "E5A")


@pytest.mark.parametrize("travel", [0.0225, 0.066, 0.080, 0.0995, 0.1215])
@pytest.mark.parametrize("wraps", [-3, 0, 14])
def test_galileo_pilot_takes_its_time_from_its_secondary_code(travel, wraps):
    # the code start with secondary chip 0 left at a multiple of 100 ms; which one follows
    # from the receiver clock, the travel time taken between 22 and 122 ms
    track, ln = _still_channel(SFS)
    pilot, true_travel = _still_e5a(travel, wraps=wraps)
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=GALILEO)
    single = make_observables([track], [ln])
    assert [(o.week, o.tow) for o in joint] == [(e.week, e.tow) for e in single]
    seen = 0
    for o in joint:
        assert o.signals == ("G1C", "G5Q", "E5Q")
        if "E11" not in o.sats:
            assert o.sats == ("G04",)
            continue
        seen += 1
        assert o.sats == ("E11", "G04")
        assert _e5q(o)[0] == pytest.approx(C_LIGHT * true_travel, abs=1e-3)
        assert np.all(np.isnan(o.pseudorange[0, :2])) and np.isnan(o.pseudorange[1, 2])
    assert seen > 20 and abs(true_travel - travel) < 1e-3


@pytest.mark.parametrize("travel", [0.0215, 0.1225])
def test_a_travel_time_outside_the_window_is_taken_a_code_length_off(travel):
    # 100 ms of ambiguity: outside 22 to 122 ms the neighboring multiple is the one taken
    track, ln = _still_channel(SFS)
    pilot, true_travel = _still_e5a(travel)
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=GALILEO)
    off = 0.1 if travel < 0.1 else -0.1
    assert _e5q(joint[-1])[0] == pytest.approx(C_LIGHT * (true_travel + off), abs=1e-3)


@pytest.mark.parametrize("polarity", [1, -1])
def test_joint_e5q_values(polarity):
    track, ln = _still_channel(SFS)
    pilot, _ = _still_e5a(polarity=polarity)
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=GALILEO)
    rx0 = receiver_time0([track], [ln])
    seen = 0
    for o in joint:
        t = o.tow - rx0
        if t <= pilot.t[pilot.sec_sync + 1]:
            assert "E11" not in o.sats  # nothing before the sync
            continue
        seen += 1
        _, phase, doppler, cn0, lli = _e5q(o)
        assert doppler == -420.0 and cn0 == 44.0
        # the negated accumulated phase, the half cycle of an inverted pilot, and RINEX's
        # quarter cycle that aligns E5a-Q with E5a-I
        adr = -420.0 * (t - 0.2e-3)
        half = 0.5 if polarity < 0 else 0.0
        assert phase == pytest.approx(-(adr + half) - 0.25, abs=1e-6)
        assert lli == (LLI_SLIP if seen == 1 or t <= 2.0 else 0)
    assert seen > 20 and SIGNALS["E5A"].phase_align == -0.25


def test_each_pilot_is_aligned_by_its_own_description(monkeypatch):
    # a Galileo pilot beside a GPS one: each phase gets its own signal's alignment
    track, ln = _still_channel(SFS)
    pilots = [_still_pilot(), _still_e5a()[0]]
    usual = make_joint_observables([track], [ln], pilots, pilot_signals=GALILEO)
    monkeypatch.setitem(SIGNALS, "E5A", dataclasses.replace(SIGNALS["E5A"], phase_align=0.125))
    moved = make_joint_observables([track], [ln], pilots, pilot_signals=GALILEO)
    for o, u in zip(moved, usual, strict=True):
        if "E11" in o.sats:
            assert _e5q(o)[1] - _e5q(u)[1] == pytest.approx(0.375, abs=1e-9)
        np.testing.assert_array_equal(_g5q(o)[0][1], _g5q(u)[0][1])
    assert sum("E11" in o.sats for o in moved) > 20


def test_galileo_pilot_flags_follow_the_gps_pilot_rules():
    # a window without phase lock blanks the phase and flags the next one; a coasted sum
    # flags the phases up to the one that holds the loop's return
    lock = np.ones(30000)
    lock[9000:10000] = 0.1
    held = np.zeros(30000, dtype=bool)
    held[15100:15301:20] = True
    track, ln = _still_channel(SFS)
    pilot, _ = _still_e5a(phase_lock=lock, held=held)
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=GALILEO)
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0, _e5q(o)) for o in joint if "E11" in o.sats]  # by file time
    blank = [n for n, (_, r) in enumerate(rows) if np.isnan(r[1])]
    assert len(blank) == 1 and 9.0 <= rows[blank[0]][0] <= 10.0
    assert np.isfinite(rows[blank[0]][1][0])
    after_coast = next(n for n, (t, _) in enumerate(rows) if t > 15.301)
    flagged = [n for n, (t, r) in enumerate(rows) if r[4] and t > 3.0]
    assert flagged == [blank[0], blank[0] + 1, after_coast]


def test_a_galileo_and_a_gps_satellite_of_one_number_are_two_rows():
    track, ln = _still_channel(SFS)
    e5a, _ = _still_e5a(prn=4)
    joint = make_joint_observables([track], [ln], [_still_pilot(), e5a], pilot_signals=GALILEO)
    last = joint[-1]
    assert last.sats == ("E04", "G04")
    assert np.isnan(last.pseudorange[0, :2]).all() and np.isfinite(last.pseudorange[0, 2])
    assert np.isfinite(last.pseudorange[1, :2]).all() and np.isnan(last.pseudorange[1, 2])
    # and the GPS rows are what they are without Galileo
    plain = make_joint_observables([track], [ln], [_still_pilot()])
    for o, u in zip(joint, plain, strict=True):
        i = o.sats.index("G04")
        np.testing.assert_array_equal(o.pseudorange[i, :2], u.pseudorange[0])
        np.testing.assert_array_equal(o.carrier_phase[i, :2], u.carrier_phase[0])
        np.testing.assert_array_equal(o.lli[i, :2], u.lli[0])


@pytest.mark.parametrize("k_start", [0, 6000, 6037])
def test_galileo_segments_time_themselves_wherever_they_begin(k_start):
    # a segment that begins late, also off the 100 ms grid of the first: its own secondary
    # origin and its own first code start give the same time as a segment from the start
    track, ln = _still_channel(SFS)
    whole, travel = _still_e5a()
    late, late_travel = _still_e5a(k_start=k_start, n=30000 - k_start)
    assert late_travel == pytest.approx(travel, abs=1e-9)
    a = make_joint_observables([track], [ln], [whole], pilot_signals=GALILEO)
    b = make_joint_observables([track], [ln], [late], pilot_signals=GALILEO)
    rx0 = receiver_time0([track], [ln])
    seen = 0
    for o, u in zip(b, a, strict=True):
        if "E11" in o.sats:
            seen += 1
            assert o.tow - rx0 > late.t[late.sec_sync + 1]
            assert _e5q(o)[0] == pytest.approx(_e5q(u)[0], abs=1e-3)
    assert seen > 15


def test_two_segments_of_a_galileo_satellite_found_with_opposite_polarity():
    # re-acquired, the pilot can be found inverted where it was upright before: each
    # segment's phase gets its own half cycle, and both are on the same time
    track, ln = _still_channel(SFS)
    first, travel = _still_e5a(n=12000, polarity=1)
    second, _ = _still_e5a(k_start=14000, n=16000, polarity=-1)
    joint = make_joint_observables([track], [ln], [first, second], pilot_signals=GALILEO)
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0, _e5q(o)) for o in joint if "E11" in o.sats]
    early = [(t, r) for t, r in rows if t < 12.0]
    late = [(t, r) for t, r in rows if t > 14.0 + 1.6]
    assert len(early) > 8 and len(late) > 10 and len(early) + len(late) == len(rows)
    assert all(r[0] == pytest.approx(C_LIGHT * travel, abs=1e-3) for _, r in rows)
    for t, r in early:
        assert r[1] == pytest.approx(420.0 * (t - 0.2e-3) - 0.25, abs=1e-6)
    for t, r in late:  # the second segment's phase begins with it, half a cycle put back
        assert r[1] == pytest.approx(420.0 * (t - 14.0 - 0.2e-3) - 0.5 - 0.25, abs=1e-6)
    assert late[0][1][4] == LLI_SLIP  # and its first phase is a slip


def test_galileo_pilot_without_the_secondary_code_is_left_out():
    track, ln = _still_channel(SFS)
    pilot, _ = _still_e5a()
    joint = make_joint_observables(
        [track], [ln], [dataclasses.replace(pilot, sec_sync=None)], pilot_signals=GALILEO
    )
    assert all(o.sats == ("G04",) and o.signals == ("G1C", "G5Q", "E5Q") for o in joint)


def test_galileo_pilot_across_the_week_rollover():
    track, ln = _still_channel(SFS_ACROSS)
    rx0 = receiver_time0([track], [ln])
    t = np.arange(30000) * CA_PERIOD
    ms0 = round((rx0 + 0.2e-3 - 0.080) / CA_PERIOD)
    travel = rx0 + 0.2e-3 - ms0 * CA_PERIOD
    pilot = dataclasses.replace(_still_e5a()[0], sec_origin=(-ms0) % 100)
    assert (pilot.t == t).all()
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=GALILEO)
    rows = [(o.tow, _e5q(o)[0]) for o in joint if "E11" in o.sats]
    assert rows[0][0] < 604800.0 < rows[-1][0] and len(rows) > 20
    # near 604800 s a double resolves 3.5 cm of range
    assert all(pr == pytest.approx(C_LIGHT * travel, abs=0.05) for _, pr in rows)


def _still_e1(true_travel, prn=11, n=30000, k_start=0, sec_sync=1599, polarity=1, wraps=3,
              phase_lock=None, doppler=-560.0):  # fmt: skip
    """A Galileo E1 pilot segment beside `_still_channel(SFS)` whose code arrives
    `true_travel` s after it was sent: epochs of 1 ms, each a quarter of the 4 ms code, the
    segment beginning with a code period (up to three epochs after `k_start` for that), and
    epoch k carrying the entry of the secondary code, 25 chips held four epochs each, that
    the millisecond of its transmit time names."""
    track, ln = _still_channel(SFS)
    rx0 = receiver_time0([track], [ln])
    # the code starts' place against the receiver's millisecond, within half of one
    delay = (true_travel - rx0 + CA_PERIOD / 2) % CA_PERIOD - CA_PERIOD / 2
    ms0 = round((rx0 + k_start * CA_PERIOD + delay - true_travel) / CA_PERIOD)
    k_start += (-ms0) % 4
    ms0 += (-ms0) % 4  # the millisecond of week of epoch 0, the start of a code period
    t = (k_start + np.arange(n)) * CA_PERIOD
    return TrackResult(
        prn, t, np.zeros((n, 4), np.complex64), (t + delay) % CA_PERIOD, np.full(n, doppler),
        doppler * CA_PERIOD * np.arange(1, n + 1), np.zeros(n), np.full(n, 41.0),
        np.ones(n) if phase_lock is None else phase_lock, True, None, np.zeros(n), np.zeros(n),
        signal="E1", sec_sync=sec_sync, sec_origin=(-ms0) % 100 + 100 * wraps,
        sec_polarity=polarity, data_prompt=np.zeros(n, np.complex64),
    )  # fmt: skip


def _e1c(epoch, sat="E11"):
    """(pseudorange, phase, doppler, cn0, lli) of a satellite's E1C at an epoch."""
    i, j = epoch.sats.index(sat), epoch.signals.index("E1C")
    return [getattr(epoch, f)[i, j] for f in ("pseudorange", "carrier_phase", "doppler", "cn0", "lli")]


BOTH_BANDS = ("L5", "E1", "E5A")


@pytest.mark.parametrize("travel", [0.0225, 0.066, 0.0801234, 0.0995, 0.1215])
@pytest.mark.parametrize("wraps", [-2, 0, 3])
def test_e1_pilot_takes_its_time_from_its_secondary_code(travel, wraps):
    # the E1 pilot's secondary code lasts 100 ms too, one chip per 4 ms code period: the
    # same rule on the tracker's 1 ms epochs
    track, ln = _still_channel(SFS)
    pilot = _still_e1(travel, wraps=wraps)
    assert pilot.sec_origin % 4 == 0
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=BOTH_BANDS)
    seen = 0
    for o in joint:
        assert o.signals == ("G1C", "G5Q", "E1C", "E5Q")
        if "E11" not in o.sats:
            continue
        seen += 1
        assert _e1c(o)[0] == pytest.approx(C_LIGHT * travel, abs=1e-3)
        assert np.isnan(o.pseudorange[0, [0, 1, 3]]).all()
    assert seen > 20


@pytest.mark.parametrize("polarity", [1, -1])
def test_joint_e1c_values(polarity):
    track, ln = _still_channel(SFS)
    travel = 0.0801234
    pilot = _still_e1(travel, polarity=polarity)
    joint = make_joint_observables([track], [ln], [pilot], pilot_signals=BOTH_BANDS)
    rx0 = receiver_time0([track], [ln])
    seen = 0
    for o in joint:
        t = o.tow - rx0
        if t <= pilot.t[pilot.sec_sync + 1]:
            assert "E11" not in o.sats  # nothing before the sync
            continue
        seen += 1
        _, phase, doppler, cn0, lli = _e1c(o)
        assert doppler == -560.0 and cn0 == 41.0
        # the negated accumulated phase, the half cycle of an inverted pilot, and RINEX's
        # half cycle that aligns E1-C with E1-B
        adr = -560.0 * (t - code_starts(pilot)[0])
        half = 0.5 if polarity < 0 else 0.0
        assert phase == pytest.approx(-(adr + half) + 0.5, abs=1e-6)
        assert lli == (LLI_SLIP if seen == 1 or t - pilot.t[0] <= 2.0 else 0)
    assert seen > 20 and SIGNALS["E1"].phase_align == 0.5


def test_e1_and_e5a_of_a_satellite_are_two_columns_of_its_row():
    track, ln = _still_channel(SFS)
    e5a, travel = _still_e5a()
    joint = make_joint_observables(
        [track], [ln], [_still_pilot(), e5a, _still_e1(travel)], pilot_signals=BOTH_BANDS
    )
    plain = make_joint_observables([track], [ln], [_still_pilot()])
    both = 0
    for o, u in zip(joint, plain, strict=True):
        assert o.signals == ("G1C", "G5Q", "E1C", "E5Q")
        if "E11" in o.sats and np.isfinite(_e1c(o)[0]):
            both += 1
            # one satellite, one transmit time: the two bands' pseudoranges are the same
            assert _e1c(o)[0] == pytest.approx(_e5q(o)[0], abs=1e-3)
            assert _e1c(o)[0] == pytest.approx(C_LIGHT * travel, abs=1e-3)
            assert np.isfinite(_e1c(o)[1]) and np.isfinite(_e5q(o)[1])
        # the GPS row is what it is without Galileo
        i = o.sats.index("G04")
        np.testing.assert_array_equal(o.pseudorange[i, :2], u.pseudorange[0])
        np.testing.assert_array_equal(o.carrier_phase[i, :2], u.carrier_phase[0])
        assert np.isnan(o.pseudorange[i, 2:]).all()
    assert both > 20


@pytest.mark.parametrize("apart", [0.9e-6, -0.9e-6, 1.1e-6, -1.1e-6, 1e-3, 4e-3, -4e-3])
def test_e1_and_e5a_that_disagree_write_neither_where_both_are(apart):
    # nothing says which of the two is wrong, so where both pilots of a satellite are tracked
    # and their transmit times stand more than a microsecond apart, neither is written; each
    # is written where it is alone, and the other satellites are not touched
    track, ln = _still_channel(SFS)
    e5a, travel = _still_e5a(n=20000)  # 0 to 20 s
    e1 = _still_e1(travel + apart, k_start=8000, n=18000)  # 8 to 26 s
    other, _ = _still_e5a(prn=19)
    pilots = [e5a, e1, other]
    joint = make_joint_observables([track], [ln], pilots, pilot_signals=BOTH_BANDS)
    alone = make_joint_observables([track], [ln], [e5a, other], pilot_signals=BOTH_BANDS)
    rx0 = receiver_time0([track], [ln])
    agree = abs(apart) < PILOT_AGREEMENT
    counts = {"e5a": 0, "both": 0, "e1": 0, "neither": 0}
    for o, u in zip(joint, alone, strict=True):
        t = o.tow - rx0
        here = [np.isfinite(_e1c(o)[0]), np.isfinite(_e5q(o)[0])] if "E11" in o.sats else [False] * 2
        if t <= e1.t[e1.sec_sync + 1]:
            assert here == [False, t > e5a.t[e5a.sec_sync + 1]]
            counts["e5a"] += here[1]
        elif t < e5a.t[-1]:
            assert here == [agree, agree], (t, here)
            counts["both" if agree else "neither"] += 1
        elif t < e1.t[-1]:
            assert here == [True, False]
            counts["e1"] += 1
            assert _e1c(o)[0] == pytest.approx(C_LIGHT * (travel + apart), abs=1e-3)
        if here[1]:
            # what is written of E5a is what it is without the E1 pilot, flags too: a stretch
            # not written is no loss of lock
            np.testing.assert_array_equal(_e5q(o), _e5q(u))
        if "E19" in u.sats:
            np.testing.assert_array_equal(_e5q(o, "E19"), _e5q(u, "E19"))
    assert counts["e5a"] > 5 and counts["e1"] > 4
    assert counts["both" if agree else "neither"] > 8 and counts["neither" if agree else "both"] == 0
    assert PILOT_AGREEMENT == 1e-6


@pytest.mark.parametrize("event", ["none", "unlocked", "coasted", "near the wrap"])
def test_a_stretch_withheld_is_not_a_loss_of_lock(event):
    # E5a tracked for 30 s; from 8 to 18 s an E1 pilot of the satellite stands a code period
    # off, so neither is written while both are there. E5a's first phase after the stretch
    # carries on its line without a flag when the loop held all through; an unlocked window,
    # a coasted sum or a sum near the wrap inside the stretch is carried to that phase as it
    # would be to the next phase written
    lock = np.ones(30000)
    held = np.zeros(30000, dtype=bool)
    error = np.zeros(30000)
    if event == "unlocked":
        lock[12000:13000] = 0.1
    if event == "coasted":
        held[13100:13301:20] = True
    if event == "near the wrap":
        error[13100] = -NEAR_WRAP
    track, ln = _still_channel(SFS)
    e5a, travel = _still_e5a(phase_lock=lock, held=held)
    e5a = dataclasses.replace(e5a, carrier_error=error)
    e1 = _still_e1(travel + 4e-3, k_start=8000, n=10000)
    joint = make_joint_observables([track], [ln], [e5a, e1], pilot_signals=BOTH_BANDS)
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0, _e5q(o)) for o in joint if "E11" in o.sats]
    assert all(np.isnan(_e1c(o)[0]) for o in joint if "E11" in o.sats)  # E1 is never alone
    before = [(t, r) for t, r in rows if t < 12.0]
    after = [(t, r) for t, r in rows if t > 12.0]
    # nothing of the satellite from E1's sync to its end, both written sides of it there
    assert len(before) > 6 and len(after) > 10 and len(before) + len(after) == len(rows)
    assert before[-1][0] < e1.t[e1.sec_sync + 1] + 1.0 and e1.t[-1] < after[0][0] < e1.t[-1] + 1.0
    first = after[0][1]
    assert first[1] == pytest.approx(420.0 * (after[0][0] - 0.2e-3) - 0.25, abs=1e-6)
    assert first[4] == (0 if event == "none" else LLI_SLIP)
    assert not any(r[4] for _, r in after[1:])


def test_a_pilot_without_phase_lock_still_speaks_for_its_time():
    # the time of a pilot comes from its code and its secondary code, which hold without
    # phase lock: an E1 pilot whose phase is blank still withholds E5a where their times
    # disagree
    track, ln = _still_channel(SFS)
    e5a, travel = _still_e5a()
    e1 = _still_e1(travel + 4e-3, k_start=8000, n=10000, phase_lock=np.full(10000, 0.1))
    joint = make_joint_observables([track], [ln], [e5a, e1], pilot_signals=BOTH_BANDS)
    rx0 = receiver_time0([track], [ln])
    inside = [o for o in joint if e1.t[e1.sec_sync + 1] + 1.0 < o.tow - rx0 < e1.t[-1] - 1.0]
    assert len(inside) > 5 and all("E11" not in o.sats for o in inside)


def test_the_first_e1_phase_written_after_a_disagreement_is_flagged_as_a_first_phase():
    # an E1 segment whose every epoch beside E5a was withheld has written nothing yet when
    # E5a ends: its first written phase is a first phase
    track, ln = _still_channel(SFS)
    e5a, travel = _still_e5a(n=20000)
    e1 = _still_e1(travel + 4e-3, k_start=8000, n=18000)
    joint = make_joint_observables([track], [ln], [e5a, e1], pilot_signals=BOTH_BANDS)
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0, _e1c(o)) for o in joint if "E11" in o.sats and np.isfinite(_e1c(o)[0])]
    assert len(rows) > 4 and rows[0][0] > e5a.t[-1] > e1.t[0] + 10.0
    assert [r[4] for _, r in rows] == [LLI_SLIP] + [0] * (len(rows) - 1)


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


def _pilot_epoch_after(seconds):
    """The still L1 channel with its decode, the joint observations with the still pilot,
    the index of the first observation later than `seconds` into the recording and the
    pilot epoch it falls in."""
    track, ln = _still_channel(SFS)
    base = make_joint_observables([track], [ln], [_still_pilot()])
    rx0 = receiver_time0([track], [ln])
    times = [o.tow - rx0 for o in base]  # s into the recording, one per second
    i = next(n for n, t in enumerate(times) if t > seconds)
    k = int(np.searchsorted(code_starts(_still_pilot()), times[i], side="right")) - 1
    return track, ln, base, i, k


def _slip_changes(joint, base):
    pairs = list(zip(joint, base, strict=True))
    return [n for n, (o, b) in enumerate(pairs) if _g5q(o)[0][4] != _g5q(b)[0][4]]


def test_pilot_phase_after_a_coasted_stretch_is_a_slip():
    # while the pilot loop coasted the signal was gone or too faded to use and the phase is
    # an extrapolation: it may be whole cycles away though every window still reads locked,
    # so a phase after such a stretch begins a new ambiguity
    track, ln, base, i, k = _pilot_epoch_after(10.0)
    held = np.zeros(30000, dtype=bool)  # the tracker marks the epoch completing a coasted sum
    held[k + 100 : k + 501 : PILOT_SUM] = True  # wholly between observations i and i + 1
    held[k + 2900 : k + 3101 : PILOT_SUM] = True  # around observation i + 3
    pilot = dataclasses.replace(_still_pilot(), held=held)
    joint = make_joint_observables([track], [ln], [pilot])
    flagged = _slip_changes(joint, base)
    assert flagged == [i + 1, i + 3, i + 4]
    assert all(_g5q(joint[n])[0][4] == LLI_SLIP for n in flagged)
    for o, b in zip(joint, base, strict=True):
        # the values themselves are untouched, and so is L1's indicator
        np.testing.assert_array_equal(_g5q(o)[0][:4], _g5q(b)[0][:4])
        assert _g5q(o)[1][4] == _g5q(b)[1][4]


@pytest.mark.parametrize("offset", [-PILOT_SUM - 1, -PILOT_SUM, -1, 0, 1, PILOT_SUM, PILOT_SUM + 1])
def test_the_first_pilot_phase_holding_the_return_from_a_coast_is_a_slip(offset):
    # the loop returns to the signal with the first sum it takes after the coasted ones, and
    # that update reaches the phase one epoch after the sum is complete: an observation
    # between the last coasted sum and that epoch is still on the old cycle, so the one
    # after it has to be flagged too
    track, ln, base, i, k = _pilot_epoch_after(10.0)
    last = k + offset  # the epoch completing the last coasted sum
    held = np.zeros(30000, dtype=bool)
    held[last - 10 * PILOT_SUM : last + 1 : PILOT_SUM] = True
    still = _still_pilot()
    phase = still.carrier_phase.copy()
    phase[last + PILOT_SUM + 1 :] += 1.0  # the loop comes back a whole cycle away
    pilot = dataclasses.replace(still, held=held, carrier_phase=phase)
    joint = make_joint_observables([track], [ln], [pilot])
    moved = [
        n for n, (o, b) in enumerate(zip(joint, base, strict=True))
        if _g5q(o)[0][1] != _g5q(b)[0][1]
    ]  # fmt: skip
    assert moved[0] == (i if offset == -PILOT_SUM - 1 else i + 1)
    assert _g5q(joint[moved[0]])[0][4] == LLI_SLIP
    assert _slip_changes(joint, base) == list(range(i, moved[0] + 1))


def test_a_pilot_phase_after_a_sum_near_the_wrap_is_a_slip():
    # the pilot loop works on the four-quadrant phase of its coherent sums, which wraps at
    # half a cycle: after a sum that close to the wrap the loop may be pulling the other
    # way, and the phase then continues a whole cycle off with every window reading locked
    track, ln, base, i, k = _pilot_epoch_after(10.0)
    error = np.zeros(30000)  # the tracker writes a sum's phase on the epoch completing it
    error[k + 300] = NEAR_WRAP  # between observations i and i + 1
    error[k + 2100] = -0.46  # between i + 2 and i + 3
    error[k + 4100 : k + 4200 : PILOT_SUM] = 0.3  # a loop far off, but not near the wrap
    pilot = dataclasses.replace(_still_pilot(), carrier_error=error)
    joint = make_joint_observables([track], [ln], [pilot])
    flagged = _slip_changes(joint, base)
    assert flagged == [i + 1, i + 3]
    assert all(_g5q(joint[n])[0][4] == LLI_SLIP for n in flagged)
    for o, b in zip(joint, base, strict=True):
        # the values themselves are untouched, and so is L1's indicator
        np.testing.assert_array_equal(_g5q(o)[0][:4], _g5q(b)[0][:4])
        assert _g5q(o)[1][4] == _g5q(b)[1][4]


@pytest.mark.parametrize("offset,after", [(-1000, 0), (-1, 0), (0, 1), (999, 1), (1000, 2)])
def test_a_sum_near_the_wrap_is_the_business_of_the_phases_after_it(offset, after):
    # the sum's update moves the carrier from the next epoch on: an observation in the
    # epoch that completes the sum is the loop's state before it, and the one after it
    # carries the flag (the observations are 1000 epochs apart)
    track, ln, base, i, k = _pilot_epoch_after(10.0)
    error = np.zeros(30000)
    error[k + offset] = 0.4
    pilot = dataclasses.replace(_still_pilot(), carrier_error=error)
    joint = make_joint_observables([track], [ln], [pilot])
    assert _slip_changes(joint, base) == [i + after]


@pytest.mark.parametrize(
    "phase,near",
    [
        (NEAR_WRAP, True),
        (-NEAR_WRAP, True),
        (0.5, True),
        (-0.5, True),
        (np.nextafter(NEAR_WRAP, 0.0), False),
        (-np.nextafter(NEAR_WRAP, 0.0), False),
        (0.25, False),
        (np.nan, False),  # a sum the loop coasted on has no phase; the coasting rule has it
    ],
)
def test_near_the_wrap_is_a_sum_phase_of_that_size_or_more(phase, near):
    track, ln, base, i, k = _pilot_epoch_after(10.0)
    error = np.zeros(30000)
    error[k + 300] = phase
    pilot = dataclasses.replace(_still_pilot(), carrier_error=error)
    assert _near_wrap(pilot).tolist() == [n == k + 300 and near for n in range(30000)]
    joint = make_joint_observables([track], [ln], [pilot])
    assert _slip_changes(joint, base) == ([i + 1] if near else [])


def test_near_the_wrap_begins_at_0_35_cycle():
    # under the smallest sum phase the three whole-cycle slips of the car-roof captures
    # passed (0.41 cycle) by 0.06, and twice what a steady pilot's sums reach at 26 dB-Hz
    assert NEAR_WRAP == 0.35
    track, ln, base, i, k = _pilot_epoch_after(10.0)
    for phase, near in ((0.34, False), (0.36, True)):
        error = np.zeros(30000)
        error[k + 300] = phase
        pilot = dataclasses.replace(_still_pilot(), carrier_error=error)
        joint = make_joint_observables([track], [ln], [pilot])
        assert _slip_changes(joint, base) == ([i + 1] if near else [])


def test_only_a_coherent_sum_can_be_near_the_wrap():
    # up to the epoch of the secondary sync the record holds the pull-in's frequency
    # discriminator and the Costas phase error, whatever their size; a segment that never
    # found its secondary code has no sums
    error = np.full(30000, 0.4)
    synced = dataclasses.replace(_still_pilot(sec_sync=1500), carrier_error=error)
    marks = _near_wrap(synced)
    assert not marks[:1501].any() and marks[1501:].all()
    never = dataclasses.replace(_still_pilot(), carrier_error=error, sec_sync=None)
    assert not _near_wrap(never).any()


def _lli_changes(pilots):
    """(time of week, satellite, signal) of every indicator that differs from the same
    pilots without any phase error in their record."""
    track, ln = _still_channel(SFS)
    clean = [dataclasses.replace(p, carrier_error=np.zeros(len(p.t))) for p in pilots]
    runs = [make_joint_observables([track], [ln], ps, pilot_signals=GALILEO) for ps in (pilots, clean)]
    out = []
    for a, b in zip(*runs, strict=True):
        assert a.sats == b.sats
        out += [(a.tow, a.sats[i], a.signals[j]) for i, j in np.argwhere(a.lli != b.lli)]
    return out


def test_a_sum_near_the_wrap_flags_its_own_pilot_among_several():
    error = np.zeros(30000)
    error[15100] = -0.41
    e5a = _still_e5a()[0]
    # the second pilot's sum: its phase alone carries the flag, once
    changes = _lli_changes([_still_pilot(), dataclasses.replace(e5a, carrier_error=error)])
    assert [(sat, signal) for _, sat, signal in changes] == [("E11", "E5Q")]
    # a pilot that takes no part (its code starts fit no L1 code period, so it has no time)
    # does not lend its sums to the pilot listed after it
    lost = dataclasses.replace(_still_pilot(delay=0.3e-3), carrier_error=error)
    assert _lli_changes([lost, e5a]) == []


@pytest.mark.parametrize("band", ["E5A", "E1"])
def test_a_galileo_pilot_phase_after_a_sum_near_the_wrap_is_a_slip(band):
    track, ln = _still_channel(SFS)
    still = _still_e5a()[0] if band == "E5A" else _still_e1(0.080)
    row, names = (_e5q, GALILEO) if band == "E5A" else (_e1c, BOTH_BANDS)
    error = np.zeros(30000)
    error[15100] = -0.41
    pilot = dataclasses.replace(still, carrier_error=error)
    rx0 = receiver_time0([track], [ln])
    runs = [
        [(o.tow - rx0, row(o)) for o in make_joint_observables([track], [ln], [p], pilot_signals=names) if "E11" in o.sats]
        for p in (still, pilot)
    ]  # fmt: skip
    changed = [t for (t, a), (_, b) in zip(*runs, strict=True) if a[4] != b[4]]
    assert len(changed) == 1 and 15.1 < changed[0] <= 16.11
    for (_, a), (_, b) in zip(*runs, strict=True):
        np.testing.assert_array_equal(a[:4], b[:4])


def test_pilot_observations_in_the_first_two_seconds_are_flagged():
    track, ln = _still_channel(SFS)
    joint = make_joint_observables([track], [ln], [_still_pilot(k_start=3000)], interval=0.25)
    rx0 = receiver_time0([track], [ln])
    rows = [(o.tow - rx0 - 3.0, _g5q(o)[0]) for o in joint]  # time into the pilot segment
    early = [l5[4] for t, l5 in rows if 1.502 < t <= 2.0]
    later = [l5[4] for t, l5 in rows if t > 2.25 and np.isfinite(l5[0])]
    assert len(early) == 2 and set(early) == {LLI_SLIP} and later and not any(later)


def test_observation_builders_refuse_a_track_of_the_wrong_signal():
    # an L5 pilot has no data bits to time it and an L1 track no secondary code; an E5a
    # pilot would be written as G5Q
    track, ln = _still_channel(SFS)
    pilot = _still_pilot()
    with pytest.raises(ValueError, match="L5 is not an L1 C/A track"):
        make_observables([pilot], [ln])
    with pytest.raises(ValueError, match="L5 is not an L1 C/A track"):
        make_joint_observables([pilot], [ln], [pilot])
    with pytest.raises(ValueError, match="L1CA is not an L5 pilot track"):
        make_joint_observables([track], [ln], [track])
    with pytest.raises(ValueError, match="E5A is not an L5 pilot track"):
        make_joint_observables([track], [ln], [dataclasses.replace(pilot, signal="E5A")])
    # and with Galileo asked for, only pilots are taken
    with pytest.raises(ValueError, match="L1CA is not an L5 or E5A pilot track"):
        make_joint_observables([track], [ln], [track], pilot_signals=GALILEO)
