import dataclasses

import numpy as np
import pytest
from test_navdecode import _frame

from pyrx.codes import CA_PERIOD
from pyrx.navdecode import decode_lnav
from pyrx.observables import (
    C_LIGHT,
    full_week,
    make_observables,
    receiver_time0,
    transmit_times,
)
from pyrx.samples import ArraySamples
from pyrx.synth import synth_l1ca
from pyrx.tracking import L1_FREQ, track_l1ca

T0 = 5044 * 6 - 6 - 100 * 0.020  # transmit time (s of week) of bit 0, 100 bits before the frame


def _channel(prn, code_offset, doppler, fs=2.5e6, tow0=5044, seconds=20.1):
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), _frame(tow0)])
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
    # new week, whichever side sample 0 was on
    c = _channel(4, 500, 800.0, fs=fs, tow0=100800 + 1)
    later = make_observables([c[0]], [c[1]])
    assert later[0].week == 2190 and 0.0 <= later[0].tow < 100.0


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
