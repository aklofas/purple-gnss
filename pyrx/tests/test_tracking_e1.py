"""Galileo E1: the BOC(1,1) replica, a code period of four epochs, the side-peak guard."""

import numpy as np
import pytest

from pyrx.acquisition import acquire_signal
from pyrx.codes import CA_PERIOD
from pyrx.correlator import correlate
from pyrx.samples import ArraySamples
from pyrx.signalspec import E1
from pyrx.synth import synth_pilot
from pyrx.tracking import SIDE_PEAK, track

FS = 12e6
N = round(FS * CA_PERIOD)
PRN = 11
CHIP = FS / E1.chip_rate  # samples per chip


def _signal(seconds, cn0, doppler=0.0, code_offset=700, **kwargs):
    return synth_pilot(E1, PRN, FS, 0.0, doppler, code_offset, int(FS * seconds), cn0, **kwargs)


def _track(x, code_offset=700, doppler=0.0, cn0=45.0, **kwargs):
    return track(ArraySamples(x, FS), 0.0, E1, PRN, code_offset / FS, doppler, cn0, **kwargs)


def _period(x, start, replica, shift=0.0):
    """The sum of one 4 ms code period from sample `start` against a replica moved `shift`
    chips late."""
    rate = E1.replica_rate
    return correlate(x[start : start + 4 * N], FS, 0.0, 0.0, replica, rate, -shift * 2, np.zeros(1))[0]


def test_synthetic_e1_is_two_boc_components_on_one_axis():
    symbols = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
    x = _signal(0.05, 80.0, symbols=symbols, sec_index=3)
    pilot, data = E1.replica(PRN), E1.data_replica(PRN)
    cs25 = E1.secondary(PRN)[::4]
    for j in range(8):  # code periods from the generated offset
        start = 700 + j * 4 * N
        p, d = _period(x, start, pilot), _period(x, start, data)
        # the pilot under chip 3 + j of CS25, the data component with the opposite sign; a
        # symbol is a code period, counted with the secondary chips
        assert (p * cs25[(3 + j) % 25]).real < -0.99 * abs(p)
        assert (d * (1 - 2 * symbols[3 + j])).real > 0.99 * abs(d)
        assert abs(abs(p) / abs(d) - 1.0) < 0.02
    # the correlation of BOC(1,1): minus a half at half a chip, nothing a chip away
    peak = _period(x, 700, pilot)
    half = [_period(x, 700, pilot, s) / peak for s in (0.5, -0.5)]
    full = [_period(x, 700, pilot, s) / peak for s in (1.0, -1.0)]
    # (within the code's own sidelobes, one in sixty-four)
    assert all(abs(h.real + 0.5) < 0.05 for h in half) and all(abs(f) < 0.06 for f in full)


@pytest.mark.parametrize("cn0", [38.0, 40.0, 45.0])
def test_e1_acquisition_finds_the_main_peak(cn0):
    code_offset = 31337
    x = _signal(0.021, cn0, 1234.0, code_offset, seed=3)
    acq = acquire_signal(x[: int(FS * 0.020)], FS, 0.0, E1, PRN)
    assert acq.detected and abs(acq.cn0 - cn0) < 3.0
    assert abs(acq.doppler - 1234.0) < 70.0
    # the offset of a code period's start, anywhere in its 4 ms, to the sample (a twelfth
    # of a chip at this rate): not the side peak half a chip away
    assert abs(acq.code_offset * FS - code_offset) <= 1.0
    assert not acquire_signal(x[: int(FS * 0.020)], FS, 0.0, E1, 12).detected


def test_e1_epochs_are_quarters_of_the_code_period():
    x = _signal(3.2, 45.0, 500.0, seed=1, sec_index=6)
    r = _track(x, doppler=500.0)
    assert r.locked and r.sec_sync is not None and not r.side_peak
    assert (r.sec_sync + 1) % 4 == 0 and r.sec_origin % 4 == 0
    assert (r.sec_origin + 6 * 4) % 100 == 0
    # every epoch is 1 ms and correlates its quarter: the prompt is as strong in each
    after = np.arange(r.sec_sync + 1, len(r.t))
    power = [np.mean(np.abs(r.corr[after[after % 4 == q], 0]) ** 2) for q in range(4)]
    assert max(power) / min(power) < 1.1
    assert np.allclose(np.diff(r.t), CA_PERIOD, rtol=0, atol=2 / FS)
    # and reads the signal's strength: half the E1 power is the pilot's
    assert abs(np.median(r.cn0[2000:]) - 45.0) < 1.0


def _code_error(r, code_offset=700):
    """The code start of every epoch against the generated one, in chips."""

    def wrap(x):
        return (x + CA_PERIOD / 2) % CA_PERIOD - CA_PERIOD / 2

    return wrap(wrap(r.code_offset - r.t) + r.t - code_offset / FS) * E1.chip_rate


@pytest.mark.parametrize("start", [0.08, -0.08])
def test_e1_code_loop_closes_an_offset_as_a_plain_code_does(start):
    # The early-minus-late discriminator of BOC(1,1) at a quarter chip is nine times as steep
    # as a plain code's, and divided by that the loop closes an offset at the rate of its
    # 0.25 Hz bandwidth: measured from a tenth of a chip, 0.062 chip left after 1 s and 0.035
    # after 2 s, where L1 C/A and E5a have 0.052 and 0.027. Undivided, the offset would be
    # gone within a quarter second and the code would follow the noise nine times as far.
    x = _signal(4.2, 50.0, seed=2)
    r = _track(x, code_offset=700 + start * CHIP, cn0=50.0)
    assert r.locked and not r.side_peak
    left = _code_error(r) / start
    assert abs(left[0] - 1.0) < 0.05
    assert 0.45 < np.mean(left[950:1050]) < 0.75 and 0.2 < np.mean(left[1950:2050]) < 0.45
    assert abs(np.mean(left[3900:])) < 0.2


@pytest.mark.parametrize("shift", [0.5, -0.5])
@pytest.mark.parametrize("cn0", [38.0, 45.0])
def test_a_start_on_a_side_peak_is_ended(shift, cn0):
    # half a chip from the code start the correlation has a peak of half the size and the
    # opposite sign, and the loops stay off the main peak from there; the tap toward the main
    # peak reads twice the prompt, and the track ends before the window that showed it
    x = _signal(5.2, cn0, 800.0, seed=4)
    r = _track(x, code_offset=700 + shift * CHIP, doppler=800.0, cn0=cn0)
    # the first window shows it, so nothing of the track is left
    assert r.side_peak and not r.locked and len(r.t) == 0
    assert r.lost_at == pytest.approx(700 / FS, abs=1e-4) and r.sec_sync is None
    assert r.symbols is None and r.data_prompt.shape == (0,)


@pytest.mark.parametrize("duration", [None, 4.95, 4.5])
def test_a_track_that_moves_to_a_side_peak_ends_a_window_before_the_one_that_shows_it(duration):
    # the signal's code jumps half a chip 3.4 s in and the loops are on a side peak from
    # there. A window reads it once two thirds of it are on the side peak, here the one from
    # 4 to 5 s, so the track ends before the window in which the jump may lie. Where the
    # data end inside that window, what there is of it shows the side peak as well
    x = _signal(6.2, 45.0, seed=8)
    jump = int(3.4 * FS)
    x[jump:] = _signal(6.2, 45.0, code_offset=700 + round(0.5 * CHIP), seed=8)[jump:]
    r = _track(x, duration=duration)
    assert r.side_peak and not r.locked and len(r.t) == 3000 and r.sec_sync is not None
    assert r.lost_at == pytest.approx(3.0 + 700 / FS, abs=1e-3)
    assert r.symbols is not None and len(r.held) == 3000


def test_a_move_to_a_side_peak_in_the_last_window_of_the_data_is_not_seen():
    # the limit of the rule: the data end with the window the jump lies in, 3 to 4 s, of
    # which less than two thirds are on the side peak, and no later window reads it
    x = _signal(6.2, 45.0, seed=8)
    jump = int(3.4 * FS)
    x[jump:] = _signal(6.2, 45.0, code_offset=700 + round(0.5 * CHIP), seed=8)[jump:]
    r = _track(x, duration=4.0 + 700 / FS + 0.0005)
    assert r.locked and not r.side_peak and len(r.t) == 4000


@pytest.mark.parametrize("epochs", [2001, 3002])
def test_a_few_epochs_at_the_end_of_the_data_do_not_trip_the_guard(epochs):
    # the data end one or two epochs into a window. On so few epochs the reading of a signal
    # on its main peak is anything (here, at 36 dB-Hz, above the threshold), so the part of
    # a window is judged only when it holds the signal energy a whole window has at the loss
    # threshold
    x = _signal(6.2, 36.0, seed=7)[: 700 + round(FS * 1e-3) * epochs + 300]
    r = _track(x, cn0=36.0)
    assert r.locked and not r.side_peak and len(r.t) == epochs


@pytest.mark.parametrize("cn0", [32.0, 36.0, 40.0, 50.0])
def test_the_main_peak_never_trips_the_guard(cn0):
    # the reading is 0.39 to 0.70 on the main peak in every window measured, down to windows
    # of 22 dB-Hz, and 1.55 or more on a side peak: the threshold stands in the gap
    x = _signal(6.2, cn0, -1500.0, seed=5)
    r = _track(x, doppler=-1500.0, cn0=cn0)
    assert not r.side_peak and len(r.t) > 6000 and SIDE_PEAK == 1.0


@pytest.mark.parametrize("cn0", [26.0, 28.0, 30.0])
@pytest.mark.parametrize("seed", [5, 6])
def test_a_weak_main_peak_ends_by_the_loss_rule_not_the_guard(cn0, seed):
    # near the loss threshold the guard's sums are mostly noise; a window below the loss
    # rule's C/N0 is not judged by the guard at all
    x = _signal(8.2, cn0, 200.0, seed=seed)
    r = _track(x, doppler=200.0, cn0=cn0)
    assert not r.side_peak


@pytest.mark.parametrize("seed", [11, 12, 14])
def test_the_guard_reads_a_weak_side_peak(seed):
    # The projection is taken with the noise out of both sums, so it reads the correlation's
    # own ratio at any strength: a 33 dB-Hz pilot that jumps to a side peak, where its prompt
    # reads 27 dB-Hz and the loss rule lets it live, is ended as a strong one is. Projected
    # on the prompt's whole power, noise and all, it read 0.9 to 1.0 here and the track ran
    # on for seconds or for good.
    x = _signal(9.2, 33.0, seed=seed)
    jump = int(3.4 * FS)
    x[jump:] = _signal(9.2, 33.0, code_offset=700 + round(0.5 * CHIP), seed=seed)[jump:]
    r = _track(x, cn0=33.0)
    assert r.side_peak and len(r.t) == 3000


def test_a_side_peak_in_the_only_window_is_seen_in_it():
    # the first window alone decides: a track of one window and a half that starts on a side
    # peak is ended by that window, and nothing of it is left
    x = _signal(1.7, 45.0, 800.0, seed=4)
    r = _track(x, code_offset=700 + 0.5 * CHIP, doppler=800.0, duration=1.5)
    assert r.side_peak and len(r.t) == 0 and r.lost_at == pytest.approx(700 / FS, abs=1e-4)


def test_a_secondary_code_found_in_the_part_cut_away_is_not_kept():
    # the code jumps at 1.7 s, the secondary code is found at 2.0 s on the side peak, the
    # window from 2 to 3 s shows the side peak: the track is cut to its first second, before
    # the sync, and reports none
    x = _signal(5.2, 45.0, seed=8)
    jump = int(1.7 * FS)
    x[jump:] = _signal(5.2, 45.0, code_offset=700 + round(0.5 * CHIP), seed=8)[jump:]
    r = _track(x)
    assert r.side_peak and len(r.t) == 1000
    assert r.sec_sync is None and r.symbols is None and r.symbol_epoch is None
    assert (r.sec_origin, r.sec_polarity) == (0, 0)


def test_an_outage_does_not_trip_the_guard():
    # without signal the half-chip taps' noise is the prompt's noise times the code's own
    # minus a half, the same as on the main peak: the loss rule ends such a track, not the guard
    x = _signal(6.2, 45.0, seed=6)
    clean = _signal(6.2, 300.0, seed=6)
    a = int(2.5 * FS)
    x[a:] -= clean[a:]
    r = _track(x)
    assert not r.side_peak and not r.locked and 3.5 < r.lost_at < 5.6
