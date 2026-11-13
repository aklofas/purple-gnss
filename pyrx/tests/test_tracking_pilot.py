"""The pilot stages of the tracker on synthetic L5 and E5a signals: secondary-code sync,
the coherent four-quadrant loop, the data component's symbols."""

from math import atan2, cos, pi, sin, sqrt

import numpy as np
import pytest
from support import assert_tracks_equal

from pyrx.acquisition import acquire_signal
from pyrx.codes import CA_PERIOD
from pyrx.samples import ArraySamples, SampleFile
from pyrx.signalspec import E5A, L5
from pyrx.synth import synth_pilot
from pyrx.tracking import (
    B_PLL,
    COAST_COHERENCE,
    PILOT_FLL_STAGES,
    PLI_LOCKED,
    SEC_SYNC_RATIO,
    T_DLL,
    T_FLL_PULLIN,
    T_PILOT_COH,
    T_SEC_SYNC,
    CarrierLoop,
    track,
    track_channels,
)

FS = 12e6
N_COH = round(T_PILOT_COH / CA_PERIOD)
N_DLL = round(T_DLL / CA_PERIOD)
K0 = round(T_FLL_PULLIN / CA_PERIOD)  # the first Costas epoch
W = B_PLL / 0.7845  # the third-order loop's natural frequency
SPECS = {"L5": (L5, 6), "E5A": (E5A, 2)}


def _signal(spec, prn, seconds, cn0, doppler=0.0, code_offset=700, **kwargs):
    return synth_pilot(spec, prn, FS, 0.0, doppler, code_offset, int(FS * seconds), cn0, **kwargs)


def _track(x, spec, prn, seed=None, **kwargs):
    """Track from an acquisition on the first 20 ms, or from the given (offset, Doppler, C/N0)."""
    if seed is None:
        acq = acquire_signal(x[: int(FS * 0.02)], FS, 0.0, spec, prn)
        assert acq.detected
        seed = (acq.code_offset, acq.doppler, acq.cn0)
    return track(ArraySamples(x, FS), 0.0, spec, prn, *seed, **kwargs)


def _chips(r, spec, prn):
    """The pilot's secondary chip of every epoch with the sync's polarity, as the tracker wipes it."""
    sec = spec.secondary(prn)
    return sec[(np.arange(len(r.t)) - r.sec_origin) % len(sec)] * r.sec_polarity


def _sum(phase, coherence, power=400.0):
    """A coherent sum of N_COH prompts with the given phase (cycles) and coherence, and the
    prompts' summed power."""
    size = sqrt(coherence * N_COH * power)
    return size * complex(cos(2.0 * pi * phase), sin(2.0 * pi * phase)), power


def _with_fade(spec, prn, seconds, cn0, fade_from, fade_len, depth_db=None, fs=FS, **kwargs):
    """A pilot signal whose amplitude is down by `depth_db` (gone when None) from `fade_from`
    for `fade_len` seconds; the code starts at sample 700 with secondary chip 5 unless
    `sec_index` says otherwise."""
    args = (spec, prn, fs, 0.0, kwargs.pop("doppler", 0.0), 700, int(fs * seconds))
    kwargs.setdefault("sec_index", 5)
    x = synth_pilot(*args, cn0, **kwargs)
    clean = synth_pilot(*args, 300.0, **kwargs)
    a, b = int(fade_from * fs), int((fade_from + fade_len) * fs)
    drop = 1.0 if depth_db is None else 1.0 - 10.0 ** (-depth_db / 20.0)
    x[a:b] -= np.complex64(drop) * clean[a:b]
    return x


def _phase_step(r, doppler, rate, t_from, t_to):
    """The change of the carrier-phase error across (t_from, t_to), in cycles: its median over
    the second that starts 1 s after `t_to` minus that over the second before `t_from`. The
    truth is the Doppler the signal was generated with, `doppler + rate * t`."""
    te = np.append(r.t[1:], r.t[-1] + CA_PERIOD)
    err = r.carrier_phase - (doppler * (te - r.t[0]) + 0.5 * rate * (te**2 - r.t[0] ** 2))
    before = np.median(err[(r.t > t_from - 1.0) & (r.t < t_from - 0.1)])
    after = np.median(err[(r.t > t_to + 1.0) & (r.t < t_to + 2.0)])
    return after - before


def test_pilot_stage_constants():
    assert PILOT_FLL_STAGES == ((0.5, 5.0, 1), (0.5, 2.0, 1))
    assert (T_SEC_SYNC, SEC_SYNC_RATIO, T_PILOT_COH) == (1.5, 0.8, 0.020)


@pytest.mark.parametrize("name", SPECS)
@pytest.mark.parametrize("cn0", [40.0, 45.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pilot_syncs_and_wipes(name, cn0, seed):
    spec, prn = SPECS[name]
    n_sec = len(spec.secondary(prn))
    sec_index = 7 + 11 * seed
    doppler = (-2000.0, 350.0, 2000.0)[seed]
    x = _signal(spec, prn, 3.2, cn0, doppler, seed=seed, sec_index=sec_index)
    r = _track(x, spec, prn)
    assert r.signal == spec.name and r.locked
    assert r.sec_sync is not None and r.t[r.sec_sync] - r.t[0] <= 2.0
    # epoch 0 is the code period at the generated offset, which carries chip sec_index
    assert (r.sec_origin + sec_index) % n_sec == 0
    assert r.sec_polarity in (1, -1)
    wiped = r.corr[r.sec_sync + 1 :, 0] * _chips(r, spec, prn)[r.sec_sync + 1 :]
    assert np.mean(wiped.real > 0) >= 0.99
    assert abs(r.doppler[-1] - doppler) < 2.0
    assert r.data_prompt.shape == r.t.shape and r.data_prompt.dtype == np.complex64


def test_pilot_loop_update_is_the_third_order_update_on_coherent_sums():
    # rebuilt from the recorded errors in the loop's own order of operations: the Costas
    # update per epoch up to the sync, then the same update with dt = 20 ms on the phase of
    # every 20 wiped prompts, the previous error and the integrated term carried over; a
    # Doppler near zero keeps an ulp of the update from being absorbed into the Doppler
    x = _signal(L5, 6, 3.0, 45.0, sec_index=3)
    r = _track(x, L5, 6, seed=(700 / FS, 3.0, 45.0))
    assert r.sec_sync is not None and r.sec_sync > K0
    chips = _chips(r, L5, 6)
    w = B_PLL / 0.7845
    fd, acc, prev = r.doppler[K0 - 1], 0.0, 0.0
    total, count, updates = 0j, 0, 0
    for k in range(K0, len(r.t)):
        e = r.carrier_error[k]
        if k <= r.sec_sync:
            dt = CA_PERIOD
            assert np.isfinite(e), k
        else:
            total += complex(r.corr[k, 0]) * chips[k]
            count += 1
            if count < N_COH:
                assert np.isnan(e), k
                assert r.doppler[k] == fd, k
                continue
            dt = N_COH * CA_PERIOD
            assert e == atan2(total.imag, total.real) / (2.0 * pi), k
            total, count, updates = 0j, 0, updates + 1
        acc += w * w * w * e * dt
        fd += 2.4 * w * (e - prev) + 1.1 * w * w * e * dt + acc * dt
        prev = e
        assert r.doppler[k] == fd, k
    assert updates > 50


@pytest.mark.parametrize("seed", range(20))
def test_no_false_lock_after_the_sync(seed):
    # the four-quadrant loop on 20 ms sums would stand 25 Hz off with alternating errors of
    # a quarter cycle (cos 2 phi = -1); the handover from the Costas loop must never put it there
    rng = np.random.default_rng(100 + seed)
    spec, prn = (L5, 6) if seed % 2 else (E5A, 2)
    doppler = float(rng.uniform(-3000.0, 3000.0))
    code_offset = int(rng.integers(0, 12000))
    x = _signal(spec, prn, 5.0, 43.0, doppler, code_offset, seed=seed, sec_index=int(rng.integers(100)))
    r = _track(x, spec, prn, seed=(code_offset / FS, doppler + float(rng.uniform(-30.0, 30.0)), 43.0))
    assert r.locked and r.sec_sync is not None
    after = r.t >= r.t[r.sec_sync] + 1.0
    assert np.max(np.abs(r.doppler[after] - doppler)) < 2.0
    ends = np.arange(999, len(r.t), 1000)
    windows = r.phase_lock[ends[ends - 999 > r.sec_sync]]  # the complete windows after the sync
    assert len(windows) >= 2 and windows.min() > 0.8


def test_pilot_tracks_a_doppler_ramp():
    # the car profile of the L1 ramp test at the L5 carrier: 8 Hz/s at L1 is 6 Hz/s here
    rate = 8.0 * L5.carrier / 1575.42e6
    x = _signal(L5, 6, 6.2, 40.0, 1000.0, seed=1, doppler_rate=rate)
    r = _track(x, L5, 6, seed=(700 / FS, 1005.0, 40.0))
    assert r.locked and r.sec_sync is not None
    m = r.t >= 4.0
    errors = r.carrier_error[m]
    assert np.sqrt(np.nanmean(errors**2)) < 0.02
    assert abs(np.nanmean(errors)) < 0.005
    assert abs(np.mean(r.doppler[m] - (1000.0 + rate * r.t[m]))) < 0.5


@pytest.mark.parametrize("name", SPECS)
def test_symbols_are_the_generated_ones(name):
    spec, prn = SPECS[name]
    sec_index = 13
    symbols = np.random.default_rng(5).integers(0, 2, size=400)
    x = _signal(spec, prn, 3.2, 45.0, 800.0, seed=2, sec_index=sec_index, symbols=symbols)
    r = _track(x, spec, prn)
    assert r.sec_sync is not None and r.symbol_epoch > r.sec_sync
    assert (r.symbol_epoch - r.sec_origin) % spec.symbol_epochs == 0
    assert len(r.symbols) == (len(r.t) - r.symbol_epoch) // spec.symbol_epochs >= 50
    first = (sec_index + r.symbol_epoch) // spec.symbol_epochs
    assert np.array_equal(r.symbols < 0, symbols[first : first + len(r.symbols)] == 1)
    # the data component lies on the axis the specification gives it, not in phase with the pilot
    dsec = spec.data_secondary(prn)
    k = np.arange(r.symbol_epoch, len(r.t))
    data = r.data_prompt[k] * dsec[(k - r.sec_origin) % len(dsec)] * r.sec_polarity
    along = (data * np.conj(spec.data_axis)).real
    assert np.sum(along**2) >= 0.9 * np.sum(np.abs(data) ** 2)
    # as strong as the pilot
    ratio = np.mean(np.abs(r.data_prompt[k]) ** 2) / np.mean(np.abs(r.corr[k, 0]) ** 2)
    assert abs(10 * np.log10(ratio)) < 0.5


def test_weak_pilot_never_syncs():
    # 28 dB-Hz: the Costas loop does not phase-lock, so no alignment of the secondary code is
    # accepted, and the track ends by the loss rule
    x = _signal(L5, 6, 4.2, 28.0, 500.0, seed=3)
    r = _track(x, L5, 6, seed=(700 / FS, 500.0, 28.0))
    assert r.sec_sync is None and r.symbols is None and r.symbol_epoch is None
    assert r.sec_polarity == 0
    assert not r.locked and r.lost_at <= 3.1
    assert np.all(np.isfinite(r.carrier_error[K0:]))  # the Costas loop ran to the end


def test_pilot_track_shorter_than_the_secondary_code():
    x = _signal(E5A, 2, 0.06, 45.0)
    r = _track(x, E5A, 2, seed=(700 / FS, 0.0, 45.0))
    assert 50 <= len(r.t) < 100 and r.sec_sync is None and r.symbols is None and r.locked


def test_noise_does_not_sync():
    # without the phase-lock condition the 20-chip code matches noise within a second
    x = _signal(L5, 6, 4.2, 45.0, seed=4)
    r = _track(x, L5, 11, seed=(700 / FS, 0.0, 33.0))  # another PRN's signal: noise to PRN 11
    assert r.sec_sync is None


def test_incoherent_sum_coasts():
    # a sum whose prompts do not add up carries no signal: the loop takes nothing from it,
    # gives back the proportional term of the last error and runs on at its Doppler rate
    loop = CarrierLoop(err_phase_prev=0.1, phase_acc=2.0)
    increment, err = loop.update_coherent(*_sum(0.3, 0.19), T_PILOT_COH)
    assert increment == pytest.approx(-2.4 * W * 0.1 + 2.0 * T_PILOT_COH, rel=1e-12)
    assert np.isnan(err) and loop.coasting
    assert (loop.err_phase_prev, loop.phase_acc) == (0.0, 2.0)


def test_coherence_at_the_threshold_is_good():
    assert COAST_COHERENCE == 0.20
    loop = CarrierLoop()
    _, err = loop.update_coherent(*_sum(0.0, COAST_COHERENCE), T_PILOT_COH)
    assert err == 0.0 and not loop.coasting


def test_first_good_sum_after_a_coast_moves_the_phase_only():
    # the integrated terms take a sum's error only when the sum before it was good too, so
    # a lone noise sum over the threshold leaves nothing behind
    dt = T_PILOT_COH
    loop = CarrierLoop(err_phase_prev=0.1, phase_acc=2.0)
    loop.update_coherent(*_sum(0.3, 0.05), dt)
    increment, err = loop.update_coherent(*_sum(0.05, 0.9), dt)
    assert err == pytest.approx(0.05, rel=1e-12) and not loop.coasting
    assert increment == pytest.approx(2.4 * W * 0.05 + 2.0 * dt, rel=1e-12)
    assert loop.phase_acc == 2.0 and loop.err_phase_prev == err
    increment, err = loop.update_coherent(*_sum(0.03, 0.9), dt)
    acc = 2.0 + W**3 * 0.03 * dt
    assert loop.phase_acc == pytest.approx(acc, rel=1e-12)
    full = 2.4 * W * (0.03 - 0.05) + 1.1 * W * W * 0.03 * dt + acc * dt
    assert increment == pytest.approx(full, rel=1e-12)


def test_zero_power_sum_coasts():
    loop = CarrierLoop(err_phase_prev=0.1, phase_acc=2.0)
    increment, err = loop.update_coherent(0j, 0.0, T_PILOT_COH)
    assert increment == pytest.approx(-2.4 * W * 0.1 + 2.0 * T_PILOT_COH, rel=1e-12)
    assert np.isnan(err) and loop.coasting


def test_coherent_update_holds_the_integrator_in_a_fade():
    # the integrator hold runs on after the sync: a sum taken while the prompt power is below
    # half its recent mean moves the Doppler and leaves the rate estimate alone
    dt = T_PILOT_COH
    loop = CarrierLoop(err_phase_prev=0.01, phase_acc=2.0, fading=True)
    increment, err = loop.update_coherent(*_sum(0.02, 0.9), dt)
    assert err == pytest.approx(0.02, rel=1e-12) and not loop.coasting
    assert loop.phase_acc == 2.0
    full = 2.4 * W * (0.02 - 0.01) + 1.1 * W * W * 0.02 * dt + 2.0 * dt
    assert increment == pytest.approx(full, rel=1e-12)


def test_bad_first_sum_after_the_sync_takes_the_proportional_term_back():
    loop = CarrierLoop()
    for _ in range(5):
        loop.update(complex(1.0, 0.3), False)
    prev, acc = loop.err_phase_prev, loop.phase_acc
    assert prev > 0.04 and acc > 0.0
    increment, _ = loop.update_coherent(*_sum(0.0, 0.1), T_PILOT_COH)
    assert increment == pytest.approx(-2.4 * W * prev + acc * T_PILOT_COH, rel=1e-12)
    assert loop.coasting and loop.phase_acc == acc


@pytest.mark.parametrize(
    ("depth_db", "sec_index", "on_close", "expected"),
    [
        (None, 5, False, {"full", "coasted", "phase only"}),
        (10.0, 5, False, {"full", "integrator held"}),
        (4.0, 0, True, {"full", "integrator held"}),
    ],
)
def test_coherent_loop_is_rebuilt_from_the_prompts(depth_db, sec_index, on_close, expected):
    # the Doppler of every epoch from the recorded prompts, by the loop's rules restated: a
    # sum is good when its coherence reaches COAST_COHERENCE; a sum that is not is coasted
    # (the last error's proportional term taken back, the integrated term still steering);
    # the first good sum after a coasted one moves the Doppler by its proportional term
    # only; a good sum completed while the fade window holds the integrator does not move
    # the integrated term. The signal is gone for 0.4 s, or 10 dB down for as long, or 4 dB
    # down with the sync falling so that every sum ends on the epoch that closes a fade
    # window: there the window is closed first and the sum reads the new flag. All three
    # settle on the inverted pilot, so the rules are seen to work on the wiped prompts
    x = _with_fade(L5, 6, 4.5, 45.0, 3.0, 0.4, depth_db, seed=7, sec_index=sec_index)
    r = _track(x, L5, 6, seed=(700 / FS, 3.0, 45.0))
    assert r.locked and r.sec_sync is not None and r.sec_polarity == -1
    assert ((r.sec_sync + 1) % N_DLL == 0) == on_close
    chips = _chips(r, L5, 6)
    loop, fd = CarrierLoop(), r.doppler[K0 - 1]
    dt = T_PILOT_COH
    total, power, count, coasted, kinds = 0j, 0.0, 0, False, set()
    for k in range(K0, len(r.t)):
        prompt = complex(r.corr[k, 0])
        if k <= r.sec_sync:
            fd += loop.update(prompt, (k + 1) % N_DLL == 0)[0]
            prev, acc = loop.err_phase_prev, loop.phase_acc
        else:
            loop.fade_window(prompt, (k + 1) % N_DLL == 0)
            total += prompt * chips[k]
            power += abs(prompt) ** 2
            count += 1
            if count < N_COH:
                assert not r.held[k], k
            else:
                if power > 0.0 and abs(total) ** 2 >= COAST_COHERENCE * N_COH * power:
                    e = atan2(total.imag, total.real) / (2.0 * pi)
                    if coasted:
                        fd += 2.4 * W * (e - prev) + acc * dt
                        kinds.add("phase only")
                    else:
                        if not loop.fading:
                            acc += W * W * W * e * dt
                        fd += 2.4 * W * (e - prev) + 1.1 * W * W * e * dt + acc * dt
                        kinds.add("integrator held" if loop.fading else "full")
                    prev, coasted = e, False
                else:
                    fd += -2.4 * W * prev + acc * dt
                    prev, coasted = 0.0, True
                    kinds.add("coasted")
                assert r.held[k] == coasted, k
                total, power, count = 0j, 0.0, 0
        assert r.doppler[k] == fd, k
    assert kinds >= expected and (depth_db is None or "coasted" not in kinds)
    assert r.held.dtype == bool and not r.held[: r.sec_sync + 1].any()


@pytest.mark.parametrize("name", SPECS)
def test_pilot_returns_on_the_same_cycle_after_an_outage(name):
    # the signal gone for 0.8 s under a 3 Hz/s ramp. A loop that keeps updating on the noise
    # comes back whole cycles away (three on this seed, on both signals); one that coasts at
    # constant Doppler is a cycle off by the ramp alone
    spec, prn = SPECS[name]
    x = _with_fade(spec, prn, 6.0, 45.0, 3.0, 0.8, doppler=100.0, doppler_rate=3.0, seed=2)
    r = _track(x, spec, prn, seed=(700 / FS, 100.0, 45.0))
    assert r.locked and r.sec_sync is not None
    assert abs(_phase_step(r, 100.0, 3.0, 3.0, 3.8)) < 0.1


def test_a_lone_noise_sum_in_an_outage_leaves_the_estimates_alone():
    # 1.5 s without signal at 38 dB-Hz, from 5 s so that the rate estimate has settled: one
    # noise sum in seventy reads coherent enough to pass the gate, and on this seed one
    # does. Were its error to reach the integrated terms, the loop would return a cycle
    # away
    fs = 6e6
    x = _with_fade(L5, 6, 8.7, 38.0, 5.0, 1.5, fs=fs, doppler=100.0, doppler_rate=0.6, seed=6)
    r = track(ArraySamples(x, fs), 0.0, L5, 6, 700 / fs, 100.0, 38.0)
    assert r.sec_sync is not None
    sums = np.arange(r.sec_sync + N_COH, len(r.t), N_COH)  # the epochs completing a sum
    inside = sums[(r.t[sums] > 5.05) & (r.t[sums] < 6.45)]
    assert np.sum(~r.held[inside]) == 1
    assert abs(_phase_step(r, 100.0, 0.6, 5.0, 6.5)) < 0.1


def test_a_fade_the_loop_can_follow_is_not_coasted():
    # 10 dB down for 1.5 s under the ramp: 35 dB-Hz is a signal the 20 ms loop tracks, and
    # its sums stay far above the gate
    x = _with_fade(L5, 6, 6.7, 45.0, 3.0, 1.5, 10.0, doppler=100.0, doppler_rate=3.0, seed=1)
    r = _track(x, L5, 6, seed=(700 / FS, 100.0, 45.0))
    assert r.locked and not r.held.any()
    assert abs(_phase_step(r, 100.0, 3.0, 3.0, 4.5)) < 0.05


def test_a_long_outage_still_ends_the_track():
    # coasting does not keep a channel alive: two windows without signal end it as before
    x = _with_fade(L5, 6, 6.2, 45.0, 3.0, 3.2, doppler=100.0, seed=1)
    r = _track(x, L5, 6, seed=(700 / FS, 100.0, 45.0))
    assert not r.locked and 4.0 < r.lost_at < 6.0


def test_an_outage_is_marked_as_held():
    # the signal gone for 0.4 s after the sync: the window's indicator stays above the lock
    # threshold, and the coasted sums are what marks the stretch
    seconds, gone_from, gone_len = 4.5, 3.0, 0.4
    noisy = _signal(L5, 6, seconds, 45.0, seed=7, sec_index=5)
    clean = _signal(L5, 6, seconds, 300.0, seed=7, sec_index=5)
    t = np.arange(len(noisy)) / FS
    gone = (t >= gone_from) & (t < gone_from + gone_len)
    x = noisy.copy()
    x[gone] -= clean[gone]
    r = _track(x, L5, 6, seed=(700 / FS, 3.0, 45.0))
    marked = r.t[r.held]
    assert len(marked) >= 15 and gone_from <= marked.min() < gone_from + 0.05
    assert marked.max() < gone_from + gone_len + 0.05
    steady = _track(noisy, L5, 6, seed=(700 / FS, 3.0, 45.0))
    assert steady.held is not None and not steady.held.any()


def test_track_channels_takes_the_signal(tmp_path):
    seconds = 2.2
    x = _signal(L5, 6, seconds, 48.0, 1234.0, 700, seed=1) + _signal(L5, 11, seconds, 48.0, -500.0, 2500, seed=2)
    raw = np.empty(2 * len(x), dtype=np.int8)
    raw[0::2] = (x.real / 2).clip(-127, 127)
    raw[1::2] = (-x.imag / 2).clip(-127, 127)
    path = tmp_path / "two.bin"
    raw.tofile(path)
    seeds = [(6, 700 / FS, 1234.0, 45.0), (11, 2500 / FS, -500.0, 45.0)]
    results = track_channels(path, FS, True, 0.0, seeds, workers=2, signal="L5")
    assert [r.prn for r in results] == [6, 11]
    for r, (prn, coff, dop, cn0) in zip(results, seeds):
        assert_tracks_equal(r, track(SampleFile(path, FS, True), 0.0, L5, prn, coff, dop, cn0))
        assert r.signal == "L5" and r.sec_sync is not None and abs(r.doppler[-1] - dop) < 2.0
        assert np.nanmin(r.phase_lock[r.sec_sync :]) > PLI_LOCKED
