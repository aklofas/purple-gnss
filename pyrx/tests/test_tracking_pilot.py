"""The pilot stages of the tracker on synthetic L5 and E5a signals: secondary-code sync,
the coherent four-quadrant loop, the data component's symbols."""

from math import atan2, pi

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


def test_hold_flag_is_read_by_the_coherent_update():
    # a 10 dB fade of 0.3 s after the sync: the hold's 10 ms power windows keep running and
    # the 20 ms update leaves the integrated term alone while they report a fade
    seconds, fade_from, fade_len = 4.5, 3.0, 0.3
    noisy = _signal(L5, 6, seconds, 45.0, seed=6, sec_index=5)
    clean = _signal(L5, 6, seconds, 300.0, seed=6, sec_index=5)
    t = np.arange(len(noisy)) / FS
    fade = (t >= fade_from) & (t < fade_from + fade_len)
    x = noisy.copy()
    x[fade] -= clean[fade] * np.complex64(1.0 - 10 ** (-10.0 / 20.0))
    r = _track(x, L5, 6, seed=(700 / FS, 3.0, 45.0))
    assert r.locked and r.sec_sync is not None
    chips = _chips(r, L5, 6)
    loop, fd = CarrierLoop(), r.doppler[K0 - 1]
    total, count, held = 0j, 0, []
    for k in range(K0, len(r.t)):
        prompt, closes = complex(r.corr[k, 0]), (k + 1) % N_DLL == 0
        if k <= r.sec_sync:
            fd += loop.update(prompt, closes)[0]
        else:
            loop.fade_window(prompt, closes)
            total += prompt * chips[k]
            count += 1
            if count == N_COH:
                before = loop.phase_acc
                fd += loop.update_coherent(total, T_PILOT_COH)[0]
                if loop.fading:
                    assert loop.phase_acc == before, k
                    held.append(r.t[k])
                total, count = 0j, 0
        assert r.doppler[k] == fd, k
    assert held and fade_from <= min(held) < fade_from + 0.1 and max(held) < fade_from + fade_len + 0.05


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
