from math import atan, pi

import numpy as np
import pytest
from support import assert_tracks_equal, synth_l1ca_faded, synth_l1ca_ramp

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD
from pyrx.samples import ArraySamples, SampleFile
from pyrx.synth import synth_l1ca
from pyrx.tracking import (
    B_PLL,
    CN0_LOST,
    FADE_HOLD,
    FADE_REF_GAIN,
    FLL_STAGES,
    PLI_LOCKED,
    T_CN0,
    T_DLL,
    T_FLL_PULLIN,
    track_channels,
    track_l1ca,
)

FS = 4e6


def _src(
    prn=7,
    doppler=1234.0,
    code_offset=1000,
    cn0=45.0,
    seconds=3.0,
    f_if=0.0,
    iq=True,
    fs=FS,
    seed=0,
    data_bits=True,
):
    n = int(fs * seconds)
    x = synth_l1ca(prn, fs, f_if, doppler, code_offset, n, cn0, iq, seed=seed, data_bits=data_bits)
    return ArraySamples(x, fs)


def _steady(r, t0=2.0):
    m = r.t >= t0
    return r.doppler[m], r.code_offset[m], r.cn0[m]


def test_track_synthetic_with_data_bits():
    r = track_l1ca(_src(), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    dop, coff, cn0 = _steady(r)
    assert r.locked and r.lost_at is None
    assert r.corr.shape == (len(r.t), 4)
    assert abs(dop.mean() - 1234.0) < 1.0
    assert np.std(dop) < 2.0
    # true code offset at time t: 1000/FS - t * 1234/1575.42e6 (code Doppler), mod T
    truth = (1000 / FS - r.t[r.t >= 2.0] * 1234.0 / 1575.42e6) % CA_PERIOD
    assert np.max(np.abs(coff - truth)) * CA_CHIP_RATE < 0.05  # chips
    assert abs(cn0[-1] - 45.0) < 2.0


def test_track_pulls_in_from_coarse_seed():
    # half a sample of code offset and 100 Hz of Doppler off: inside the ±125 Hz fold of the
    # 2 ms sums of the first FLL stage (a weak satellite's acquisition peak can be 100 Hz off)
    r = track_l1ca(_src(), 0.0, 7, (1000 + 0.5) / FS, 1234.0 + 100.0, 45.0)
    dop, _, _ = _steady(r)
    assert r.locked and abs(dop.mean() - 1234.0) < 1.0


def test_track_real_if():
    fs, f_if = 8e6, 2.0e6
    r = track_l1ca(
        _src(f_if=f_if, iq=False, fs=fs, code_offset=3000, doppler=-2000.0),
        f_if,
        7,
        3000 / fs,
        -2000.0,
        45.0,
    )
    dop, _, cn0 = _steady(r)
    assert r.locked and abs(dop.mean() + 2000.0) < 1.0 and abs(cn0[-1] - 45.0) < 2.0


def test_track_phase_lock_is_the_windows_indicator():
    r = track_l1ca(_src(seconds=3.5), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert r.locked and r.phase_lock.shape == r.t.shape
    w1, w2, w3 = r.phase_lock[500], r.phase_lock[1500], r.phase_lock[2500]
    assert np.all(r.phase_lock[:1000] == w1) and np.all(r.phase_lock[1000:2000] == w2)
    assert np.all(r.phase_lock[2000:3000] == w3)
    assert w2 > 0.8 and w3 > 0.9  # 45 dB-Hz: rho/(1+rho) = 0.97; the PLL settles inside window 2
    assert np.isfinite(w1)  # the pull-in window is recorded too
    assert np.all(np.isnan(r.phase_lock[3000:]))  # incomplete last window


def test_track_shorter_than_a_window_has_no_indicator_and_no_loss():
    r = track_l1ca(_src(seconds=0.8), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert r.locked and r.lost_at is None and np.all(np.isnan(r.phase_lock))


def test_track_declares_loss():
    # Track PRN 9 where only PRN 7 is present: two bad windows after the pull-in window.
    r = track_l1ca(_src(prn=7, seconds=4.0), 0.0, 9, 1000 / FS, 1234.0, 45.0)
    assert not r.locked and r.lost_at is not None and 2.95 <= r.lost_at <= 3.05
    assert r.cn0[-1] < CN0_LOST
    assert r.phase_lock[1500] < PLI_LOCKED and r.phase_lock[2500] < PLI_LOCKED
    assert len(r.t) == 3000  # the losing epoch kept, nothing after it
    assert r.carrier_error.shape == r.code_error.shape == r.t.shape
    assert np.isfinite(r.carrier_error[-1])


def test_one_bad_cn0_window_is_not_a_loss():
    # seed 4 dB-Hz on a 45 dB-Hz signal: the first window updates C/N0 to 24.5 (bad),
    # the second to 34.75 (good); the seed alone must not end the track
    r = track_l1ca(_src(seconds=4.0), 0.0, 7, 1000 / FS, 1234.0, 4.0)
    assert r.locked and r.cn0[999] < CN0_LOST < r.cn0[1999]
    assert r.cn0[-1] > 38.0  # climbing toward 45 by half the gap per window


def test_phase_lock_rule_drops_a_spinning_carrier_at_full_cn0(monkeypatch):
    # a correlator double whose prompt rotates 0.3 cycle per epoch whatever the NCO does:
    # prompt power stays high, so C/N0 is high, while I**2 - Q**2 averages to zero
    from pyrx import tracking

    k = 0

    def spinning(samples, fs, fc, phi, code, chip_rate, residual, taps):
        nonlocal k
        k += 1
        p = 1000.0 * np.exp(2j * np.pi * 0.3 * k)
        return np.array([p, 500.0, 500.0, 1.0], dtype=np.complex64)

    monkeypatch.setattr(tracking, "correlate", spinning)
    r = track_l1ca(_src(seconds=5.0), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert not r.locked and 2.95 <= r.lost_at <= 3.05
    assert r.cn0[-1] > 60.0 and abs(r.phase_lock[2500]) < PLI_LOCKED


def test_track_stops_at_end_of_data():
    r = track_l1ca(_src(seconds=0.5), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert 495 <= len(r.t) <= 500 and r.locked
    r2 = track_l1ca(_src(seconds=0.5), 0.0, 7, 1000 / FS, 1234.0, 45.0, duration=0.2)
    assert 195 <= len(r2.t) <= 200
    for x in (r, r2):
        assert x.carrier_error.shape == x.code_error.shape == x.t.shape


def test_track_cn0_held_between_updates():
    r = track_l1ca(_src(seconds=2.5), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert np.all(r.cn0[:999] == 45.0)  # seeded value until the first 1 s update
    assert r.cn0[1000] != 45.0 or r.cn0[1999] != 45.0


def test_track_channels_matches_sequential(tmp_path):
    fs = FS
    x7 = synth_l1ca(7, fs, 0.0, 1234.0, 1000, int(fs * 1.5), 48.0, True, seed=1, data_bits=True)
    x9 = synth_l1ca(9, fs, 0.0, -500.0, 2500, int(fs * 1.5), 48.0, True, seed=2, data_bits=True)
    x = x7 + x9
    raw = np.empty(2 * len(x), dtype=np.int8)
    raw[0::2] = (x.real / 2).clip(-127, 127)
    raw[1::2] = (-x.imag / 2).clip(-127, 127)
    path = tmp_path / "two.bin"
    raw.tofile(path)
    seeds = [(7, 1000 / fs, 1234.0, 45.0), (9, 2500 / fs, -500.0, 45.0)]
    results = track_channels(path, fs, True, 0.0, seeds, workers=2)
    assert [r.prn for r in results] == [7, 9]
    for r, (prn, coff, dop, cn0) in zip(results, seeds):
        single = track_l1ca(SampleFile(path, fs, True), 0.0, prn, coff, dop, cn0)
        assert_tracks_equal(r, single)
        assert r.locked and abs(r.doppler[-1] - dop) < 2.0


def test_track_carrier_phase_integrates_doppler_over_epoch_spacing():
    # carrier_phase[k] is the Doppler phase at the end of epoch k, so it must equal
    # the sum of doppler * actual epoch spacing, not doppler * nominal 1 ms
    r = track_l1ca(_src(doppler=4000.0, data_bits=False), 0.0, 7, 1000 / FS, 4000.0, 45.0)
    expected = np.cumsum(r.doppler[:-1] * np.diff(r.t))
    assert abs(r.carrier_phase[-2] - expected[-1]) < 0.01  # cycles, over 3 s


def test_track_code_offset_is_this_epochs_code_start():
    r = track_l1ca(_src(data_bits=False), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    # the window start t[k] is the code start rounded to a sample: the two differ by
    # less than half a sample once code_offset is unwrapped around t[k]
    residual = (r.code_offset - r.t + 0.5 * CA_PERIOD) % CA_PERIOD - 0.5 * CA_PERIOD
    assert np.max(np.abs(residual)) <= 0.5 / FS + 1e-12
    truth = (1000 / FS - r.t * 1234.0 / 1575.42e6) % CA_PERIOD
    m = r.t >= 2.0
    assert np.max(np.abs(r.code_offset[m] - truth[m])) * CA_CHIP_RATE < 0.05


def test_track_float32_seed_matches_float64():
    # same seed values either way (rounded to float32 once); only the dtype differs
    src = _src(seconds=1.5)
    f_if, coff, dop, cn0 = (np.float32(v) for v in (0.0, 1000 / FS, 1234.0, 45.0))
    a = track_l1ca(src, float(f_if), 7, float(coff), float(dop), float(cn0))
    b = track_l1ca(src, f_if, 7, coff, dop, cn0)
    np.testing.assert_array_equal(a.code_offset, b.code_offset)
    np.testing.assert_array_equal(a.doppler, b.doppler)


def test_track_result_phase_is_nco_phase_at_window_start():
    fs, f_if = 8e6, 2.0001e6  # 2000.1 cycles per code period: the origin is not a whole cycle
    src = _src(f_if=f_if, iq=False, fs=fs, code_offset=3000, doppler=-2000.0, seconds=1.5)
    r = track_l1ca(src, f_if, 7, 3000 / fs, -2000.0, 45.0)
    assert r.phase.shape == r.t.shape and np.all((0 <= r.phase) & (r.phase < 1))
    assert r.phase[0] == pytest.approx((f_if * r.t[0]) % 1.0, abs=1e-9)
    # from one window start to the next the NCO advances (f_if + doppler) cycles per second,
    # with the Doppler in force during that epoch (doppler[k] applies from epoch k+1 on)
    adv = (f_if + r.doppler[:-2]) * np.diff(r.t)[1:]
    step = r.phase[2:] - r.phase[1:-1]
    np.testing.assert_allclose((step - adv + 0.5) % 1.0 - 0.5, 0, atol=1e-6)


def test_carrier_phase_origin_common_across_start_times():
    # the same signal tracked from its first code period and from the next one: with a
    # common origin the accumulated Doppler phases agree to a fraction of a cycle (mod 0.5,
    # Costas); with a per-channel origin they differ by frac(f_if * 1 ms) = 0.1 cycles.
    # Compared from 4 s: the third-order loop rings for 2 to 4 s after the handover (the two
    # starts differ by 0.059 cycle over 2 to 3 s, 0.006 from 4 s)
    fs, f_if = 8e6, 2.0001e6
    src = _src(f_if=f_if, iq=False, fs=fs, code_offset=3000, doppler=-2000.0, seconds=5.0)
    a = track_l1ca(src, f_if, 7, 3000 / fs, -2000.0, 45.0)
    b = track_l1ca(src, f_if, 7, 3000 / fs, -2000.0, 45.0, t_start=CA_PERIOD)
    assert b.t[0] == pytest.approx(a.t[1], abs=1 / fs)
    ta, tb = a.t[1:], b.t[1:]  # carrier_phase[k] is the phase at the end of epoch k
    k = slice(4000, 4980)
    pb = np.interp(ta[k], tb, b.carrier_phase[1:])
    d = (a.carrier_phase[1:][k] - pb + 0.25) % 0.5 - 0.25
    assert np.abs(d).max() < 0.03


def test_carrier_phase_is_the_nco_doppler_phase():
    # a seed Doppler 200 Hz off: once locked, the accumulated Doppler phase at a code start
    # must equal the NCO phase there minus the IF phase, else every channel carries a
    # fractional bias of (seed error) x 1 ms cycles
    fs, f_if = 8e6, 2.0001e6
    src = _src(f_if=f_if, iq=False, fs=fs, code_offset=3000, doppler=-2000.0)
    r = track_l1ca(src, f_if, 7, 3000 / fs, -1800.0, 45.0)
    k = np.arange(2000, len(r.t))
    d = (r.phase[k] - (f_if * r.t[k]) % 1.0 - r.carrier_phase[k - 1] + 0.5) % 1.0 - 0.5
    assert np.abs(d).max() < 1e-3


def test_discriminators_are_recorded_and_settle():
    fs, doppler, offset = 2.5e6, 800.0, 500
    x = synth_l1ca(4, fs, 0.0, doppler, offset, int(fs * 3.0), 45.0, True)
    track = track_l1ca(ArraySamples(x, fs), 0.0, 4, offset / fs, doppler, 45.0)
    n = len(track.doppler)
    assert track.locked
    assert track.carrier_error.shape == track.code_error.shape == (n,)
    assert np.isnan(track.carrier_error[0])
    # each recorded value is the discriminator of the recorded correlator sums
    p = track.corr[:, 0].astype(np.complex128)
    n_pullin = round(T_FLL_PULLIN / CA_PERIOD)
    k0 = 0
    finite = []
    for seconds, _, n_sum in FLL_STAGES:
        n_stage = round(seconds / CA_PERIOD)
        sums = p[k0 : k0 + n_stage].reshape(-1, n_sum).sum(axis=1)
        fll = k0 + np.arange(2 * n_sum - 1, n_stage, n_sum)  # sums with a predecessor
        dot = (sums[1:] * sums[:-1].conj()).real
        cross = (sums[:-1] * sums[1:].conj()).imag
        np.testing.assert_allclose(track.carrier_error[fll], np.arctan(cross / dot) / (2 * np.pi))
        finite.extend(fll)
        k0 += n_stage
    assert k0 == n_pullin
    others = np.setdiff1d(np.arange(n_pullin), finite)  # first sums, between updates
    assert np.isnan(track.carrier_error[others]).all()
    pll = np.arange(n_pullin, n)
    np.testing.assert_allclose(
        track.carrier_error[pll], np.arctan(p[pll].imag / p[pll].real) / (2 * np.pi)
    )
    n_dll = round(T_DLL / CA_PERIOD)
    updates = np.arange(n_dll - 1, n, n_dll)
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(track.code_error)), updates)
    mag = np.abs(track.corr[: len(updates) * n_dll].astype(np.complex128))
    e, lt = mag.reshape(len(updates), n_dll, 4).sum(axis=1)[:, 1:3].T
    np.testing.assert_allclose(
        track.code_error[updates], (e - lt) / (e + lt) / 2.0 * CA_PERIOD / 1023
    )
    late = track.t > track.t[0] + 2.0
    assert np.nanmean(np.abs(track.carrier_error[late])) < 0.02  # cycles
    assert np.nanmean(np.abs(track.code_error[late])) < 0.1 / CA_CHIP_RATE  # a tenth of a chip


def test_loop_timing_constants_keep_their_relations():
    # pinned here as well as asserted at import: the loss rule exempts only the first C/N0
    # window from the phase-lock test (the pull-in lies within it) and the first fade sum
    # after the handover is a whole DLL period (the pull-in ends on a DLL boundary)
    n_pullin = round(T_FLL_PULLIN / CA_PERIOD)
    assert sum(round(s / CA_PERIOD) for s, _, _ in FLL_STAGES) == n_pullin
    assert n_pullin <= round(T_CN0 / CA_PERIOD)
    assert n_pullin % round(T_DLL / CA_PERIOD) == 0


def test_pull_in_doppler_follows_the_fll_update_exactly():
    # rebuilt from the recorded prompts in the loop's own order of operations: per stage the
    # sums of its epochs, the dot/cross of consecutive sums, one update per completed sum,
    # the sums restarted at each stage; a Doppler near zero keeps an ulp of the update from
    # being absorbed into the Doppler
    fs, seed = 2.5e6, 4.0
    x = synth_l1ca(4, fs, 0.0, 0.0, 500, int(fs * 1.2), 45.0, True)
    track = track_l1ca(ArraySamples(x, fs), 0.0, 4, 500 / fs, seed, 45.0)
    fd = seed
    k = 0
    checked = 0
    for seconds, b, n_sum in FLL_STAGES:
        acc, prev, n_in = 0j, None, 0
        for _ in range(round(seconds / CA_PERIOD)):
            acc += complex(track.corr[k, 0])
            n_in += 1
            if n_in == n_sum:
                if prev is not None:
                    dot = acc.real * prev.real + acc.imag * prev.imag
                    cross = acc.real * prev.imag - acc.imag * prev.real
                    if dot != 0.0:
                        fd -= b / 0.25 * atan(cross / dot) / (2.0 * pi)
                prev, acc, n_in = acc, 0j, 0
            assert track.doppler[k] == fd, k
            checked = k
            k += 1
    assert checked == round(T_FLL_PULLIN / CA_PERIOD) - 1  # the whole pull-in was checked
    assert track.doppler[k] != fd  # the Costas loop takes over on the next epoch


def test_weak_signal_locks_after_the_pull_in():
    # 30 dB-Hz with data bits and a seed 10 Hz off, and 33 dB-Hz with a seed 90 Hz off (a weak
    # satellite's acquisition peak can be that biased): phase-locked by 3 s with the Doppler
    # right. The 1 ms two-stage FLL this replaces handed the 30 dB-Hz seeds 1 and 2 over 9.9
    # and 4.1 Hz off and they never locked (seed 0 handed over 0.4 Hz off and locked)
    for cn0, seed_error in ((30.0, 10.0), (33.0, 90.0)):
        for seed in (0, 1, 2):
            n = int(FS * 3.2)
            x = synth_l1ca(7, FS, 0.0, 1234.0, 1000, n, cn0, True, seed=seed, data_bits=True)
            r = track_l1ca(ArraySamples(x, FS), 0.0, 7, 1000 / FS, 1234.0 + seed_error, cn0)
            rho = 10 ** (cn0 / 10) * 1e-3
            case = (cn0, seed_error, seed)
            assert r.phase_lock[2999] >= 0.6 * rho / (1 + rho), case  # the window ending at 3 s
            assert abs(r.doppler[2000:3000].mean() - 1234.0) < 1.0, case


def test_seed_from_the_search_edge_pulls_in():
    # the acquisition grid is ±5000 Hz in 500 Hz bins with the peak interpolated between
    # bins, except at the two edge bins, which seed as they are: a signal 90 Hz beyond the
    # edge seeds 5000 Hz (90 Hz off, inside the ±125 Hz fold), one just inside seeds within
    # a bin; both must lock
    from pyrx.acquisition import acquire_l1ca

    for doppler in (5090.0, -4990.0):
        x = synth_l1ca(7, FS, 0.0, doppler, 1000, int(FS * 3.0), 45.0, True, data_bits=True)
        acq = acquire_l1ca(x[: int(FS * 0.020)], FS, 0.0, 7)
        assert acq.detected and abs(acq.doppler) <= 5000.0, doppler
        r = track_l1ca(ArraySamples(x, FS), 0.0, 7, acq.code_offset, acq.doppler, acq.cn0)
        dop, _, _ = _steady(r)
        assert r.locked and abs(dop.mean() - doppler) < 1.0, (doppler, acq.doppler)


def test_track_shorter_than_two_sums_runs_no_fll_update():
    # 3 epochs: one complete 2 ms sum with no predecessor and a partial second; no update
    r = track_l1ca(_src(seconds=0.0035), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert 2 <= len(r.t) <= 3 and r.locked
    assert np.all(r.doppler == 1234.0) and np.all(np.isnan(r.carrier_error))


def test_pll_update_is_pocket_sdr_third_order():
    # rebuilt from the recorded Costas errors in the loop's own order of operations: Pocket
    # SDR's C tracker, a3 = 1.1, b3 = 2.4, W = B_PLL / 0.7845, an integrated phase term that
    # starts at zero at the handover; a Doppler near zero keeps an ulp of the update from
    # being absorbed into the Doppler
    fs = 2.5e6
    x = synth_l1ca(4, fs, 0.0, 0.0, 500, int(fs * 2.0), 45.0, True)
    track = track_l1ca(ArraySamples(x, fs), 0.0, 4, 500 / fs, 3.0, 45.0)
    k0 = round(T_FLL_PULLIN / CA_PERIOD)  # the first Costas epoch
    fd = track.doppler[k0 - 1]
    w = B_PLL / 0.7845
    acc, prev = 0.0, 0.0
    for k in range(k0, len(track.t)):
        e = track.carrier_error[k]
        if np.isfinite(e):
            acc += w * w * w * e * CA_PERIOD
            fd += 2.4 * w * (e - prev) + 1.1 * w * w * e * CA_PERIOD + acc * CA_PERIOD
            prev = e
        assert track.doppler[k] == fd, k
    assert k0 < len(track.t) - 100  # the loop ran well past the handover


def test_pll_tracks_a_doppler_ramp_without_lag():
    # a car's Doppler rate reaches 10 Hz/s; a 5 Hz second-order loop lagged that by 0.11
    # cycle and lost the satellite. Third-order: no steady-state lag on a ramp. Judged from
    # 4 s, after the handover transient (the loop rings for 2 to 4 s). The second-order loop
    # read a mean error of 0.09 cycle and window indicators of 0.69 here.
    fs, rate = 4e6, 8.0
    x = synth_l1ca_ramp(7, fs, 1000.0, rate, 1000, int(fs * 6.2), 40.0, seed=1)
    r = track_l1ca(ArraySamples(x, fs), 0.0, 7, 1000 / fs, 1005.0, 40.0)
    m = r.t >= 4.0
    assert abs(np.nanmean(r.carrier_error[m])) < 0.01
    windows = r.phase_lock[np.arange(999, len(r.t), 1000)][4:6]  # the windows ending at 5 and 6 s
    assert len(windows) == 2 and windows.min() > 0.8
    truth = 1000.0 + rate * r.t[m]
    assert abs(np.mean(r.doppler[m] - truth)) < 0.5


def _rebuild_third_order(track):
    # the Costas loop's Doppler rebuilt from the recorded prompts and errors in the loop's own order
    # of operations: Pocket SDR's third-order update with the integrator held while the 10 ms prompt
    # power is below FADE_HOLD of its running mean (reference seeded by the first sum after the
    # handover, then FADE_REF_GAIN per sum). Returns the Doppler per epoch and the held-epoch mask.
    k0 = round(T_FLL_PULLIN / CA_PERIOD)
    n_dll = round(T_DLL / CA_PERIOD)
    fd = track.doppler[k0 - 1]
    w = B_PLL / 0.7845
    acc = prev = fade_sum = fade_ref = 0.0
    fading = False
    out, held = np.full(len(track.t), np.nan), np.zeros(len(track.t), bool)
    for k in range(k0, len(track.t)):
        e = track.carrier_error[k]
        # the power window runs on every Costas epoch, a zero prompt included
        fade_sum += abs(complex(track.corr[k, 0])) ** 2
        if (k + 1) % n_dll == 0:
            fading = fade_ref > 0.0 and fade_sum < FADE_HOLD * fade_ref
            fade_ref = (
                fade_sum if fade_ref == 0.0 else fade_ref + FADE_REF_GAIN * (fade_sum - fade_ref)
            )
            fade_sum = 0.0
        if np.isfinite(e):  # the discriminator is undefined on a zero in-phase prompt
            if not fading:
                acc += w * w * w * e * CA_PERIOD
            held[k] = fading
            fd += 2.4 * w * (e - prev) + 1.1 * w * w * e * CA_PERIOD + acc * CA_PERIOD
            prev = e
        out[k] = fd
    return out, held


def test_zero_prompts_skip_the_costas_update_but_not_the_fade_window():
    # 57 ms of zero samples: the prompt is exactly zero there, the Costas discriminator
    # undefined and no carrier update made, while the 10 ms power windows keep their
    # schedule (the model mirrors this on integer dumps, where a zero in-phase sum is routine)
    fs = 2.5e6
    x = synth_l1ca(4, fs, 0.0, 0.0, 500, int(fs * 5.0), 40.0, False, seed=0, data_bits=True)
    x[int(fs * 3.0) : int(fs * 3.057)] = 0.0
    track = track_l1ca(ArraySamples(x, fs), 0.0, 4, 500 / fs, 3.0, 40.0)
    assert track.locked
    zero = (track.t >= 3.0) & (track.t <= 3.0559)  # windows wholly inside the zero stretch
    assert zero.sum() >= 52 and np.isnan(track.carrier_error[zero]).all()
    fd, held = _rebuild_third_order(track)
    k0 = round(T_FLL_PULLIN / CA_PERIOD)
    np.testing.assert_array_equal(track.doppler[k0:], fd[k0:])
    assert held[np.flatnonzero(zero)[-1] + 1]  # the hold set in the stretch outlasts it


def test_pll_hold_rule_is_rebuilt_from_the_prompts():
    # a 40 dB-Hz signal with a 10 dB fade for 2 s from 8 s (30 dB-Hz inside it, above the loss rule):
    # the hold triggers inside the fade and releases before it ends (the reference adapts with a
    # 0.5 s time constant); every Costas epoch's Doppler equals the rebuilt rule bit for bit. Without
    # a fade at the same C/N0 the rule never holds.
    fs = 2.5e6
    x = synth_l1ca_faded(4, fs, 0.0, 500, int(fs * 12.0), 40.0, 8.0, 2.0, 10.0, seed=0)
    track = track_l1ca(ArraySamples(x, fs), 0.0, 4, 500 / fs, 3.0, 40.0)
    assert track.locked
    fd, held = _rebuild_third_order(track)
    k0 = round(T_FLL_PULLIN / CA_PERIOD)
    np.testing.assert_array_equal(track.doppler[k0:], fd[k0:])
    # nothing held before the reference is seeded
    assert not held[: k0 + round(T_DLL / CA_PERIOD)].any()
    t_held = track.t[held]
    # set in the fade, released inside it
    assert len(t_held) and 8.0 <= t_held.min() < 8.2 and t_held.max() < 10.0
    y = synth_l1ca(4, fs, 0.0, 0.0, 500, int(fs * 4.0), 40.0, True, seed=0, data_bits=True)
    steady = track_l1ca(ArraySamples(y, fs), 0.0, 4, 500 / fs, 3.0, 40.0)
    _, held_steady = _rebuild_third_order(steady)
    assert not held_steady.any()
