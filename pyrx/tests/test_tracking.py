import numpy as np
import pytest

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD
from pyrx.samples import ArraySamples, SampleFile
from pyrx.synth import synth_l1ca
from pyrx.tracking import CN0_LOST, track_channels, track_l1ca

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
    r = track_l1ca(_src(), 0.0, 7, (1000 + 0.5) / FS, 1234.0 + 200.0, 45.0)
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


def test_track_declares_loss():
    # Track PRN 9 where only PRN 7 is present: C/N0 must fall below the loss threshold.
    r = track_l1ca(_src(prn=7, seconds=4.0), 0.0, 9, 1000 / FS, 1234.0, 45.0)
    assert not r.locked and r.lost_at is not None and r.lost_at <= 3.5
    assert r.cn0[-1] < CN0_LOST
    assert len(r.t) < 4000


def test_track_stops_at_end_of_data():
    r = track_l1ca(_src(seconds=0.5), 0.0, 7, 1000 / FS, 1234.0, 45.0)
    assert 495 <= len(r.t) <= 500 and r.locked
    r2 = track_l1ca(_src(seconds=0.5), 0.0, 7, 1000 / FS, 1234.0, 45.0, duration=0.2)
    assert 195 <= len(r2.t) <= 200


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
        np.testing.assert_array_equal(r.doppler, single.doppler)
        np.testing.assert_array_equal(r.code_offset, single.code_offset)
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
    # Costas); with a per-channel origin they differ by frac(f_if * 1 ms) = 0.1 cycles
    fs, f_if = 8e6, 2.0001e6
    src = _src(f_if=f_if, iq=False, fs=fs, code_offset=3000, doppler=-2000.0)
    a = track_l1ca(src, f_if, 7, 3000 / fs, -2000.0, 45.0)
    b = track_l1ca(src, f_if, 7, 3000 / fs, -2000.0, 45.0, t_start=CA_PERIOD)
    assert b.t[0] == pytest.approx(a.t[1], abs=1 / fs)
    ta, tb = a.t[1:], b.t[1:]  # carrier_phase[k] is the phase at the end of epoch k
    k = slice(2000, 2980)
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
    assert track.carrier_error.shape == track.code_error.shape == (n,)
    assert np.isnan(track.carrier_error[0])
    late = track.t > track.t[0] + 2.0
    assert np.nanmean(np.abs(track.carrier_error[late])) < 0.02  # cycles
    updates = np.isfinite(track.code_error)
    assert abs(updates.sum() - n // 10) <= 1
    assert np.nanmean(np.abs(track.code_error[late])) < 0.1 / CA_CHIP_RATE  # a tenth of a chip
