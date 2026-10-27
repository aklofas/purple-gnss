import numpy as np
import pytest

from pyrx.acquisition import acquire_l1ca
from pyrx.synth import synth_l1ca

FS = 4e6


def _sig(prn=7, doppler=1234.0, code_offset=1000, cn0=45.0, f_if=0.0, iq=True, fs=FS, ms=10):
    return synth_l1ca(prn, fs, f_if, doppler, code_offset, int(fs * 1e-3 * ms), cn0, iq)


def test_acquire_finds_synthetic_signal():
    r = acquire_l1ca(_sig(), FS, 0.0, 7)
    assert r.detected
    assert abs(r.code_offset * FS - 1000) <= 1.0
    assert abs(r.doppler - 1234.0) <= 500.0
    assert 40.0 <= r.cn0 <= 50.0


def test_acquire_absent_prn_not_detected():
    r = acquire_l1ca(_sig(prn=7), FS, 0.0, 9)
    assert not r.detected and r.cn0 < 36.0


def test_acquire_real_samples_with_if():
    fs, f_if = 24e6, 6.0e6
    x = _sig(prn=3, doppler=-2000.0, code_offset=5000, f_if=f_if, iq=False, fs=fs)
    r = acquire_l1ca(x, fs, f_if, 3)
    assert r.detected
    assert abs(r.code_offset * fs - 5000) <= 1.0
    assert abs(r.doppler + 2000.0) <= 500.0


def test_acquire_noninteger_block():
    # 4000.2 samples per code period: the peak drifts 0.2 sample per block,
    # so the 10-block sum smears it by about one sample.
    fs = 4.0002e6
    x = _sig(prn=7, code_offset=2000, fs=fs)
    r = acquire_l1ca(x, fs, 0.0, 7)
    assert r.detected and abs(r.code_offset * fs - 2000) <= 2.0


def test_acquire_doppler_at_edge():
    r = acquire_l1ca(_sig(doppler=5000.0), FS, 0.0, 7)
    assert r.detected and abs(r.doppler - 5000.0) <= 500.0


def test_acquire_short_buffer():
    with pytest.raises(ValueError):
        acquire_l1ca(np.zeros(3999, dtype=np.complex64), FS, 0.0, 7)


def test_acquire_grid_shape_and_bins():
    r = acquire_l1ca(_sig(), FS, 0.0, 7)
    assert r.grid.shape == (21, 4000)
    np.testing.assert_allclose(r.dopplers[[0, 10, 20]], [-5000.0, 0.0, 5000.0])
    assert r.grid.max() == pytest.approx(1.0)
