import numpy as np
import pytest

from pyrx.acquisition import (
    DEFAULT_WINDOW,
    DETECT_THRESHOLD,
    acquire,
    acquire_l1ca,
    acquire_signal,
)
from pyrx.signalspec import E5A, L1CA, L5
from pyrx.synth import synth_l1ca, synth_pilot

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


def test_default_window_and_threshold():
    assert DEFAULT_WINDOW == 0.020 and DETECT_THRESHOLD == 36.0


PILOT_FS = 12e6
PILOT_N = round(PILOT_FS * 1e-3)


def _pilot(spec, prn, code_offset, cn0=42.0, doppler=1234.0):
    return synth_pilot(spec, prn, PILOT_FS, 0.0, doppler, code_offset, 20 * PILOT_N, cn0)


def _lag_error(acq, code_offset):
    """Samples between the reported code start and the generated one, modulo a period."""
    return abs((acq.code_offset * PILOT_FS - code_offset + PILOT_N / 2) % PILOT_N - PILOT_N / 2)


def test_pilot_acquisition_any_code_phase():
    # wherever the code period starts within a block, the secondary code's sign changes
    # must not cost signal
    cn0 = []
    for quarter in range(4):
        code_offset = quarter * PILOT_N // 4 + 7
        acq = acquire_signal(_pilot(L5, 6, code_offset), PILOT_FS, 0.0, L5, 6)
        assert acq.detected and acq.prn == 6, quarter
        assert abs(acq.doppler - 1234.0) <= 250.0, quarter
        assert _lag_error(acq, code_offset) <= 1.0, quarter
        cn0.append(acq.cn0)
    assert max(cn0) - min(cn0) <= 1.0, cn0
    # the estimate reads up to 2 dB low between two Doppler bins, as the one-period search does
    assert max(abs(c - 42.0) for c in cn0) <= 2.5, cn0


def test_pilot_acquisition_e5a():
    code_offset = PILOT_N // 2 + 3
    acq = acquire_signal(_pilot(E5A, 2, code_offset), PILOT_FS, 0.0, E5A, 2)
    assert acq.detected and abs(acq.doppler - 1234.0) <= 250.0
    assert _lag_error(acq, code_offset) <= 1.0
    assert abs(acq.cn0 - 42.0) <= 2.5


def test_pilot_acquisition_rejects_absent_prn():
    x = _pilot(L5, 6, 500, cn0=48.0)
    assert not acquire_signal(x, PILOT_FS, 0.0, L5, 11).detected
    assert not acquire_signal(x, PILOT_FS, 0.0, E5A, 6).detected


def test_one_period_search_loses_to_the_secondary_code():
    # a code period starting mid-block: every sign change of the secondary code cancels
    # the two halves of a one-period block (the Doppler on a bin, so that no residual
    # rotation between the halves hides it)
    x = _pilot(L5, 6, PILOT_N // 2, doppler=1000.0)
    args = (x, PILOT_FS, 0.0, L5.code(6), L5.chip_rate, 1e-3)
    assert acquire(*args, two_period=True).cn0 - acquire(*args).cn0 >= 2.0


def test_acquire_signal_l1ca_is_acquire_l1ca():
    x = synth_l1ca(7, 4e6, 0.0, 1234.0, 1000, int(4e6 * 0.02), 45.0, True)
    a, b = acquire_signal(x, 4e6, 0.0, L1CA, 7), acquire_l1ca(x, 4e6, 0.0, 7)
    assert (a.code_offset, a.doppler, a.cn0, a.detected) == (b.code_offset, b.doppler, b.cn0, b.detected)
    assert np.array_equal(a.grid, b.grid)


def test_two_period_search_needs_two_periods():
    with pytest.raises(ValueError, match="two code periods"):
        acquire_signal(np.zeros(PILOT_N + 5, dtype=np.complex64), PILOT_FS, 0.0, L5, 6)
