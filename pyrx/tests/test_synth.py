import numpy as np
import pytest

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca, sample_code
from pyrx.synth import synth_l1ca

FS = 4e6
N = round(FS * CA_PERIOD)


def _prompt_signs(x, code_offset, n_ms=60):
    code = sample_code(gps_ca(7), CA_CHIP_RATE, FS, N)
    out = []
    for m in range(n_ms):  # 60 ms = 3 bits
        start = code_offset + m * N
        out.append(np.sign(np.real(np.dot(x[start : start + N], code))))
    return np.array(out)


def test_synth_without_bits_is_constant_sign():
    x = synth_l1ca(7, FS, 0.0, 0.0, 1000, 80 * N, 60.0, True)
    assert np.all(_prompt_signs(x, 1000) == 1)


def test_synth_data_bits_flip_every_20ms():
    x = synth_l1ca(7, FS, 0.0, 0.0, 1000, 80 * N, 60.0, True, seed=3, data_bits=True)
    s = _prompt_signs(x, 1000)
    assert np.all(s[0:20] == s[0]) and np.all(s[20:40] == s[20]) and np.all(s[40:60] == s[40])
    assert len(set(s[[0, 20, 40]])) > 1  # seed 3 must produce at least one flip


def test_synth_explicit_bits():
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    x = synth_l1ca(7, FS, 0.0, 0.0, 1000, 100 * N, 60.0, True, bits=bits)
    s = _prompt_signs(x, 1000, n_ms=80)
    assert list(s[[0, 20, 40, 60]]) == [1, -1, -1, 1]
    with pytest.raises(ValueError):
        synth_l1ca(7, FS, 0.0, 0.0, 1000, 100 * N, 60.0, True, data_bits=True, bits=bits)
