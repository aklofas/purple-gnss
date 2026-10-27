import numpy as np
import pytest

from pyrx.samples import read_samples


def test_read_iq_negates_q(tmp_path):
    raw = np.array([1, 2, 3, 4, -5, 6], dtype=np.int8)  # I,Q,I,Q,I,Q
    p = tmp_path / "x.bin"
    raw.tofile(p)
    x = read_samples(p, fs=3.0, iq=True)
    assert x.dtype == np.complex64
    np.testing.assert_array_equal(x, np.array([1 - 2j, 3 - 4j, -5 - 6j], dtype=np.complex64))


def test_read_real_file(tmp_path):
    raw = np.array([7, -8, 9], dtype=np.int8)
    p = tmp_path / "x.bin"
    raw.tofile(p)
    x = read_samples(p, fs=3.0, iq=False)
    np.testing.assert_array_equal(x, np.array([7, -8, 9], dtype=np.complex64))


def test_read_offset_and_duration(tmp_path):
    raw = np.arange(10, dtype=np.int8)
    p = tmp_path / "x.bin"
    raw.tofile(p)
    x = read_samples(p, fs=10.0, iq=False, t_start=0.3, duration=0.4)
    np.testing.assert_array_equal(x.real, [3, 4, 5, 6])


def test_read_samples_past_end(tmp_path):
    raw = np.arange(10, dtype=np.int8)
    p = tmp_path / "x.bin"
    raw.tofile(p)
    with pytest.raises(ValueError):
        read_samples(p, fs=10.0, iq=False, t_start=0.5, duration=1.0)
