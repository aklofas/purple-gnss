import numpy as np
import pytest

from pyrx.codes import CA_CHIP_RATE, gps_ca, sample_code

FIRST_10_CHIPS_OCTAL = {  # IS-GPS-200 Table 3-Ia, PRN: first 10 chips as octal
    1: 0o1440, 2: 0o1620, 3: 0o1710, 4: 0o1744, 5: 0o1133, 6: 0o1455,
    7: 0o1131, 8: 0o1454, 9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}  # fmt: skip


@pytest.mark.parametrize("prn,octal", FIRST_10_CHIPS_OCTAL.items())
def test_gps_ca_first_chips(prn, octal):
    code = gps_ca(prn)
    assert code.shape == (1023,) and code.dtype == np.int8
    bits = (code[:10] == -1).astype(int)  # chip 1 -> -1
    value = int("".join(map(str, bits)), 2)
    assert value == octal


def test_gps_ca_balance():
    assert np.sum(gps_ca(1) == -1) == 512  # 512 ones, 511 zeros


def test_gps_ca_autocorrelation_sidelobes():
    c = gps_ca(5).astype(float)
    full = np.array([np.dot(c, np.roll(c, k)) for k in range(1023)])
    assert full[0] == 1023
    assert np.max(np.abs(full[1:])) <= 65


@pytest.mark.parametrize("prn", [0, 33])
def test_gps_ca_bad_prn(prn):
    with pytest.raises(ValueError):
        gps_ca(prn)


def test_sample_code_integer_ratio():
    code = np.array([1, -1, -1, 1], dtype=np.int8)
    s = sample_code(code, chip_rate=1.0, fs=2.0, n=10)
    np.testing.assert_array_equal(s, [1, 1, -1, -1, -1, -1, 1, 1, 1, 1])


def test_sample_code_phase_offset():
    code = np.array([1, -1, -1, 1], dtype=np.int8)
    s = sample_code(code, chip_rate=1.0, fs=1.0, n=4, phase_chips=1.5)
    np.testing.assert_array_equal(s, [-1, -1, 1, 1])


def test_sample_code_one_period_length():
    assert len(sample_code(gps_ca(1), CA_CHIP_RATE, 4e6, 4000)) == 4000
