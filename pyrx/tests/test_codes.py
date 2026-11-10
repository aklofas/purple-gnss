import numpy as np
import pytest

from pyrx.codes import (
    CA_CHIP_RATE,
    GAL_CS20,
    L5_LENGTH,
    NH10,
    NH20,
    gal_cs100,
    gal_e5ai,
    gal_e5aq,
    gps_ca,
    gps_l5i,
    gps_l5q,
    sample_code,
)

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


# IS-GPS-705J Table 3-Ia, PRN: XB code advance in chips (I5, Q5) and initial XB code state
# (I5, Q5). A state lists the present output last.
L5_PHASES = {
    1: (266, 1701, "0101011100100", "1001011001100"),
    2: (365, 323, "1100000110101", "0100011110110"),
    3: (804, 5292, "0100000001000", "1111000100011"),
    4: (1138, 2020, "1011000100110", "0011101101010"),
    5: (1509, 5429, "1110111010111", "0011110110010"),
    6: (1559, 7136, "0110011111010", "0101010101001"),
    7: (1756, 1041, "1010010011111", "1111110000001"),
    8: (2084, 5947, "1011110100100", "0110101101000"),
    9: (2170, 4315, "1111100101011", "1011101000011"),
    10: (2303, 148, "0111111011110", "0010010000110"),
    11: (2527, 535, "0000100111010", "0001000000101"),
    12: (2687, 1939, "1110011111001", "0101011000101"),
    13: (2930, 5206, "0001110011100", "0100110100101"),
    14: (3471, 5910, "0100000100111", "1010000111111"),
    15: (3940, 3595, "0110101011010", "1011110001111"),
    16: (4132, 5135, "0001111001001", "1101001011111"),
    17: (4332, 6082, "0100110001111", "1110011001000"),
    18: (4924, 6990, "1111000011110", "1011011100100"),
    19: (5343, 3546, "1100100011111", "0011001011011"),
    20: (5443, 1523, "0110101101101", "1100001110001"),
    21: (5641, 4548, "0010000001000", "0110110010000"),
    22: (5816, 4484, "1110111101111", "0010110001110"),
    23: (5898, 1893, "1000011111110", "1000101111101"),
    24: (5918, 3961, "1100010110100", "0110111110011"),
    25: (5955, 7106, "1101001101101", "0100010011011"),
    26: (6243, 5299, "1010110010110", "0101010111100"),
    27: (6345, 4660, "0101011011110", "1000011111010"),
    28: (6477, 276, "0111101010110", "1111101000010"),
    29: (6518, 4389, "0101111100001", "0101000100100"),
    30: (6875, 3783, "1000010110111", "1000001111001"),
    31: (7168, 1591, "0001010011110", "0101111100101"),
    32: (7187, 1601, "0000010111001", "1001000101010"),
}


def _register(taps, length, stages=13):
    """Chips (0/1) of a shift register seeded with all ones: the output is the last stage,
    the input the modulo-2 sum of the 1-based tapped stages."""
    state = [1] * stages
    out = np.empty(length, dtype=int)
    for i in range(length):
        out[i] = state[-1]
        feedback = 0
        for t in taps:
            feedback ^= state[t - 1]
        state = [feedback] + state[:-1]
    return out


def _l5_registers():
    xa = _register((9, 10, 12, 13), 8190)  # 1 + x^9 + x^10 + x^12 + x^13, short-cycled
    xb = _register((1, 3, 4, 6, 7, 8, 12, 13), 8191)  # its natural period
    return xa, xb


def _chips(code):
    return (code == -1).astype(int)


@pytest.mark.parametrize("prn", sorted(L5_PHASES))
def test_l5_xb_start_states(prn):
    xa, xb = _l5_registers()
    adv_i, adv_q, state_i, state_q = L5_PHASES[prn]
    for gen, adv, state in ((gps_l5i, adv_i, state_i), (gps_l5q, adv_q, state_q)):
        code = gen(prn)
        assert code.shape == (L5_LENGTH,) and code.dtype == np.int8
        first = "".join(map(str, _chips(code)[:13] ^ xa[:13]))  # the code's XB factor
        assert first == state[::-1]
        assert "".join(map(str, xb[(np.arange(13) + adv) % 8191])) == state[::-1]


@pytest.mark.parametrize("gen,prn,column", [(gps_l5q, 6, 1), (gps_l5i, 32, 0)])
def test_l5_xb_runs_its_own_period(gen, prn, column):
    # the XB register is restarted at its natural completion (8191 chips) inside the 1 ms
    # code and reset with it; a code built by rotating a 10230-chip array is wrong in the tail
    xa, xb = _l5_registers()
    adv = L5_PHASES[prn][column]
    i = np.arange(L5_LENGTH)
    expected = xa[i % 8190] ^ xb[(i + adv) % 8191]
    np.testing.assert_array_equal(_chips(gen(prn)), expected)
    assert 8191 - adv < 8190 < L5_LENGTH  # both wraps lie inside the period


def test_l5_cross_correlation():
    def peak(a, b):
        spectrum = np.fft.fft(a.astype(float)) * np.conj(np.fft.fft(b.astype(float)))
        return np.abs(np.fft.ifft(spectrum)).max()

    assert peak(gps_l5q(6), gps_l5q(6)) == pytest.approx(L5_LENGTH)
    assert peak(gps_l5i(6), gps_l5q(6)) < 0.1 * L5_LENGTH
    assert peak(gps_l5q(6), gps_l5q(11)) < 0.1 * L5_LENGTH
    assert peak(gal_e5aq(2), gal_e5ai(2)) < 0.1 * L5_LENGTH
    assert peak(gal_e5aq(2), gps_l5q(6)) < 0.1 * L5_LENGTH


def test_neuman_hofman():
    assert "".join(map(str, _chips(NH10))) == "0000110101"
    assert "".join(map(str, _chips(NH20))) == "00000100110101001110"
    assert NH10.dtype == np.int8 and NH20.dtype == np.int8


# Galileo OS SIS ICD 2.1 Tables 16 and 17: the first 24 chips of each primary code in hex
E5AI_FIRST_24 = (
    0x3CEA9D, 0x9D8CF1, 0x45D1C8, 0x7A0133, 0x64D423, 0x23300D, 0x91CEF2, 0xAA82DC, 0xF2A17D,
    0x3D84AE, 0x446D38, 0xC514F2, 0x0C0184, 0x8767E0, 0xCB8EFF, 0x93EBCD, 0x5D55CE, 0xB19B7C,
    0x5805FC, 0xF99EA1, 0xB23CE5, 0x8515E8, 0x436822, 0x30F77B, 0xA7D629, 0x9BFAC7, 0x18A25B,
    0x69A39F, 0x39B27D, 0x454598, 0xF2BC62, 0x9DDBC6, 0x332827, 0x6E2FCA, 0x22C6D5, 0xE881D9,
)  # fmt: skip
E5AQ_FIRST_24 = (
    0x515537, 0xD67539, 0x58B2E5, 0x305914, 0x442710, 0x593CF8, 0x214AD7, 0x435EA6, 0xC1A7D5,
    0xF0E94A, 0xA8C239, 0x2EB63B, 0x8F0A46, 0x896DD4, 0x0245F1, 0xEB0160, 0xED28B3, 0xCB9F5B,
    0x576592, 0xA88811, 0xDD3649, 0xB59F42, 0xFF81F6, 0xCE8128, 0xDCD55C, 0x79E450, 0xB63460,
    0x6D562B, 0x3A9010, 0x59CD72, 0xC0211A, 0x28EB96, 0xD7554B, 0x425126, 0xE0DAFB, 0xEF79F2,
)  # fmt: skip


def _hex(code, n):
    return f"{int(''.join(map(str, _chips(code)[:n])), 2):0{n // 4}X}"


@pytest.mark.parametrize("prn", range(1, 37))
def test_e5a_start_values(prn):
    for gen, table in ((gal_e5ai, E5AI_FIRST_24), (gal_e5aq, E5AQ_FIRST_24)):
        code = gen(prn)
        assert code.shape == (L5_LENGTH,) and code.dtype == np.int8
        assert int(_hex(code, 24), 16) == table[prn - 1]
        assert abs(int(code.sum())) < 300  # a balanced code, not a stuck register


def test_e5a_secondary_codes():
    # Table 20: CS20_1 for E5a-I, CS100_n for E5a-Q of code number n
    assert _hex(GAL_CS20, 20) == "842E9"
    assert _hex(gal_cs100(1), 100) == "83F6F69D8F6E15411FB8C9B1C"
    assert _hex(gal_cs100(36), 100) == "A8C3DE1AC668089B0B45B3579"
    assert len(gal_cs100(7)) == 100 and gal_cs100(7).dtype == np.int8


@pytest.mark.parametrize("gen,bad", [(gps_l5i, 0), (gps_l5q, 33), (gal_e5ai, 37), (gal_e5aq, 0), (gal_cs100, 37)])
def test_pilot_codes_bad_prn(gen, bad):
    with pytest.raises(ValueError):
        gen(bad)
