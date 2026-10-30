import numpy as np
from pyrx.codes import gps_ca

from fpgamodel.quantize import code_bits, pack_words, to_sign_mag, values


def test_recording_values_round_trip():
    x = np.array([-3, -1, 1, 3], dtype=np.int8)
    codes = to_sign_mag(x)
    assert codes.dtype == np.uint8
    assert codes.tolist() == [0b11, 0b10, 0b00, 0b01]
    assert values(codes).tolist() == [-3, -1, 1, 3]


def test_float_threshold_is_std():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000)
    codes = to_sign_mag(x)
    assert abs((codes & 1).mean() - 0.3173) < 0.01  # P(|x| >= sigma)
    assert abs((codes >> 1).mean() - 0.5) < 0.01


def test_explicit_threshold():
    x = np.array([0.5, 1.5, -0.5, -1.5])
    assert to_sign_mag(x, threshold=1.0).tolist() == [0b00, 0b01, 0b10, 0b11]


def test_pack_words_layout():
    assert pack_words(np.array([3, 1, 2], dtype=np.uint8)).tolist() == [0b10_01_11]
    words = pack_words(np.arange(20, dtype=np.uint8) & 3)
    assert words.dtype == np.uint32
    assert len(words) == 2
    assert words[1] == 0b11_10_01_00


def test_code_bits_prn1():
    chips = gps_ca(1)
    words = code_bits(chips)
    assert words.dtype == np.uint32
    assert len(words) == 32
    for i in (0, 1, 31, 32, 1022):
        assert (words[i >> 5] >> (i & 31)) & 1 == (chips[i] < 0)
    assert words[31] >> 31 == 0  # chip 1023 does not exist: padding is zero
