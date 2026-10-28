from math import pi

import numpy as np
import pytest

from pyrx.lnav import (
    PREAMBLE,
    Ephemeris,
    assemble_ephemeris,
    data_bits,
    decode_how,
    decode_subframe,
    encode_subframe,
    encode_word,
    parity_ok,
)

SF1 = {
    "wn": 142,
    "code_l2": 1,
    "ura": 0,
    "health": 0,
    "iodc": 917,
    "l2p": 0,
    "tgd": -4.19095158577e-09,
    "toc": 28800,
    "af2": 2 * 2**-55,
    "af1": -1.02318153949e-12,
    "af0": -1.97644811124e-04,
}
SF2 = {
    "iode": 149,
    "crs": -37.5625,
    "delta_n": 4.6734e-09,
    "m0": -1.2345678,
    "cuc": -2.0e-06,
    "e": 0.001602733,
    "cus": 5.8e-06,
    "sqrt_a": 5153.668436,
    "toe": 28800,
    "fit": 0,
    "aodo": 0,
}
SF3 = {
    "cic": 1.2e-07,
    "omega0": 2.1,
    "cis": -9.3e-08,
    "i0": 0.96,
    "crc": 250.0,
    "omega": -0.5,
    "omega_dot": -8.1e-09,
    "iode": 149,
    "idot": 1.5e-10,
}
# exact multiples of the LSBs (12*2^-30, -1*2^-27, -1*2^-24, 2*2^-24; 56*2^11, -10*2^14,
# -3*2^16, 14*2^16), which is what the broadcast values of 2022-01-14 round to in RINEX
IONO = {
    "alpha0": 12 * 2**-30,
    "alpha1": -(2**-27),
    "alpha2": -(2**-24),
    "alpha3": 2 * 2**-24,
    "beta0": 114688.0,
    "beta1": -163840.0,
    "beta2": -196608.0,
    "beta3": 917504.0,
}


# LSB of every scaled field in the decoded units (IS-GPS-200 Tables 20-I, 20-III and
# 20-X; angles in semicircles become radians); fields not listed are integers
SC = 2**-31 * pi
LSB = {
    "tgd": 2**-31, "toc": 2**4, "af2": 2**-55, "af1": 2**-43, "af0": 2**-31,
    "crs": 2**-5, "delta_n": 2**-43 * pi, "m0": SC, "cuc": 2**-29, "e": 2**-33,
    "cus": 2**-29, "sqrt_a": 2**-19, "toe": 2**4,
    "cic": 2**-29, "omega0": SC, "cis": 2**-29, "i0": SC, "crc": 2**-5, "omega": SC,
    "omega_dot": 2**-43 * pi, "idot": 2**-43 * pi,
    "alpha0": 2**-30, "alpha1": 2**-27, "alpha2": 2**-24, "alpha3": 2**-24,
    "beta0": 2**11, "beta1": 2**14, "beta2": 2**16, "beta3": 2**16,
}  # fmt: skip


def _quantised(fields, decoded):
    # the encoder rounds to the field's LSB, so a decoded value is within half an LSB of
    # the value given; integer fields are exact
    for k, v in fields.items():
        if k in LSB:
            assert abs(decoded[k] - v) <= 0.5 * LSB[k] * (1 + 1e-12), (k, decoded[k], v)
        else:
            assert decoded[k] == v, k


def test_encode_word_roundtrip_and_parity():
    rng = np.random.default_rng(0)
    d29 = d30 = 0
    for _ in range(50):
        data = rng.integers(0, 2, 24).astype(np.uint8)
        word = encode_word(data, d29, d30)
        assert word.shape == (30,)
        assert parity_ok(word, d29, d30)
        np.testing.assert_array_equal(data_bits(word, d30), data)
        d29, d30 = int(word[28]), int(word[29])


def test_parity_detects_single_bit_error():
    data = np.ones(24, dtype=np.uint8)
    word = encode_word(data, 1, 0)
    for i in range(30):
        bad = word.copy()
        bad[i] ^= 1
        assert not parity_ok(bad, 1, 0), i


@pytest.mark.parametrize("sf_id,fields", [(1, SF1), (2, SF2), (3, SF3)])
def test_subframe_roundtrip(sf_id, fields):
    bits = encode_subframe(sf_id, tow_count=5044, fields=fields)
    assert bits.shape == (300,)
    np.testing.assert_array_equal(bits[:8], PREAMBLE)
    assert int(bits[298]) == 0 and int(bits[299]) == 0
    sf = decode_subframe(bits, 0, 0, prn=4)
    assert sf is not None and sf.sf_id == sf_id and sf.prn == 4
    assert sf.tow_s == 5044 * 6 - 6
    _quantised(fields, sf.fields)


def test_subframe4_page18_roundtrip():
    bits = encode_subframe(4, 5045, dict(page_sv_id=56, **IONO))
    sf = decode_subframe(bits, 0, 0)
    assert sf.sf_id == 4 and sf.fields["page_sv_id"] == 56
    _quantised(IONO, sf.fields)


def _tc(value, nbits):
    return f"{value & ((1 << nbits) - 1):0{nbits}b}"


def test_subframe4_page18_decodes_the_broadcast_layout():
    # Subframe 4 page 18 laid out word by word from IS-GPS-200 Figure 20-1 (sheet 8) and
    # Table 20-X, without the decoder's field table: word 3 carries data ID, SV ID 56,
    # alpha0 and alpha1; word 4 alpha2, alpha3, beta0; word 5 beta1, beta2, beta3; each an
    # 8-bit two's complement. The integers are the GPSA/GPSB values broadcast on
    # 2022-01-14 (12, -1, -1, 2 and 56, -10, -3, 14 in units of the LSB).
    alpha, beta = (12, -1, -1, 2), (56, -10, -3, 14)
    words = [
        "".join(str(b) for b in PREAMBLE) + "0" * 16,  # TLM: preamble, message, reserved
        _tc(5045, 17) + "00" + _tc(4, 3) + "00",  # HOW: TOW count, flags, ID 4, free bits
        "01" + _tc(56, 6) + _tc(alpha[0], 8) + _tc(alpha[1], 8),
        _tc(alpha[2], 8) + _tc(alpha[3], 8) + _tc(beta[0], 8),
        _tc(beta[1], 8) + _tc(beta[2], 8) + _tc(beta[3], 8),
    ] + ["0" * 24] * 5
    d29 = d30 = 0
    bits = []
    for w in words:
        word = encode_word(np.array([int(c) for c in w], dtype=np.uint8), d29, d30)
        bits.extend(word)
        d29, d30 = int(word[28]), int(word[29])
    sf = decode_subframe(np.array(bits, dtype=np.uint8), 0, 0, prn=10)
    assert sf is not None and sf.sf_id == 4 and sf.tow_s == 5045 * 6 - 6
    assert sf.fields["page_sv_id"] == 56
    expect = {
        "alpha0": 12 * 2**-30, "alpha1": -(2**-27), "alpha2": -(2**-24), "alpha3": 2 * 2**-24,
        "beta0": 56 * 2**11, "beta1": -10 * 2**14, "beta2": -3 * 2**16, "beta3": 14 * 2**16,
    }  # fmt: skip
    assert {k: sf.fields[k] for k in expect} == expect


def test_decode_rejects_bad_parity_and_bad_id():
    bits = encode_subframe(2, 5044, SF2)
    bad = bits.copy()
    bad[100] ^= 1
    assert decode_subframe(bad, 0, 0) is None
    assert decode_subframe(bits, 1, 0) is None  # wrong D29* changes the parity
    assert decode_subframe(encode_subframe(6, 5044, SF2), 0, 0) is None


def test_assemble_ephemeris_requires_matching_iode():
    sfs = [
        decode_subframe(encode_subframe(i, 5044 + i, f), 0, 0, prn=4)
        for i, f in ((1, SF1), (2, SF2), (3, SF3))
    ]
    eph = assemble_ephemeris(sfs)
    assert isinstance(eph, Ephemeris) and eph.prn == 4 and eph.iode == 149
    assert eph.sqrt_a == pytest.approx(SF2["sqrt_a"], rel=1e-9)
    stale = dict(SF3, iode=148)
    sfs[2] = decode_subframe(encode_subframe(3, 5047, stale), 0, 0, prn=4)
    assert assemble_ephemeris(sfs) is None


def test_decode_how_reads_tow_and_id():
    bits = encode_subframe(2, 5044, SF2)
    assert decode_how(bits[:30], bits[30:60], 0, 0) == (5044, 2)
    assert decode_how(bits[:30], bits[30:60], 1, 0) is None
    bad = bits.copy()
    bad[40] ^= 1
    assert decode_how(bad[:30], bad[30:60], 0, 0) is None
