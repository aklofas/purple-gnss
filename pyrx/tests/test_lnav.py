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
    "af2": 0.0,
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


def _quantised(fields, decoded):
    # every decoded value must equal the encoded one to within one LSB of its scale
    for k, v in fields.items():
        assert decoded[k] == pytest.approx(v, rel=1e-6, abs=2e-9), k


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
