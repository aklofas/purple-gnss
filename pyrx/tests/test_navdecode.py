import numpy as np
import pytest
from test_lnav import SF1, SF2, SF3

from pyrx.lnav import encode_subframe
from pyrx.navdecode import decode_lnav
from pyrx.samples import ArraySamples
from pyrx.synth import synth_l1ca
from pyrx.tracking import track_l1ca

FS = 2.5e6


def _frame(tow0=5044):
    out, d29, d30 = [], 0, 0
    for i, f in ((1, SF1), (2, SF2), (3, SF3)):
        sf = encode_subframe(i, tow0 + i - 1, f, d29, d30)
        d29, d30 = int(sf[298]), int(sf[299])
        out.append(sf)
    return np.concatenate(out)


@pytest.mark.parametrize("invert", [False, True])
def test_decode_synthetic_frame(invert):
    bits = _frame()
    lead = np.zeros(100, dtype=np.uint8)  # 2 s of zeros covers pull-in before the frame
    seq = np.concatenate([lead, bits])
    n = int(FS * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, FS, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    if invert:
        x = -x
    r = track_l1ca(ArraySamples(x, FS), 0.0, 4, 500 / FS, 800.0, 45.0)
    res = decode_lnav(r)
    assert [s.sf_id for s in res.subframes] == [1, 2, 3]
    assert res.ephemeris is not None and res.ephemeris.iode == 149
    assert res.ephemeris.af0 == pytest.approx(SF1["af0"], rel=1e-6)
    # first bit of subframe 1 starts 100 bits after the code offset: epoch index 2000
    assert abs(res.subframes[0].epoch - 2000) <= 1
    assert res.subframes[0].tow_s == 5044 * 6 - 6


def test_decode_too_short_returns_nothing():
    bits = _frame()[:250]  # less than one subframe after pull-in
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), bits])
    n = int(FS * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, FS, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    r = track_l1ca(ArraySamples(x, FS), 0.0, 4, 500 / FS, 800.0, 45.0)
    res = decode_lnav(r)
    assert res.subframes == [] and res.ephemeris is None
