import numpy as np
import pytest

from pyrx.ephemeris import Ionosphere
from pyrx.lnav import decode_subframe, encode_subframe
from pyrx.navdecode import (
    LnavResult,
    decode_lnav,
    ephemerides_by_prn,
    ionosphere_of,
    tow1_by_prn,
)
from pyrx.samples import ArraySamples
from pyrx.synth import SF1, SF2, SF3, frame, synth_l1ca
from pyrx.tracking import track_l1ca

FS = 2.5e6


@pytest.mark.parametrize("invert", [False, True])
def test_decode_synthetic_frame(invert):
    bits = frame()
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
    bits = frame()[:250]  # less than one subframe after pull-in
    seq = np.concatenate([np.zeros(100, dtype=np.uint8), bits])
    n = int(FS * (len(seq) * 0.020 + 0.1))
    x = synth_l1ca(4, FS, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    r = track_l1ca(ArraySamples(x, FS), 0.0, 4, 500 / FS, 800.0, 45.0)
    res = decode_lnav(r)
    assert res.subframes == [] and res.ephemeris is None


def _sf(sf_id, tow_count, fields, prn=4):
    return decode_subframe(encode_subframe(sf_id, tow_count, fields), 0, 0, prn=prn)


def _lnav(prn, subframes):
    return LnavResult(prn, None, subframes, None, None)


def test_ephemerides_by_prn_unites_segments():
    a = _lnav(4, [_sf(1, 5045, SF1), _sf(2, 5046, SF2)])
    b = _lnav(4, [_sf(3, 5047, SF3)])
    assert ephemerides_by_prn([a, b])[4].iode == SF2["iode"]
    assert 4 not in ephemerides_by_prn([a])
    assert 4 not in ephemerides_by_prn([b])


def test_ephemerides_by_prn_keeps_the_last_complete_set():
    a = _lnav(4, [_sf(1, 5045, SF1), _sf(2, 5046, SF2), _sf(3, 5047, SF3)])
    other = SF2["iode"] + 1
    b = _lnav(4, [_sf(2, 5056, dict(SF2, iode=other)), _sf(3, 5057, dict(SF3, iode=other))])
    assert ephemerides_by_prn([a, b])[4].iode == SF2["iode"]  # b's set is not complete yet
    c = _lnav(4, [_sf(1, 5065, dict(SF1, iodc=other))])
    assert ephemerides_by_prn([a, b, c])[4].iode == other
    assert ephemerides_by_prn([a])[4].iode == SF2["iode"]


def test_ephemerides_by_prn_keeps_prns_apart():
    a = _lnav(4, [_sf(1, 5045, SF1), _sf(2, 5046, SF2)])
    b = _lnav(7, [_sf(3, 5047, SF3, prn=7)])
    assert ephemerides_by_prn([a, b]) == {}


def test_tow1_by_prn_is_the_subframe_1_of_the_selected_ephemeris():
    a = _lnav(4, [_sf(1, 5045, SF1), _sf(2, 5046, SF2), _sf(3, 5047, SF3)])
    b = _lnav(4, [_sf(1, 5065, dict(SF1, iodc=SF2["iode"] + 1))])  # a new set, incomplete
    c = _lnav(7, [_sf(2, 5046, SF2, prn=7)])
    assert tow1_by_prn([a, b, c]) == {4: 5045 * 6 - 6}
    assert tow1_by_prn([a, a]) == {4: 5045 * 6 - 6} and tow1_by_prn([b]) == {}


def test_ionosphere_of_takes_the_first_present_in_list_order():
    iono = Ionosphere((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))
    other = Ionosphere((1.5, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))
    assert ionosphere_of([_lnav(4, []), LnavResult(7, None, [], None, iono)]) is iono
    # two segments with a set: the first in list order wins, whatever their times
    assert ionosphere_of([LnavResult(7, None, [], None, iono), LnavResult(4, None, [], None, other)]) is iono
    assert ionosphere_of([LnavResult(4, None, [], None, other), LnavResult(7, None, [], None, iono)]) is other
    assert ionosphere_of([_lnav(4, [])]) is None
