"""LNAV decode of the two 30 s recordings against the day's broadcast ephemeris."""

from math import pi

import pytest
from conftest import capture_file

from pyrx.acquisition import acquire_l1ca
from pyrx.navdecode import decode_lnav
from pyrx.rinex import read_rinex_nav
from pyrx.samples import read_samples
from pyrx.tracking import track_channels

# File start times are GPS seconds of week decoded from the signal; the file names
# (UTC) agree to 0.5 s for the first recording and are 4 s early for the second.
# Each 30 s window holds four whole subframes, so the subframe sequence is fixed by
# the start time; the second window starts inside subframe 1 and so holds no
# complete ephemeris.
CASES = [
    (
        "L1_20211226_082401_24MHz_I.bin",
        24e6,
        6.0e6,
        [4, 16, 18, 26, 27, 29, 31],
        "brdc/BRDC00WRD_R_20213600000_01D_GN.rnx",
        None,
        2190,
        30259.5,
        [5, 1, 2, 3],
    ),
    (
        "L1_20220114_004440_8MHz_I.bin",
        8e6,
        2.0e6,
        [10, 12, 15, 23, 24, 32],
        "brdc/BRDC00WRD_R_20220140000_01D_GN.rnx",
        "brdc/BRDC00IGS_R_20220140000_01D_MN.rnx",
        2192,
        434702.1,
        [2, 3, 4, 5],
    ),
]
# (field, LSB in the decoded units) per subframe, from IS-GPS-200 Tables 20-I and 20-III.
# A decoded value within half an LSB of the reference is the same transmitted integer;
# the relative check covers the many records the file stores to 13 digits.
SC = 2**-31 * pi
SF_FIELDS = {
    1: [
        ("tgd", 2**-31),
        ("toc", 2**4),
        ("af2", 2**-55),
        ("af1", 2**-43),
        ("af0", 2**-31),
        ("iodc", 1),
        ("health", 1),
    ],
    2: [
        ("iode", 1),
        ("crs", 2**-5),
        ("delta_n", 2**-43 * pi),
        ("m0", SC),
        ("cuc", 2**-29),
        ("e", 2**-33),
        ("cus", 2**-29),
        ("sqrt_a", 2**-19),
        ("toe", 2**4),
    ],
    3: [
        ("cic", 2**-29),
        ("omega0", SC),
        ("cis", 2**-29),
        ("i0", SC),
        ("crc", 2**-5),
        ("omega", SC),
        ("omega_dot", 2**-43 * pi),
        ("iode", 1),
        ("idot", 2**-43 * pi),
    ],
}


@pytest.mark.parametrize(
    "name,fs,f_if,prns,nav_name,iono_name,week,tow0,sf_ids", CASES, ids=[c[0] for c in CASES]
)
def test_lnav_matches_broadcast(
    captures, name, fs, f_if, prns, nav_name, iono_name, week, tow0, sf_ids
):
    ephs, _ = read_rinex_nav(capture_file(captures, nav_name))
    iono = read_rinex_nav(capture_file(captures, iono_name))[1] if iono_name else None
    path = capture_file(captures, name)
    head = read_samples(path, fs, iq=False, duration=0.010)
    seeds = []
    for prn in prns:
        a = acquire_l1ca(head, fs, f_if, prn)
        seeds.append((prn, a.code_offset, a.doppler, a.cn0))
    results = [decode_lnav(r) for r in track_channels(path, fs, False, f_if, seeds)]
    complete = 0
    starts = []  # file start in GPS time implied by each subframe's time tag
    for res in results:
        prn = res.prn
        assert [sf.sf_id for sf in res.subframes] == sf_ids, prn
        for sf in res.subframes:
            t_epoch = sf.epoch * 1e-3
            starts.append(sf.tow_s - t_epoch)
            assert abs(sf.tow_s - t_epoch - tow0) <= 0.1, (prn, sf.sf_id, sf.tow_s, t_epoch)
            if sf.sf_id not in SF_FIELDS:
                continue
            rec = _find(ephs, prn, sf)
            assert rec is not None, (prn, sf.sf_id, sf.fields)
            for f, lsb in SF_FIELDS[sf.sf_id]:
                assert sf.fields[f] == pytest.approx(getattr(rec, f), rel=1e-11, abs=0.5 * lsb), (
                    prn,
                    sf.sf_id,
                    f,
                )
            if sf.sf_id == 1:
                assert sf.fields["wn"] == rec.wn, prn
        if res.ephemeris is not None:
            complete += 1
        if res.ionosphere is not None and iono is not None:
            # the header stores five significant digits
            assert res.ionosphere.alpha == pytest.approx(iono.alpha, rel=1e-4)
            assert res.ionosphere.beta == pytest.approx(iono.beta, rel=1e-4)
    # satellites differ in range by up to 20 ms of light time
    assert max(starts) - min(starts) <= 0.03
    assert complete == (len(prns) if 1 in sf_ids else 0)


def _find(ephs, prn, sf):
    """The record whose IODE (sf2/sf3) or IODC (sf1) matches the subframe."""
    for rec in ephs:
        if rec.prn != prn:
            continue
        if sf.sf_id == 1 and rec.iodc == sf.fields["iodc"]:
            return rec
        if sf.sf_id in (2, 3) and rec.iode == sf.fields["iode"]:
            return rec
    return None
