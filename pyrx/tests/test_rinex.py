from datetime import datetime

import pytest
from rinex_nav import load_gps_iono, load_gps_nav
from test_lnav import SF1, SF2, SF3

from pyrx.lnav import Ionosphere, assemble_ephemeris, decode_subframe, encode_subframe
from pyrx.observables import Epoch, Observation
from pyrx.rinex import gps_datetime, ura_metres, write_rinex_nav, write_rinex_obs


def test_gps_datetime():
    assert gps_datetime(2190, 30259.5) == datetime(2021, 12, 26, 8, 24, 19, 500000)


def test_ura_metres():
    assert [ura_metres(i) for i in (0, 1, 6, 7, 14, 15)] == [
        2.0,
        2.8284271247461903,
        16.0,
        32.0,
        4096.0,
        6144.0,
    ]


@pytest.mark.filterwarnings("ignore::FutureWarning")  # xarray, raised inside georinex
def test_obs_round_trip(tmp_path):
    georinex = pytest.importorskip("georinex")
    epochs = [
        Epoch(
            2190,
            30260.0,
            [
                Observation(4, 22235454.255, -116848778.204, 421.142, 43.3),
                Observation(16, 20653742.563, 108536284.235, -698.28, 46.9),
            ],
        ),
        Epoch(2190, 30261.0, [Observation(4, 22235373.830, -116848357.260, 420.9, 43.2)]),
        Epoch(2190, 30262.0, []),
    ]
    write_rinex_obs(tmp_path / "t.obs", epochs)
    text = (tmp_path / "t.obs").read_text()
    lines = text.splitlines()
    assert (
        lines[0].rstrip()
        == "     3.05           OBSERVATION DATA    G: GPS              RINEX VERSION / TYPE"
    )
    assert "G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES" in text
    assert "> 2021 12 26 08 24 20.0000000  0  2" in text
    assert "G04  22235454.255  -116848778.204         421.142          43.300" in text
    ds = georinex.load(tmp_path / "t.obs")
    assert ds["C1C"].sel(sv="G16").dropna("time").item() == pytest.approx(20653742.563, abs=5e-4)
    assert ds["L1C"].sel(sv="G04").isel(time=1).item() == pytest.approx(-116848357.260, abs=5e-4)
    assert len(ds.time) == 2
    with pytest.raises(ValueError, match="week"):
        write_rinex_obs(tmp_path / "u.obs", [Epoch(None, 1.0, epochs[0].obs)])


def test_nav_round_trip(tmp_path):
    pytest.importorskip("georinex")
    sfs = [
        decode_subframe(encode_subframe(i, 5044 + i, f), 0, 0, prn=4)
        for i, f in ((1, SF1), (2, SF2), (3, SF3))
    ]
    eph = assemble_ephemeris(sfs)
    iono = Ionosphere(
        (12 * 2**-30, -(2**-27), -(2**-24), 2 * 2**-24),
        (114688.0, -163840.0, -196608.0, 917504.0),
    )
    write_rinex_nav(tmp_path / "t.nav", [eph], {4: sfs[0].tow_s}, iono)
    text = (tmp_path / "t.nav").read_text()
    assert text.startswith(
        "     3.05           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE"
    )
    assert "GPSA   1.1176D-08 -7.4506D-09 -5.9605D-08  1.1921D-07       IONOSPHERIC CORR" in text
    nav = load_gps_nav(tmp_path / "t.nav")
    rec = nav[(4, 149, 917)]
    for name in (
        "af0", "af1", "af2", "crs", "delta_n", "m0", "cuc", "e", "cus", "sqrt_a", "toe",
        "cic", "omega0", "cis", "i0", "crc", "omega", "omega_dot", "idot", "tgd",
    ):  # fmt: skip
        assert rec[name] == pytest.approx(getattr(eph, name), rel=1e-11, abs=1e-20), name
    assert rec["week"] == 2190 and rec["toc"] == 28800 and rec["ura"] == 2.0
    assert rec["health"] == 0
    assert load_gps_iono(tmp_path / "t.nav") == (
        pytest.approx(iono.alpha, rel=1e-4),
        pytest.approx(iono.beta, rel=1e-4),
    )
