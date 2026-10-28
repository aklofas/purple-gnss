from datetime import datetime
from math import isnan

import pytest
from conftest import capture_file
from test_lnav import SF1, SF2, SF3

from pyrx.lnav import Ionosphere, assemble_ephemeris, decode_subframe, encode_subframe
from pyrx.observables import Epoch, Observation
from pyrx.rinex import (
    URA_BOUNDS,
    gps_datetime,
    read_rinex_header,
    read_rinex_nav,
    read_rinex_obs,
    ura_index,
    ura_metres,
    write_rinex_nav,
    write_rinex_obs,
)


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
    [rec], iono_read = read_rinex_nav(tmp_path / "t.nav")
    for name in (
        "af0", "af1", "af2", "crs", "delta_n", "m0", "cuc", "e", "cus", "sqrt_a", "toe",
        "cic", "omega0", "cis", "i0", "crc", "omega", "omega_dot", "idot", "tgd",
    ):  # fmt: skip
        assert getattr(rec, name) == pytest.approx(getattr(eph, name), rel=1e-11, abs=1e-20), name
    assert (rec.prn, rec.wn, rec.toc, rec.ura, rec.health, rec.iode, rec.iodc, rec.fit) == (
        4, 142, 28800, 0, 0, 149, 917, 0
    )  # fmt: skip
    assert iono_read.alpha == pytest.approx(iono.alpha, rel=1e-4)
    assert iono_read.beta == pytest.approx(iono.beta, rel=1e-4)


def test_ura_index_round_trips_the_nominal_values():
    assert [ura_index(ura_metres(i)) for i in range(15)] == list(range(15))
    assert ura_index(2.4) == 0 and ura_index(2.41) == 1
    assert ura_index(6144.0) == 14 and ura_index(7000.0) == 15  # 6144 m is the top interval's bound
    assert len(URA_BOUNDS) == 15 and URA_BOUNDS[0] == 2.4 and URA_BOUNDS[14] == 6144.0


def test_read_rinex_obs_round_trip(tmp_path):
    epochs = [
        Epoch(2190, 30260.0, [
            Observation(4, 22235454.255, -116848778.204, 421.142, 43.3),
            Observation(16, 20653742.563, 108536284.235, -698.28, 46.9),
        ]),
        Epoch(2190, 30261.0, [Observation(4, 22235373.830, -116848357.260, 420.9, 43.2)]),
    ]  # fmt: skip
    write_rinex_obs(tmp_path / "t.obs", epochs)
    assert read_rinex_obs(tmp_path / "t.obs") == epochs
    assert (
        read_rinex_obs(
            tmp_path / "t.obs",
            (datetime(2021, 12, 26, 8, 24, 20, 500000), datetime(2021, 12, 26, 9)),
        )
        == epochs[1:]
    )


def test_read_rinex_obs_without_doppler(tmp_path):
    lines = [
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
        "G    2 C1C L1C                                              SYS / # / OBS TYPES",
        "  2021    12    26     8    24   20.0000000     GPS         TIME OF FIRST OBS",
        "                                                            END OF HEADER",
        "> 2021 12 26 08 24 20.0000000  0  2",
        "G04  22235454.255  -116848778.204",
        "G16  20653742.563",
    ]
    (tmp_path / "s.obs").write_text("\n".join(lines) + "\n")
    [epoch] = read_rinex_obs(tmp_path / "s.obs")
    assert (epoch.week, epoch.tow) == (2190, 30260.0)
    assert [o.prn for o in epoch.obs] == [4, 16]
    assert epoch.obs[0].pseudorange == 22235454.255 and isnan(epoch.obs[0].doppler)
    assert isnan(epoch.obs[1].carrier_phase) and isnan(epoch.obs[1].cn0)


def test_write_rinex_obs_leaves_missing_values_blank(tmp_path):
    nan = float("nan")
    write_rinex_obs(
        tmp_path / "m.obs", [Epoch(2190, 30260.0, [Observation(4, 22235454.255, 0.0, nan, nan)])]
    )
    text = (tmp_path / "m.obs").read_text()
    assert "nan" not in text
    [epoch] = read_rinex_obs(tmp_path / "m.obs")
    o = epoch.obs[0]
    assert (o.prn, o.pseudorange, o.carrier_phase) == (4, 22235454.255, 0.0)
    assert isnan(o.doppler) and isnan(o.cn0)


def test_read_rinex_nav_without_ionosphere(tmp_path):
    sfs = [
        decode_subframe(encode_subframe(i, 5044 + i, f), 0, 0, prn=4)
        for i, f in ((1, SF1), (2, SF2), (3, SF3))
    ]
    eph = assemble_ephemeris(sfs)
    write_rinex_nav(tmp_path / "n.nav", [eph], {4: sfs[0].tow_s})
    ephs, iono = read_rinex_nav(tmp_path / "n.nav")
    assert iono is None and len(ephs) == 1


def test_read_rinex_header_of_a_hatanaka_file(captures):
    h = read_rinex_header(capture_file(captures, "igs/TLSG00FRA_R_20213600000_01D_30S_MO.crx.gz"))
    assert h.marker == "TLSG" and h.delta_enu[2] == 0.441
