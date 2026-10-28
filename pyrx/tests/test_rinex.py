from datetime import datetime
from math import isnan

import pytest
from conftest import capture_file
from test_lnav import SF1, SF2, SF3

from pyrx.lnav import Ionosphere, assemble_ephemeris, decode_subframe, encode_subframe
from pyrx.observables import Epoch, Observation
from pyrx.orbits import sat_id
from pyrx.rinex import (
    URA_BOUNDS,
    gps_datetime,
    read_rinex_header,
    read_rinex_nav,
    read_rinex_obs,
    read_rinex_observations,
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
    assert (o.prn, o.pseudorange) == (4, 22235454.255)
    assert isnan(o.carrier_phase) and isnan(o.doppler) and isnan(o.cn0)  # 0.000 is "missing"


ZERO_OBS = """\
     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
  2021    12    26     8    24   20.0000000     GPS         TIME OF FIRST OBS
                                                            END OF HEADER
> 2021 12 26 08 24 20.0000000  0  3
G04         0.000  -116848778.204           0.000          43.300
G16  20653742.563           0.000        -698.280           0.000
G25         0.000           0.000           0.000           0.000
"""


def test_readers_take_numeric_zero_as_missing(tmp_path):
    (tmp_path / "z.obs").write_text(ZERO_OBS)
    [epoch] = read_rinex_obs(tmp_path / "z.obs")  # a zero pseudorange has no single-point use
    assert [o.prn for o in epoch.obs] == [16]
    o = epoch.obs[0]
    assert o.pseudorange == 20653742.563 and o.doppler == -698.28
    assert isnan(o.carrier_phase) and isnan(o.cn0)
    [obs] = read_rinex_observations(tmp_path / "z.obs", ("G1C",))
    assert obs.sats == ("G04", "G16")  # G25 has neither code nor phase
    assert isnan(obs.pseudorange[0, 0]) and obs.carrier_phase[0, 0] == -116848778.204
    assert isnan(obs.doppler[0, 0]) and obs.cn0[0, 0] == 43.3
    assert obs.pseudorange[1, 0] == 20653742.563 and isnan(obs.carrier_phase[1, 0])


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


GAL_NAV = """\
     3.05           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE
                                                            END OF HEADER
E01 2021 12 26 00 10 00-5.458160885610e-04-8.071765478235e-12 0.000000000000e+00
     1.000000000000e+00 3.565625000000e+01 3.232277494475e-09 1.742238364867e+00
     1.659616827965e-06 2.746939426288e-04 5.688518285751e-06 5.440608743668e+03
     6.000000000000e+02 2.235174179077e-08 3.002451169377e+00 6.519258022308e-08
     9.756833324185e-01 2.242500000000e+02-2.166523904760e-01-5.679879446920e-09
    -1.050043738536e-10 5.170000000000e+02 2.190000000000e+03 0.000000000000e+00
     3.120000000000e+00 0.000000000000e+00 9.313225746155e-10 6.984919309616e-10
     1.775000000000e+03
E01 2021 12 26 00 10 00-5.458170000000e-04-8.071765478235e-12 0.000000000000e+00
     1.000000000000e+00 3.565625000000e+01 3.232277494475e-09 1.742238364867e+00
     1.659616827965e-06 2.746939426288e-04 5.688518285751e-06 5.440608743668e+03
     6.000000000000e+02 2.235174179077e-08 3.002451169377e+00 6.519258022308e-08
     9.756833324185e-01 2.242500000000e+02-2.166523904760e-01-5.679879446920e-09
    -1.050043738536e-10 2.580000000000e+02 2.190000000000e+03 0.000000000000e+00
     3.120000000000e+00 0.000000000000e+00 9.313225746155e-10 0.000000000000e+00
     1.775000000000e+03
G04 2021 12 26 00 00 00-1.976881176233e-04 1.477928890381e-12 0.000000000000e+00
     1.440000000000e+02-5.093750000000e+01 4.405897809043e-09 1.002758171737e+00
    -2.583488821983e-06 1.603024313226e-03 9.605661034584e-06 5.153669235229e+03
     0.000000000000e+00-1.117587089539e-08 1.083721621721e+00 6.519258022308e-08
     9.607089453838e-01 1.943750000000e+02-3.125643079512e+00-7.928187383685e-09
     9.464679956196e-11 1.000000000000e+00 2.190000000000e+03 0.000000000000e+00
     2.000000000000e+00 0.000000000000e+00-4.190951585770e-09 9.120000000000e+02
    -7.182000000000e+03 4.000000000000e+00
"""


def test_read_rinex_nav_galileo_record(tmp_path):
    (tmp_path / "m.nav").write_text(GAL_NAV)
    ephs, iono = read_rinex_nav(tmp_path / "m.nav")
    assert iono is None and sorted(sat_id(e) for e in ephs) == ["E01", "G04"]
    e = next(e for e in ephs if e.system == "E")
    assert (e.prn, e.wn, e.toe, e.iode, e.iodc, e.health) == (1, 2190 % 1024, 600.0, 1, 1, 0)
    assert e.ura == ura_index(3.12) == 1
    assert e.tgd == 9.313225746155e-10 and e.tgd2 == 6.984919309616e-10
    assert e.toc == 600.0 and e.sqrt_a == pytest.approx(5440.608743668)


def test_read_rinex_nav_skips_fnav_records(tmp_path):
    (tmp_path / "m.nav").write_text(GAL_NAV)
    ephs, _ = read_rinex_nav(tmp_path / "m.nav")
    assert len([e for e in ephs if e.system == "E"]) == 1  # the 258 (F/NAV) record is dropped


def test_nav_writer_dates_a_next_week_data_set_in_its_own_week(tmp_path):
    # an ephemeris broadcast in the last two hours of a week has toe and toc at the start
    # of the next week: the record's epoch and week field belong to that week, with the
    # transmission time left in the broadcast week (negative relative to it)
    sf1 = dict(SF1, wn=142, toc=0)
    sf2 = dict(SF2, toe=0)
    sfs = [
        decode_subframe(encode_subframe(i, 99600 + i, f), 0, 0, prn=4)
        for i, f in ((1, sf1), (2, sf2), (3, SF3))
    ]
    eph = assemble_ephemeris(sfs)
    write_rinex_nav(tmp_path / "w.nav", [eph], {4: sfs[0].tow_s})
    text = (tmp_path / "w.nav").read_text()
    assert "G04 2022 01 02 00 00 00" in text  # Sunday of week 2191, not week 2190
    rows = text.split("END OF HEADER\n")[1].splitlines()
    assert float(rows[5].split()[2].replace("D", "E")) == 2191.0
    assert (
        float(rows[7].split()[0].replace("D", "E")) == sfs[0].tow_s - 604800
    )  # in the record's week
    [rec], _ = read_rinex_nav(tmp_path / "w.nav")
    assert (rec.wn, rec.toe, rec.toc) == (143, 0.0, 0.0)


def test_obs_writer_carries_the_loss_of_lock_indicator(tmp_path):
    epochs = [
        Epoch(
            2190,
            30260.0,
            [
                Observation(4, 22235454.255, -116848778.204, 421.142, 43.3, lli=3),
                Observation(16, 20653742.563, 108536284.235, -698.28, 46.9),
            ],
        )
    ]
    write_rinex_obs(tmp_path / "l.obs", epochs)
    text = (tmp_path / "l.obs").read_text()
    assert "G04  22235454.255  -116848778.2043        421.142          43.300" in text
    assert "G16  20653742.563   108536284.235        -698.280          46.900" in text
    [back] = read_rinex_observations(tmp_path / "l.obs", ("G1C",))
    assert back.sats == ("G04", "G16") and back.lli[:, 0].tolist() == [3, 0]
