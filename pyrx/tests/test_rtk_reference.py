"""Our RTK fix against RTKLIB's and the EUREF coordinates on two IGS station pairs."""

import dataclasses
from datetime import datetime

import numpy as np
import pytest
from conftest import capture_file
from rtklib import rnx2rtkp_rtk, rnx2rtkp_ttff

from pyrx.gpstime import C_LIGHT
from pyrx.models import Station, geometric_range
from pyrx.observations import to_epoch
from pyrx.orbits import satellite_state, select_ephemeris, usable
from pyrx.pvt import solve_epoch
from pyrx.rinex import read_rinex_header, read_rinex_nav, read_rinex_observations
from pyrx.rtk import RtkOptions, solve_rtk
from pyrx.signals import wavelength

NAV = "igs/BRDC00WRD_R_20213600000_01D_MN.rnx.gz"
# base file, rover file, base marker, rover marker (EUREF combined SINEX, GPS week 2190)
PAIRS = {
    "MATE": (
        "igs/MATE00ITA_R_20213600000_01D_30S_MO.crx.gz",
        "igs/MAT100ITA_R_20213600000_01D_30S_MO.crx.gz",
        (4641949.24236861, 1393045.74884337, 4133287.70969356),
        (4641951.13677723, 1393053.98279794, 4133281.10948056),
    ),
    "TLSE": (
        "igs/TLSE00FRA_R_20213600000_01D_30S_MO.crx.gz",
        "igs/TLSG00FRA_R_20213600000_01D_30S_MO.crx.gz",
        (4627851.62640963, 119640.344437304, 4372993.74586133),
        (4628684.66467592, 119997.313092678, 4372110.44549263),
    ),
}
WINDOW = (datetime(2021, 12, 26, 0, 0), datetime(2021, 12, 26, 6, 0))
BASE_WINDOW = (datetime(2021, 12, 25, 23, 59), WINDOW[1])  # a base epoch for the first rover epoch
TS, TE = "2021/12/26 00:00:00", "2021/12/26 06:00:00"
CASES = [
    ("MATE", "kinematic", ("G1C", "G2W"), "l1+2"),
    ("MATE", "kinematic", ("G1C",), "l1"),
    ("MATE", "static", ("G1C", "G2W"), "l1+2"),
    ("TLSE", "kinematic", ("G1C", "G2W"), "l1+2"),
    ("TLSE", "kinematic", ("G1C",), "l1"),
    ("TLSE", "static", ("G1C", "G2W"), "l1+2"),
]
_CACHE: dict = {}


def _load(captures, pair, signals):
    """Rover and base observations of the window, ephemerides and headers, once per session."""
    key = (pair, signals)
    if key not in _CACHE:
        base_name, rover_name = PAIRS[pair][:2]
        rover_path, base_path = (
            capture_file(captures, rover_name),
            capture_file(captures, base_name),
        )
        if "ephs" not in _CACHE:
            _CACHE["ephs"] = read_rinex_nav(capture_file(captures, NAV))[0]
        _CACHE[key] = (
            read_rinex_observations(rover_path, signals, WINDOW),
            read_rinex_observations(base_path, signals, BASE_WINDOW),
            _CACHE["ephs"],
            read_rinex_header(base_path),
            read_rinex_header(rover_path),
        )
    return _CACHE[key]


def _key(f):
    return (f.week, round(f.tow, 1))


@pytest.mark.parametrize(
    "pair,mode,signals,freq", CASES, ids=[f"{c[0]}-{c[1]}-{c[3]}" for c in CASES]
)
def test_rtk_matches_rtklib_and_the_marker(captures, tmp_path, pair, mode, signals, freq):
    base_name, rover_name, base_xyz, rover_xyz = PAIRS[pair]
    rover, base, ephs, hb, hr = _load(captures, pair, signals)
    station = Station(np.array(base_xyz), hb.delta_enu)
    fixes = solve_rtk(rover, base, ephs, station, hr.delta_enu, RtkOptions(mode=mode))
    ref = rnx2rtkp_rtk(
        capture_file(captures, rover_name), capture_file(captures, base_name), capture_file(captures, NAV),
        tmp_path / "rtk", base_xyz, hb.delta_enu[2], hr.delta_enu[2], mode, freq, 1, ts=TS, te=TE,
    )  # fmt: skip
    fixed = [f for f in fixes if f.status == "fix"]
    ref_fixed = {k: v for k, v in ref.items() if v["q"] == 1}
    assert len(rover) == 721 and len(fixes) == len(rover)
    # the same epochs fix in both engines
    assert {_key(f) for f in fixed} == set(ref_fixed), (len(fixed), len(ref_fixed))
    err = np.array([np.linalg.norm(f.position - rover_xyz) for f in fixed])
    # single-epoch kinematic fixes scatter up to 2.5 cm from the marker on both pairs, in
    # RTKLIB as well (its worst epochs 20-24 mm); the carried static position within 1 cm
    limits = (0.005, 0.02) if mode == "static" else (0.01, 0.03)
    assert err.mean() <= limits[0] and np.percentile(err, 95) <= 0.02, (
        err.mean(),
        np.percentile(err, 95),
    )
    assert err.max() <= limits[1], err.max()
    d = np.array([np.linalg.norm(f.position - ref_fixed[_key(f)]["xyz"]) for f in fixed])
    # same models, same integers: 2.2 mm at the 95th percentile and 2.6 mm worst measured
    assert np.percentile(d, 95) <= 0.005 and d.max() <= 0.01, (np.percentile(d, 95), d.max())


@pytest.mark.parametrize(
    "pair,signals,freq",
    [("MATE", ("G1C", "G2W"), "l1+2"), ("MATE", ("G1C",), "l1"), ("TLSE", ("G1C", "G2W"), "l1+2"), ("TLSE", ("G1C",), "l1")],
    ids=["MATE-l1+2", "MATE-l1", "TLSE-l1+2", "TLSE-l1"],
)  # fmt: skip
def test_time_to_first_fix(captures, tmp_path, pair, signals, freq):
    base_name, rover_name, base_xyz, _ = PAIRS[pair]
    rover, base, ephs, hb, hr = _load(captures, pair, signals)
    station = Station(np.array(base_xyz), hb.delta_enu)
    # restarts every 30 min on windows of 60 epochs (the day starts the GPS week, so the
    # time of week of a restart is its offset into the day)
    starts = [1800.0 * k for k in range(12)]
    ours = []
    for t0 in starts:
        rov = [o for o in rover if t0 <= o.tow < t0 + 1800.0]
        bas = [o for o in base if t0 - 30.0 <= o.tow < t0 + 1800.0]
        fixes = solve_rtk(rov, bas, ephs, station, hr.delta_enu)
        ours.append(next((f.tow for f in fixes if f.status == "fix"), None))
    theirs = rnx2rtkp_ttff(
        capture_file(captures, rover_name), capture_file(captures, base_name), capture_file(captures, NAV),
        tmp_path / "ttff", base_xyz, hb.delta_enu[2], hr.delta_enu[2], "kinematic", freq, 1,
        [f"2021/12/26 {int(t0) // 3600:02d}:{int(t0) % 3600 // 60:02d}:00" for t0 in starts], 1770,
    )  # fmt: skip
    # time to first fix in elapsed epochs from the restart, the first epoch counting as 1;
    # every restart must fix within its window on both sides
    assert None not in ours and None not in theirs, (ours, theirs)
    ours = [(tow - t0) / 30.0 + 1 for tow, t0 in zip(ours, starts)]
    theirs = [(tow - t0) / 30.0 + 1 for tow, t0 in zip(theirs, starts)]
    # the design allows twice RTKLIB's; measured, the two agree within 0.8 epoch in the mean
    assert np.mean(ours) <= min(2.0 * np.mean(theirs), np.mean(theirs) + 1.0), (ours, theirs)


def test_rtk_multi_gnss_matches_rtklib(captures, tmp_path):
    # RTKLIB with pos1-frequency=l1+2+3 and gps+gal uses GPS L1/L2/L5 and Galileo E1/E5b/E5a
    signals = ("G1C", "G2W", "G5Q", "E1C", "E7Q", "E5Q")
    base_name, rover_name, base_xyz, rover_xyz = PAIRS["MATE"]
    rover, base, ephs, hb, hr = _load(captures, "MATE", signals)
    assert any(e.system == "E" for e in ephs)
    fixes = solve_rtk(rover, base, ephs, Station(np.array(base_xyz), hb.delta_enu), hr.delta_enu)
    ref = rnx2rtkp_rtk(
        capture_file(captures, rover_name), capture_file(captures, base_name), capture_file(captures, NAV),
        tmp_path / "rtk", base_xyz, hb.delta_enu[2], hr.delta_enu[2], "kinematic", "l1+2+3", 9, ts=TS, te=TE,
    )  # fmt: skip
    fixed = [f for f in fixes if f.status == "fix"]
    ref_fixed = {k: v for k, v in ref.items() if v["q"] == 1}
    assert np.mean([f.ns for f in fixed]) >= 10  # GPS and Galileo both in the solution
    assert len(fixed) >= len(ref_fixed) - 0.01 * len(rover), (len(fixed), len(ref_fixed))
    err = np.array([np.linalg.norm(f.position - rover_xyz) for f in fixed])
    assert err.mean() <= 0.01 and np.percentile(err, 95) <= 0.02 and err.max() <= 0.02
    common = [(f, ref_fixed[_key(f)]) for f in fixed if _key(f) in ref_fixed]
    assert len(common) >= len(ref_fixed) - 0.01 * len(rover)
    # against RTKLIB's fixed positions: 1.1 mm at the 95th percentile and 7.0 mm worst
    # measured (its Galileo models and reference choice differ from ours in detail)
    d = np.array([np.linalg.norm(f.position - r["xyz"]) for f, r in common])
    assert np.percentile(d, 95) <= 0.01 and d.max() <= 0.02, (np.percentile(d, 95), d.max())


def test_rtk_l1_l5_e1_e5a_fixes_the_marker(captures):
    # the hardware's signal set; RTKLIB cannot run exactly this combination
    _, _, base_xyz, rover_xyz = PAIRS["MATE"]
    rover, base, ephs, hb, hr = _load(captures, "MATE", ("G1C", "G5Q", "E1C", "E5Q"))
    fixes = solve_rtk(rover, base, ephs, Station(np.array(base_xyz), hb.delta_enu), hr.delta_enu)
    fixed = [f for f in fixes if f.status == "fix"]
    assert len(fixed) >= len(rover) - 0.01 * len(rover), len(fixed)  # 721 of 721 measured
    err = np.array([np.linalg.norm(f.position - rover_xyz) for f in fixed])
    assert err.mean() <= 0.01 and np.percentile(err, 95) <= 0.02 and err.max() <= 0.02


# The project's own receiver on a car roof against a public base station 15 km away, 1 s,
# GPS L1 and L5: the observation and navigation files pyrx-rinex formed from a capture
# driven through fades, and the base's half hour as RINEX 3 with its position in the header
DRIVING = "sky/20261103_031515_carroof-driving"
CAR_BASE = "cors/base_20261103_0300_0330_gps.rnx"


def _car(captures, capture, signals):
    """Rover and base observations of a car-roof capture, ephemerides and ionosphere, the
    base station, the rover's antenna offset and the three file paths, once per session."""
    key = (capture, signals)
    if key not in _CACHE:
        files = (
            capture_file(captures, f"{capture}/pyrx_l1l5.obs"),
            capture_file(captures, CAR_BASE),
            capture_file(captures, f"{capture}/pyrx_l1l5.nav"),
        )
        hb = read_rinex_header(files[1])
        _CACHE[key] = (
            read_rinex_observations(files[0], signals),
            read_rinex_observations(files[1], signals),
            *read_rinex_nav(files[2]),
            Station(hb.position, hb.delta_enu),
            read_rinex_header(files[0]).delta_enu,
            files,
        )
    return _CACHE[key]


def _same_as_rtklib(fixes, ref):
    """Every epoch solved by both, with the same status, at the same position."""
    ours = {_key(f): f for f in fixes}
    assert set(ours) == set(ref) and len(ours) > 250, (len(ours), len(ref))
    assert all((f.status == "fix") == (ref[k]["q"] == 1) for k, f in ours.items())
    d = max(np.abs(f.position - ref[k]["xyz"]).max() for k, f in ours.items())
    assert d <= 0.001, d  # measured 0.1 mm, the rounding of the solution files
    return sum(f.status == "fix" for f in fixes)


def test_backward_pass_matches_rtklib_on_a_drive(captures, tmp_path):
    # L1 alone, where the two engines have the same slip rules: 250 of the file's 2238 L1
    # phases carry the slip flag and the half-cycle flag changes 56 times
    rover, base, ephs, iono, station, delta, files = _car(captures, DRIVING, ("G1C",))
    fixes = solve_rtk(rover, base, ephs, station, delta, RtkOptions(backward=True), iono)
    ref = rnx2rtkp_rtk(
        *files, tmp_path / "rtk", station.marker, 0.0, 0.0, "kinematic", "l1", soltype="backward"
    )
    assert _same_as_rtklib(fixes, ref) >= 40


@pytest.mark.parametrize("soltype", ["forward", "backward"])
def test_lock_count_matches_rtklib_on_a_drive(captures, tmp_path, soltype):
    # the file restarts an L1 ambiguity 270 times; with RTKLIB's pos2-arlockcnt the same
    # epochs fix, at the same positions, in both directions
    rover, base, ephs, iono, station, delta, files = _car(captures, DRIVING, ("G1C",))
    options = RtkOptions(min_lock=5, backward=soltype == "backward")
    fixes = solve_rtk(rover, base, ephs, station, delta, options, iono)
    ref = rnx2rtkp_rtk(
        *files, tmp_path / "rtk", station.marker, 0.0, 0.0, "kinematic", "l1",
        soltype=soltype, arlockcnt=5,
    )  # fmt: skip
    assert _same_as_rtklib(fixes, ref) >= 20
    # and the count is at work: without it other epochs fix
    plain = solve_rtk(
        rover, base, ephs, station, delta, RtkOptions(backward=options.backward), iono
    )
    assert {_key(f) for f in fixes if f.status == "fix"} != {
        _key(f) for f in plain if f.status == "fix"
    }


@pytest.mark.parametrize("signals,freq", [(("G1C", "G2W"), "l1+2"), (("G1C",), "l1")])
def test_lock_count_matches_rtklib_on_a_station_pair(captures, tmp_path, signals, freq):
    # six hours with satellites rising and setting: an ambiguity that begins after the first
    # epochs waits in RTKLIB like one begun again, and a few do each hour
    base_name, rover_name, base_xyz, _ = PAIRS["MATE"]
    rover, base, ephs, hb, hr = _load(captures, "MATE", signals)
    station = Station(np.array(base_xyz), hb.delta_enu)
    fixes = solve_rtk(rover, base, ephs, station, hr.delta_enu, RtkOptions(min_lock=5))
    ref = rnx2rtkp_rtk(
        capture_file(captures, rover_name), capture_file(captures, base_name), capture_file(captures, NAV),
        tmp_path / "rtk", base_xyz, hb.delta_enu[2], hr.delta_enu[2], "kinematic", freq, 1, ts=TS, te=TE,
        arlockcnt=5,
    )  # fmt: skip
    assert _same_as_rtklib(fixes, ref) >= 700


STATIC = "sky/20261103_030551_carroof-static"  # the same car parked there ten minutes earlier
# what pyrx-rtk turns on
COMMAND = {"slips": "check", "min_lock": 5, "min_fix_sats": 5, "change_limit": 0.03}
NO_LIMIT = {**COMMAND, "change_limit": 0.0}
L1_L5 = ("G1C", "G5Q")


def _both_ways(data, **options):
    """The fixes forward and backward by epoch, and the epochs both fix: those within 5 cm
    horizontally and 10 cm vertically of each other, and those further apart."""
    rover, base, ephs, iono, station, delta, _ = data
    fwd, bwd = (
        {_key(f): f for f in solve_rtk(rover, base, ephs, station, delta, RtkOptions(backward=b, **options), iono)}
        for b in (False, True)
    )  # fmt: skip
    together, apart = [], []
    for k in sorted(set(fwd) & set(bwd)):
        if fwd[k].status == "fix" and bwd[k].status == "fix":
            d = fwd[k].position - bwd[k].position
            up = d @ fwd[k].position / np.linalg.norm(fwd[k].position)
            near = np.sqrt(max(d @ d - up * up, 0.0)) < 0.05 and abs(up) < 0.10
            (together if near else apart).append(k)
    return fwd, bwd, together, apart


def test_a_drive_fixes_more_where_both_directions_agree(captures):
    # There is no truth for this drive. What can be checked is that the filter run backward
    # arrives at the same position: a wrong set of integers rarely survives both ways.
    # Measured: RTKLIB's rules fix 157 epochs forward and 122 backward, 84 of them together
    # and one 5 m apart (four satellites in an obstruction); the command's settings 202 and
    # 178, 144 together and one apart (five satellites in the search); a floor of six
    # satellites instead of five 198 and 177, 144 together, none apart.
    data = _car(captures, DRIVING, L1_L5)
    assert len(data[0]) == 291
    fwd, bwd, together, apart = _both_ways(data, **COMMAND)
    _, _, stock_together, stock_apart = _both_ways(data)
    assert 75 <= len(stock_together) <= 95 and len(stock_apart) <= 2
    assert len(together) >= 130 and len(together) >= len(stock_together) + 40
    assert len(apart) <= 1
    fixed = [sum(f.status == "fix" for f in run.values()) for run in (fwd, bwd)]
    assert fixed[0] >= 185 and fixed[1] >= 165, fixed
    six = _both_ways(data, **{**COMMAND, "min_fix_sats": 6})
    assert len(six[2]) >= 130 and not six[3]
    # without the change limit the counts are the same, 202 and 178, 144 and one: nine
    # satellites, five of them on two frequencies, seldom leave it a phase to judge
    plain = _both_ways(data, **NO_LIMIT)
    assert len(plain[2]) >= 130 and len(plain[3]) <= 1


def test_the_parked_start_of_the_drive_fixes_where_the_parked_capture_did(captures):
    # two recordings of a car that did not move between them, solved apart: the static
    # filter on the first, the kinematic one on the first seconds of the second
    rover, base, ephs, iono, station, delta, _ = _car(captures, STATIC, L1_L5)
    options = RtkOptions(mode="static", **COMMAND)
    parked = [
        f for f in solve_rtk(rover, base, ephs, station, delta, options, iono) if f.status == "fix"
    ]
    plain = solve_rtk(rover, base, ephs, station, delta, RtkOptions(mode="static"), iono)
    plain = [f for f in plain if f.status == "fix"]
    # the check costs the parked capture nothing: as many fixed epochs, at the same place
    # (the medians: RTKLIB's rules fix the very first epoch 2 m off, on a ratio of 3.1)
    assert len(parked) >= len(plain) >= 130
    home = np.median([f.position for f in parked], axis=0)
    assert np.linalg.norm(home - np.median([f.position for f in plain], axis=0)) < 0.005
    assert max(np.linalg.norm(f.position - home) for f in parked) < 0.02
    assert max(np.linalg.norm(f.position - home) for f in plain) > 1.0
    rover, base, ephs, iono, station, delta, _ = _car(captures, DRIVING, L1_L5)
    start = [
        f
        for f in solve_rtk(rover, base, ephs, station, delta, RtkOptions(**COMMAND), iono)
        if f.status == "fix" and f.tow < rover[0].tow + 22.0  # the car pulls out at 22 s
    ]
    # measured: twelve fixed epochs, 3 to 7 mm from the parked capture's fix horizontally
    # and 1 to 17 mm below it
    assert len(start) >= 10
    for f in start:
        d = f.position - home
        up = d @ home / np.linalg.norm(home)
        assert np.sqrt(max(d @ d - up * up, 0.0)) < 0.01 and abs(up) < 0.025, (f.tow, d)


# The same captures with the Galileo E5a pilots beside GPS (pyrx-rinex --l5 --galileo), the
# base's Galileo beside its GPS, and the day's public broadcast file for the Galileo orbits
COLD_START = "sky/20261103_032110_carroof-coldstart"  # begun in motion, an obstruction at 40 s
E21_SLIP = 185076  # s of week of the first observation after E21's E5a pilot slipped a cycle
CAR_BASE_GALILEO = "cors/base_20261103_0300_0330.rnx"
PUBLIC_NAV = "brdc/BRDC00IGS_R_20263070000_01D_MN.rnx"
WITH_E5A = ("G1C", "G5Q", "E5Q")


def _galileo_ephemerides(captures):
    if "galileo" not in _CACHE:
        ephs, _ = read_rinex_nav(capture_file(captures, PUBLIC_NAV))
        _CACHE["galileo"] = [e for e in ephs if e.system == "E"]
    return _CACHE["galileo"]


def _car_with_galileo(captures, capture, signals=WITH_E5A, name="pyrx_gal.obs"):
    """As `_car`, from the capture's file with Galileo: the decoded GPS ephemerides with the
    public file's Galileo records added."""
    key = (capture, signals, name)
    if key not in _CACHE:
        rover_file = capture_file(captures, f"{capture}/{name}")
        base_file = capture_file(captures, CAR_BASE_GALILEO)
        ephs, iono = read_rinex_nav(capture_file(captures, f"{capture}/pyrx_l1l5.nav"))
        hb = read_rinex_header(base_file)
        _CACHE[key] = (
            read_rinex_observations(rover_file, signals),
            read_rinex_observations(base_file, signals),
            ephs + _galileo_ephemerides(captures),
            iono,
            Station(hb.position, hb.delta_enu),
            read_rinex_header(rover_file).delta_enu,
            None,
        )
    return _CACHE[key]


@pytest.mark.parametrize("capture", [STATIC, DRIVING, COLD_START])
def test_the_file_with_galileo_has_the_gps_rows_of_the_gps_file(captures, capture):
    # what the GPS-only tests above measure stands for the file with Galileo in it
    plain = read_rinex_observations(capture_file(captures, f"{capture}/pyrx_l1l5.obs"), L1_L5)
    mixed = read_rinex_observations(capture_file(captures, f"{capture}/pyrx_gal.obs"), L1_L5)
    assert len(plain) == len(mixed) > 280
    for a, b in zip(plain, mixed, strict=True):
        assert (a.week, a.tow, a.sats) == (b.week, b.tow, b.sats)
        for field in ("pseudorange", "carrier_phase", "doppler", "cn0", "lli"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("backward", [False, True])
def test_galileo_rows_without_a_base_for_them_change_nothing(captures, backward):
    # the rover's file with Galileo against the base's GPS-only file: a system only one
    # receiver has takes no part, and every epoch comes out as from the GPS file
    rover, base, ephs, iono, station, delta, _ = _car(captures, DRIVING, L1_L5)
    options = RtkOptions(backward=backward, **COMMAND)
    plain = solve_rtk(rover, base, ephs, station, delta, options, iono)
    mixed = read_rinex_observations(capture_file(captures, f"{DRIVING}/pyrx_gal.obs"), WITH_E5A)
    gps_base = read_rinex_observations(capture_file(captures, CAR_BASE), WITH_E5A)
    more = ephs + _galileo_ephemerides(captures)
    fixes = solve_rtk(mixed, gps_base, more, station, delta, options, iono)
    assert len(fixes) == len(plain) > 270
    for a, b in zip(fixes, plain, strict=True):
        assert (a.tow, a.status, a.ns, a.ratio) == (b.tow, b.status, b.ns, b.ratio)
        np.testing.assert_array_equal(a.position, b.position)


@pytest.mark.parametrize("capture", [STATIC, COLD_START])
def test_galileo_pseudoranges_fit_the_broadcast_orbits_at_the_gps_fix(captures, capture):
    # A Galileo pilot takes its time from its own secondary code and the receiver's GPS
    # clock; a wrong multiple of the code's 100 ms would be 30 000 km and a wrong millisecond
    # 300 km. At the single-point fix of the GPS L1 rows every E5a pseudorange is the range
    # to the satellite's broadcast position, within the atmosphere, the satellite's group
    # delay and the front end's delay between its two bands (measured over every epoch of
    # the three captures: -37 to +15 m).
    rover = _car_with_galileo(captures, capture)[0]
    ephs, iono = read_rinex_nav(capture_file(captures, f"{capture}/pyrx_l1l5.nav"))
    galileo = _galileo_ephemerides(captures)
    seen = set()
    for obs in rover[20::30]:
        fix = solve_epoch(to_epoch(obs), ephs, iono)
        j = obs.signals.index("E5Q")
        for i, sat in enumerate(obs.sats):
            pr = obs.pseudorange[i, j]
            if sat[0] != "E" or not np.isfinite(pr):
                continue
            t_tx = obs.tow - pr / C_LIGHT
            state = satellite_state(select_ephemeris(galileo, int(sat[1:]), t_tx, "E"), t_tx)
            rho, _ = geometric_range(fix.position, state.position)
            assert abs(pr - (rho + C_LIGHT * (fix.clock_bias - state.clock))) < 100.0, sat
            seen.add(sat)
    assert len(seen) >= 8


def test_galileo_pilots_give_the_cold_start_fixes_both_directions_agree_on(captures):
    # The capture begun in motion: seven GPS satellites, four of them with a pilot, the L1
    # phases of the others flagged every few seconds. With GPS alone the two filter
    # directions never fix the same epoch at the same place. The six Galileo satellites the
    # base also has, on their E5a pilots, are the carrier redundancy that was missing.
    # Measured with the command's settings, forward fixed / backward fixed / together /
    # apart: GPS 1 / 40 / 0 / 0, with Galileo 193 / 188 / 126 / 0, twelve satellites or
    # more in every fix both directions agree on. No truth for it, and the 126 depend on
    # the satellite set: without the strongest Galileo satellite 82 are together, 54 of them
    # the same epochs, and two stand apart. They also rest on one flag: E21's pilot slips a
    # whole cycle at 188 s, with one Galileo band nothing but its own lost-lock bit says so,
    # and without that bit the counts are 105 / 177 / 54 / 0.
    gps = _car(captures, COLD_START, L1_L5)
    _, _, together, _ = _both_ways(gps, **COMMAND)
    assert together == []
    data = _car_with_galileo(captures, COLD_START)
    fwd, bwd, together, apart = _both_ways(data, **COMMAND)
    assert len(together) >= 115 and len(apart) <= 1, (len(together), len(apart))
    fixed = [sum(f.status == "fix" for f in run.values()) for run in (fwd, bwd)]
    assert fixed[0] >= 180 and fixed[1] >= 175, fixed
    assert min(fwd[k].ns for k in together) >= 12
    # RTKLIB's rules get a seventh of that (18): the lock count and the check are needed too
    assert len(_both_ways(data)[2]) <= 25
    # the slip's flag taken off again: the ambiguity keeps the wrong cycle to the end
    k = next(n for n, e in enumerate(data[0]) if round(e.tow) == E21_SLIP)
    i, j = data[0][k].sats.index("E21"), WITH_E5A.index("E5Q")
    assert data[0][k].lli[i, j] & 1
    lli = data[0][k].lli.copy()
    lli[i, j] &= ~1
    unflagged = [*data[0][:k], dataclasses.replace(data[0][k], lli=lli), *data[0][k + 1 :]]
    _, _, together, _ = _both_ways((unflagged, *data[1:]), **COMMAND)
    assert len(together) <= 70, len(together)


def test_galileo_pilots_add_agreed_fixes_to_the_drive_and_the_parked_car(captures):
    # Measured with the command's settings: the drive 242 / 255 / 226 / 5 where GPS alone has
    # 202 / 178 / 144 / 1 (the five apart are 5.1 to 6.9 cm horizontal at 64, 65, 68 and 69 s and
    # 10.1 cm vertical at 58 s, just past the bounds; without the change limit 242 / 255 /
    # 227 / 4). The parked car in kinematic mode 294 / 299 / 294 / 0 where GPS alone fixes 14
    # epochs forward and none backward.
    _, _, together, apart = _both_ways(_car_with_galileo(captures, DRIVING), **COMMAND)
    assert len(together) >= 220 and len(apart) <= 5, (len(together), len(apart))
    data = _car_with_galileo(captures, STATIC)
    fwd, _, together, apart = _both_ways(data, **COMMAND)
    assert len(together) >= 288 and not apart
    # and where the parked car's GPS-only fix is (static mode, 132 epochs): the medians 3 mm
    # apart, every kinematic fix within 2 cm of its own median
    rover, base, ephs, iono, station, delta, _ = _car(captures, STATIC, L1_L5)
    options = RtkOptions(mode="static", **COMMAND)
    gps = [f for f in solve_rtk(rover, base, ephs, station, delta, options, iono) if f.status == "fix"]
    home = np.median([f.position for f in gps], axis=0)
    fixes = [f.position for f in fwd.values() if f.status == "fix"]
    assert np.linalg.norm(np.median(fixes, axis=0) - home) < 0.006
    assert max(np.linalg.norm(p - np.median(fixes, axis=0)) for p in fixes) < 0.025


BOTH_BANDS = ("G1C", "G5Q", "E1C", "E5Q")
E1_E5A_FILE = "pyrx_e1e5a.obs"  # a whole run of pyrx-rinex --l5 --galileo on each capture


@pytest.mark.parametrize("capture", [STATIC, DRIVING, COLD_START])
def test_the_file_with_both_galileo_bands_holds_the_earlier_files(captures, capture):
    # every GPS value of the GPS file and every E5a value of the file with E5a alone, formed
    # again by a whole run with the E1 pilots beside them: the agreement rule of the two
    # Galileo bands withheld nothing on these captures
    for name, signals in (("pyrx_l1l5.obs", L1_L5), ("pyrx_gal.obs", WITH_E5A)):
        old = read_rinex_observations(capture_file(captures, f"{capture}/{name}"), signals)
        new = read_rinex_observations(capture_file(captures, f"{capture}/{E1_E5A_FILE}"), signals)
        assert len(old) == len(new) > 280
        for a, b in zip(old, new, strict=True):
            assert (a.week, a.tow) == (b.week, b.tow)
            for i, sat in enumerate(a.sats):
                j = b.sats.index(sat)
                for field in ("pseudorange", "carrier_phase", "doppler", "cn0", "lli"):
                    np.testing.assert_array_equal(getattr(a, field)[i], getattr(b, field)[j])


def test_the_whole_cycle_slips_of_the_capture_started_while_driving_carry_the_flag(captures):
    # Three pilots slip a whole cycle in that capture, each in a fade null in which a 20 ms
    # sum stood 0.41 to 0.46 cycle from the loop's phase, near the discriminator's wrap, and
    # was still coherent enough for the loop to take it: G24's L5 pilot
    # 146 s into the capture, E21's E5a pilot at 188 s, E30's E1 pilot at 231 s. The
    # phase after each carries the lost-lock bit. On the two Galileo satellites the other
    # band shows the cycle: the geometry-free difference of the two phases steps by one
    # wavelength of the band that slipped, within 4 cm (the fade moves both by centimeters).
    path = capture_file(captures, f"{COLD_START}/{E1_E5A_FILE}")
    epochs = {round(e.tow): e for e in read_rinex_observations(path, BOTH_BANDS)}
    for tow, sat, signal in ((185035, "G24", "G5Q"), (E21_SLIP, "E21", "E5Q"), (185119, "E30", "E1C")):
        e = epochs[tow]
        assert e.lli[e.sats.index(sat), BOTH_BANDS.index(signal)] & 1, (tow, sat, signal)
        if sat[0] == "E":
            gf = []
            for t in (tow - 1, tow):
                row = epochs[t].carrier_phase[epochs[t].sats.index(sat)]
                gf.append(wavelength("E1C") * row[2] - wavelength("E5Q") * row[3])
            assert abs(abs(gf[1] - gf[0]) - wavelength(signal)) < 0.04, (sat, gf[1] - gf[0])
    # and the parked capture's pilots carry no flag after their first seconds, as before
    parked = read_rinex_observations(capture_file(captures, f"{STATIC}/{E1_E5A_FILE}"), BOTH_BANDS)
    assert not any((e.lli[:, 1:] & 1).any() for e in parked[3:])


def _galileo_pairs(captures, capture):
    """Per Galileo satellite: (tow, C1C, C5Q, L1C, L5Q, flagged) where both bands have a
    pseudorange."""
    path = capture_file(captures, f"{capture}/{E1_E5A_FILE}")
    rows: dict[str, list] = {}
    for e in read_rinex_observations(path, ("E1C", "E5Q")):
        for i, sat in enumerate(e.sats):
            if np.isfinite(e.pseudorange[i]).all():
                rows.setdefault(sat, []).append(
                    (e.tow, *e.pseudorange[i], *e.carrier_phase[i], e.lli[i].any())
                )
    return {sat: np.array(r) for sat, r in rows.items()}


@pytest.mark.parametrize("capture,satellites", [(STATIC, 7), (DRIVING, 10), (COLD_START, 8)])
def test_e1_and_e5a_pseudoranges_differ_by_one_value_on_every_satellite(captures, capture, satellites):
    # The two bands of a satellite are tracked in two recordings by channels that share
    # nothing but the sample clock, and each takes its time from its own secondary code.
    # E1C minus E5Q is the front end's delay between its bands, 31 m, with meters of
    # ionosphere and code noise: every epoch of every satellite within 7 m of it on the
    # three captures. An E1 channel beside a side peak of its correlation would stand 147
    # to 183 m off, one a quarter chip off 73 m, a wrong millisecond 300 km.
    pairs = _galileo_pairs(captures, capture)
    assert len(pairs) == satellites
    common = np.median(np.concatenate([r[:, 1] - r[:, 2] for r in pairs.values()]))
    assert 25.0 < common < 37.0, common
    for sat, r in pairs.items():
        assert len(r) > 50, sat
        assert np.max(np.abs(r[:, 1] - r[:, 2] - common)) < 30.0, sat


def test_e1_and_e5a_phases_of_the_parked_car_follow_a_line_together(captures):
    # the geometry-free combination of the two phases changes only with the ionosphere: over
    # the five minutes of the parked capture a line, with 1.4 to 2.9 mm of scatter about it
    # and 11 mm at most, on each of the seven satellites that have both bands, and no flag
    # after the first second (in motion the antenna's turning is in that combination too)
    pairs = _galileo_pairs(captures, STATIC)
    for sat, r in pairs.items():
        assert len(r) == 299 and np.isfinite(r[:, 3:5]).all() and not r[2:, 5].any(), sat
        x = r[:, 0] - r[0, 0]
        gf = wavelength("E1C") * r[:, 3] - wavelength("E5Q") * r[:, 4]
        residual = gf - np.polyval(np.polyfit(x, gf, 1), x)
        assert residual.std() < 0.005 and np.abs(residual).max() < 0.02, (sat, residual.std())


def test_both_galileo_bands_give_the_cold_start_more_fixes_both_directions_agree_on(captures):
    # Measured with the command's settings, forward fixed / backward fixed / together / apart:
    # with the E5a pilots alone 193 / 188 / 126 / 0, with E1 beside them 223 / 189 / 144 / 0
    # (206 / 185 / 128 / 0 without the change limit). Six Galileo satellites take part on
    # both bands (two more are not at the base), four GPS satellites on L1 and L5 and three
    # on L1. No truth for it, and the gain is thin: forward it follows from one ambiguity
    # that is kept at 182 s, after which thirteen epochs pass the ratio test at 3.0 to 3.5
    # where they fail it at 2.9 to 3.0. Left out one satellite at a time, 79 to 148 epochs
    # stay together and none to five stand apart, and single forward fixes lie up to a
    # meter from where they lay with every satellite.
    data = _car_with_galileo(captures, COLD_START, BOTH_BANDS, E1_E5A_FILE)
    fwd, bwd, together, apart = _both_ways(data, **COMMAND)
    assert len(together) >= 136 and not apart, (len(together), len(apart))
    fixed = [sum(f.status == "fix" for f in run.values()) for run in (fwd, bwd)]
    assert fixed[0] >= 196 and fixed[1] >= 180, fixed
    _, _, together, apart = _both_ways(data, **NO_LIMIT)
    assert 120 <= len(together) <= 135 and not apart, (len(together), len(apart))
    # RTKLIB's rules: 40 / 52 / 11 / 0
    assert len(_both_ways(data)[2]) <= 20


def test_both_galileo_bands_on_the_drive_and_the_parked_car(captures):
    # Measured with the command's settings: the drive 247 / 251 / 223 / 0 where GPS alone has
    # 202 / 178 / 144 / 1 and E5a beside it 242 / 255 / 226 / 5. Without the change limit 245
    # / 250 / 217 / 3, the three apart 5.4 to 5.6 cm horizontal at 53, 66 and 67 s, in twenty
    # seconds of a turn in which G24 is 5 to 13 dB down with its L1 phase flagged at every
    # epoch and both its phases moving by centimeters against the other satellites. With the
    # limit those three are 4.2 cm apart, under the bound of 5: the directions do not agree
    # there, they disagree a little less. The parked car in kinematic mode 294 / 299 / 294 /
    # 0 either way.
    data = _car_with_galileo(captures, DRIVING, BOTH_BANDS, E1_E5A_FILE)
    _, _, together, apart = _both_ways(data, **COMMAND)
    assert len(together) >= 216 and len(apart) <= 1, (len(together), len(apart))
    _, _, together, apart = _both_ways(data, **NO_LIMIT)
    assert len(together) >= 208 and 2 <= len(apart) <= 4, (len(together), len(apart))
    data = _car_with_galileo(captures, STATIC, BOTH_BANDS, E1_E5A_FILE)
    fwd, _, together, apart = _both_ways(data, **COMMAND)
    assert len(together) >= 288 and not apart
    # where the parked car's GPS-only fix is (static mode): the medians 2 mm apart
    rover, base, ephs, iono, station, delta, _ = _car(captures, STATIC, L1_L5)
    options = RtkOptions(mode="static", **COMMAND)
    gps = [f for f in solve_rtk(rover, base, ephs, station, delta, options, iono) if f.status == "fix"]
    home = np.median([f.position for f in gps], axis=0)
    fixes = [f.position for f in fwd.values() if f.status == "fix"]
    assert np.linalg.norm(np.median(fixes, axis=0) - home) < 0.006
    assert max(np.linalg.norm(p - np.median(fixes, axis=0)) for p in fixes) < 0.025


def test_every_galileo_satellite_of_the_base_has_a_usable_ephemeris(captures):
    # the public file writes a spare field in some Galileo records and not in others, which
    # once read the strongest satellite's record of the hour as unhealthy
    rover = _car_with_galileo(captures, COLD_START)[0]
    galileo = _galileo_ephemerides(captures)
    for prn in (13, 15, 21, 27, 29, 30, 34):
        assert usable(select_ephemeris(galileo, prn, rover[100].tow, "E")), prn


@pytest.mark.parametrize("runner", [rnx2rtkp_rtk, rnx2rtkp_ttff])
def test_reference_runners_turn_away_a_frequency_rtklib_would_ignore(runner, tmp_path):
    # this build of RTKLIB reads l1, l1+2, l1+2+3 and runs its default on anything else
    args = ("r.obs", "b.obs", "n.nav", tmp_path, (0.0, 0.0, 0.0), 0.0, 0.0, "kinematic", "l1+l2")
    with pytest.raises(ValueError, match=r"l1\+l2"):
        runner(*args, 1, [], 0) if runner is rnx2rtkp_ttff else runner(*args)
