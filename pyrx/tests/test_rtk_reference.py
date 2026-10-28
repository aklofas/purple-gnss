"""Our RTK fix against RTKLIB's and the EUREF coordinates on two IGS station pairs."""

from datetime import datetime

import numpy as np
import pytest
from conftest import capture_file
from rtklib import rnx2rtkp_rtk, rnx2rtkp_ttff

from pyrx.models import Station
from pyrx.rinex import read_rinex_header, read_rinex_nav, read_rinex_observations
from pyrx.rtk import RtkOptions, solve_rtk

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
    ("MATE", "kinematic", ("G1C", "G2W"), "l1+l2"),
    ("MATE", "kinematic", ("G1C",), "l1"),
    ("MATE", "static", ("G1C", "G2W"), "l1+l2"),
    ("TLSE", "kinematic", ("G1C", "G2W"), "l1+l2"),
    ("TLSE", "kinematic", ("G1C",), "l1"),
    ("TLSE", "static", ("G1C", "G2W"), "l1+l2"),
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
    assert len(rover) == 721 and len(fixes) >= 700
    assert len(fixed) / len(rover) >= len(ref_fixed) / len(rover) - 0.10
    err = np.array([np.linalg.norm(f.position - rover_xyz) for f in fixed])
    # single-epoch kinematic fixes scatter up to 2.5 cm from the marker on both pairs, in
    # RTKLIB as well (its worst epochs 20-24 mm); the carried static position stays under 1 cm
    limits = (0.005, 0.02) if mode == "static" else (0.01, 0.03)
    assert err.mean() <= limits[0] and np.percentile(err, 95) <= 0.02, (
        err.mean(),
        np.percentile(err, 95),
    )
    assert err.max() <= limits[1], err.max()
    common = [(f, ref_fixed[_key(f)]) for f in fixed if _key(f) in ref_fixed]
    assert len(common) >= 0.9 * len(ref_fixed)
    d = np.array([np.linalg.norm(f.position - r["xyz"]) for f, r in common])
    # same models, same integers: 2.2 mm at the 95th percentile and 2.6 mm worst measured
    assert np.percentile(d, 95) <= 0.005 and d.max() <= 0.01, (np.percentile(d, 95), d.max())


@pytest.mark.parametrize(
    "pair,signals,freq",
    [("MATE", ("G1C", "G2W"), "l1+l2"), ("MATE", ("G1C",), "l1"), ("TLSE", ("G1C", "G2W"), "l1+l2"), ("TLSE", ("G1C",), "l1")],
    ids=["MATE-l1+l2", "MATE-l1", "TLSE-l1+l2", "TLSE-l1"],
)  # fmt: skip
def test_time_to_first_fix(captures, tmp_path, pair, signals, freq):
    base_name, rover_name, base_xyz, _ = PAIRS[pair]
    rover, base, ephs, hb, hr = _load(captures, pair, signals)
    station = Station(np.array(base_xyz), hb.delta_enu)
    ours = []
    for k in range(12):  # restarts every 30 min, 30 min windows (61 epochs)
        t0 = 1800.0 * k
        rov = [o for o in rover if t0 <= o.tow < t0 + 1800.0]
        bas = [o for o in base if t0 - 30.0 <= o.tow < t0 + 1800.0]
        fixes = solve_rtk(rov, bas, ephs, station, hr.delta_enu)
        first = next((i for i, f in enumerate(fixes) if f.status == "fix"), 60)
        ours.append(first + 1)
    starts = [f"2021/12/26 {k // 2:02d}:{30 * (k % 2):02d}:00" for k in range(12)]
    theirs = rnx2rtkp_ttff(
        capture_file(captures, rover_name), capture_file(captures, base_name), capture_file(captures, NAV),
        tmp_path / "ttff", base_xyz, hb.delta_enu[2], hr.delta_enu[2], "kinematic", freq, 1, starts, 1800,
    )  # fmt: skip
    assert np.mean(ours) <= 2.0 * np.mean(theirs), (ours, theirs)


def test_rtk_multi_gnss_matches_rtklib(captures, tmp_path):
    # RTKLIB with pos1-frequency=l1+l2+l5 and gps+gal uses GPS L1/L2/L5 and Galileo E1/E5b/E5a
    signals = ("G1C", "G2W", "G5Q", "E1C", "E7Q", "E5Q")
    base_name, rover_name, base_xyz, rover_xyz = PAIRS["MATE"]
    rover, base, ephs, hb, hr = _load(captures, "MATE", signals)
    assert any(e.system == "E" for e in ephs)
    fixes = solve_rtk(rover, base, ephs, Station(np.array(base_xyz), hb.delta_enu), hr.delta_enu)
    ref = rnx2rtkp_rtk(
        capture_file(captures, rover_name), capture_file(captures, base_name), capture_file(captures, NAV),
        tmp_path / "rtk", base_xyz, hb.delta_enu[2], hr.delta_enu[2], "kinematic", "l1+l2+l5", 9, ts=TS, te=TE,
    )  # fmt: skip
    fixed = [f for f in fixes if f.status == "fix"]
    ref_fixed = {k: v for k, v in ref.items() if v["q"] == 1}
    assert np.mean([f.ns for f in fixed]) >= 10  # GPS and Galileo both in the solution
    assert len(fixed) / len(rover) >= len(ref_fixed) / len(rover) - 0.10
    err = np.array([np.linalg.norm(f.position - rover_xyz) for f in fixed])
    assert err.mean() <= 0.01 and np.percentile(err, 95) <= 0.02 and err.max() <= 0.02
    common = [(f, ref_fixed[_key(f)]) for f in fixed if _key(f) in ref_fixed]
    assert len(common) >= 0.9 * len(ref_fixed)
    # against RTKLIB's fixed positions: 4.6 mm at the 95th percentile and 7.8 mm worst
    # measured (its Galileo models and reference choice differ from ours in detail)
    d = np.array([np.linalg.norm(f.position - r["xyz"]) for f, r in common])
    assert np.percentile(d, 95) <= 0.01 and d.max() <= 0.02, (np.percentile(d, 95), d.max())


def test_rtk_l1_l5_e1_e5a_fixes_the_marker(captures):
    # the hardware's signal set; RTKLIB cannot run exactly this combination
    _, _, base_xyz, rover_xyz = PAIRS["MATE"]
    rover, base, ephs, hb, hr = _load(captures, "MATE", ("G1C", "G5Q", "E1C", "E5Q"))
    fixes = solve_rtk(rover, base, ephs, Station(np.array(base_xyz), hb.delta_enu), hr.delta_enu)
    fixed = [f for f in fixes if f.status == "fix"]
    assert len(fixed) / len(rover) >= 0.90
    err = np.array([np.linalg.norm(f.position - rover_xyz) for f in fixed])
    assert err.mean() <= 0.01 and np.percentile(err, 95) <= 0.02 and err.max() <= 0.02
