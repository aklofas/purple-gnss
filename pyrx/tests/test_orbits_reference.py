"""Broadcast satellite positions and clocks of 2021-12-26 against the IGS final orbit."""

import warnings

import numpy as np
import pytest
from conftest import capture_file

from pyrx.gpstime import GPS_EPOCH
from pyrx.orbits import MAX_TOE_AGE, satellite_clock, satellite_state, select_ephemeris, wrap_week
from pyrx.rinex import read_rinex_nav

NAV = "igs/USUD00JPN_R_20213600000_01D_GN.rnx.gz"
SP3 = "igs/igs21900.sp3"


def test_broadcast_orbit_matches_igs_final(captures):
    georinex = pytest.importorskip("georinex")
    ephs, _ = read_rinex_nav(capture_file(captures, NAV))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sp3 = georinex.load(capture_file(captures, SP3))
    pairs = 0
    for t in sp3.time.values:
        tow = ((t - np.datetime64(GPS_EPOCH)) / np.timedelta64(1, "s")) % 604800
        for sv in sp3.sv.values:
            if not sv.startswith("G"):
                continue
            eph = select_ephemeris(ephs, int(sv[1:]), float(tow))
            if eph is None or eph.health != 0:
                continue
            p = sp3.sel(sv=sv, time=t)
            pos = p["position"].values * 1e3  # km -> m
            clk = float(p["clock"].values) * 1e-6  # us -> s
            if not np.all(np.isfinite(pos)) or not np.isfinite(clk) or abs(clk) > 0.9:
                continue
            s = satellite_state(eph, float(tow))
            assert abs(wrap_week(float(tow) - eph.toe)) <= MAX_TOE_AGE
            assert np.linalg.norm(s.position - pos) < 6.0, (sv, str(t))
            # SP3 clocks exclude the relativistic term and carry the IGS clock datum (a few ns)
            assert abs(satellite_clock(eph, float(tow)) - clk) < 20e-9, (sv, str(t))
            pairs += 1
    assert pairs >= 1300
