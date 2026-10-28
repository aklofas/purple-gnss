from dataclasses import replace

import numpy as np
import pytest
from test_lnav import SF1, SF2, SF3

from pyrx.lnav import assemble_ephemeris, decode_subframe, encode_subframe
from pyrx.orbits import OMEGA_E, satellite_clock, satellite_state, select_ephemeris, wrap_week


def _eph(**changes):
    sfs = [
        decode_subframe(encode_subframe(i, 5044 + i, f), 0, 0, prn=4)
        for i, f in ((1, SF1), (2, SF2), (3, SF3))
    ]
    return replace(assemble_ephemeris(sfs), **changes)


def test_wrap_week():
    assert wrap_week(10.0) == 10.0
    assert wrap_week(604790.0 - 20.0) == pytest.approx(-30.0)
    assert wrap_week(-604790.0) == pytest.approx(10.0)
    assert wrap_week(302400.0) == -302400.0


def test_select_ephemeris_across_week_rollover():
    old = _eph(prn=4, toe=597600.0, iode=10)  # 2 h before the rollover
    new = _eph(prn=4, toe=0.0, iode=11)
    other = _eph(prn=5, toe=0.0, iode=12)
    ephs = [old, new, other]
    assert select_ephemeris(ephs, 4, 600000.0) is old
    assert select_ephemeris(ephs, 4, 603000.0) is new  # 1800 s before toe 0 of the next week
    assert select_ephemeris(ephs, 4, 1800.0) is new
    assert select_ephemeris(ephs, 4, 7201.0 + 3600.0) is None  # more than 2 h from both
    assert select_ephemeris(ephs, 6, 0.0) is None


def test_satellite_state_is_a_near_circular_gps_orbit():
    eph = _eph()
    s = satellite_state(eph, eph.toe)
    r = np.linalg.norm(s.position)
    assert r == pytest.approx(eph.sqrt_a**2, rel=3e-3)  # e = 0.0016, harmonics of a few hundred m
    # the ECEF velocity plus the Earth's rotation is the inertial speed, about sqrt(mu / a)
    inertial = s.velocity + np.cross([0.0, 0.0, OMEGA_E], s.position)
    assert np.linalg.norm(inertial) == pytest.approx(3870.0, abs=30.0)
    # velocity is the time derivative of position
    dt = 1.0
    s2 = satellite_state(eph, eph.toe + dt)
    np.testing.assert_allclose(
        (s2.position - s.position) / dt, (s.velocity + s2.velocity) / 2, atol=1e-3
    )


def test_satellite_state_at_toe_matches_closed_form_for_zero_eccentricity():
    eph = _eph(e=0.0, cuc=0.0, cus=0.0, crc=0.0, crs=0.0, cic=0.0, cis=0.0, idot=0.0, delta_n=0.0)
    s = satellite_state(eph, eph.toe)
    u = eph.m0 + eph.omega
    big_omega = eph.omega0 - OMEGA_E * eph.toe
    r = eph.sqrt_a**2
    expected = r * np.array([
        np.cos(u) * np.cos(big_omega) - np.sin(u) * np.cos(eph.i0) * np.sin(big_omega),
        np.cos(u) * np.sin(big_omega) + np.sin(u) * np.cos(eph.i0) * np.cos(big_omega),
        np.sin(u) * np.sin(eph.i0),
    ])  # fmt: skip
    np.testing.assert_allclose(s.position, expected, atol=1e-6)
    assert s.clock == pytest.approx(satellite_clock(eph, eph.toe))  # no relativity when e = 0


def test_satellite_clock_and_relativity():
    eph = _eph(af0=1e-5, af1=2e-12, af2=0.0)
    assert satellite_clock(eph, eph.toc + 100.0) == pytest.approx(1e-5 + 2e-10, rel=1e-12)
    s = satellite_state(eph, eph.toe + 1000.0)
    rel = s.clock - satellite_clock(eph, eph.toe + 1000.0)
    assert 0 < abs(rel) < 2e-8  # e = 0.0016: a few nanoseconds
    assert s.clock_drift == pytest.approx(2e-12, abs=1e-12)


def test_satellite_state_wraps_week():
    eph = _eph(toe=604000.0, toc=604000.0)
    a = satellite_state(eph, 604100.0)
    b = satellite_state(eph, 604100.0 - 604800.0)  # the same instant written in the next week
    np.testing.assert_allclose(a.position, b.position, atol=1e-6)
