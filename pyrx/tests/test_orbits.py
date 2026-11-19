from dataclasses import replace

import numpy as np
import pytest

from pyrx.orbits import (
    OMEGA_E,
    sat_id,
    satellite_clock,
    satellite_state,
    select_ephemeris,
    usable,
    wrap_week,
)
from pyrx.synth import ephemeris


def test_wrap_week():
    assert wrap_week(10.0) == 10.0
    assert wrap_week(604790.0 - 20.0) == pytest.approx(-30.0)
    assert wrap_week(-604790.0) == pytest.approx(10.0)
    assert wrap_week(302400.0) == -302400.0


def test_select_ephemeris_across_week_rollover():
    old = ephemeris(prn=4, toe=597600.0, iode=10)  # 2 h before the rollover
    new = ephemeris(prn=4, toe=0.0, iode=11)
    other = ephemeris(prn=5, toe=0.0, iode=12)
    ephs = [old, new, other]
    assert select_ephemeris(ephs, 4, 600000.0) is old
    assert select_ephemeris(ephs, 4, 603000.0) is new  # 1800 s before toe 0 of the next week
    assert select_ephemeris(ephs, 4, 1800.0) is new
    assert select_ephemeris(ephs, 4, 7201.0 + 3600.0) is None  # more than 2 h from both
    assert select_ephemeris(ephs, 6, 0.0) is None


def test_satellite_state_is_a_near_circular_gps_orbit():
    eph = ephemeris()
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
        (s2.position - s.position) / dt, (s.velocity + s2.velocity) / 2, atol=1e-3, rtol=0
    )


def test_satellite_state_at_toe_matches_closed_form_for_zero_eccentricity():
    eph = ephemeris(e=0.0, cuc=0.0, cus=0.0, crc=0.0, crs=0.0, cic=0.0, cis=0.0, idot=0.0, delta_n=0.0)
    s = satellite_state(eph, eph.toe)
    u = eph.m0 + eph.omega
    big_omega = eph.omega0 - OMEGA_E * eph.toe
    r = eph.sqrt_a**2
    expected = r * np.array([
        np.cos(u) * np.cos(big_omega) - np.sin(u) * np.cos(eph.i0) * np.sin(big_omega),
        np.cos(u) * np.sin(big_omega) + np.sin(u) * np.cos(eph.i0) * np.cos(big_omega),
        np.sin(u) * np.sin(eph.i0),
    ])  # fmt: skip
    np.testing.assert_allclose(s.position, expected, atol=1e-6, rtol=0)
    assert s.clock == pytest.approx(satellite_clock(eph, eph.toe))  # no relativity when e = 0


def test_satellite_clock_and_relativity():
    eph = ephemeris(af0=1e-5, af1=2e-12, af2=0.0)
    assert satellite_clock(eph, eph.toc + 100.0) == pytest.approx(1e-5 + 2e-10, rel=1e-12)
    s = satellite_state(eph, eph.toe + 1000.0)
    rel = s.clock - satellite_clock(eph, eph.toe + 1000.0)
    assert 0 < abs(rel) < 2e-8  # e = 0.0016: a few nanoseconds
    assert s.clock_drift == pytest.approx(2e-12, abs=1e-12)


def test_satellite_state_wraps_week():
    eph = ephemeris(toe=604000.0, toc=604000.0)
    a = satellite_state(eph, 604100.0)
    b = satellite_state(eph, 604100.0 - 604800.0)  # the same instant written in the next week
    np.testing.assert_allclose(a.position, b.position, atol=1e-6, rtol=0)


def test_select_ephemeris_by_system():
    g = ephemeris(prn=4, toe=1000.0)
    e = replace(ephemeris(prn=4, toe=1000.0), system="E")
    assert select_ephemeris([g, e], 4, 1500.0) is g
    assert select_ephemeris([g, e], 4, 1500.0, system="E") is e
    assert select_ephemeris([g], 4, 1500.0, system="E") is None
    assert sat_id(e) == "E04" and sat_id(g) == "G04"


def test_the_age_limits_are_those_of_rtklib():
    g = ephemeris(prn=4, toe=20000.0)
    e = replace(g, system="E")
    # GPS: 7201 s either side of toe
    for t, found in ((12799.0, True), (12798.5, False), (27201.0, True), (27201.5, False)):
        assert (select_ephemeris([g], 4, t) is g) == found, t
    # Galileo: after toe alone, for 14400 s
    for t, found in ((19000.0, False), (20000.0, False), (20000.5, True), (34400.0, True)):
        assert (select_ephemeris([e], 4, t, system="E") is e) == found, t
    assert select_ephemeris([e], 4, 34400.5, system="E") is None


def test_a_galileo_record_is_not_selected_before_its_toe():
    # the nearest toe may lie ahead: RTKLIB then takes the nearest one behind, and a
    # receiver at that time has no other
    old = replace(ephemeris(prn=4, toe=600.0, iode=1), system="E")
    new = replace(ephemeris(prn=4, toe=1200.0, iode=2), system="E")
    for ephs in ([old, new], [new, old]):
        assert select_ephemeris(ephs, 4, 1100.0, system="E") is old
        assert select_ephemeris(ephs, 4, 1200.0, system="E") is old
        assert select_ephemeris(ephs, 4, 1200.5, system="E") is new
        assert select_ephemeris(ephs, 4, 600.0, system="E") is None


def test_of_two_records_at_the_same_distance_the_one_transmitted_last_is_selected():
    # exactly between two toes RTKLIB takes the last of its records, which it has sorted
    # by transmission time; without transmission times the list's order stands for it
    a, b = ephemeris(prn=4, toe=0.0, iode=1), ephemeris(prn=4, toe=7200.0, iode=2)
    assert select_ephemeris([a, b], 4, 3600.0) is b and select_ephemeris([b, a], 4, 3600.0) is a
    a, b = replace(a, ttr=-7170.0), replace(b, ttr=30.0)
    assert select_ephemeris([a, b], 4, 3600.0) is b and select_ephemeris([b, a], 4, 3600.0) is b
    assert select_ephemeris([a, b], 4, 3599.0) is a and select_ephemeris([a, b], 4, 3601.0) is b


def test_an_issue_broadcast_again_is_taken_as_last_transmitted_by_the_time():
    # the health and group delays of an I/NAV record have no issue of data: the same issue
    # comes again with its health changed. Before the change the first stands, after it the
    # second, in whichever order the file lists them
    first = replace(ephemeris(prn=1, toe=600.0, iode=1), system="E", ttr=1775.0)
    again = replace(first, ttr=2375.0, health=2)
    for ephs in ([first, again], [again, first]):
        assert select_ephemeris(ephs, 1, 1000.0, system="E") is first  # before either was sent
        assert select_ephemeris(ephs, 1, 2374.0, system="E") is first
        assert select_ephemeris(ephs, 1, 2375.0, system="E") is again
        assert select_ephemeris(ephs, 1, 9000.0, system="E") is again
        assert not usable(select_ephemeris(ephs, 1, 9000.0, system="E"))
    # without transmission times the later in the list stands
    first, again = replace(first, ttr=None), replace(again, ttr=None)
    assert select_ephemeris([first, again], 1, 2000.0, system="E") is again


def test_satellite_state_uses_galileo_constants():
    g = ephemeris()
    e = replace(g, system="E")
    dg = satellite_state(g, g.toe + 3600.0).position
    de = satellite_state(e, g.toe + 3600.0).position
    # the mean motions differ by (MU_GAL / MU_GPS)**0.5 - 1 = -7.3e-8: meters after an hour
    assert 0.5 < np.linalg.norm(dg - de) < 5.0


@pytest.mark.parametrize(("ura", "ok"), [(0, True), (9, True), (10, False), (15, False)])
@pytest.mark.parametrize("system", ["G", "E"])
def test_a_record_of_poor_or_no_stated_accuracy_is_not_usable(ura, ok, system):
    # RTKLIB's limit of (300 m)^2 on the variance: URA index 9 is 192 m, 10 is 384 m
    assert usable(replace(ephemeris(), ura=ura, system=system)) is ok


@pytest.mark.parametrize(
    ("e", "ok"), [(0.0, True), (0.02, True), (1.0, False), (1.1, False), (-0.01, False)]
)
def test_only_an_elliptical_orbit_is_usable(e, ok):
    # Kepler's equation as solved here, and sqrt(1 - e^2), need 0 <= e < 1
    assert usable(replace(ephemeris(), e=e)) is ok
