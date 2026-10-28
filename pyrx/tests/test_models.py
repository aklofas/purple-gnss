from math import pi, radians, sin

import numpy as np
import pytest

from pyrx.lnav import Ionosphere
from pyrx.models import (
    Station,
    azimuth_elevation,
    geodetic,
    geometric_range,
    klobuchar,
    marker_from_arp,
    nmf_dry,
    saastamoinen,
    station_arp,
    zenith_hydrostatic,
)
from pyrx.observables import C_LIGHT
from pyrx.orbits import OMEGA_E

USUD = np.array([-3855263.0361, 3427432.5571, 3741020.2929])  # ITRF, from the IGS site log


def test_geodetic_of_usud():
    lat, lon, h = geodetic(USUD)
    assert np.degrees(lat) == pytest.approx(36.1331, abs=1e-4)
    assert np.degrees(lon) == pytest.approx(138.3620, abs=1e-4)
    assert h == pytest.approx(1508.6, abs=0.5)


def test_azimuth_elevation_of_zenith_north_and_east():
    lat, lon, _ = geodetic(USUD)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    assert azimuth_elevation(USUD, USUD + 1e6 * up)[1] == pytest.approx(pi / 2, abs=1e-6)
    az, el = azimuth_elevation(USUD, USUD + 1e6 * north + 1e5 * up)
    assert min(az, 2 * pi - az) == pytest.approx(0.0, abs=1e-6)  # rounding may give 2 pi - eps
    assert el == pytest.approx(np.arctan(0.1), abs=1e-6)
    az, el = azimuth_elevation(USUD, USUD + 1e6 * east + 1e5 * up)
    assert az == pytest.approx(pi / 2, abs=1e-6) and el == pytest.approx(np.arctan(0.1), abs=1e-6)
    az, _ = azimuth_elevation(USUD, USUD - 1e6 * east + 1e5 * up)
    assert az == pytest.approx(1.5 * pi, abs=1e-6)


def test_geometric_range_has_the_sagnac_term():
    sat = np.array([15e6, 10e6, 20e6])
    rho, e = geometric_range(USUD, sat)
    plain = np.linalg.norm(sat - USUD)
    assert rho - plain == pytest.approx(OMEGA_E * (sat[0] * USUD[1] - sat[1] * USUD[0]) / C_LIGHT)
    np.testing.assert_allclose(e, (sat - USUD) / plain)


def test_klobuchar_night_and_day():
    iono = Ionosphere((1e-8, 0.0, 0.0, 0.0), (72000.0, 0.0, 0.0, 0.0))
    f = 1 + 16 * (0.53 - 0.5) ** 3  # slant factor at the zenith
    night = klobuchar(iono, 0.0, 0.0, 0.0, pi / 2, 7200.0)
    assert night == pytest.approx(f * 5e-9 * C_LIGHT, rel=1e-9)
    noon = klobuchar(iono, 0.0, 0.0, 0.0, pi / 2, 50400.0)
    assert noon == pytest.approx(f * 1.5e-8 * C_LIGHT, rel=1e-9)
    assert klobuchar(iono, 0.0, 0.0, 0.0, 0.05, 50400.0) > 2 * noon  # 3 deg: slant factor 3.2
    assert klobuchar(iono, 0.0, 0.0, 0.0, 0.0, 50400.0) == 0.0
    assert klobuchar(iono, 0.0, 0.0, 0.0, -0.1, 50400.0) == 0.0


def test_saastamoinen_zenith_and_mapping():
    dry = saastamoinen(pi / 4, 0.0, pi / 2, humidity=0.0)
    assert dry == pytest.approx(2.30697, abs=1e-3)
    wet = saastamoinen(pi / 4, 0.0, pi / 2) - dry
    assert 0.10 < wet < 0.15
    zenith = saastamoinen(pi / 4, 0.0, pi / 2)
    assert saastamoinen(pi / 4, 0.0, pi / 6) == pytest.approx(2 * zenith, rel=1e-12)
    assert saastamoinen(pi / 4, 1500.0, pi / 2) < dry
    assert saastamoinen(pi / 4, 20000.0, pi / 2) == 0.0 and saastamoinen(pi / 4, 0.0, 0.0) == 0.0


def test_station_arp_and_back():
    st = Station(USUD, np.array([0.0, 0.0, 1.053]))
    arp = station_arp(st)
    assert np.linalg.norm(arp - USUD) == pytest.approx(1.053, abs=1e-6)
    assert np.linalg.norm(arp) > np.linalg.norm(USUD)  # up
    np.testing.assert_allclose(marker_from_arp(arp, st.delta_enu), USUD, atol=1e-6, rtol=0)


def test_zenith_hydrostatic_matches_saastamoinen_dry():
    dry = saastamoinen(pi / 4, 0.0, pi / 2, humidity=0.0)
    assert zenith_hydrostatic(pi / 4, 0.0) == pytest.approx(dry, rel=1e-12)
    assert zenith_hydrostatic(pi / 4, 1500.0) < zenith_hydrostatic(pi / 4, 0.0)


def test_nmf_dry_at_zenith_and_low_elevation():
    assert nmf_dry(pi / 4, 0.0, pi / 2, 360) == pytest.approx(1.0, abs=1e-9)
    m15 = nmf_dry(pi / 4, 0.0, radians(15.0), 360)
    assert 3.80 < m15 < 3.87 and m15 < 1 / sin(radians(15.0))  # Niell at 15 deg is below 1/sin el
    assert nmf_dry(pi / 4, 1500.0, radians(15.0), 360) > m15  # the height term is positive
    south = nmf_dry(-pi / 4, 0.0, radians(15.0), 360)
    assert south == pytest.approx(nmf_dry(pi / 4, 0.0, radians(15.0), 360 - 183), abs=2e-3)
    assert nmf_dry(pi / 4, 0.0, 0.0, 360) == 0.0
