"""Receiver-satellite geometry and the broadcast atmospheric corrections.

Klobuchar follows IS-GPS-200 20.3.3.5.2.5; Saastamoinen uses the standard
atmosphere at the receiver height with 70 % humidity, as RTKLIB's single-point
mode does, so the two solvers correct by the same amounts. Relative positioning
uses the hydrostatic zenith delay alone, mapped with the Niell dry function, as
RTKLIB's RTK does. A station is its marker plus the antenna offset of the RINEX
header; ranges go to the antenna reference point.
"""

from dataclasses import dataclass
from math import asin, atan2, cos, exp, pi, sin, sqrt

import numpy as np
import pymap3d as pm

from pyrx.ephemeris import Ionosphere
from pyrx.gpstime import C_LIGHT
from pyrx.orbits import OMEGA_E

# Niell 1996 hydrostatic mapping coefficients at latitudes 15, 30, 45, 60, 75 deg:
# averages a, b, c; seasonal amplitudes; height correction
_NMF_LAT = (15.0, 30.0, 45.0, 60.0, 75.0)
_NMF_AVG = (
    (1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3),
    (2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3),
    (62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3),
)
_NMF_AMP = (
    (0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5),
    (0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5),
    (0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5),
)
_NMF_HEIGHT = (2.53e-5, 5.49e-3, 1.14e-3)


@dataclass(frozen=True)
class Station:
    marker: np.ndarray  # ECEF m
    delta_enu: np.ndarray  # m, marker to antenna reference point, east north up


def station_arp(station: Station) -> np.ndarray:
    """ECEF position of the antenna reference point."""
    lat, lon, h = geodetic(station.marker)
    e, n, u = station.delta_enu
    return np.array(pm.enu2ecef(e, n, u, lat, lon, h, deg=False), dtype=float)


def marker_from_arp(arp: np.ndarray, delta_enu: np.ndarray) -> np.ndarray:
    """ECEF position of the marker below an antenna reference point."""
    lat, lon, h = geodetic(arp)
    e, n, u = delta_enu
    return np.array(pm.enu2ecef(-e, -n, -u, lat, lon, h, deg=False), dtype=float)


def geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    """Latitude and longitude (rad) and ellipsoidal height (m) of an ECEF position."""
    lat, lon, h = pm.ecef2geodetic(xyz[0], xyz[1], xyz[2], deg=False)
    return float(lat), float(lon), float(h)


def azimuth_elevation(rx_xyz: np.ndarray, sat_xyz: np.ndarray) -> tuple[float, float]:
    """Azimuth in [0, 2 pi) and elevation (rad) of a satellite seen from a receiver."""
    lat, lon, h = geodetic(rx_xyz)
    e, n, u = pm.ecef2enu(sat_xyz[0], sat_xyz[1], sat_xyz[2], lat, lon, h, deg=False)
    return atan2(e, n) % (2 * pi), asin(u / sqrt(e * e + n * n + u * u))


def geometric_range(rx_xyz: np.ndarray, sat_xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """Range with the Earth-rotation (Sagnac) term, and the unit line-of-sight vector."""
    d = sat_xyz - rx_xyz
    norm = float(np.linalg.norm(d))
    sagnac = OMEGA_E * (sat_xyz[0] * rx_xyz[1] - sat_xyz[1] * rx_xyz[0]) / C_LIGHT
    return norm + sagnac, d / norm


def klobuchar(iono: Ionosphere, lat: float, lon: float, az: float, el: float, tow: float) -> float:
    """L1 ionospheric delay (m) from the broadcast Klobuchar coefficients; zero below the horizon."""
    if el <= 0.0:
        return 0.0
    e = el / pi  # semicircles
    psi = 0.0137 / (e + 0.11) - 0.022
    phi_i = lat / pi + psi * cos(az)
    phi_i = min(max(phi_i, -0.416), 0.416)
    lam_i = lon / pi + psi * sin(az) / cos(phi_i * pi)
    phi_m = phi_i + 0.064 * cos((lam_i - 1.617) * pi)
    t = (4.32e4 * lam_i + tow) % 86400.0
    f = 1.0 + 16.0 * (0.53 - e) ** 3
    amp = max(sum(a * phi_m**n for n, a in enumerate(iono.alpha)), 0.0)
    per = max(sum(b * phi_m**n for n, b in enumerate(iono.beta)), 72000.0)
    x = 2.0 * pi * (t - 50400.0) / per
    delay = 5e-9
    if abs(x) < 1.57:
        delay += amp * (1.0 - x**2 / 2.0 + x**4 / 24.0)
    return C_LIGHT * f * delay


def saastamoinen(lat: float, height: float, el: float, humidity: float = 0.7) -> float:
    """Tropospheric delay (m) with the standard atmosphere; zero off the Earth or below the horizon."""
    if height < -100.0 or height > 1e4 or el <= 0.0:
        return 0.0
    h = max(height, 0.0)
    p = 1013.25 * (1.0 - 2.2557e-5 * h) ** 5.2568
    temp = 15.0 - 6.5e-3 * h + 273.16
    e = 6.108 * humidity * exp((17.15 * temp - 4684.0) / (temp - 38.45))
    z = pi / 2 - el
    dry = 0.0022768 * p / (1.0 - 0.00266 * cos(2.0 * lat) - 0.00028 * h / 1e3) / cos(z)
    wet = 0.002277 * (1255.0 / temp + 0.05) * e / cos(z)
    return dry + wet


def zenith_hydrostatic(lat: float, height: float) -> float:
    """Saastamoinen hydrostatic delay (m) at the zenith with the standard atmosphere."""
    h = max(height, 0.0)
    p = 1013.25 * (1.0 - 2.2557e-5 * h) ** 5.2568
    return 0.0022768 * p / (1.0 - 0.00266 * cos(2.0 * lat) - 0.00028 * h / 1e3)


def _interp_lat(coef, lat_deg: float) -> float:
    i = int(lat_deg / 15.0)
    if i < 1:
        return coef[0]
    if i > 4:
        return coef[4]
    frac = lat_deg / 15.0 - i
    return coef[i - 1] * (1.0 - frac) + coef[i] * frac


def _mapf(el: float, a: float, b: float, c: float) -> float:
    s = sin(el)
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (s + a / (s + b / (s + c)))


def nmf_dry(lat: float, height: float, el: float, doy: float) -> float:
    """Niell hydrostatic mapping function; zero below the horizon."""
    if el <= 0.0:
        return 0.0
    y = (doy - 28.0) / 365.25 + (0.5 if lat < 0.0 else 0.0)
    cosy = cos(2.0 * pi * y)
    lat_deg = abs(lat) * 180.0 / pi
    a, b, c = (
        _interp_lat(_NMF_AVG[i], lat_deg) - _interp_lat(_NMF_AMP[i], lat_deg) * cosy
        for i in range(3)
    )
    dm = (1.0 / sin(el) - _mapf(el, *_NMF_HEIGHT)) * height / 1e3
    return _mapf(el, a, b, c) + dm
