"""Receiver-satellite geometry and the broadcast atmospheric corrections.

Klobuchar follows IS-GPS-200 20.3.3.5.2.5; Saastamoinen uses the standard
atmosphere at the receiver height with 70 % humidity, as RTKLIB's single-point
mode does, so the two solvers correct by the same amounts.
"""

from math import asin, atan2, cos, exp, pi, sin, sqrt

import numpy as np
import pymap3d as pm

from pyrx.lnav import Ionosphere
from pyrx.observables import C_LIGHT
from pyrx.orbits import OMEGA_E


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
