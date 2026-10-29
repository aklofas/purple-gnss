"""Standalone position, clock, velocity and clock drift of one epoch by least squares.

Pseudoranges give position and receiver clock bias, then Dopplers give velocity
and clock drift. Measurement weights, the elevation mask, the chi-square and
GDOP checks and the atmospheric models are those of RTKLIB's single-point mode,
so the two solvers can be compared number for number on the same files.
"""

from dataclasses import dataclass
from math import isfinite, radians, sin, sqrt

import numpy as np

from pyrx.ephemeris import URA_BOUNDS, Ephemeris, Ionosphere
from pyrx.gpstime import C_LIGHT
from pyrx.models import azimuth_elevation, geodetic, geometric_range, klobuchar, saastamoinen
from pyrx.observations import Epoch
from pyrx.orbits import OMEGA_E, satellite_clock, satellite_state, select_ephemeris, usable
from pyrx.signals import wavelength

ELEVATION_MASK = 15.0  # deg
MAX_ITER = 10
MAX_GDOP = 30.0
# residual test thresholds at significance 0.001 by degrees of freedom, RTKLIB's rounded
# table (rtkcmn.c chisqr) so that both solvers accept the same epochs
_CHI2 = (
    10.8, 13.8, 16.3, 18.5, 20.5, 22.5, 24.3, 26.1, 27.9, 29.6,
    31.3, 32.9, 34.5, 36.1, 37.7, 39.3, 40.8, 42.3, 43.8, 45.3,
    46.8, 48.3, 49.7, 51.2, 52.6, 54.1, 55.5, 56.9, 58.3, 59.7,
    61.1, 62.5, 63.9, 65.2, 66.6, 68.0, 69.3, 70.7, 72.1, 73.4,
    74.7, 76.0, 77.3, 78.6, 80.0, 81.3, 82.6, 84.0, 85.4, 86.7,
    88.0, 89.3, 90.6, 91.9, 93.3, 94.7, 96.0, 97.4, 98.7, 100,
    101, 102, 103, 104, 105, 107, 108, 109, 110, 112,
    113, 114, 115, 116, 118, 119, 120, 122, 123, 125,
    126, 127, 128, 129, 131, 132, 133, 134, 135, 137,
    138, 139, 140, 142, 143, 144, 145, 147, 148, 149,
)  # fmt: skip
LAMBDA_L1 = wavelength("G1C")  # m
_ON_EARTH = 6e6  # m, estimates nearer the center get no mask, weights or atmosphere
_ERR_CODE = 0.003 * 100.0  # m, RTKLIB err[0] * err[1]: the a priori code error
_ERR_CBIAS = 0.3  # m, code bias error
_ERR_BRDCI = 0.5  # fraction of the broadcast ionospheric delay
_ERR_SAAS = 0.3  # m, Saastamoinen error at the zenith
_MIN_EL_WEIGHT = radians(5.0)  # elevation floor of the weights, as RTKLIB's varerr
# Klobuchar coefficients used when the navigation file carries none (RTKLIB's ion_default)
IONO_DEFAULT = Ionosphere(
    (0.1118e-07, -0.7451e-08, -0.5961e-07, 0.1192e-06),
    (0.1167e06, -0.2294e06, -0.1311e06, 0.1049e07),
)
_CONVERGED_POS = 1e-4  # m
_CONVERGED_VEL = 1e-6  # m/s


@dataclass(frozen=True)
class Solution:
    """One epoch's fix; GPS time of the epoch is `tow - clock_bias`."""

    week: int | None
    tow: float  # s, the epoch's receiver time stamp
    position: np.ndarray  # ECEF m
    clock_bias: float  # s
    velocity: np.ndarray | None  # ECEF m/s
    clock_drift: float | None  # s/s
    prns: tuple[int, ...]  # satellites in the solution
    residuals: np.ndarray  # m, pseudorange residuals per PRN at the solution
    gdop: float


def chi2_threshold(dof: int) -> float:
    """Residual test threshold (significance 0.001) for `dof` degrees of freedom, 1 to 100."""
    return _CHI2[dof - 1]


def _variance(ura: int, el: float, dion: float) -> float:
    # the a priori error's elevation is floored (RTKLIB varerr); the troposphere term's is not
    s = sin(max(el, _MIN_EL_WEIGHT))
    return (
        _ERR_CODE**2 * (1.0 + 1.0 / s)
        + URA_BOUNDS[min(ura, 14)] ** 2
        + _ERR_CBIAS**2
        + (_ERR_BRDCI * dion) ** 2
        + (_ERR_SAAS / (sin(el) + 0.1)) ** 2
    )


def _position(epoch, sats, iono, mask, x):
    """Iterate the pseudorange solution from `x`; None when it does not converge."""
    dx = np.full(4, np.inf)
    for _ in range(MAX_ITER + 1):
        on_earth = np.linalg.norm(x[:3]) > _ON_EARTH
        if on_earth:
            lat, lon, h = geodetic(x[:3])
        rows, res, sig, used = [], [], [], []
        for o, eph, s in sats:
            rho, e = geometric_range(x[:3], s.position)
            dion = dtrp = 0.0
            var = 1.0
            if on_earth:
                az, el = azimuth_elevation(x[:3], s.position)
                if el < mask:
                    continue
                dion = klobuchar(iono, lat, lon, az, el, epoch.tow)
                dtrp = saastamoinen(lat, h, el)
                var = _variance(eph.ura, el, dion)
            p = o.pseudorange - C_LIGHT * eph.tgd
            res.append(p - (rho + x[3] - C_LIGHT * s.clock + dion + dtrp))
            rows.append([-e[0], -e[1], -e[2], 1.0])
            sig.append(sqrt(var))
            used.append((o, eph, s, e))
        if len(res) < 4:
            return None
        if np.linalg.norm(dx) < _CONVERGED_POS:
            return x, np.array(rows), np.array(res), np.array(sig), used
        w = 1.0 / np.array(sig)
        try:
            dx = np.linalg.lstsq(np.array(rows) * w[:, None], np.array(res) * w, rcond=None)[0]
        except np.linalg.LinAlgError:  # degenerate geometry
            return None
        x = x + dx
    return None


def _velocity(x, used):
    """Doppler solution over the satellites of the position solution.

    None with fewer than four Dopplers or when the iteration does not converge. A
    Doppler of exactly zero is a missing value (RINEX writes one as 0.000).
    """
    sats = [(o, s, e) for o, _, s, e in used if isfinite(o.doppler) and o.doppler != 0.0]
    if len(sats) < 4:
        return None
    xv = np.zeros(4)
    for _ in range(MAX_ITER):
        rows, res = [], []
        for o, s, e in sats:
            rate = e @ (s.velocity - xv[:3]) + OMEGA_E / C_LIGHT * (
                s.velocity[1] * x[0]
                + s.position[1] * xv[0]
                - s.velocity[0] * x[1]
                - s.position[0] * xv[1]
            )
            res.append(-LAMBDA_L1 * o.doppler - (rate + xv[3] - C_LIGHT * s.clock_drift))
            rows.append([-e[0], -e[1], -e[2], 1.0])
        dx = np.linalg.lstsq(np.array(rows), np.array(res), rcond=None)[0]
        xv = xv + dx
        if np.linalg.norm(dx) < _CONVERGED_VEL:
            return xv
    return None


def solve_epoch(
    epoch: Epoch,
    ephs: list[Ephemeris],
    iono: Ionosphere | None = None,
    mask_deg: float = ELEVATION_MASK,
    x0: np.ndarray | None = None,
) -> Solution | None:
    """Least-squares fix of one epoch, or None when it has none that passes validation.

    `x0` seeds the position (ECEF m); from None the iteration starts at the Earth's
    center. Satellites without an ephemeris within 2 h, with an unhealthy or
    unusable record (a blank or zero orbit field) and those below `mask_deg` are
    left out; degenerate geometry gives no solution. `iono` None or all zeros (a
    navigation file without GPSA/GPSB) means RTKLIB's default Klobuchar
    coefficients, as `rnx2rtkp` does.
    """
    if iono is None or not any(iono.alpha + iono.beta):
        iono = IONO_DEFAULT
    sats = []
    for o in epoch.obs:
        if not isfinite(o.pseudorange):
            continue
        eph = select_ephemeris(ephs, o.prn, epoch.tow)
        if eph is None or not usable(eph):
            continue
        t = epoch.tow - o.pseudorange / C_LIGHT
        sats.append((o, eph, satellite_state(eph, t - satellite_clock(eph, t))))
    if len(sats) < 4:
        return None
    x = np.zeros(4) if x0 is None else np.array([x0[0], x0[1], x0[2], 0.0])
    fix = _position(epoch, sats, iono, radians(mask_deg), x)
    if fix is None:
        return None
    x, rows, res, sig, used = fix
    n = len(res)
    if n > 4 and np.sum((res / sig) ** 2) > chi2_threshold(n - 4):
        return None
    try:
        gdop = sqrt(np.trace(np.linalg.inv(rows.T @ rows)))
    except np.linalg.LinAlgError:
        return None
    if gdop > MAX_GDOP:
        return None
    vel = _velocity(x, used)
    return Solution(
        week=epoch.week,
        tow=epoch.tow,
        position=x[:3],
        clock_bias=x[3] / C_LIGHT,
        velocity=None if vel is None else vel[:3],
        clock_drift=None if vel is None else vel[3] / C_LIGHT,
        prns=tuple(o.prn for o, _, _, _ in used),
        residuals=res,
        gdop=gdop,
    )


def solve(
    epochs: list[Epoch],
    ephs: list[Ephemeris],
    iono: Ionosphere | None = None,
    mask_deg: float = ELEVATION_MASK,
) -> list[Solution]:
    """Fix every epoch that has one, each seeded with the previous fix's position."""
    out = []
    x0 = None
    for epoch in epochs:
        sol = solve_epoch(epoch, ephs, iono, mask_deg, x0)
        if sol is not None:
            out.append(sol)
            x0 = sol.position
    return out
