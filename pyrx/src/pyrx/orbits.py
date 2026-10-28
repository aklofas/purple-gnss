"""Satellite position, velocity and clock from the broadcast ephemeris.

Position and clock follow IS-GPS-200 Table 20-IV and 20.3.3.3.3.1 (clock
polynomial plus the relativistic term); velocity and clock drift are central
differences over one millisecond.

Time: every `t` is GPS seconds of week. A caller's seconds of week may run past
604800 across a week boundary, and an ephemeris time of week (`toe`, `toc`) may
belong to the week before or after, so every difference against one is reduced
into [-302400, 302400) with `wrap_week`; callers never wrap their own time.
"""

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin, sqrt

import numpy as np

from pyrx.lnav import Ephemeris
from pyrx.observables import WEEK_S

MU_GPS = 3.986005e14  # m^3/s^2, WGS 84 value of the ICD
MU_GAL = 3.986004418e14  # m^3/s^2, GTRF value of the Galileo OS SIS ICD
OMEGA_E = 7.2921151467e-5  # rad/s, WGS 84 Earth rotation rate
F_REL = -4.442807633e-10  # s/sqrt(m), relativistic clock constant
MAX_TOE_AGE = 7200.0  # s, oldest GPS ephemeris used, as RTKLIB
MAX_TOE_AGE_GAL = 14400.0  # s, the same for Galileo
_DT = 0.5e-3  # s, half the central-difference step


@dataclass(frozen=True)
class SatState:
    position: np.ndarray  # ECEF m
    velocity: np.ndarray  # ECEF m/s
    clock: float  # s, including the relativistic term
    clock_drift: float  # s/s


def wrap_week(dt: float) -> float:
    """Reduce a difference of two seconds-of-week into [-302400, 302400)."""
    return (dt + WEEK_S / 2) % WEEK_S - WEEK_S / 2


def sat_id(eph: Ephemeris) -> str:
    """RINEX satellite id of a record, e.g. `G04`."""
    return f"{eph.system}{eph.prn:02d}"


def usable(eph: Ephemeris) -> bool:
    """A healthy record whose orbit can be evaluated (finite fields, a positive semi-major axis)."""
    fields = [v for v in vars(eph).values() if isinstance(v, (int, float))]
    return eph.health == 0 and eph.sqrt_a > 0.0 and all(isfinite(v) for v in fields)


def select_ephemeris(
    ephs: list[Ephemeris], prn: int, t: float, system: str = "G"
) -> Ephemeris | None:
    """The satellite's record with `toe` nearest to `t`, or None beyond the age limit."""
    mine = [e for e in ephs if e.prn == prn and e.system == system]
    if not mine:
        return None
    best = min(mine, key=lambda e: abs(wrap_week(t - e.toe)))
    limit = MAX_TOE_AGE_GAL if system == "E" else MAX_TOE_AGE
    return best if abs(wrap_week(t - best.toe)) <= limit else None


def satellite_clock(eph: Ephemeris, t: float) -> float:
    """Clock bias (s) from the polynomial alone, the quantity a precise clock product gives."""
    dtc = wrap_week(t - eph.toc)
    return eph.af0 + eph.af1 * dtc + eph.af2 * dtc**2


def _position_clock(eph: Ephemeris, t: float) -> tuple[np.ndarray, float]:
    tk = wrap_week(t - eph.toe)
    a = eph.sqrt_a**2
    n = sqrt((MU_GAL if eph.system == "E" else MU_GPS) / a**3) + eph.delta_n
    m = eph.m0 + n * tk
    big_e = m
    for _ in range(30):
        d = (big_e - eph.e * sin(big_e) - m) / (1.0 - eph.e * cos(big_e))
        big_e -= d
        if abs(d) < 1e-14:
            break
    nu = atan2(sqrt(1.0 - eph.e**2) * sin(big_e), cos(big_e) - eph.e)
    phi = nu + eph.omega
    s2, c2 = sin(2.0 * phi), cos(2.0 * phi)
    u = phi + eph.cus * s2 + eph.cuc * c2
    r = a * (1.0 - eph.e * cos(big_e)) + eph.crs * s2 + eph.crc * c2
    i = eph.i0 + eph.cis * s2 + eph.cic * c2 + eph.idot * tk
    big_omega = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe
    xp, yp = r * cos(u), r * sin(u)
    position = np.array(
        [
            xp * cos(big_omega) - yp * cos(i) * sin(big_omega),
            xp * sin(big_omega) + yp * cos(i) * cos(big_omega),
            yp * sin(i),
        ]
    )
    clock = satellite_clock(eph, t) + F_REL * eph.e * eph.sqrt_a * sin(big_e)
    return position, clock


def satellite_state(eph: Ephemeris, t: float) -> SatState:
    """Position, velocity, clock bias and clock drift at GPS seconds of week `t`."""
    position, clock = _position_clock(eph, t)
    p0, c0 = _position_clock(eph, t - _DT)
    p1, c1 = _position_clock(eph, t + _DT)
    return SatState(position, (p1 - p0) / (2 * _DT), clock, (c1 - c0) / (2 * _DT))
