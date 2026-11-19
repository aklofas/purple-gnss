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
from math import atan2, cos, inf, isfinite, sin, sqrt

import numpy as np

from pyrx.ephemeris import Ephemeris
from pyrx.gpstime import WEEK_S

MU_GPS = 3.986005e14  # m^3/s^2, WGS 84 value of the ICD
MU_GAL = 3.986004418e14  # m^3/s^2, GTRF value of the Galileo OS SIS ICD
OMEGA_E = 7.2921151467e-5  # rad/s, WGS 84 Earth rotation rate
F_REL = -4.442807633e-10  # s/sqrt(m), relativistic clock constant of IS-GPS-200
F_REL_GAL = -4.442807309e-10  # s/sqrt(m), that of the Galileo OS SIS ICD (its own mu)
MAX_TOE_AGE = 7201.0  # s, a GPS record is used this far from its toe, as RTKLIB
MAX_TOE_AGE_GAL = 14400.0  # s, a Galileo record this long after its toe
# RTKLIB leaves out a record whose accuracy has a variance over (300 m)^2: a GPS URA
# index from 10 (384 m), and a Galileo record without an accuracy prediction or with an
# SISA over 6 m (500 m to RTKLIB; read as index 15 here)
MAX_URA = 9
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
    """A healthy record of stated accuracy (`MAX_URA`) whose orbit can be evaluated (finite
    fields, a positive semi-major axis, an eccentricity of an ellipse)."""
    fields = [v for v in vars(eph).values() if isinstance(v, (int, float))]
    return (
        eph.health == 0
        and eph.ura <= MAX_URA
        and eph.sqrt_a > 0.0
        and 0.0 <= eph.e < 1.0
        and all(isfinite(v) for v in fields)
    )


def select_ephemeris(
    ephs: list[Ephemeris], prn: int, t: float, system: str = "G"
) -> Ephemeris | None:
    """The satellite's record for time `t` as RTKLIB selects it, or None when there is none.

    The record with `toe` nearest to `t` within the age limit; of Galileo only a record
    with `toe` before `t` (its data is in force from then on, and a receiver has no
    later one). Of several at the same distance the one transmitted last, records
    without a transmission time in list order. An issue broadcast more than once (the
    same `toe` and issue of data; the health and group delays of a Galileo record may
    change between repeats) is taken as it was last transmitted by `t`, or as first
    transmitted when `t` is before all of them.
    """
    limit = MAX_TOE_AGE_GAL if system == "E" else MAX_TOE_AGE
    nearest: list[Ephemeris] = []
    for e in ephs:
        if e.prn != prn or e.system != system:
            continue
        dt = wrap_week(t - e.toe)
        if abs(dt) > limit or (system == "E" and dt <= 0.0):
            continue
        if nearest and abs(dt) > abs(wrap_week(t - nearest[0].toe)):
            continue
        if nearest and abs(dt) < abs(wrap_week(t - nearest[0].toe)):
            nearest = []
        nearest.append(e)
    if not nearest:
        return None

    def sent(e):  # s after `t`: negative for a record on the air by then
        return -inf if e.ttr is None else wrap_week(e.ttr - t)

    last = max(reversed(nearest), key=sent)
    repeats = [e for e in nearest if (e.toe, e.iode) == (last.toe, last.iode)]
    by_now = [e for e in repeats if sent(e) <= 0.0]
    return max(reversed(by_now), key=sent) if by_now else min(repeats, key=sent)


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
    f_rel = F_REL_GAL if eph.system == "E" else F_REL
    clock = satellite_clock(eph, t) + f_rel * eph.e * eph.sqrt_a * sin(big_e)
    return position, clock


def satellite_state(eph: Ephemeris, t: float) -> SatState:
    """Position, velocity, clock bias and clock drift at GPS seconds of week `t`."""
    position, clock = _position_clock(eph, t)
    p0, c0 = _position_clock(eph, t - _DT)
    p1, c1 = _position_clock(eph, t + _DT)
    return SatState(position, (p1 - p0) / (2 * _DT), clock, (c1 - c0) / (2 * _DT))
