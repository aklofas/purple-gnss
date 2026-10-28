"""Relative positioning: a float Kalman filter over double differences and LAMBDA fixing.

State: rover antenna position and one single-differenced (rover minus base)
carrier-phase ambiguity in cycles per satellite and signal. Measurements:
double-differenced carrier phase and code against the highest satellite of each
system. The seed, the residual model, the weights, the slip and outage rules, the
filter and the ratio test are those of RTKLIB's single-baseline kinematic and
static modes with its default options, so the two solvers compare number for
number on the same files.

Deliberate differences from RTKLIB, kept because they are at least as sound and
do not change the default-configuration comparisons: the phase-minus-code offset
that aligns existing ambiguity states to the current clock is averaged per signal
(RTKLIB: per frequency index across systems); the geometry-free slip test runs
against the first signal for every other signal present (RTKLIB stops at the
first missing one); and of two reference candidates at exactly the same
elevation the first by satellite id is taken (RTKLIB the last).
"""

from dataclasses import dataclass
from math import isfinite, radians, sin

import numpy as np

from pyrx.ambiguity import lambda_fix
from pyrx.lnav import Ephemeris, Ionosphere
from pyrx.models import (
    Station,
    azimuth_elevation,
    geodetic,
    geometric_range,
    marker_from_arp,
    nmf_dry,
    station_arp,
    zenith_hydrostatic,
)
from pyrx.observables import C_LIGHT, WEEK_S
from pyrx.orbits import satellite_clock, satellite_state, select_ephemeris, usable
from pyrx.pvt import solve_epoch
from pyrx.rinex import gps_datetime
from pyrx.signals import Observations, to_epoch, wavelength

DTTOL = 0.025  # s, base and rover epochs closer than this are the same epoch


@dataclass(frozen=True)
class RtkOptions:
    mode: str = "kinematic"  # or "static"
    mask_deg: float = 15.0
    ratio_threshold: float = 3.0
    max_age: float = 30.0  # s, oldest base epoch used
    max_outage: int = 5  # epochs a satellite may be missing before its ambiguity is dropped
    slip_threshold: float = 0.05  # m, geometry-free jump
    max_innovation: float = 30.0  # m, a double difference beyond this is dropped; 0 for no limit
    phase_error: float = 0.003  # m, a priori phase error (constant and per sine of elevation)
    code_ratio: float = 100.0  # code error relative to phase
    bias_sigma: float = 30.0  # cycles, initial ambiguity sigma
    bias_noise: float = 1e-4  # cycles per sqrt(s), ambiguity process noise
    position_sigma: float = 30.0  # m, kinematic position reset
    clock_stability: float = 5e-12  # s/s, receiver clock, times the differential age


@dataclass(frozen=True)
class Fix:
    week: int
    tow: float
    position: np.ndarray  # rover marker, ECEF m
    status: str  # "fix" or "float"
    ns: int  # satellites used on the first signal of their system
    ratio: float
    age: float  # s, rover minus base epoch
    nb: int  # double-differenced ambiguities searched


class _Bias:
    __slots__ = ("half", "lock", "outc", "used")

    def __init__(self):
        self.lock = 0
        self.outc = 0
        self.half = False
        self.used = False


def _time(obs: Observations) -> float:
    return obs.week * WEEK_S + obs.tow


def _seed_signal(signals: tuple[str, ...]) -> str:
    """The GPS signal the single-point seed solves on: L1 C/A when present.

    The seed is an L1 solver (L1 group delay and ionosphere), so another GPS
    signal only stands in when L1 C/A is absent.
    """
    gps = [s for s in signals if s[0] == "G"]
    if not gps:
        raise ValueError("no GPS signal to seed the position from")
    return "G1C" if "G1C" in gps else gps[0]


def pair_epochs(
    rover: list[Observations], base: list[Observations], max_age: float
) -> list[tuple[Observations, Observations | None]]:
    """Each rover epoch with the latest base epoch not later than it, None beyond `max_age`."""
    out = []
    j = 0
    for r in rover:
        t = _time(r)
        while j + 1 < len(base) and _time(base[j + 1]) <= t + DTTOL:
            j += 1
        b = base[j] if base and _time(base[j]) <= t + DTTOL else None
        out.append((r, b if b is not None and t - _time(b) <= max_age else None))
    return out


class Rtk:
    """The filter of one baseline; call `step` once per rover epoch in time order."""

    def __init__(
        self,
        ephs: list[Ephemeris],
        base: Station,
        rover_delta_enu: np.ndarray,
        options: RtkOptions | None = None,
        iono: Ionosphere | None = None,
    ):
        self.ephs = ephs
        self.base_arp = station_arp(base)
        self.rover_delta = np.asarray(rover_delta_enu, dtype=float)
        self.opt = options or RtkOptions()
        self.iono = iono
        self.keys: list[
            tuple[str, str]
        ] = []  # bias states in state-vector order after the position
        self.biases: dict[tuple[str, str], _Bias] = {}
        self.gf: dict[
            tuple[str, str], float
        ] = {}  # geometry-free against the system's first signal
        self.lli: dict[tuple[str, str], tuple[int, int]] = {}  # last rover and base indicators
        self.x = np.zeros(3)
        self.P = np.zeros((3, 3))
        self.last_time: float | None = None
        self.last_base_time: float | None = None

    # --- state bookkeeping ---------------------------------------------------------------

    def _remove(self, key):
        i = 3 + self.keys.index(key)
        self.x = np.delete(self.x, i)
        self.P = np.delete(np.delete(self.P, i, 0), i, 1)
        self.keys.remove(key)
        del self.biases[key]

    def _add(self, key, value, var):
        n = len(self.x)
        x = np.append(self.x, value)
        P = np.zeros((n + 1, n + 1))
        P[:n, :n] = self.P
        P[n, n] = var
        self.x, self.P = x, P
        self.keys.append(key)
        self.biases[key] = _Bias()

    def _index(self, key):
        return 3 + self.keys.index(key)

    def _age(self):
        for key in list(self.keys):
            b = self.biases[key]
            b.outc += 1
            b.used = False
            if b.outc > self.opt.max_outage:
                self._remove(key)

    # --- geometry ---------------------------------------------------------------------------

    def _satellites(self, rover, base):
        """Per common satellite: ephemeris and state at each receiver's transmit time."""
        out = {}
        for i, sat in enumerate(rover.sats):
            if sat not in base.sats:
                continue
            eph = select_ephemeris(self.ephs, int(sat[1:]), rover.tow, sat[0])
            if eph is None or not usable(eph):
                continue
            states = []
            for obs, row in ((rover, i), (base, base.sats.index(sat))):
                p = next((v for v in obs.pseudorange[row] if isfinite(v)), None)
                if p is None:
                    break
                t = obs.tow - p / C_LIGHT
                states.append(satellite_state(eph, t - satellite_clock(eph, t)))
            if len(states) == 2:
                out[sat] = (eph, states[0], states[1])
        return out

    def _residuals(self, obs, xyz, sats, which, doy, mask):
        """Undifferenced phase and code residuals (m) per satellite and signal, with geometry."""
        lat, _, h = geodetic(xyz)
        zhd = zenith_hydrostatic(lat, h)
        out = {}
        for sat, entry in sats.items():
            s = entry[which]
            _, el = azimuth_elevation(xyz, s.position)
            if el < mask:
                continue
            rho, e = geometric_range(xyz, s.position)
            corr = rho - C_LIGHT * s.clock + zhd * nmf_dry(lat, h, el, doy)
            row = obs.sats.index(sat)
            lam = np.array([wavelength(sig) for sig in obs.signals])
            y_phase = lam * obs.carrier_phase[row] - corr
            y_code = obs.pseudorange[row] - corr
            out[sat] = (y_phase, y_code, e, el, obs.lli[row])
        return out

    # --- the epoch ------------------------------------------------------------------------

    def step(self, rover: Observations, base: Observations | None) -> Fix | None:
        opt = self.opt
        t = _time(rover)
        tt = 0.0 if self.last_time is None else t - self.last_time
        self.last_time = t
        if base is None:
            return None
        seed = solve_epoch(
            to_epoch(rover, _seed_signal(rover.signals)),
            self.ephs,
            self.iono,
            opt.mask_deg,
        )
        if seed is None:
            return None
        if opt.mode == "kinematic" or not self.P[:3, :3].any():
            self.x[:3] = seed.position
            self.P[:3, :] = 0.0
            self.P[:, :3] = 0.0
            self.P[:3, :3] = np.eye(3) * opt.position_sigma**2
        sats = self._satellites(rover, base)
        doy = gps_datetime(rover.week, rover.tow).timetuple().tm_yday
        mask = radians(opt.mask_deg)
        yb = self._residuals(base, self.base_arp, sats, 2, doy, mask)
        yr = self._residuals(rover, self.x[:3], sats, 1, doy, mask)
        common = sorted(set(yr) & set(yb))
        if not common:
            return None
        # outages count only on epochs that reach the relative solution (RTKLIB returns
        # before udbias without a base, beyond the age limit, without a seed or without
        # a common satellite)
        self._age()
        signals = rover.signals
        cols = [base.signals.index(s) if s in base.signals else None for s in signals]
        age = t - _time(base)
        # a base epoch reused for several rover epochs reports its slips once
        base_new = _time(base) != self.last_base_time
        self.last_base_time = _time(base)

        # single differences, slips and ambiguity states (RTKLIB udbias)
        sd = {}  # (sat, signal) -> (phase m, code m)
        halves = {}
        first_sigs = {sig[0]: sig for sig in reversed(signals)}  # first signal of each system
        for sat in common:
            yr_p, yr_c, _, _, lli_r = yr[sat]
            yb_p, yb_c, _, _, lli_b = yb[sat]
            ref_sig = first_sigs[sat[0]]
            ref_phase = None
            for j, sig in enumerate(signals):
                jb = cols[j]
                if jb is None or not (isfinite(yr_p[j]) and isfinite(yb_p[jb])):
                    continue
                key = (sat, sig)
                phase = yr_p[j] - yb_p[jb]
                code = yr_c[j] - yb_c[jb] if isfinite(yr_c[j]) and isfinite(yb_c[jb]) else None
                # a loss-of-lock flag, or a half-cycle flag changing, at either receiver
                prev_r, prev_b = self.lli.get(key, (lli_r[j], lli_b[jb]))
                slip = (
                    bool(lli_r[j] & 1)
                    or (base_new and bool(lli_b[jb] & 1))
                    or bool(((prev_r ^ lli_r[j]) | (prev_b ^ lli_b[jb])) & 2)
                )
                self.lli[key] = (int(lli_r[j]), int(lli_b[jb]))
                halves[key] = bool((lli_r[j] | lli_b[jb]) & 2)
                if sig == ref_sig:
                    ref_phase = phase
                elif ref_phase is not None:  # geometry-free against the first signal, when present
                    gf = ref_phase - phase
                    prev = self.gf.get(key)
                    if prev is not None and abs(gf - prev) > opt.slip_threshold:
                        slip = True
                        if (sat, ref_sig) in self.biases:
                            self._remove((sat, ref_sig))
                    self.gf[key] = gf
                if slip and key in self.biases:
                    self._remove(key)
                if code is not None:
                    sd[key] = (phase, code)
                    if key in self.biases:
                        i = self._index(key)
                        self.P[i, i] += opt.bias_noise**2 * abs(tt)
        # align existing states to the current single-differenced clock, create the rest
        for sig in signals:
            lam = wavelength(sig)
            news = {k: (v[0] - v[1]) / lam for k, v in sd.items() if k[1] == sig}
            existing = [k for k in news if k in self.biases]
            if existing:
                offset = np.mean([news[k] - self.x[self._index(k)] for k in existing])
                for k in self.keys:
                    if k[1] == sig:
                        self.x[self._index(k)] += offset
            for k, v in news.items():
                if k not in self.biases:
                    self._add(k, v, opt.bias_sigma**2)
                self.biases[k].half = halves[k]

        # double differences (RTKLIB ddres)
        rows, v, blocks = [], [], []
        used = set()
        n = len(self.x)
        for system in sorted({s[0] for s in common}):
            for kind in ("phase", "code"):
                for sig in signals:
                    if sig[0] != system:
                        continue
                    valid = [sat for sat in common if sat[0] == system and (sat, sig) in sd]
                    if len(valid) < 2:
                        continue
                    ref = max(valid, key=lambda sat: yr[sat][3])
                    var_ref = self._variance(yr[ref][3], age, kind == "code")
                    block = []
                    for sat in valid:
                        if sat == ref:
                            continue
                        idx = 0 if kind == "phase" else 1
                        res = sd[(sat, sig)][idx] - sd[(ref, sig)][idx]
                        h = np.zeros(n)
                        h[:3] = -yr[sat][2] + yr[ref][2]
                        if kind == "phase":
                            lam = wavelength(sig)
                            i, j = self._index((sat, sig)), self._index((ref, sig))
                            res -= lam * (self.x[i] - self.x[j])
                            h[i], h[j] = lam, -lam
                        if opt.max_innovation > 0.0 and abs(res) > opt.max_innovation:
                            continue
                        rows.append(h)
                        v.append(res)
                        block.append(self._variance(yr[sat][3], age, kind == "code"))
                        used.update({(sat, sig), (ref, sig)})
                    if block:
                        blocks.append((var_ref, block))
        if not rows:
            return None
        H = np.array(rows)
        v = np.array(v)
        R = np.zeros((len(v), len(v)))
        k = 0
        for var_ref, block in blocks:
            m = len(block)
            R[k : k + m, k : k + m] = var_ref + np.diag(block)
            k += m
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None
        self.x = self.x + K @ v
        self.P = (np.eye(n) - K @ H) @ self.P
        for key in used:
            b = self.biases[key]
            b.lock += 1
            b.outc = 0
            b.used = True
        first_sigs = {sig[0]: sig for sig in reversed(signals)}  # first signal of each system
        ns = len({sat for sat, sig in used if sig == first_sigs[sig[0]]})
        if ns < 4:  # the update stands, the epoch has no solution (RTKLIB SOLQ_NONE)
            return None

        # ambiguity resolution (RTKLIB resamb_LAMBDA, continuous mode)
        pairs = []
        for system in sorted({s[0] for s in common}):
            for sig in signals:
                if sig[0] != system:
                    continue
                cands = [
                    key
                    for key in self.keys
                    if key[1] == sig
                    and key[0][0] == system
                    and self.biases[key].used
                    and self.biases[key].lock > 0
                    and not self.biases[key].half
                ]
                if len(cands) >= 2:
                    ref = cands[0]
                    pairs.extend((ref, key) for key in cands[1:])
        nb = len(pairs)
        ratio = 0.0
        status = "float"
        position = self.x[:3]
        if nb >= 2:
            D = np.zeros((nb, n))
            for r, (ref, key) in enumerate(pairs):
                D[r, self._index(ref)] = 1.0
                D[r, self._index(key)] = -1.0
            y = D @ self.x
            Qb = D @ self.P @ D.T
            Qab = self.P[:3] @ D.T
            try:
                cands, s = lambda_fix(y, Qb, 2)
                ratio = 0.0 if s[0] <= 0.0 else min(s[1] / s[0], 999.9)
                if s[0] <= 0.0 or ratio >= opt.ratio_threshold:
                    position = self.x[:3] - Qab @ np.linalg.solve(Qb, y - cands[0])
                    status = "fix"
            except (ValueError, np.linalg.LinAlgError):
                ratio = 0.0
        return Fix(
            rover.week,
            rover.tow,
            marker_from_arp(position, self.rover_delta),
            status,
            ns,
            float(ratio),
            age,
            nb,
        )

    def _variance(self, el, age, code):
        a = b = self.opt.phase_error * (self.opt.code_ratio if code else 1.0)
        return (
            2.0 * (a * a + b * b / sin(el) ** 2) + (C_LIGHT * self.opt.clock_stability * age) ** 2
        )


def solve_rtk(
    rover: list[Observations],
    base: list[Observations],
    ephs: list[Ephemeris],
    base_station: Station,
    rover_delta_enu: np.ndarray,
    options: RtkOptions | None = None,
    iono: Ionosphere | None = None,
) -> list[Fix]:
    """Fix every rover epoch that has one, pairing it with the latest base epoch."""
    rtk = Rtk(ephs, base_station, rover_delta_enu, options, iono)
    out = []
    for r, b in pair_epochs(rover, base, rtk.opt.max_age):
        fix = rtk.step(r, b)
        if fix is not None:
            out.append(fix)
    return out
