"""Relative positioning: a float Kalman filter over double differences and LAMBDA fixing.

State: rover antenna position and one single-differenced (rover minus base)
carrier-phase ambiguity in cycles per satellite and signal. Measurements:
double-differenced carrier phase and code against the highest satellite of each
system. The residual model, the weights, the slip and outage rules, the filter and
the ratio test are those of RTKLIB's single-baseline kinematic and static modes with
its default options, and so is the seed on GPS, so the two solvers compare number
for number on the same files.

Deliberate differences from RTKLIB, kept because they are at least as sound and
do not change the default-configuration comparisons: the phase-minus-code offset
that aligns existing ambiguity states to the current clock is averaged per signal
(RTKLIB: per frequency index across systems); the geometry-free slip test runs
against the first signal for every other signal present (RTKLIB stops at the
first missing one); of two reference candidates at exactly the same
elevation the first by satellite id is taken (RTKLIB the last); and a slip the
base reports in an epoch that pairs with no rover epoch (a base faster than the
rover, or a gap in the rover's epochs) is applied at the next paired base epoch (RTKLIB
reads only the paired epochs, so that flag is lost); and a slip either receiver flags
in an epoch that ends before the ambiguities are looked at (no base epoch in reach, the
single-point seed rejected, no common satellite) is applied at the next epoch that gets
there, whichever satellites are common in it (RTKLIB returns before it reads those
indicators, and the ambiguity survives the slip); with a base slower than the rover,
a base phase whose half-cycle bit is set keeps its ambiguity out of the integer search
at every rover epoch it is paired with (RTKLIB reads a base observation's indicator
once and admits the ambiguity at the later ones); and `Fix.ns`, with the four
satellites an epoch needs, counts each system's satellites on the first signal listed
for it (RTKLIB on its first frequency, L1 or E1, whatever the list), so a list without
L1 or E1 still has a count; and of a broadcast issue transmitted more than once the
record last transmitted by the epoch is used (`orbits.select_ephemeris`), where RTKLIB
uses the one transmitted last in the file.

Not RTKLIB's, and outside the comparison: the single-point seed is solved on one GPS
signal (`_seed_signal`) and needs four GPS satellites whatever other systems the
signals hold, where RTKLIB seeds on every system with an offset between them, so an
epoch with fewer GPS satellites has no solution here; the ratio reported on a float
epoch with a single double-differenced ambiguity, or after the search failed on an
ill-conditioned covariance, is 0 (RTKLIB reports the value of a search it does not
use); and two integer candidates of exactly equal norm come back in the other order
(the ratio is then 1 in both, never a fix).

Kept as in RTKLIB: the loss-of-lock indicators are read only for the satellites
both receivers contribute at an epoch (observed, with an ephemeris, above the
mask at the base: RTKLIB's selsat). A slip one receiver flags while the other
lacks the satellite is therefore not seen, and the ambiguity survives it when the
satellite is common again within `max_outage` epochs; only the geometry-free test
can still catch the jump. The rover's own mask does not select: a satellite below
it at the rover alone keeps its indicators read and its ambiguity looked after,
and only its rows stay out of the update.

Beyond RTKLIB's defaults, all off unless asked for: `slips="check"` looks at a phase
only the rover flags before its ambiguity restarts, against the satellite's other
signal and against a trial update of the rest of the epoch (`_update_biases`,
`_test_asked`), for a receiver that flags every phase that may have slipped, and
with `change_limit` the other signal's word is not taken when either of the two
changed against the other satellites since the previous epoch (`_moved`);
`min_lock` is RTKLIB's pos2-arlockcnt, counted as RTKLIB counts it (`_add`);
`min_fix_sats` keeps an epoch float unless the integer search spans that many
satellites (of one system: a further system brings its own reference satellite, which
does not count), since a search over a few double differences passes the ratio test
without pinning the position; `backward` runs the filter from the last epoch to the
first, reading the indicators as RTKLIB does backward, over the pairs of the forward
pass (RTKLIB pairs a rover epoch with the next base epoch when it runs backward; the
same epoch at equal rates).
"""

from dataclasses import dataclass, replace
from math import isfinite, radians, sin, sqrt

import numpy as np

from pyrx.ambiguity import lambda_fix
from pyrx.ephemeris import Ephemeris, Ionosphere
from pyrx.gpstime import C_LIGHT, WEEK_S, gps_datetime
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
from pyrx.observations import Observations, to_epoch
from pyrx.orbits import satellite_clock, satellite_state, select_ephemeris, usable
from pyrx.pvt import solve_epoch
from pyrx.signals import wavelength

DTTOL = 0.025  # s, base and rover epochs closer than this are the same epoch
# cycles: a phase in question continues its ambiguity when it fits the other phases of the
# epoch within the first (half the distance to the nearest slip of a Costas loop, which
# settles half a cycle away) and that fit is predicted, state and noise, within the second
CHECK_INNOVATION = 0.25
CHECK_SIGMA = 0.125
# a verdict on which phases changed against the others since the previous epoch (`_moved`):
CHANGE_FIT = 0.015  # m, a phase within this of the solution is explained by it
CHANGE_MIN_PHASES = 12  # phases no receiver flags that it takes
CHANGE_MIN_BAND = 3  # satellites explained on a frequency for its phases to be judged
CHANGE_MAX_STEP = 5.0  # s between the two epochs


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
    backward: bool = False  # the epochs last to first, the flags read from the other side
    min_lock: int = 0  # updates an ambiguity waits before the integer search (see `_add`)
    # satellites the searched ambiguities must span for a fix, counted for one system: each
    # further system's reference satellite does not count
    min_fix_sats: int = 0
    slips: str = "restart"  # or "check": a phase the rover flags is looked at first
    # m, with slips="check"; 0 for no limit: a phase in question is not settled by its
    # satellite's other signal when either of the two changed by more than this against
    # the other satellites since the previous epoch (see `_moved`)
    change_limit: float = 0.0


@dataclass(frozen=True)
class Ambiguity:
    """One double-differenced ambiguity of the search: reference satellite minus satellite.

    The reference is the first eligible state of its system and signal, not the satellite
    highest in the sky that the measurement double differences use; it and the order of the
    records can change when a state is reset, so identify a record by `ref`, `sat` and
    `signal`.
    """

    ref: str  # reference satellite, RINEX id
    sat: str
    signal: str  # e.g. "G1C"
    value: float  # cycles, the float estimate
    std: float  # cycles
    integer: int | None  # the integer applied; None on a float epoch


@dataclass(frozen=True)
class Fix:
    week: int
    tow: float
    position: np.ndarray  # rover marker, ECEF m
    status: str  # "fix" or "float"
    ns: int  # satellites used on the first signal of their system
    ratio: float
    age: float  # s, rover minus base epoch (negative before the first base epoch)
    nb: int  # double-differenced ambiguities eligible for the search (LAMBDA runs from 2)
    float_position: np.ndarray  # rover marker from the float states, ECEF m
    position_std: np.ndarray  # m, standard deviation of the float position per ECEF axis
    ambiguities: tuple[Ambiguity, ...]  # the nb eligible double differences, in search order


class _Bias:
    __slots__ = ("half", "lock", "outc", "used", "used_before")

    def __init__(self):
        self.lock = 0
        self.outc = 0
        self.half = False
        self.used = False
        self.used_before = False  # in the update of the previous epoch that had one


def _time(obs: Observations) -> float:
    return obs.week * WEEK_S + obs.tow


def _least_deviations(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A solution close to the one with the least sum of absolute residuals, by thirty
    rounds of reweighted least squares: a few rows far off do not steer it as they steer
    least squares."""
    weight = np.ones(len(y))
    for _ in range(30):
        root = np.sqrt(weight)
        x = np.linalg.lstsq(a * root[:, None], y * root, rcond=None)[0]
        weight = 1.0 / np.maximum(np.abs(y - a @ x), 1e-4)
    return x


def _moved(
    change: np.ndarray,
    los: np.ndarray,
    band: np.ndarray,
    sat: np.ndarray,
    clean: np.ndarray,
    limit: float,
) -> np.ndarray | None:
    """Per phase, whether its change (m) since the previous epoch stands more than `limit`
    from what the other phases explain; None without a verdict.

    Between two epochs the single-differenced residual of a phase changes by the change of
    the position error along its line of sight `los`, by a receiver clock term that is one
    number per carrier frequency (`band`, an index; the term also holds the antenna's turn,
    one cycle a turn on every phase of a frequency), and by what the phase did on its own.
    The unknowns are fitted to the `clean` phases, those neither receiver flags at either
    epoch (the lost-lock bit, or a change of the half-cycle bit between the two), with the
    least sum of absolute residuals. A phase that moved still pulls on
    such a solution when few phases are there, so one is formed without each satellite in
    turn (`sat`, an index), and the one that explains the most clean phases within
    CHANGE_FIT gives the verdict. A solution is formed only where the phases left
    determine the unknowns: three satellites, or several at one elevation, cannot tell a
    change of position from the clocks (the test is of rank: satellites a few degrees
    from one elevation pass it and still determine the height poorly). There is no verdict with fewer than
    CHANGE_MIN_PHASES clean phases, or when the phases the best solution explains, less
    one per unknown (any solution explains that many for nothing), do not outnumber the
    clean phases it leaves out: those may be the receiver's motion and the others the
    ones that moved. A phase of a frequency with fewer than CHANGE_MIN_BAND satellites
    among the explained is not judged: a clock fitted to one or two satellites explains
    them whatever they did, on however many signals.

    The position error meant is that of the position the residuals were formed at. In
    kinematic mode that is each epoch's single-point seed, meters off, and the lines of
    sight turn between the epochs: the change then also holds that turn times the earlier
    error, a millimeter over a second and centimeters over half a minute, which is why the
    engine asks for no verdict across more than CHANGE_MAX_STEP.
    """
    if clean.sum() < CHANGE_MIN_PHASES:
        return None
    a = np.zeros((len(change), 3 + int(band.max()) + 1))
    a[:, :3] = los  # the sign of the unknown change of position does not matter
    a[np.arange(len(band)), 3 + band] = 1.0
    best = None
    for left_out in np.unique(sat[clean]):
        rows = clean & (sat != left_out)
        known = a[rows][:, a[rows].any(axis=0)]  # less the clocks of frequencies not there
        if np.linalg.matrix_rank(known) < known.shape[1]:
            continue
        off = np.abs(change - a @ _least_deviations(a[rows], change[rows]))
        fit = clean & (off <= CHANGE_FIT)
        if best is None or fit.sum() > best[0]:
            best = (fit.sum(), off, fit)
    if best is None:
        return None
    _, off, fit = best
    if fit.sum() - a.shape[1] <= clean.sum() - fit.sum():
        return None
    on_band = np.unique(np.stack([band[fit], sat[fit]]), axis=1)[0]  # one per satellite
    judged = np.bincount(on_band, minlength=band.max() + 1)[band] >= CHANGE_MIN_BAND
    return (off > limit) & judged


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
    """Each rover epoch with the latest base epoch not later than it, None beyond `max_age`.
    A rover epoch before the first base epoch is paired with that one when it is within
    `max_age` ahead, as RTKLIB pairs it forward; the age is then negative.

    A base epoch that pairs with no rover epoch still reports its slips: an event
    there (the lost-lock bit set, or the half-cycle bit changed from the base's
    previous phase of that satellite and signal) sets the lost-lock bit in a copy of
    the next paired base epoch that has the phase.

    Both lists must be in time order without a repeated time, as the readers return
    them; ValueError otherwise.
    """
    for name, epochs in (("rover", rover), ("base", base)):
        if np.any(np.diff([_time(e) for e in epochs]) <= 0.0):
            raise ValueError(f"{name} epochs are not in time order, or one is repeated")
    out = []
    j = 0
    half: dict[tuple[str, str], int] = {}  # half-cycle bit of the base's last phase
    events: set[tuple[str, str]] = set()  # slip events of base epoch j
    pending: set[tuple[str, str]] = set()  # events of skipped epochs, not yet carried
    paired = None  # base epoch j as it was paired, once it has been

    def scan(b):
        found = set()
        for i, k in zip(*np.nonzero(np.isfinite(b.carrier_phase)), strict=True):
            key, lli = (b.sats[i], b.signals[k]), int(b.lli[i, k])
            if lli & 1 or (half.get(key, lli) ^ lli) & 2:
                found.add(key)
            half[key] = lli
        return found

    if base:
        events = scan(base[0])
    for r in rover:
        t = _time(r)
        while j + 1 < len(base) and _time(base[j + 1]) <= t + DTTOL:
            if paired is None:
                pending |= events
            j += 1
            events, paired = scan(base[j]), None
        b = None
        # base[j] is later than the rover epoch only while it is the first one
        if base and abs(t - _time(base[j])) <= max_age:
            if paired is None:
                paired = base[j]
                lli = paired.lli.copy()
                for sat, sig in sorted(pending):
                    if sat in paired.sats and sig in paired.signals:
                        i, k = paired.sats.index(sat), paired.signals.index(sig)
                        if isfinite(paired.carrier_phase[i, k]):
                            lli[i, k] |= 1
                            pending.discard((sat, sig))
                if (lli != paired.lli).any():
                    paired = replace(paired, lli=lli)
            b = paired
        out.append((r, b))
    return out


class Rtk:
    """The filter of one baseline; call `step` once per rover epoch, in time order or, with
    `backward`, in reverse."""

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
        if self.opt.slips not in ("restart", "check"):
            raise ValueError(f"slips is 'restart' or 'check', not {self.opt.slips!r}")
        if self.opt.change_limit < 0.0:
            raise ValueError(f"change_limit is a length in m, not {self.opt.change_limit}")
        self.iono = iono
        self.keys: list[
            tuple[str, str]
        ] = []  # bias states in state-vector order after the position
        self.biases: dict[tuple[str, str], _Bias] = {}
        self.gf: dict[
            tuple[str, str], float
        ] = {}  # geometry-free against the system's first signal
        self.lli_rover: dict[tuple[str, str], int] = {}  # each receiver's last indicator
        self.lli_base: dict[tuple[str, str], int] = {}
        self.base_seen: dict[tuple[str, str], float] = {}  # base epoch whose indicator was read
        self.held: set[tuple[str, str]] = set()  # slips of epochs that ended early, to apply
        # every ambiguity there has been a state for, or a slip seen on before it had one
        self.begun: set[tuple[str, str]] = set()
        self.epochs = 0  # epochs that reached the relative solution
        self.asked: set[tuple[str, str]] = set()  # phases in question at this epoch
        self.fresh: dict[tuple[str, str], tuple[float, bool]] = {}  # what a restart begins with
        self.gf_now: dict[tuple[str, str], float] = {}  # geometry-free values of this epoch
        # per phase of the previous epoch that came to the ambiguities: its single-differenced
        # residual (m), the rover's line of sight, whether no receiver's lost-lock bit was
        # on it, and the two receivers' half-cycle bits; and that epoch's rover and base time
        self.residuals: dict[tuple[str, str], tuple] = {}
        self.residuals_at = (0.0, 0.0)
        self.moved: set[tuple[str, str]] = set()  # phases the other satellites contradict now
        self.x = np.zeros(3)
        self.P = np.zeros((3, 3))
        self.last_time: float | None = None

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
        # RTKLIB's lock counter: -minlock for an ambiguity begun again, for one whose
        # phase slipped at this epoch or before it had a state (a flag of either receiver,
        # a held one, a geometry-free step), and for one begun after max_outage solved
        # epochs without it; zero only for a first, unflagged ambiguity of the first epochs
        if key in self.begun or self.epochs > self.opt.max_outage:
            self.biases[key].lock = -self.opt.min_lock
        self.begun.add(key)

    def _index(self, key):
        return 3 + self.keys.index(key)

    def _flagged(self, last, key, lli):
        """Whether one receiver's loss-of-lock indicator marks a slip: value 1 (bit 0) set, or the
        half-cycle bit changed since that receiver's previous indicator (RTKLIB detslp_ll).

        Bit 0 says the phase is not continuous with the one before it in time. A backward
        filter comes to that pair from the later phase, so it reads the bit from the
        indicator it read last and restarts at the earlier one."""
        prev = last.get(key, lli)
        last[key] = lli
        return bool((prev if self.opt.backward else lli) & 1) or bool((prev ^ lli) & 2)

    def _hold_slips(self, rover, base):
        """Keep the slips of an epoch that ends before the ambiguities are looked at: every
        phase either receiver flags (as `_flagged` reads the indicator) waits in `held` for
        the next epoch that reaches `_update_biases`."""
        for i, k in zip(*np.nonzero(np.isfinite(rover.carrier_phase)), strict=True):
            key = (rover.sats[i], rover.signals[k])
            if self._flagged(self.lli_rover, key, int(rover.lli[i, k])):
                self.held.add(key)
        if base is None:
            return
        t_base = _time(base)
        for i, k in zip(*np.nonzero(np.isfinite(base.carrier_phase)), strict=True):
            key = (base.sats[i], base.signals[k])
            if self.base_seen.get(key) != t_base:  # a base epoch reports its slips once
                self.base_seen[key] = t_base
                if self._flagged(self.lli_base, key, int(base.lli[i, k])):
                    self.held.add(key)

    def _age(self):
        self.epochs += 1
        for key in list(self.keys):
            b = self.biases[key]
            b.outc += 1
            b.used_before, b.used = b.used, False
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
        llh = geodetic(xyz)
        lat, _, h = llh
        zhd = zenith_hydrostatic(lat, h)
        lam = np.array([wavelength(sig) for sig in obs.signals])
        out = {}
        for sat, entry in sats.items():
            s = entry[which]
            _, el = azimuth_elevation(xyz, s.position, llh)
            if mask is not None and el < mask:
                continue
            rho, e = geometric_range(xyz, s.position)
            corr = rho - C_LIGHT * s.clock + zhd * nmf_dry(lat, h, el, doy)
            row = obs.sats.index(sat)
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
            self._hold_slips(rover, None)
            return None
        seed = solve_epoch(
            to_epoch(rover, _seed_signal(rover.signals)),
            self.ephs,
            self.iono,
            opt.mask_deg,
        )
        if seed is None:
            self._hold_slips(rover, base)
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
        yr = self._residuals(rover, self.x[:3], sats, 1, doy, None)
        # RTKLIB's selsat takes the satellites above the mask at the base, and its slip
        # and ambiguity bookkeeping runs on all of them; the rover's mask acts on the
        # rows of the update alone (zdres)
        selected = sorted(set(yr) & set(yb))
        common = [sat for sat in selected if yr[sat][3] >= mask]
        if not selected:
            self._hold_slips(rover, base)
            return None
        # outages count only on epochs that reach the relative solution (RTKLIB returns
        # before udbias without a base, beyond the age limit, without a seed or without
        # a common satellite)
        self._age()
        signals = rover.signals
        cols = [base.signals.index(s) if s in base.signals else None for s in signals]
        age = t - _time(base)
        first_sigs = {sig[0]: sig for sig in reversed(signals)}  # first signal of each system

        sd = self._update_biases(signals, cols, selected, yr, yb, tt, _time(base), first_sigs)
        if self.asked:
            self._test_asked(signals, common, sd, yr, age, first_sigs)
        dd = self._double_differences(signals, common, sd, yr, age)
        if dd is None:
            return None
        H, v, R, used = dd
        if not self._update_float(H, v, R, used):
            return None
        ns = len({sat for sat, sig in used if sig == first_sigs[sig[0]]})
        if ns < 4:  # the update stands, the epoch has no solution (RTKLIB SOLQ_NONE)
            return None
        status, ratio, position, ambiguities = self._resolve_ambiguities(signals, common)
        return Fix(
            rover.week,
            rover.tow,
            marker_from_arp(position, self.rover_delta),
            status,
            ns,
            float(ratio),
            age,
            len(ambiguities),
            marker_from_arp(self.x[:3].copy(), self.rover_delta),
            np.sqrt(np.diag(self.P[:3, :3])),
            ambiguities,
        )

    def _update_biases(self, signals, cols, common, yr, yb, tt, t_base, first_sigs):
        """Single differences, slips and ambiguity states (RTKLIB udbias).

        Removes slipped and flags half-cycle states, adds process noise, aligns the
        existing states of each signal to the current single-differenced clock and
        creates the missing ones. Returns the single differences (phase m, code m) by
        (satellite, signal); the code is None where an existing state has a phase alone.

        With `slips="check"` a phase only the rover flags, on an ambiguity updated at the
        previous epoch, is in question instead: left in `asked` for `_test_asked` unless
        the satellite's other signal settles it here. The geometry-free difference of the
        two is then compared with its value at the last epoch neither was in question (or
        at the last restart of either), so a slip spread over flagged epochs is measured
        whole: a step restarts both, as always; without one, a phase in question beside an
        unflagged one is continuous. The first signal's word for another holds only while
        no signal of the satellite steps against the first at this epoch: when one does,
        the phases it settled are in question again. (A step that shows at a later epoch,
        the other signal having had no phase at this one, takes nothing back.) With `change_limit` the word of either holds only while
        neither of the two changed against the other satellites since the previous epoch
        (`_judge_changes`): two signals that move together leave their difference where it
        was, and the phase in question then stays one.
        """
        opt = self.opt
        check = opt.slips == "check"
        self.asked, self.gf_now = set(), {}
        self.moved = set()
        if check and opt.change_limit > 0.0:
            self.moved = self._judge_changes(signals, cols, common, yr, yb, t_base)
        for key in self.held:  # slips of the epochs since the last one that came this far
            self.begun.add(key)
            if key in self.biases:
                self._remove(key)
            # nor can its phase vouch for the satellite's other signal now (a base flag may
            # be of a system the rover's signals do not have)
            if check and key[0][0] in first_sigs:
                self._forget_references(*key, first_sigs[key[0][0]], signals)
        self.held.clear()
        sd = {}  # (sat, signal) -> (phase m, code m)
        halves = {}
        for sat in common:
            yr_p, yr_c, _, _, lli_r = yr[sat]
            yb_p, yb_c, _, _, lli_b = yb[sat]
            ref_sig = first_sigs[sat[0]]
            ref_phase = None
            ref_asked = ref_clear = False  # the first signal: in question, unflagged
            restarted = []
            waiting = []  # in question beside a first signal that was in question too
            vouched = []  # settled on the first signal's word, which a later signal may refute
            for j, sig in enumerate(signals):
                jb = cols[j]
                if jb is None:
                    continue
                key = (sat, sig)
                # backward, RTKLIB takes the half-cycle bit too from the indicators read last
                before = self.lli_rover.get(key, 0) | self.lli_base.get(key, 0)
                late = opt.backward and not check
                # each receiver's indicator counts on its own, whether or not the other
                # receiver has a phase; a base epoch reused for several rover epochs
                # reports its slips once
                slip = isfinite(yr_p[j]) and self._flagged(self.lli_rover, key, int(lli_r[j]))
                asked = check and slip
                if isfinite(yb_p[jb]) and self.base_seen.get(key) != t_base:
                    self.base_seen[key] = t_base
                    if self._flagged(self.lli_base, key, int(lli_b[jb])):
                        slip, asked = True, False
                if slip:
                    self.begun.add(key)
                phase = None
                if isfinite(yr_p[j]) and isfinite(yb_p[jb]):
                    phase = yr_p[j] - yb_p[jb]
                    code = yr_c[j] - yb_c[jb] if isfinite(yr_c[j]) and isfinite(yb_c[jb]) else None
                    now = int(lli_r[j] | lli_b[jb])
                    # (in check mode the bit of the phase itself, in both directions: an
                    # ambiguity begun on a phase is searched or not by that phase's bit)
                    halves[key] = bool((before if late else now) & 2)
                    clear = not slip
                    # in question: its ambiguity was updated at the previous epoch (one
                    # begun there that no row used has nothing to continue)
                    asked = asked and code is not None and key in self.biases
                    asked = asked and self.biases[key].used_before
                    slip = slip and not asked
                    if sig == ref_sig:
                        ref_phase, ref_asked, ref_clear = phase, asked, clear
                    elif ref_phase is not None:  # geometry-free against the first signal
                        gf = ref_phase - phase
                        prev = self.gf.get(key)
                        question = ref_asked or asked
                        if prev is not None and abs(gf - prev) > opt.slip_threshold:
                            slip, asked, ref_asked, question = True, False, False, False
                            ref_clear = False  # which of the two stepped is not known
                            self.begun.update({key, (sat, ref_sig)})
                            self.asked.discard((sat, ref_sig))
                            self.asked.update(vouched)  # nor whether those are continuous
                            vouched.clear()
                            if (sat, ref_sig) in self.biases:
                                self._remove((sat, ref_sig))
                                restarted.append(ref_sig)
                        elif prev is not None and (
                            key in self.moved or (sat, ref_sig) in self.moved
                        ):
                            pass  # the two may have moved together: neither vouches
                        elif prev is not None and ref_asked and clear:
                            ref_asked, ref_clear = False, True
                            self.asked.discard((sat, ref_sig))
                        elif prev is not None and asked and ref_clear:
                            asked = False
                            vouched.append(key)
                        elif prev is not None and asked and ref_asked:
                            waiting.append(key)
                        # the reference waits while a phase of the pair is in question; a
                        # restart of either moves it on below
                        if prev is None or not question:
                            self.gf[key] = gf
                        self.gf_now[key] = gf
                else:
                    asked = False
                if slip and key in self.biases:
                    self._remove(key)
                    restarted.append(sig)
                if asked:
                    self.asked.add(key)
                # a phase without its code keeps the row of the state it has (RTKLIB's
                # validobs asks a phase row for the phases alone); none begins without
                if phase is not None and (code is not None or key in self.biases):
                    sd[key] = (phase, code)
                    if key in self.biases:
                        i = self._index(key)
                        self.P[i, i] += opt.bias_noise**2 * abs(tt)
            if ref_clear:  # the first signal was settled after these were looked at
                self.asked.difference_update(waiting)
            if check:
                for sig in restarted:
                    self._forget_references(sat, sig, ref_sig, signals)
        # a phase without its code rested on the ambiguity it had: where a step of the
        # satellite's other signal took that away, there is no code to begin another on
        for key in [k for k, v in sd.items() if v[1] is None and k not in self.biases]:
            del sd[key]
        # align existing states to the current single-differenced clock, create the rest
        self.fresh = {}
        for sig in signals:
            lam = wavelength(sig)
            news = {
                k: (v[0] - v[1]) / lam for k, v in sd.items() if k[1] == sig and v[1] is not None
            }
            existing = [k for k in news if k in self.biases]
            # a phase in question may have slipped: it follows the others, it does not lead.
            # With every one in question there is nothing to align to: `_test_asked` finds
            # none to difference against and restarts them, and the states of satellites
            # without a phase now stay where they are
            existing = [k for k in existing if k not in self.asked]
            if existing:
                offset = np.mean([news[k] - self.x[self._index(k)] for k in existing])
                for k in self.keys:
                    if k[1] == sig:
                        self.x[self._index(k)] += offset
            for k in [k for k in sd if k[1] == sig]:
                created = k not in self.biases
                if created:
                    self._add(k, news[k], opt.bias_sigma**2)
                # a phase that continues through a flag keeps an established half cycle
                established = check and not created and not self.biases[k].half
                self.biases[k].half = halves[k] and not established
                if k in self.asked:
                    self.fresh[k] = (news[k], halves[k])
        return sd

    def _judge_changes(self, signals, cols, common, yr, yb, t_base) -> set[tuple[str, str]]:
        """The phases whose single-differenced residual changed by more than `change_limit`
        against the other satellites since the previous epoch that came this far (`_moved`);
        empty without a verdict. Keeps this epoch's residuals for the next.

        No verdict either when the two epochs are more than CHANGE_MAX_STEP apart, or when
        the base epoch changed by another step than the rover's (a base at a lower rate:
        the base's own residuals over that step are then in the change; between its epochs
        the base's part stands still and the verdict is the rover's)."""
        now = {}
        for sat in common:
            for j, sig in enumerate(signals):
                jb = cols[j]
                if jb is None:
                    continue
                residual = yr[sat][0][j] - yb[sat][0][jb]
                if isfinite(residual):
                    # a flag held from an epoch that ended early lies between the two epochs
                    lli = (int(yr[sat][4][j]), int(yb[sat][4][jb]))
                    clean = not (lli[0] | lli[1]) & 1 and (sat, sig) not in self.held
                    now[(sat, sig)] = (residual, yr[sat][2], clean, (lli[0] & 2, lli[1] & 2))
        at = (self.last_time, t_base)
        (before, before_at), self.residuals, self.residuals_at = (
            (self.residuals, self.residuals_at),
            now,
            at,
        )
        step, base_step = at[0] - before_at[0], at[1] - before_at[1]
        keys = [key for key in now if key in before]
        if (
            not keys
            or abs(step) > CHANGE_MAX_STEP
            or min(abs(base_step), abs(base_step - step)) > DTTOL
        ):
            return set()
        lams = sorted({wavelength(sig) for _, sig in keys})
        sats = sorted({sat for sat, _ in keys})
        verdict = _moved(
            np.array([now[key][0] - before[key][0] for key in keys]),
            np.array([now[key][1] for key in keys]),
            np.array([lams.index(wavelength(sig)) for _, sig in keys]),
            np.array([sats.index(sat) for sat, _ in keys]),
            # clean: no lost-lock bit at either epoch, no change of a half-cycle bit between
            np.array([now[k][2] and before[k][2] and now[k][3] == before[k][3] for k in keys]),
            self.opt.change_limit,
        )
        return (
            set() if verdict is None else {key for key, m in zip(keys, verdict, strict=True) if m}
        )

    def _forget_references(self, sat, sig, ref_sig, signals):
        """After a restart of one signal of a satellite: the geometry-free references it
        is part of that this epoch did not measure anew are of no use."""
        others = (
            [s for s in signals if s[0] == sat[0] and s != ref_sig] if sig == ref_sig else [sig]
        )
        for other in others:
            if (sat, other) in self.gf_now:
                self.gf[(sat, other)] = self.gf_now[(sat, other)]
            else:
                self.gf.pop((sat, other), None)

    def _test_asked(self, signals, common, sd, yr, age, first_sigs):
        """Decide the phases in question against the rest of the epoch.

        A trial update on copies of the state, with every code and the phases not in
        question, says where the rover is relative to the ambiguities that are trusted.
        Each phase in question is double-differenced against the highest satellite of its
        signal that is not: it continues when that innovation is within CHECK_INNOVATION
        and its predicted standard deviation within CHECK_SIGMA; otherwise, or with
        nothing to difference against, its ambiguity restarts on this phase. The epoch's
        one update then runs on everything as usual.
        """
        keep = set()
        trial = self._double_differences(signals, common, sd, yr, age, self.asked)
        if trial is not None:
            H, v, R, _ = trial
            try:
                K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
            except np.linalg.LinAlgError:
                K = None
            if K is not None:
                dx = K @ v
                P = (np.eye(len(self.x)) - K @ H) @ self.P
                for sat, sig in sorted(self.asked):
                    sure = [
                        s for s in common
                        if s[0] == sat[0] and (s, sig) in sd and (s, sig) not in self.asked
                    ]  # fmt: skip
                    if not sure:
                        continue
                    ref = max(sure, key=lambda s: yr[s][3])
                    lam = wavelength(sig)
                    i, j = self._index((sat, sig)), self._index((ref, sig))
                    h = np.zeros(len(self.x))
                    h[:3] = -yr[sat][2] + yr[ref][2]
                    h[i], h[j] = lam, -lam
                    res = sd[(sat, sig)][0] - sd[(ref, sig)][0] - lam * (self.x[i] - self.x[j])
                    noise = self._variance(yr[sat][3], age, False) + self._variance(
                        yr[ref][3], age, False
                    )
                    innovation = (res - h @ dx) / lam
                    sigma = sqrt(h @ P @ h + noise) / lam
                    if abs(innovation) <= CHECK_INNOVATION and sigma <= CHECK_SIGMA:
                        keep.add((sat, sig))
        for key in sorted(self.asked - keep):
            value, half = self.fresh[key]
            self._remove(key)
            self._add(key, value, self.opt.bias_sigma**2)
            self.biases[key].half = half
            self._forget_references(*key, first_sigs[key[0][0]], signals)
        self.asked = set()

    def _double_differences(self, signals, common, sd, yr, age, without=frozenset()):
        """Double-differenced phase and code against the highest satellite per system
        (RTKLIB ddres): design matrix, innovations, covariance and the (satellite, signal)
        states used, those of the phase rows that were kept (RTKLIB's vsat: lock, outage
        and satellite counts follow them); None when no difference can be formed. The
        phases of `without` are left out."""
        opt = self.opt
        rows, v, blocks = [], [], []
        used = set()
        n = len(self.x)
        for system in sorted({s[0] for s in common}):
            for kind in ("phase", "code"):
                for sig in signals:
                    if sig[0] != system:
                        continue
                    valid = [sat for sat in common if sat[0] == system and (sat, sig) in sd]
                    if kind == "phase" and without:
                        valid = [sat for sat in valid if (sat, sig) not in without]
                    if kind == "code":
                        valid = [sat for sat in valid if sd[(sat, sig)][1] is not None]
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
                        if kind == "phase":  # a code row does not use the ambiguities
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
        return H, v, R, used

    def _update_float(self, H, v, R, used):
        """The measurement update and the lock counters (RTKLIB filter); False, with the
        state untouched, when the innovation covariance is singular."""
        n = len(self.x)
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        self.x = self.x + K @ v
        self.P = (np.eye(n) - K @ H) @ self.P
        for key in used:
            b = self.biases[key]
            b.lock += 1
            b.outc = 0
            b.used = True
        return True

    def _resolve_ambiguities(self, signals, common):
        """LAMBDA on the double-differenced ambiguities with the ratio test (RTKLIB
        resamb_LAMBDA, continuous mode): status, ratio, position and the eligible
        ambiguities, searched from two on and only when they span `min_fix_sats`
        satellites, one reference per system counted once. The float states are kept
        whatever the outcome."""
        opt = self.opt
        n = len(self.x)
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
        ambiguities = ()
        if nb >= 1:
            D = np.zeros((nb, n))
            for r, (ref, key) in enumerate(pairs):
                D[r, self._index(ref)] = 1.0
                D[r, self._index(key)] = -1.0
            y = D @ self.x
            Qb = D @ self.P @ D.T
            fixed = None
            sats = {key[0] for pair in pairs for key in pair}
            # every system has its own reference: five satellites of one system are four
            # beside it, and that is what a floor of five asks of several systems together
            beside = len(sats) - len({sat[0] for sat in sats})
            if nb >= 2 and beside >= opt.min_fix_sats - 1:
                Qab = self.P[:3] @ D.T
                try:
                    cands, s = lambda_fix(y, Qb, 2)
                    ratio = 0.0 if s[0] <= 0.0 else min(s[1] / s[0], 999.9)
                    if s[0] <= 0.0 or ratio >= opt.ratio_threshold:
                        position = self.x[:3] - Qab @ np.linalg.solve(Qb, y - cands[0])
                        status = "fix"
                        fixed = cands[0]
                except (ValueError, np.linalg.LinAlgError):
                    ratio = 0.0
            std = np.sqrt(np.diag(Qb))
            ambiguities = tuple(
                Ambiguity(
                    ref[0], key[0], key[1], float(y[r]), float(std[r]),
                    None if fixed is None else round(fixed[r]),
                )
                for r, (ref, key) in enumerate(pairs)
            )  # fmt: skip
        return status, ratio, position, ambiguities

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
    """Fix every rover epoch that has one, pairing it with the latest base epoch. `rover` and
    `base` in time order without a repeated time (ValueError otherwise). With
    `options.backward` the filter runs from the last epoch to the first over the same
    pairs; the fixes come back in time order either way."""
    rtk = Rtk(ephs, base_station, rover_delta_enu, options, iono)
    pairs = pair_epochs(rover, base, rtk.opt.max_age)
    if rtk.opt.backward:
        pairs.reverse()
    out = []
    for r, b in pairs:
        fix = rtk.step(r, b)
        if fix is not None:
            out.append(fix)
    if rtk.opt.backward:
        out.reverse()
    return out
