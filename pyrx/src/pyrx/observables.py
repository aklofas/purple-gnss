"""Pseudorange, carrier-phase, Doppler and C/N0 observables at receiver epochs.

Transmit time: every tracking epoch integrates exactly one code period, so once a
subframe has tagged one epoch with its GPS time, the code start of epoch k was
transmitted at `tow_s + (k - sf.epoch) * CA_PERIOD` for every k. All subframes of
a channel must agree on that mapping.

Receiver time: one clock for the whole recording. Sample 0 is assigned the GPS
time `rx_time0`, the latest transmit time seen at sample 0 plus a nominal travel
time, so pseudoranges come out near their true values and the remaining clock
bias is left to the position solution. Receiver epochs are the instants at which
`rx_time0 + T` is a multiple of the interval; each channel is sampled there by
interpolating between the two code starts that bracket the epoch.

Conventions (RINEX): pseudorange `C_LIGHT * (t_rx - t_tx)`; carrier phase is the
negated accumulated Doppler phase (phase grows with range), less half a cycle
while the Costas loop is locked on inverted data; Doppler is positive for an
approaching satellite, as tracking reports it.

Carrier validity: an epoch whose 1 s window is not phase-locked (the channel's
`phase_lock` below `PLI_LOCKED`, or unknown) has no carrier phase (nan, a blank in
RINEX), as receivers that report a phase-lock flag withhold the phase; its
pseudorange, Doppler and C/N0 stay. RINEX loss-of-lock indicator (value 1, bit 0,
after Pocket SDR's rules): on a channel's first observation and on those within its
first 2 s (a channel's carrier phase starts at its own ambiguity, so a satellite
re-acquired after a gap begins with a slip), on the first observation with a phase
after any epoch without phase lock since the previous one with a phase (an unlocked
window can lie wholly between sparse observations), on an observation when some
`FADE_SPAN` that ends at or after the epoch of the channel's previous observation with a
phase, and before its own epoch, contains `FADE_SLIP` or more of the carrier loop's
integrator hold (a fade: the signal's phase jumps in one and the Costas loop settles on
the nearest half cycle, with every window still reading phase-locked; the flag outlasts
the fade by up to the span, while the loop settles, and says a slip was possible: most
phases that carry it are continuous; a held epoch counts for the phases after its own,
which is the loop's state before that epoch's update), and on the first observation
after the data polarity changed,
that is after a half-cycle slip of the Costas loop; value 2 (bit 1) where the polarity is not
established by a decoded subframe: before the first subframe, from the end of
the last subframe before a polarity change to the start of the first subframe
after it, and where fades leave it open (`_half_cycle_open`): from the first to the last
fade between two subframes of one polarity, and from a fade after the last subframe on. The bits are kept on an observation without a phase too, so
the slip is reported on the next observation that has one.

A pilot channel of a second band on the same sample clock (GPS L5 next to L1) decodes
no time of its own. Every one of its epochs is one code period too, so it needs the
transmit time of one code start, and it takes that from the same satellite's L1
channel: both signals leave the satellite together and arrive within a fraction of a
microsecond of each other, so the L1 transmit time at a pilot code start, rounded to
the 1 ms code period, is the pilot's (`pilot_transmit_times`). Its observations begin
once the secondary code is found. Its carrier phase has no half-cycle ambiguity; where
the tracker settled on the inverted pilot the reported phase gets the half cycle back,
and the quarter cycle that aligns L5Q with the band's reference signal L5I (RINEX 3.05
Table A23) is applied here, as a receiver does. The pilot's loop coasts through an outage
on an extrapolation and can return whole cycles away with its window indicator still
high. It returns with the first sum it takes after the coasted ones, and that update
is in the phases after the epoch that completes the sum, so every phase from a coasted
sum to the first one that holds the return is flagged as a slip. A pilot can also slip a
whole cycle with its sums coherent enough to be taken. In a fade null the received phase
can turn a whole cycle within a few sums, the loop following a third of a cycle or more
behind; or it runs past half a cycle from the loop, where the four-quadrant phase of a sum
wraps, and the loop pulls the other way. Either way a sum stands far from the loop's own
phase, so a phase after a sum the loop took at `NEAR_WRAP` or more is flagged as a slip
too. A slip the loop follows with a small phase error (a reflection that outweighs the
direct path for a moment) shows in neither rule.

A pilot whose secondary code lasts 100 ms (Galileo E5a, and Galileo E1, whose 4 ms code
the tracker follows in 1 ms epochs) needs no other channel for its time: the code start
that carries secondary chip 0 left the satellite at a multiple of 100 ms, and the receiver
clock says which (`_secondary_time0`). Its phase is aligned as its signal description
says. Two such pilots of one satellite, tracked independently, must name the same
transmit time; where they stand more than `PILOT_AGREEMENT` apart neither is reported.
"""

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from pyrx.bitsync import EPOCHS_PER_BIT
from pyrx.codes import CA_PERIOD
from pyrx.gpstime import C_LIGHT, WEEK_S, full_week
from pyrx.lnav import PREAMBLE, SUBFRAME_BITS
from pyrx.navdecode import LnavResult
from pyrx.observations import Epoch, Observation, Observations
from pyrx.signalspec import SIGNALS
from pyrx.tracking import PLI_LOCKED, T_PILOT_COH, TrackResult

TRAVEL_TIME_NOMINAL = 0.068  # s, light time to the nearest satellite, sets the receiver clock
T_LOCK = 2.0  # s into a channel over which its observations carry the slip flag
SUBFRAME_EPOCHS = SUBFRAME_BITS * EPOCHS_PER_BIT
FADE_SLIP = 0.050  # s of integrator hold within FADE_SPAN after which a phase is flagged
FADE_SPAN = 0.5  # s
LLI_SLIP = 1  # RINEX loss-of-lock indicator bits
LLI_HALF_CYCLE = 2
# s a pilot's code start may lie from the L1 channel's code period grid: the two bands'
# paths through a front end differed by 0.1 microsecond on the two front ends measured, and
# a pilot locked on something else lands anywhere in the millisecond
ANCHOR_TOLERANCE = 1e-6
# s: a pilot whose secondary code is at least this long takes its time from it. The travel
# times of the satellites in view span some 30 ms, and the receiver clock is right within 20
SECONDARY_TIME = 0.100
# s of the receiver clock: the lower end of the code length in which such a pilot's travel
# time is taken. The clock is the latest GPS transmit time at sample 0 plus
# TRAVEL_TIME_NOMINAL: a GPS satellite is 67 to 86 ms away, so the clock is up to 18 ms
# behind true time and 1 ms ahead at most. A Galileo satellite is 77 to 97 ms away, the two
# in eccentric orbits 57 to 106 ms: 39 to 107 ms by that clock. The window of 22 to 122 ms
# leaves some 15 ms either side
TRAVEL_TIME_MIN = 0.022
# s two pilots of one satellite that each took their time from their own secondary code
# may stand apart in transmit time at one instant: the bands' paths through a front end and
# the ionosphere make a tenth of a microsecond, a secondary code found wrongly a millisecond
# or more
PILOT_AGREEMENT = 1e-6
# epochs from the one completing a pilot's coasted sum to the one completing the next sum,
# whose update is in the phases after it
COAST_RETURN = round(T_PILOT_COH / CA_PERIOD)
# cycles: a pilot's coherent sum whose phase against the NCO is this large or larger is near
# the wrap of the four-quadrant discriminator at half a cycle, 0.15 cycle from it at most.
# The whole-cycle slips of the car-roof captures passed sums of 0.41 to 0.46 cycle; a steady
# signal has none over 0.35 down to 24 dB-Hz
NEAR_WRAP = 0.35


def code_starts(track: TrackResult) -> np.ndarray:
    """Receiver time (s from the source start) at which every epoch's 1 ms of code starts: a
    code period, or a quarter of Galileo E1's."""
    off = (track.code_offset - track.t + CA_PERIOD / 2) % CA_PERIOD - CA_PERIOD / 2
    return track.t + off


def transmit_times(track: TrackResult, lnav: LnavResult) -> np.ndarray:
    """GPS seconds of week at which every epoch's code start was transmitted.

    Counted in the week of the channel's first subframe: subframes decoded after a
    week rollover continue past 604800 rather than restarting at zero.
    """
    if not lnav.subframes:
        raise ValueError("no subframe decoded")
    t0 = np.array([sf.tow_s - sf.epoch * CA_PERIOD for sf in lnav.subframes])
    t0 += WEEK_S * np.round((t0[0] - t0) / WEEK_S)
    if t0.max() - t0.min() > 1e-9:
        raise ValueError("inconsistent subframe time tags")
    return t0[0] + np.arange(len(track.t)) * CA_PERIOD


def _polarities(lnav: LnavResult) -> tuple[np.ndarray, np.ndarray]:
    """First-bit epoch of every decoded subframe and whether it came with the data
    polarity opposite to the one frame sync settled on."""
    starts = np.array([sf.epoch for sf in lnav.subframes], dtype=int)
    flipped = np.array([not np.array_equal(sf.bits[:8], PREAMBLE) for sf in lnav.subframes])
    return starts, flipped


def _usable(tracks, lnavs) -> list[tuple[TrackResult, LnavResult, np.ndarray, np.ndarray]]:
    """Channels with consistent time tags, each with its code starts and transmit times.

    Every channel's transmit times are put in one week, the one in which the
    signals at sample 0 were transmitted, so a recording that crosses the week
    boundary counts on past 604800 for all channels, whichever side each decoded
    its first subframe on.
    """
    out = []
    for tr, ln in zip(tracks, lnavs):
        try:
            tt = transmit_times(tr, ln)
        except ValueError:
            continue
        out.append((tr, ln, code_starts(tr), tt))
    if out:
        at0 = [tt[0] - tc[0] for _, _, tc, tt in out]  # transmit time seen at sample 0
        shifts = WEEK_S * np.round((at0[0] - np.array(at0)) / WEEK_S)
        shifts -= WEEK_S * floor(min(at0 + shifts) / WEEK_S)
        out = [(tr, ln, tc, tt + d) for (tr, ln, tc, tt), d in zip(out, shifts)]
    return out


def receiver_time0(tracks: list[TrackResult], lnavs: list[LnavResult]) -> float:
    """GPS seconds of week assigned to sample 0 of the recording.

    In the week in which the signals at sample 0 were transmitted;
    `make_observables` moves the origin on by a week when every observation epoch
    falls past the boundary.
    """
    chans = _usable(tracks, lnavs)
    if not chans:
        raise ValueError("no channel with a decoded subframe")
    return _time0(chans)


def _time0(chans) -> float:
    t_tx0 = [tt[0] - tc[0] for _, _, tc, tt in chans]
    return round(max(t_tx0) + TRAVEL_TIME_NOMINAL, 3)


def _carrier_state(ln: LnavResult, polarity, k: int, last_k: int) -> tuple[bool, int]:
    """Data polarity in effect at epoch k and the loss-of-lock bits it earns.

    The polarity is that of the latest subframe begun at or before k (frame sync's
    before the first). A subframe of the other polarity means the Costas loop
    slipped half a cycle somewhere between the end of the previous subframe and
    the start of that one: the epochs in between may be half a cycle off, and the
    first observation after any such change since the channel's previous
    observation (at epoch `last_k`, -1 for none) is a slip.
    """
    starts, flipped = polarity
    j = int(np.searchsorted(starts, k, side="right")) - 1
    if j < 0:
        return bool(ln.stream.inverted), LLI_HALF_CYCLE
    lli = 0
    if j + 1 < len(starts) and flipped[j + 1] != flipped[j] and k >= starts[j] + SUBFRAME_EPOCHS:
        lli |= LLI_HALF_CYCLE
    if last_k >= 0:
        j_prev = int(np.searchsorted(starts, last_k, side="right")) - 1
        before = flipped[j_prev] if j_prev >= 0 else False
        if np.any(flipped[j_prev + 1 : j + 1] != before):
            lli |= LLI_SLIP
    return bool(ln.stream.inverted) != bool(flipped[j]), lli


def _faded(tr: TrackResult) -> np.ndarray | None:
    """Per epoch, whether the carrier loop's integrator was held for FADE_SLIP or more of the
    FADE_SPAN ending with it; None for a track without the record."""
    if tr.held is None:
        return None
    count = np.cumsum(tr.held)
    span = round(FADE_SPAN / CA_PERIOD)
    count[span:] -= count[:-span].copy()
    return count >= round(FADE_SLIP / CA_PERIOD)


def _coasting(tr: TrackResult) -> np.ndarray | None:
    """Per epoch of a pilot, whether a coasted sum completed within the COAST_RETURN epochs up
    to it, so that its phase may not hold the loop's return yet; None for a track without
    the record."""
    if tr.held is None:
        return None
    count = np.cumsum(tr.held)
    count[COAST_RETURN + 1 :] -= count[: -COAST_RETURN - 1].copy()
    return count > 0


def _near_wrap(tr: TrackResult) -> np.ndarray:
    """Per epoch of a pilot, whether it completes a coherent sum the loop took whose phase
    against the NCO is NEAR_WRAP or more in size. Up to the epoch of the secondary sync the
    record holds other discriminators, and a sum the loop coasted on has no phase."""
    marks = np.zeros(len(tr.t), dtype=bool)
    if tr.sec_sync is not None:
        sums = tr.carrier_error[tr.sec_sync + 1 :]
        marks[tr.sec_sync + 1 :] = np.abs(sums) >= NEAR_WRAP
    return marks


def _fades(tr: TrackResult) -> list[tuple[int, int, int]]:
    """The fades of a track as (first held, last held, last flagged) epochs: every run of
    epochs `_faded` marks, begun at the first held epoch of the span that made it. A slip in
    a fade lies between its first and last held epoch; the flag outlasts the hold."""
    faded = _faded(tr)
    if faded is None or not faded.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], faded, [False]))))
    span = round(FADE_SPAN / CA_PERIOD)
    out = []
    for a, b in zip(edges[::2], edges[1::2] - 1, strict=True):
        lo = max(int(a) - span + 1, 0)
        held = lo + np.flatnonzero(tr.held[lo : b + 1])
        out.append((int(held[0]), int(held[-1]), int(b)))
    return out


def _half_cycle_open(tr: TrackResult, polarity) -> np.ndarray | None:
    """Per epoch, whether fades leave the half cycle of its phase open; None for a track
    without the hold record.

    A slip of half a cycle in a fade shows as a subframe of the other polarity, which
    `_carrier_state` reads. Two such slips between two decoded subframes do not: the
    polarity is back where it was, and the phases between the slips are half a cycle off.
    So where fades lie between the end of one subframe and the start of the next of the
    same polarity, the half cycle is open from the first held epoch of the first fade to
    the end of the last one's flag (one long fade can hold both slips); and from a fade
    after the last subframe to the end of the track, for nothing decoded after it says
    where the loop settled. A subframe that decoded held its polarity throughout, so a
    fade whose held epochs all lie under decoded subframes is a slip of whole cycles at
    most.
    """
    if tr.held is None:
        return None
    starts, flipped = polarity
    is_open = np.zeros(len(tr.t), dtype=bool)
    fades = _fades(tr)
    for j in range(len(starts)):
        lo = starts[j] + SUBFRAME_EPOCHS  # the first epoch after subframe j
        last = j + 1 == len(starts)
        hi = len(tr.t) if last else starts[j + 1]
        # the fades with a held epoch in the gap, which is empty between subframes decoded
        # back to back
        between = [f for f in fades if tr.held[max(f[0], lo) : min(f[1] + 1, hi)].any()]
        if last and between:
            is_open[between[0][0] + 1 :] = True
        elif not last and flipped[j + 1] == flipped[j] and between:
            is_open[between[0][0] + 1 : between[-1][2] + 1] = True
    return is_open


def _weeks_on(first: int, interval: float) -> int:
    """Whole weeks between the time origin's week and the first epoch (index `first` on the
    grid): 1 when every epoch lies past the week boundary, so that `tow` restarts."""
    return floor(first * interval / WEEK_S)


@dataclass(frozen=True)
class Run:
    """What the observations of one recording share, made once by `prepare`: the L1 C/A
    channels with a time tag, each as (track, decode, code starts, transmit times) with the
    transmit times in one week; the receiver clock; and the epoch grid."""

    chans: list[tuple[TrackResult, LnavResult, np.ndarray, np.ndarray]]
    rx0: float  # GPS seconds of week assigned to sample 0 (`receiver_time0`)
    interval: float  # s; epoch i of the grid is the receiver time i * interval
    first: int  # the grid's first and last index inside the channels' span
    last: int
    weeks_on: int  # whole weeks from the week of `rx0` to the first epoch
    origin_week: int | None  # the week of `rx0` from a decoded subframe 1, None without one

    def week(self, week: int | None) -> int | None:
        """The week of the first epoch: the one given, else the decoded one, else None."""
        if week is None and self.origin_week is not None:
            return self.origin_week + self.weeks_on
        return week


def prepare(
    tracks: list[TrackResult], lnavs: list[LnavResult], interval: float = 1.0
) -> Run | None:
    """The receiver clock and the epoch grid of a recording from its L1 C/A channels; None
    when no channel has a consistent time tag. Raises ValueError for a track of another
    signal than L1 C/A."""
    for tr in tracks:
        if tr.signal != "L1CA":
            raise ValueError(f"PRN {tr.prn}: {tr.signal} is not an L1 C/A track")
    chans = _usable(tracks, lnavs)
    if not chans:
        return None
    # the week of the run's time origin: a subframe 1 decoded past the rollover carries
    # the next week's number
    wn = [
        full_week(sf.fields["wn"]) - int(tt[sf.epoch] // WEEK_S)
        for _, ln, _, tt in chans
        for sf in ln.subframes
        if sf.sf_id == 1
    ]
    rx0 = _time0(chans)
    t_begin = min(tc[int(ln.stream.epoch[0])] for _, ln, tc, _ in chans)
    t_end = max(tc[-1] for _, _, tc, _ in chans)
    first = ceil((rx0 + t_begin) / interval)
    last = floor((rx0 + t_end) / interval)
    return Run(chans, rx0, interval, first, last, _weeks_on(first, interval), wn[0] if wn else None)


def make_observables(
    tracks: list[TrackResult],
    lnavs: list[LnavResult],
    interval: float = 1.0,
    week: int | None = None,
) -> list[Epoch]:
    """Observations of every channel with a decoded subframe at a common epoch grid.

    A channel is used from its first bit-synced epoch on; channels without a
    subframe, or with subframes that disagree on the time tag, are skipped.
    `week` is the week of the first observation epoch; it defaults to the one
    implied by any decoded subframe 1 and stays None when there is none. Raises ValueError
    for a track of another signal than L1 C/A.
    """
    run = prepare(tracks, lnavs, interval)
    return [] if run is None else l1_observables(run, week)


@dataclass
class _Sampled:
    """A segment read at one receiver time."""

    k: int  # the epoch whose code start is the last before it
    f: float  # how far it lies from that code start to the next, 0 to 1
    adr: float  # cycles, the accumulated Doppler phase there
    locked: bool  # epoch k's window is phase-locked
    slip: bool  # a first observation, one in the segment's first T_LOCK, or a window without
    # phase lock since the segment's previous phase


class _Segment:
    """A track segment sampled at receiver times in order: the bracketing of a time by two
    code starts, the interpolation between them, and the ledger of the previous observation
    and the previous one with a phase, which the loss-of-lock rules of both signals read."""

    def __init__(self, tr: TrackResult, starts: np.ndarray, k_min: int):
        self.tr, self.starts, self.k_min = tr, starts, k_min
        self.last_k = -1  # epoch of the previous observation
        self.last_phase_k = -1  # epoch of the previous observation with a phase

    def at(self, t: float) -> _Sampled | None:
        """The segment at `t` (s from the source start); None before epoch `k_min` or
        outside the segment."""
        tr, tc = self.tr, self.starts
        k = int(np.searchsorted(tc, t, side="right")) - 1
        if k < self.k_min or k + 1 >= len(tc):
            return None
        f = (t - tc[k]) / (tc[k + 1] - tc[k])
        # carrier_phase[k-1] is the accumulated Doppler phase at tc[k], the end of epoch k-1
        adr = tr.carrier_phase[k - 1] + f * (tr.carrier_phase[k] - tr.carrier_phase[k - 1])
        # unlocked (or unknown, nan) at this epoch or at any epoch since the previous
        # observation with a phase: an unlocked window can lie between sparse observations
        # (the slice is empty when several observations fall in one tracking epoch)
        since_phase = tr.phase_lock[self.last_phase_k + 1 : k + 1] >= PLI_LOCKED
        locked = bool(tr.phase_lock[k] >= PLI_LOCKED)
        slip = tr.t[k] - tr.t[0] <= T_LOCK or self.last_k < 0 or not since_phase.all()
        return _Sampled(k, f, adr, locked, bool(slip))

    def since_phase(self, marks: np.ndarray | None, k: int) -> bool:
        """Whether a per-epoch mark (a fade, a coasted sum, a sum near the wrap) is set
        between the previous observation with a phase and epoch k: an event at an epoch is
        the business of the phases after it, the phase of epoch k being the loop's state
        before its update."""
        return marks is not None and bool(marks[max(self.last_phase_k, 0) : k].any())

    def observed(self, s: _Sampled) -> None:
        self.last_k = s.k
        if s.locked:
            self.last_phase_k = s.k


def l1_observables(run: Run, week: int | None = None) -> list[Epoch]:
    """`make_observables` on a prepared run."""
    rx0, interval, weeks_on = run.rx0, run.interval, run.weeks_on
    first, last = run.first, run.last
    week = run.week(week)
    per = [(tr, ln, tc, tt, int(ln.stream.epoch[0])) for tr, ln, tc, tt in run.chans]
    polarity = [_polarities(ln) for _, ln, _, _, _ in per]
    faded = [_faded(tr) for tr, _, _, _, _ in per]
    half_open = [_half_cycle_open(tr, pol) for (tr, *_), pol in zip(per, polarity, strict=True)]
    segments = [_Segment(tr, tc, max(k0, 1)) for tr, _, tc, _, k0 in per]
    epochs = []
    for i in range(first, last + 1):
        t = i * interval - rx0
        t_rx = i * interval  # in the week of the time origin, as the transmit times are
        tow = t_rx - weeks_on * WEEK_S
        obs = []
        for c, (tr, ln, _, tt, _) in enumerate(per):
            seg = segments[c]
            at = seg.at(t)
            if at is None:
                continue
            k = at.k
            t_tx = tt[k] + at.f * CA_PERIOD
            inverted, lli = _carrier_state(ln, polarity[c], k, seg.last_k)
            if at.slip or seg.since_phase(faded[c], k):
                lli |= LLI_SLIP
            if half_open[c] is not None and half_open[c][k]:
                lli |= LLI_HALF_CYCLE
            phase = -(at.adr + (0.5 if inverted else 0.0)) if at.locked else np.nan
            obs.append(
                Observation(
                    tr.prn,
                    C_LIGHT * (t_rx - t_tx),
                    float(phase),
                    float(tr.doppler[k]),
                    float(tr.cn0[k]),
                    lli,
                )
            )
            seg.observed(at)
        if obs:
            epochs.append(Epoch(week, float(tow), sorted(obs, key=lambda o: o.prn)))
    return epochs


def _pilot_time0(starts: np.ndarray, l1: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """Transmit time of a pilot segment's epoch 0 from its code starts and the (code starts,
    transmit times) of the L1 segments of its satellite; None without a consistent anchor.

    At every pilot code start inside an L1 segment the L1 transmit time, rounded to the code
    period, is that code start's transmit time. All of them must name the same time for
    epoch 0, each within ANCHOR_TOLERANCE of the grid.
    """
    origins = []
    for tc, tt in l1:
        k = np.nonzero((starts >= tc[0]) & (starts < tc[-1]))[0]
        if not len(k):
            continue
        j = np.searchsorted(tc, starts[k], side="right") - 1
        t_tx = tt[j] + (starts[k] - tc[j]) / (tc[j + 1] - tc[j]) * CA_PERIOD
        periods = np.rint(t_tx / CA_PERIOD)
        if np.max(np.abs(t_tx - periods * CA_PERIOD)) > ANCHOR_TOLERANCE:
            return None
        origins.append(periods - k)
    if not origins:
        return None
    origins = np.concatenate(origins)
    if np.any(origins != origins[0]):
        return None
    return float(origins[0]) * CA_PERIOD


def _secondary_time0(tr: TrackResult, starts: np.ndarray, rx0: float, period: float) -> float:
    """Transmit time of a pilot segment's epoch 0 from its own secondary code of `period` s.

    The code start that carries secondary chip 0 was transmitted at a multiple of the
    period, so epoch 0's code start left `sec_origin` code periods before one. Which multiple
    follows from the receiver time `rx0 + starts[0]` at which it arrived: the one that puts
    the travel time in [TRAVEL_TIME_MIN, TRAVEL_TIME_MIN + period).
    """
    travel = rx0 + starts[0] + tr.sec_origin * CA_PERIOD  # plus a whole number of periods
    multiple = floor((travel - TRAVEL_TIME_MIN) / period)
    return (multiple * round(period / CA_PERIOD) - tr.sec_origin) * CA_PERIOD


def pilot_transmit_times(
    track: TrackResult, l1: list[tuple[TrackResult, LnavResult]]
) -> np.ndarray | None:
    """Seconds of week at which every code start of a pilot segment was transmitted, from
    the L1 segments (with their decodes) of the same satellite on the same sample clock, in
    the week those segments alone put the signals at sample 0 in; None when no L1 segment
    with a time tag overlaps it or the overlap is not consistent."""
    chans = _usable([tr for tr, _ in l1], [ln for _, ln in l1])
    time0 = _pilot_time0(code_starts(track), [(tc, tt) for _, _, tc, tt in chans])
    return None if time0 is None else time0 + np.arange(len(track.t)) * CA_PERIOD


def make_joint_observables(
    tracks: list[TrackResult],
    lnavs: list[LnavResult],
    pilots: list[TrackResult],
    interval: float = 1.0,
    week: int | None = None,
    pilot_signals: tuple[str, ...] = ("L5",),
) -> list[Observations]:
    """The L1 C/A observations of `make_observables` with the pilot segments `pilots` of the
    same recording time beside them: signals `G1C` and those the pilots of `pilot_signals`
    are reported as (`G5Q` for L5, `E1C` for E1, `E5Q` for E5A), in that order.

    The epochs, the receiver clock and every L1 value are `make_observables`'. A pilot
    segment contributes from the epoch after its secondary code was found. An L5 pilot is on
    the time of its satellite's L1 channel, and one without a consistent anchor contributes
    nothing; a pilot with a secondary code of 100 ms takes its time from that code and the
    receiver clock, and where two such pilots of one satellite stand more than
    PILOT_AGREEMENT apart in transmit time, neither is reported. Raises ValueError when the week is unknown (no subframe 1 decoded and
    none given), and for a track that is not a pilot of one of `pilot_signals`.
    """
    _check_pilots(pilots, pilot_signals)
    run = prepare(tracks, lnavs, interval)
    if run is None:
        return []
    return joint_observables(run, l1_observables(run, week), pilots, pilot_signals)


def _check_pilots(pilots: list[TrackResult], pilot_signals: tuple[str, ...]) -> None:
    for tr in pilots:
        if tr.signal not in pilot_signals or not SIGNALS[tr.signal].pilot:
            wanted = " or ".join(pilot_signals)
            raise ValueError(f"PRN {tr.prn}: {tr.signal} is not an {wanted} pilot track")


def joint_observables(
    run: Run,
    epochs: list[Epoch],
    pilots: list[TrackResult],
    pilot_signals: tuple[str, ...] = ("L5",),
) -> list[Observations]:
    """`make_joint_observables` on a prepared run and the L1 epochs `l1_observables` made of
    it, so that a caller holding both forms neither again."""
    _check_pilots(pilots, pilot_signals)
    if not epochs:
        return []
    if epochs[0].week is None:
        raise ValueError("week unknown: no ephemeris decoded and none given")
    chans, rx0, weeks_on = run.chans, run.rx0, run.weeks_on
    # (segment, its code starts, transmit time of its epoch 0, its description, whether it
    # took that time from its own secondary code)
    timed = []
    for tr in pilots:
        if tr.sec_sync is None:
            continue
        spec = SIGNALS[tr.signal]
        starts = code_starts(tr)
        period = len(spec.secondary(tr.prn)) * CA_PERIOD
        if period >= SECONDARY_TIME:
            time0 = _secondary_time0(tr, starts, rx0, period)
        else:
            l1 = [(tc, tt) for l1_tr, _, tc, tt in chans if l1_tr.prn == tr.prn]
            time0 = _pilot_time0(starts, l1)
        if time0 is not None:
            timed.append((tr, starts, time0, spec, period >= SECONDARY_TIME))
    coasting = [_coasting(tr) for tr, *_ in timed]
    near_wrap = [_near_wrap(tr) for tr, *_ in timed]
    segments = [_Segment(tr, tc, tr.sec_sync + 1) for tr, tc, *_ in timed]
    signals = ("G1C", *(SIGNALS[name].rinex for name in pilot_signals))
    column = {name: 1 + j for j, name in enumerate(pilot_signals)}
    out = []
    for e in epochs:
        t_rx = e.tow + weeks_on * WEEK_S
        t = t_rx - rx0
        rows = {f"G{o.prn:02d}": [o] + [None] * len(pilot_signals) for o in e.obs}
        sampled = [(c, segments[c].at(t)) for c in range(len(timed))]
        sampled = [(c, at, timed[c][2] + (at.k + at.f) * CA_PERIOD) for c, at in sampled if at]
        # the transmit times that the self-timed pilots of each satellite name at this instant
        own: dict[str, list[float]] = {}
        for c, _, t_tx in sampled:
            tr, _, _, spec, self_timed = timed[c]
            if self_timed:
                own.setdefault(f"{spec.rinex[0]}{tr.prn:02d}", []).append(t_tx)
        for c, at, t_tx in sampled:
            tr, _, _, spec, self_timed = timed[c]
            seg = segments[c]
            sat = f"{spec.rinex[0]}{tr.prn:02d}"
            if self_timed and max(own[sat]) - min(own[sat]) > PILOT_AGREEMENT:
                continue  # nothing says which of them is wrong
            k = at.k
            slip = at.slip or seg.since_phase(coasting[c], k) or seg.since_phase(near_wrap[c], k)
            lli = LLI_SLIP if slip else 0
            half = 0.5 if tr.sec_polarity < 0 else 0.0
            phase = -(at.adr + half) + spec.phase_align if at.locked else np.nan
            pilot_obs = Observation(
                tr.prn, C_LIGHT * (t_rx - t_tx), float(phase), float(tr.doppler[k]),
                float(tr.cn0[k]), lli,
            )  # fmt: skip
            rows.setdefault(sat, [None] * len(signals))[column[tr.signal]] = pilot_obs
            seg.observed(at)
        sats = tuple(sorted(rows))
        values = np.full((5, len(sats), len(signals)), np.nan)
        values[4] = 0.0
        for i, sat in enumerate(sats):
            for j, o in enumerate(rows[sat]):
                if o is not None:
                    values[:, i, j] = (o.pseudorange, o.carrier_phase, o.doppler, o.cn0, o.lli)
        pr, phase, doppler, cn0, lli = values
        lli = lli.astype(int)
        out.append(Observations(e.week, e.tow, sats, signals, pr, phase, doppler, cn0, lli))
    return out
