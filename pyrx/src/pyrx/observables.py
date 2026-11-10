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
window can lie wholly between sparse observations), and on the first observation
after the data polarity changed,
that is after a half-cycle slip of the Costas loop; value 2 (bit 1) where the polarity is not
established by a decoded subframe: before the first subframe, and from the end of
the last subframe before a polarity change to the start of the first subframe
after it. The bits are kept on an observation without a phase too, so the slip is
reported on the next observation that has one.

A pilot channel of a second band on the same sample clock (GPS L5 next to L1) decodes
no time of its own. Every one of its epochs is one code period too, so it needs the
transmit time of one code start, and it takes that from the same satellite's L1
channel: both signals leave the satellite together and arrive within a fraction of a
microsecond of each other, so the L1 transmit time at a pilot code start, rounded to
the 1 ms code period, is the pilot's (`pilot_transmit_times`). Its observations begin
once the secondary code is found. Its carrier phase has no half-cycle ambiguity; where
the tracker settled on the inverted pilot the reported phase gets the half cycle back,
and the quarter cycle that aligns L5Q with the band's reference signal L5I (RINEX 3.05
Table A23) is applied here, as a receiver does.
"""

from math import ceil, floor

import numpy as np

from pyrx.bitsync import EPOCHS_PER_BIT
from pyrx.codes import CA_PERIOD
from pyrx.gpstime import C_LIGHT, WEEK_S, full_week
from pyrx.lnav import PREAMBLE, SUBFRAME_BITS
from pyrx.navdecode import LnavResult
from pyrx.observations import Epoch, Observation, Observations
from pyrx.tracking import PLI_LOCKED, TrackResult

TRAVEL_TIME_NOMINAL = 0.068  # s, light time to the nearest satellite, sets the receiver clock
T_LOCK = 2.0  # s into a channel over which its observations carry the slip flag
SUBFRAME_EPOCHS = SUBFRAME_BITS * EPOCHS_PER_BIT
LLI_SLIP = 1  # RINEX loss-of-lock indicator bits
LLI_HALF_CYCLE = 2
L5Q_ALIGN = -0.25  # cycles added to the L5Q phase to align it with L5I (RINEX 3.05 Table A23)
ANCHOR_TOLERANCE = 1e-4  # s a pilot's code start may lie from the L1 channel's code period grid


def code_starts(track: TrackResult) -> np.ndarray:
    """Receiver time (s from the source start) of every epoch's code-period start."""
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
    t_tx0 = [tt[0] - tc[0] for _, _, tc, tt in _usable(tracks, lnavs)]
    if not t_tx0:
        raise ValueError("no channel with a decoded subframe")
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
    implied by any decoded subframe 1 and stays None when there is none.
    """
    chans = _usable(tracks, lnavs)
    if not chans:
        return []
    # the week of the run's time origin: a subframe 1 decoded past the rollover carries
    # the next week's number
    wn = [
        full_week(sf.fields["wn"]) - int(tt[sf.epoch] // WEEK_S)
        for _, ln, _, tt in chans
        for sf in ln.subframes
        if sf.sf_id == 1
    ]
    origin_week = wn[0] if wn else None
    rx0 = receiver_time0([c[0] for c in chans], [c[1] for c in chans])
    per = [(tr, ln, tc, tt, int(ln.stream.epoch[0])) for tr, ln, tc, tt in chans]
    t_begin = min(tc[k0] for _, _, tc, _, k0 in per)
    t_end = max(tc[-1] for _, _, tc, _, _ in per)
    first = ceil((rx0 + t_begin) / interval)
    last = floor((rx0 + t_end) / interval)
    weeks_on = floor(first * interval / WEEK_S)  # every epoch past the boundary: the next week
    if week is None and origin_week is not None:
        week = origin_week + weeks_on
    polarity = [_polarities(ln) for _, ln, _, _, _ in per]
    last_k = [-1] * len(per)  # epoch of each channel's previous observation
    last_phase_k = [-1] * len(per)  # epoch of each channel's previous observation with a phase
    epochs = []
    for i in range(first, last + 1):
        t = i * interval - rx0
        t_rx = i * interval  # in the week of the time origin, as the transmit times are
        tow = t_rx - weeks_on * WEEK_S
        obs = []
        for c, (tr, ln, tc, tt, k0) in enumerate(per):
            k = int(np.searchsorted(tc, t, side="right")) - 1
            if k < max(k0, 1) or k + 1 >= len(tc):
                continue
            f = (t - tc[k]) / (tc[k + 1] - tc[k])
            t_tx = tt[k] + f * CA_PERIOD
            # carrier_phase[k-1] is the accumulated Doppler phase at tc[k], the end of epoch k-1
            adr = tr.carrier_phase[k - 1] + f * (tr.carrier_phase[k] - tr.carrier_phase[k - 1])
            inverted, lli = _carrier_state(ln, polarity[c], k, last_k[c])
            # unlocked (or unknown, nan) at this epoch or at any epoch since the previous
            # observation with a phase: an unlocked window can lie between sparse observations
            # (the slice is empty when several observations fall in one tracking epoch)
            since_phase = tr.phase_lock[last_phase_k[c] + 1 : k + 1] >= PLI_LOCKED
            locked = bool(tr.phase_lock[k] >= PLI_LOCKED)
            if tr.t[k] - tr.t[0] <= T_LOCK or last_k[c] < 0 or not since_phase.all():
                lli |= LLI_SLIP
            phase = -(adr + (0.5 if inverted else 0.0)) if locked else np.nan
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
            last_k[c] = k
            if locked:
                last_phase_k[c] = k
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


def pilot_transmit_times(
    track: TrackResult, l1: list[tuple[TrackResult, LnavResult]]
) -> np.ndarray | None:
    """Seconds of week at which every code start of a pilot segment was transmitted, from
    the L1 segments (with their decodes) of the same satellite on the same sample clock;
    None when no L1 segment with a time tag overlaps it or the overlap is not consistent."""
    chans = _usable([tr for tr, _ in l1], [ln for _, ln in l1])
    time0 = _pilot_time0(code_starts(track), [(tc, tt) for _, _, tc, tt in chans])
    return None if time0 is None else time0 + np.arange(len(track.t)) * CA_PERIOD


def make_joint_observables(
    tracks: list[TrackResult],
    lnavs: list[LnavResult],
    pilots: list[TrackResult],
    interval: float = 1.0,
    week: int | None = None,
) -> list[Observations]:
    """The L1 C/A observations of `make_observables` with the L5 pilot segments `pilots` of
    the same recording time beside them, as signals `G1C` and `G5Q`.

    The epochs, the receiver clock and every L1 value are `make_observables`'. A pilot
    segment contributes from the epoch after its secondary code was found, on the time of
    its satellite's L1 channel; one without a consistent anchor contributes nothing. Raises
    ValueError when the week is unknown (no subframe 1 decoded and none given).
    """
    epochs = make_observables(tracks, lnavs, interval, week)
    if not epochs:
        return []
    if epochs[0].week is None:
        raise ValueError("week unknown: no ephemeris decoded and none given")
    chans = _usable(tracks, lnavs)
    rx0 = receiver_time0([c[0] for c in chans], [c[1] for c in chans])
    # the L1 epochs count from the week of the time origin; tow restarts if all lie past it
    t_begin = min(tc[int(ln.stream.epoch[0])] for _, ln, tc, _ in chans)
    weeks_on = floor(ceil((rx0 + t_begin) / interval) * interval / WEEK_S)
    anchored = []
    for tr in pilots:
        starts = code_starts(tr)
        l1 = [(tc, tt) for l1_tr, _, tc, tt in chans if l1_tr.prn == tr.prn]
        time0 = _pilot_time0(starts, l1)
        if time0 is not None and tr.sec_sync is not None:
            anchored.append((tr, starts, time0))
    last_k = [-1] * len(anchored)
    last_phase_k = [-1] * len(anchored)
    out = []
    for e in epochs:
        t_rx = e.tow + weeks_on * WEEK_S
        t = t_rx - rx0
        rows = {f"G{o.prn:02d}": [o, None] for o in e.obs}
        for c, (tr, tc, time0) in enumerate(anchored):
            k = int(np.searchsorted(tc, t, side="right")) - 1
            if k <= tr.sec_sync or k + 1 >= len(tc):
                continue
            f = (t - tc[k]) / (tc[k + 1] - tc[k])
            t_tx = time0 + (k + f) * CA_PERIOD
            adr = tr.carrier_phase[k - 1] + f * (tr.carrier_phase[k] - tr.carrier_phase[k - 1])
            since_phase = tr.phase_lock[last_phase_k[c] + 1 : k + 1] >= PLI_LOCKED
            locked = bool(tr.phase_lock[k] >= PLI_LOCKED)
            lli = 0
            if tr.t[k] - tr.t[0] <= T_LOCK or last_k[c] < 0 or not since_phase.all():
                lli = LLI_SLIP
            half = 0.5 if tr.sec_polarity < 0 else 0.0
            phase = -(adr + half) + L5Q_ALIGN if locked else np.nan
            pilot_obs = Observation(
                tr.prn, C_LIGHT * (t_rx - t_tx), float(phase), float(tr.doppler[k]),
                float(tr.cn0[k]), lli,
            )  # fmt: skip
            rows.setdefault(f"G{tr.prn:02d}", [None, None])[1] = pilot_obs
            last_k[c] = k
            if locked:
                last_phase_k[c] = k
        sats = tuple(sorted(rows))
        values = np.full((5, len(sats), 2), np.nan)
        values[4] = 0.0
        for i, sat in enumerate(sats):
            for j, o in enumerate(rows[sat]):
                if o is not None:
                    values[:, i, j] = (o.pseudorange, o.carrier_phase, o.doppler, o.cn0, o.lli)
        pr, phase, doppler, cn0, lli = values
        out.append(
            Observations(e.week, e.tow, sats, ("G1C", "G5Q"), pr, phase, doppler, cn0, lli.astype(int))
        )
    return out
