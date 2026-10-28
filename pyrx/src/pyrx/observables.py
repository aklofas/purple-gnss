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

Carrier validity (RINEX loss-of-lock indicator, as Pocket SDR sets it): bit 1 on
the observations within the first 2 s of a channel and on the first observation
after the data polarity changed, that is after a half-cycle slip of the Costas
loop; bit 2 where the polarity is not established by a decoded subframe: before
the first subframe, and from the end of the last subframe before a polarity
change to the start of the first subframe after it.
"""

from math import ceil, floor

import numpy as np

from pyrx.bitsync import EPOCHS_PER_BIT
from pyrx.codes import CA_PERIOD
from pyrx.gpstime import C_LIGHT, WEEK_S, full_week
from pyrx.lnav import PREAMBLE, SUBFRAME_BITS
from pyrx.navdecode import LnavResult
from pyrx.observations import Epoch, Observation
from pyrx.tracking import TrackResult

TRAVEL_TIME_NOMINAL = 0.068  # s, light time to the nearest satellite, sets the receiver clock
T_LOCK = 2.0  # s a channel must have run before its carrier phase counts as locked
SUBFRAME_EPOCHS = SUBFRAME_BITS * EPOCHS_PER_BIT
LLI_SLIP = 1  # RINEX loss-of-lock indicator bits
LLI_HALF_CYCLE = 2


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
    epochs = []
    for i in range(first, last + 1):
        t = i * interval - rx0
        tow = i * interval - weeks_on * WEEK_S
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
            if tr.t[k] - tr.t[0] <= T_LOCK:
                lli |= LLI_SLIP
            phase = -(adr + (0.5 if inverted else 0.0))
            obs.append(
                Observation(
                    tr.prn,
                    C_LIGHT * (tow - t_tx),
                    float(phase),
                    float(tr.doppler[k]),
                    float(tr.cn0[k]),
                    lli,
                )
            )
            last_k[c] = k
        if obs:
            epochs.append(Epoch(week, float(tow), sorted(obs, key=lambda o: o.prn)))
    return epochs
