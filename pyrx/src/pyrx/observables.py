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
negated accumulated Doppler phase (phase grows with range), plus half a cycle
when the Costas loop locked on inverted data; Doppler is positive for an
approaching satellite, as tracking reports it.
"""

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from pyrx.codes import CA_PERIOD
from pyrx.navdecode import LnavResult
from pyrx.tracking import TrackResult

C_LIGHT = 299792458.0  # m/s
WEEK_S = 604800
TRAVEL_TIME_NOMINAL = 0.068  # s, light time to the nearest satellite, sets the receiver clock
GPS_WEEK_ROLLOVER = 2048  # first week of the current 1024-week era (2019-04-07)


def full_week(wn10: int) -> int:
    """Full GPS week from the broadcast 10-bit week number (valid until 2038-11-20)."""
    return wn10 + GPS_WEEK_ROLLOVER


def code_starts(track: TrackResult) -> np.ndarray:
    """Receiver time (s from the source start) of every epoch's code-period start."""
    off = (track.code_offset - track.t + CA_PERIOD / 2) % CA_PERIOD - CA_PERIOD / 2
    return track.t + off


def transmit_times(track: TrackResult, lnav: LnavResult) -> np.ndarray:
    """GPS seconds of week at which every epoch's code start was transmitted."""
    if not lnav.subframes:
        raise ValueError("no subframe decoded")
    t0 = np.array([sf.tow_s - sf.epoch * CA_PERIOD for sf in lnav.subframes])
    if t0.max() - t0.min() > 1e-9:
        raise ValueError("inconsistent subframe time tags")
    return t0[0] + np.arange(len(track.t)) * CA_PERIOD


@dataclass(frozen=True)
class Observation:
    prn: int
    pseudorange: float  # m
    carrier_phase: float  # cycles
    doppler: float  # Hz
    cn0: float  # dB-Hz


@dataclass(frozen=True)
class Epoch:
    week: int | None
    tow: float  # GPS seconds of week of the receiver epoch
    obs: list[Observation]  # ascending PRN


def _usable(lnav: LnavResult) -> bool:
    return bool(lnav.subframes) and lnav.stream is not None


def receiver_time0(tracks: list[TrackResult], lnavs: list[LnavResult]) -> float:
    """GPS seconds of week assigned to sample 0 of the recording."""
    t_tx0 = [
        transmit_times(tr, ln)[0] - code_starts(tr)[0]
        for tr, ln in zip(tracks, lnavs)
        if _usable(ln)
    ]
    if not t_tx0:
        raise ValueError("no channel with a decoded subframe")
    return round(max(t_tx0) + TRAVEL_TIME_NOMINAL, 3)


def make_observables(
    tracks: list[TrackResult],
    lnavs: list[LnavResult],
    interval: float = 1.0,
    week: int | None = None,
) -> list[Epoch]:
    """Observations of every channel with a decoded subframe at a common epoch grid.

    A channel is used from its first bit-synced epoch on. `week` defaults to the
    week of any decoded ephemeris and stays None when there is none.
    """
    chans = [(tr, ln) for tr, ln in zip(tracks, lnavs) if _usable(ln)]
    if not chans:
        return []
    if week is None:
        week = next((full_week(ln.ephemeris.wn) for _, ln in chans if ln.ephemeris), None)
    rx0 = receiver_time0([c[0] for c in chans], [c[1] for c in chans])
    per = []
    for tr, ln in chans:
        per.append((tr, ln, code_starts(tr), transmit_times(tr, ln), int(ln.stream.epoch[0])))
    t_begin = min(tc[k0] for _, _, tc, _, k0 in per)
    t_end = max(tc[-1] for _, _, tc, _, _ in per)
    first = ceil((rx0 + t_begin) / interval)
    last = floor((rx0 + t_end) / interval)
    epochs = []
    for i in range(first, last + 1):
        tow = i * interval
        t = tow - rx0
        obs = []
        for tr, ln, tc, tt, k0 in per:
            k = int(np.searchsorted(tc, t, side="right")) - 1
            if k < max(k0, 1) or k + 1 >= len(tc):
                continue
            f = (t - tc[k]) / (tc[k + 1] - tc[k])
            t_tx = tt[k] + f * CA_PERIOD
            # carrier_phase[k-1] is the accumulated Doppler phase at tc[k], the end of epoch k-1
            adr = tr.carrier_phase[k - 1] + f * (tr.carrier_phase[k] - tr.carrier_phase[k - 1])
            phase = -(adr + (0.5 if ln.stream.inverted else 0.0))
            obs.append(
                Observation(
                    tr.prn,
                    C_LIGHT * (tow - t_tx),
                    float(phase),
                    float(tr.doppler[k]),
                    float(tr.cn0[k]),
                )
            )
        if obs:
            epochs.append(Epoch(week, float(tow), sorted(obs, key=lambda o: o.prn)))
    return epochs
