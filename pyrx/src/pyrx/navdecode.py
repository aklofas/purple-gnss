"""Navigation message decode of tracked channels, and the per-PRN view over their segments."""

from dataclasses import dataclass

import numpy as np

from pyrx.bitsync import BitStream, bit_sync, frame_sync
from pyrx.ephemeris import Ephemeris, Ionosphere
from pyrx.lnav import IONO_PAGE_SV_ID, Subframe, assemble_ephemeris, assemble_ephemeris_with_tow
from pyrx.tracking import T_FLL_PULLIN, TrackResult

T_PLL_SETTLE = 0.5  # s after FLL pull-in before the prompt signs are trusted as data bits


@dataclass
class LnavResult:
    prn: int
    stream: BitStream | None
    subframes: list[Subframe]
    ephemeris: Ephemeris | None
    ionosphere: Ionosphere | None


def decode_lnav(track: TrackResult) -> LnavResult:
    """Bits, subframes, ephemeris and ionosphere parameters from a tracked channel.

    Bit sync starts once the PLL has settled, at the first epoch at least
    `T_FLL_PULLIN + T_PLL_SETTLE` after the track start.
    """
    start = 0
    if len(track.t):
        start = int(np.searchsorted(track.t, track.t[0] + T_FLL_PULLIN + T_PLL_SETTLE))
    stream = bit_sync(track.corr[start:, 0].real, start)
    if stream is None:
        return LnavResult(track.prn, None, [], None, None)
    subframes = frame_sync(stream, track.prn)
    iono = None
    for sf in subframes:
        if sf.sf_id == 4 and sf.fields.get("page_sv_id") == IONO_PAGE_SV_ID:
            f = sf.fields
            alpha = tuple(f[f"alpha{i}"] for i in range(4))
            beta = tuple(f[f"beta{i}"] for i in range(4))
            iono = Ionosphere(alpha, beta)
    return LnavResult(track.prn, stream, subframes, assemble_ephemeris(subframes), iono)


def _ephemerides_with_tow(lnavs: list[LnavResult]) -> dict[int, tuple[Ephemeris, int]]:
    subframes: dict[int, list[Subframe]] = {}
    for ln in lnavs:
        subframes.setdefault(ln.prn, []).extend(ln.subframes)
    out = {}
    for prn, sfs in subframes.items():
        found = assemble_ephemeris_with_tow(sfs)
        if found is not None:
            out[prn] = found
    return out


def ephemerides_by_prn(lnavs: list[LnavResult]) -> dict[int, Ephemeris]:
    """One ephemeris per PRN from the subframes of all its segments, in list order.

    The list is in segment time order, so each PRN gets the last ephemeris its
    subframes 1, 2 and 3 completed with an agreeing issue of data, whichever
    segments they came from; PRNs that never completed one are absent.
    """
    return {prn: eph for prn, (eph, _) in _ephemerides_with_tow(lnavs).items()}


def tow1_by_prn(lnavs: list[LnavResult]) -> dict[int, int]:
    """The time of week of the subframe 1 behind each PRN's ephemeris (`ephemerides_by_prn`)."""
    return {prn: tow for prn, (_, tow) in _ephemerides_with_tow(lnavs).items()}


def ionosphere_of(lnavs: list[LnavResult]) -> Ionosphere | None:
    """The ionosphere parameter set of the first segment in `lnavs` that decoded one: list
    order, which is the requested PRN order and then time, not the earliest in time."""
    return next((ln.ionosphere for ln in lnavs if ln.ionosphere is not None), None)
