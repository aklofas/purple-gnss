"""Navigation message decode of one tracked channel."""

from dataclasses import dataclass

import numpy as np

from pyrx.bitsync import BitStream, bit_sync, frame_sync
from pyrx.ephemeris import Ephemeris, Ionosphere
from pyrx.lnav import IONO_PAGE_SV_ID, Subframe, assemble_ephemeris
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
