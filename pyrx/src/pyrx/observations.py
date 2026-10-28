"""The observations of one receiver at one epoch, in the two forms the package uses.

`Epoch` is the single-signal GPS L1 C/A form the signal chain produces and the
single-point solver consumes; `Observations` holds every satellite by every requested
signal as arrays, which is what the relative positioning works on. A signal is the
RINEX 3 system letter plus the two-character observation attribute (`G1C`, `G2W`,
`E5Q`); a satellite is its RINEX id (`G04`, `E11`). Imports nothing from the signal
chain or the positioning.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class Observation:
    prn: int
    pseudorange: float  # m
    carrier_phase: float  # cycles
    doppler: float  # Hz
    cn0: float  # dB-Hz
    lli: int = 0  # RINEX loss-of-lock indicator of the carrier phase


@dataclass(frozen=True)
class Epoch:
    """Observations of all channels at one receiver epoch.

    `tow` is seconds from the start of `week`. Within one `make_observables` run it
    starts in [0, 604800), increases monotonically and may reach or exceed 604800
    when a recording crosses the week boundary; `week` does not change within a
    run. The RINEX reader
    returns `tow` in [0, 604800) with the week of the time stamp. Consumers form a
    continuous time as `week * WEEK_S + tow` when they need one and reduce every
    difference against an ephemeris time of week (`toe`, `toc`) into
    [-302400, 302400) with `orbits.wrap_week`.
    """

    week: int | None
    tow: float  # GPS seconds of week of the receiver epoch
    obs: list[Observation]  # ascending PRN


@dataclass(frozen=True)
class Observations:
    """All observations of one receiver at one epoch; arrays are (satellite, signal).

    `tow` follows the `Epoch` time contract. Values are `nan` where a satellite has
    no observation on a signal (including every signal of another system); `lli`
    is the RINEX loss-of-lock indicator (bit 0 lost lock, bit 1 half-cycle
    ambiguity), 0 where absent.
    """

    week: int
    tow: float
    sats: tuple[str, ...]  # ascending RINEX ids
    signals: tuple[str, ...]
    pseudorange: np.ndarray  # m
    carrier_phase: np.ndarray  # cycles
    doppler: np.ndarray  # Hz
    cn0: np.ndarray  # dB-Hz
    lli: np.ndarray  # int


def to_epoch(obs: Observations, signal: str = "G1C") -> Epoch:
    """The GPS observations of one signal as an `Epoch`, for the single-point solver."""
    j = obs.signals.index(signal)
    out = []
    for i, sat in enumerate(obs.sats):
        if sat[0] == "G" and isfinite(obs.pseudorange[i, j]):
            out.append(
                Observation(
                    int(sat[1:]),
                    float(obs.pseudorange[i, j]),
                    float(obs.carrier_phase[i, j]),
                    float(obs.doppler[i, j]),
                    float(obs.cn0[i, j]),
                    int(obs.lli[i, j]),
                )
            )
    return Epoch(obs.week, obs.tow, sorted(out, key=lambda o: o.prn))
