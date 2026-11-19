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
    obs: list[Observation]  # in the producer's order: ascending PRN from the builders


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
    sats: tuple[str, ...]  # RINEX ids, ascending from the builders and the reader
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


def from_epoch(epoch: Epoch, signal: str = "G1C") -> Observations:
    """An `Epoch` as `Observations` of one GPS signal, its rows in the order they have."""
    n = len(epoch.obs)
    values = np.array(
        [[o.pseudorange, o.carrier_phase, o.doppler, o.cn0] for o in epoch.obs], dtype=float
    ).reshape(n, 4)
    return Observations(
        epoch.week,
        epoch.tow,
        tuple(f"G{o.prn:02d}" for o in epoch.obs),
        (signal,),
        *(values[:, k].reshape(n, 1) for k in range(4)),
        np.array([o.lli for o in epoch.obs], dtype=int).reshape(n, 1),
    )
