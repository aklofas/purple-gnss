"""Signals of several systems and bands, and the observations of one receiver at one epoch.

A signal is the RINEX 3 system letter plus the two-character observation attribute
(`G1C`, `G2W`, `E5Q`); a satellite is its RINEX id (`G04`, `E11`). `Observations`
holds every satellite by every requested signal as arrays, which is what the
relative positioning works on; `Epoch` stays the single-signal GPS L1 C/A form.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from pyrx.observables import C_LIGHT, Epoch, Observation

# carrier frequencies (Hz) by system letter and band digit
FREQUENCY = {
    "G1": 1575.42e6,
    "G2": 1227.60e6,
    "G5": 1176.45e6,
    "E1": 1575.42e6,
    "E5": 1176.45e6,
    "E7": 1207.14e6,
}


def signal_frequency(signal: str) -> float:
    """Carrier frequency (Hz) of a signal such as `G2W`."""
    return FREQUENCY[signal[:2]]


def wavelength(signal: str) -> float:
    """Carrier wavelength (m) of a signal."""
    return C_LIGHT / signal_frequency(signal)


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
