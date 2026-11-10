"""What the signal chain needs to know about each signal it tracks.

A signal is tracked on one component. GPS L1 C/A is a data signal: its code carries the
navigation bits. GPS L5 and Galileo E5a are tracked on their pilot component (Q5, E5a-Q),
which carries no data but one chip of a secondary code per code period; their data
component (I5, E5a-I) is on the same carrier a quarter cycle away, with its own primary
and secondary code and one symbol per repetition of that secondary code.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyrx.codes import (
    CA_CHIP_RATE,
    CA_LENGTH,
    GAL_CS20,
    L5_CHIP_RATE,
    L5_LENGTH,
    NH10,
    NH20,
    gal_cs100,
    gal_e5ai,
    gal_e5aq,
    gps_ca,
    gps_l5i,
    gps_l5q,
)
from pyrx.signals import FREQUENCY


@dataclass(frozen=True)
class SignalSpec:
    name: str  # as the commands take it
    rinex: str  # the RINEX 3 signal the observations are reported as
    carrier: float  # Hz
    chip_rate: float  # chips/s
    code_length: int  # chips of the primary code, one period of 1 ms
    prns: range
    code: Callable[[int], np.ndarray]  # the tracked component's primary code of a PRN
    # the tracked component's secondary code of a PRN; None for a data signal
    secondary: Callable[[int], np.ndarray] | None = None
    # pilot signals: the data component's primary and secondary code of a PRN
    data_code: Callable[[int], np.ndarray] | None = None
    data_secondary: Callable[[int], np.ndarray] | None = None
    symbol_epochs: int = 0  # code periods per data symbol
    # where a symbol 0 lies on the data component's prompt when the pilot's prompt, its
    # secondary code removed, is real and positive
    data_axis: complex = 0j

    @property
    def pilot(self) -> bool:
        return self.secondary is not None


def _nh20(prn: int) -> np.ndarray:
    return NH20


def _nh10(prn: int) -> np.ndarray:
    return NH10


def _cs20(prn: int) -> np.ndarray:
    return GAL_CS20


L1CA = SignalSpec("L1CA", "G1C", FREQUENCY["G1"], CA_CHIP_RATE, CA_LENGTH, range(1, 33), gps_ca)
# IS-GPS-705 3.3.1.5: the Q5 carrier lags the I5 carrier by 90 degrees
L5 = SignalSpec(
    "L5", "G5Q", FREQUENCY["G5"], L5_CHIP_RATE, L5_LENGTH, range(1, 33), gps_l5q,
    secondary=_nh20, data_code=gps_l5i, data_secondary=_nh10, symbol_epochs=10, data_axis=1j,
)  # fmt: skip
# Galileo OS SIS ICD 2.3.1.2, Eq. 3: the E5a signal is e_E5a-I + j e_E5a-Q on its carrier
E5A = SignalSpec(
    "E5A", "E5Q", FREQUENCY["E5"], L5_CHIP_RATE, L5_LENGTH, range(1, 37), gal_e5aq,
    secondary=gal_cs100, data_code=gal_e5ai, data_secondary=_cs20, symbol_epochs=20, data_axis=-1j,
)  # fmt: skip
SIGNALS = {s.name: s for s in (L1CA, L5, E5A)}
