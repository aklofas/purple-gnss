"""What the signal chain needs to know about each signal it tracks.

A signal is tracked on one component. GPS L1 C/A is a data signal: its code carries the
navigation bits. GPS L5 and Galileo E5a are tracked on their pilot component (Q5, E5a-Q),
which carries no data but one chip of a secondary code per code period; their data
component (I5, E5a-I) is on the same carrier a quarter cycle away, with its own primary
and secondary code and one symbol per repetition of that secondary code.

The tracker works in epochs of 1 ms, the code period of those three signals. Galileo E1 is
tracked on its pilot E1-C, whose code lasts 4 ms: an epoch correlates a quarter of it, and
the secondary code, one chip per code period, is held over the four epochs of a period, so
that it is one entry per epoch like the others. Every E1 chip is multiplied by one cycle
of a square subcarrier (BOC(1,1); the signal's small BOC(6,1) part is left out), which the
replica carries as two entries per chip. The data component E1-B is on the pilot's axis
with the opposite sign, one symbol per code period.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyrx.codes import (
    CA_CHIP_RATE,
    CA_LENGTH,
    CA_PERIOD,
    E1_CHIP_RATE,
    E1_LENGTH,
    GAL_CS20,
    GAL_CS25,
    L5_CHIP_RATE,
    L5_LENGTH,
    NH10,
    NH20,
    gal_cs100,
    gal_e1b,
    gal_e1c,
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
    code_length: int  # chips of one period of the primary code
    prns: range
    code: Callable[[int], np.ndarray]  # the tracked component's primary code of a PRN
    # the tracked component's secondary code of a PRN, one entry per 1 ms epoch; None for a
    # data signal
    secondary: Callable[[int], np.ndarray] | None = None
    # pilot signals: the data component's primary and secondary code of a PRN
    data_code: Callable[[int], np.ndarray] | None = None
    data_secondary: Callable[[int], np.ndarray] | None = None
    symbol_epochs: int = 0  # 1 ms epochs per data symbol
    # where a symbol 0 lies on the data component's prompt when the pilot's prompt, its
    # secondary code removed, is real and positive
    data_axis: complex = 0j
    # cycles added to the reported carrier phase to align it with its band's reference signal
    # (RINEX 3.05 Table A23)
    phase_align: float = 0.0
    # the values every chip of the replica is multiplied by in turn: (1,) for a plain code,
    # (1, -1) for one cycle of a square subcarrier per chip, BOC(1,1)
    subcarrier: tuple[int, ...] = (1,)
    # correlator taps (chips from prompt) at the side peaks of the replica's correlation
    side_taps: tuple[float, ...] = ()
    # the slope of the early-minus-late code discriminator at the tracker's correlator spacing
    # over a plain code's: the correlation peak of BOC(1,1) falls three times as steeply and
    # early plus late is a third, nine in all at a quarter chip
    dll_slope: float = 1.0

    @property
    def pilot(self) -> bool:
        return self.secondary is not None

    @property
    def epochs(self) -> int:
        """The tracker's 1 ms epochs in one period of the primary code."""
        return round(self.code_length / self.chip_rate / CA_PERIOD)

    @property
    def replica_rate(self) -> float:
        """Entries of the replica per second: the chip rate times the subcarrier's length."""
        return self.chip_rate * len(self.subcarrier)

    def replica(self, prn: int) -> np.ndarray:
        """The tracked component's primary code of a PRN with the subcarrier on every chip;
        the code itself for a plain code."""
        return _with_subcarrier(self.code(prn), self.subcarrier)

    def data_replica(self, prn: int) -> np.ndarray:
        """As `replica`, of the data component."""
        return _with_subcarrier(self.data_code(prn), self.subcarrier)


_replicas: dict[tuple[bytes, tuple[int, ...]], np.ndarray] = {}


def _with_subcarrier(code: np.ndarray, subcarrier: tuple[int, ...]) -> np.ndarray:
    if subcarrier == (1,):
        return code
    key = (code.tobytes(), subcarrier)  # by the chips, not by the array that holds them
    if key not in _replicas:
        replica = np.repeat(code, len(subcarrier)) * np.tile(
            np.array(subcarrier, dtype=np.int8), len(code)
        )
        replica.setflags(write=False)
        _replicas[key] = replica
    return _replicas[key]


def _nh20(prn: int) -> np.ndarray:
    return NH20


def _nh10(prn: int) -> np.ndarray:
    return NH10


def _cs20(prn: int) -> np.ndarray:
    return GAL_CS20


_CS25_BY_EPOCH = np.repeat(GAL_CS25, round(E1_LENGTH / E1_CHIP_RATE / CA_PERIOD))
_CS25_BY_EPOCH.setflags(write=False)
_NO_SECONDARY = np.ones(1, dtype=np.int8)
_NO_SECONDARY.setflags(write=False)


def _cs25(prn: int) -> np.ndarray:
    return _CS25_BY_EPOCH


def _no_secondary(prn: int) -> np.ndarray:
    return _NO_SECONDARY


L1CA = SignalSpec("L1CA", "G1C", FREQUENCY["G1"], CA_CHIP_RATE, CA_LENGTH, range(1, 33), gps_ca)
# IS-GPS-705 3.3.1.5: the Q5 carrier lags the I5 carrier by 90 degrees
L5 = SignalSpec(
    "L5", "G5Q", FREQUENCY["G5"], L5_CHIP_RATE, L5_LENGTH, range(1, 33), gps_l5q,
    secondary=_nh20, data_code=gps_l5i, data_secondary=_nh10, symbol_epochs=10, data_axis=1j,
    phase_align=-0.25,
)  # fmt: skip
# Galileo OS SIS ICD 2.3.1.2, Eq. 3: the E5a signal is e_E5a-I + j e_E5a-Q on its carrier
E5A = SignalSpec(
    "E5A", "E5Q", FREQUENCY["E5"], L5_CHIP_RATE, L5_LENGTH, range(1, 37), gal_e5aq,
    secondary=gal_cs100, data_code=gal_e5ai, data_secondary=_cs20, symbol_epochs=20, data_axis=-1j,
    phase_align=-0.25,
)  # fmt: skip
# Galileo OS SIS ICD issue 2.2, 2.3.3, Eq. 12: the E1 signal is (e_E1-B - e_E1-C) / sqrt 2 on its carrier;
# RINEX 3.05 Table A23 has L1C half a cycle from L1B
E1 = SignalSpec(
    "E1", "E1C", FREQUENCY["E1"], E1_CHIP_RATE, E1_LENGTH, range(1, 37), gal_e1c,
    secondary=_cs25, data_code=gal_e1b, data_secondary=_no_secondary, symbol_epochs=4,
    data_axis=-1 + 0j, phase_align=0.5, subcarrier=(1, -1), side_taps=(0.5, -0.5), dll_slope=9.0,
)  # fmt: skip
SIGNALS = {s.name: s for s in (L1CA, L5, E5A, E1)}
