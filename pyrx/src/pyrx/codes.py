"""PRN code generation and resampling.

GPS L1 C/A codes follow IS-GPS-200: two 10-stage LFSRs (G1, G2), the output chip
being G1 XOR a delayed G2. Chips are returned as +1 for chip value 0 and -1 for
chip value 1, so a code multiplies directly onto a signal.
"""

import numpy as np

CA_CHIP_RATE = 1.023e6
CA_PERIOD = 1e-3
CA_LENGTH = 1023

# G2 delay in chips per PRN 1..32, IS-GPS-200 Table 3-Ia.
_CA_G2_DELAY = (
    5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252, 254, 255, 256, 257, 258,
    469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860, 861, 862,
)  # fmt: skip

_ca_cache: dict[int, np.ndarray] = {}


def _lfsr(taps: tuple[int, ...], length: int) -> np.ndarray:
    """Output of a 10-stage LFSR seeded with all ones; `taps` are 1-based stage numbers."""
    state = [1] * 10
    out = np.empty(length, dtype=np.int8)
    for i in range(length):
        out[i] = state[9]
        feedback = 0
        for t in taps:
            feedback ^= state[t - 1]
        state = [feedback] + state[:9]
    return out


def gps_ca(prn: int) -> np.ndarray:
    """One period (1023 chips) of the GPS L1 C/A code for `prn` in 1..32, as int8 +1/-1."""
    if not 1 <= prn <= len(_CA_G2_DELAY):
        raise ValueError(f"PRN {prn} outside 1..{len(_CA_G2_DELAY)}")
    if prn not in _ca_cache:
        g1 = _lfsr((3, 10), CA_LENGTH)
        g2 = _lfsr((2, 3, 6, 8, 9, 10), CA_LENGTH)
        chips = g1 ^ np.roll(g2, _CA_G2_DELAY[prn - 1])
        _ca_cache[prn] = (1 - 2 * chips).astype(np.int8)
    return _ca_cache[prn]


def sample_code(
    code: np.ndarray, chip_rate: float, fs: float, n: int, phase_chips: float = 0.0
) -> np.ndarray:
    """`n` samples of `code` at sample rate `fs`, sample k holding chip floor(phase + k*rate/fs)."""
    chip_index = np.floor(phase_chips + np.arange(n, dtype=np.float64) * (chip_rate / fs))
    return code[chip_index.astype(np.int64) % len(code)].astype(np.float32)
