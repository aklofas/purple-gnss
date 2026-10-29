"""Correlation of one coherent interval against carrier and code replicas."""

import numpy as np

_ramp_cache: dict[int, np.ndarray] = {}


def _ramp(n: int) -> np.ndarray:
    if n not in _ramp_cache:
        _ramp_cache[n] = np.arange(n, dtype=np.float64)
    return _ramp_cache[n]


def correlate(
    block: np.ndarray,
    fs: float,
    carrier_freq: float,
    carrier_phase: float,
    code: np.ndarray,
    chip_rate: float,
    code_phase: float,
    taps: np.ndarray,
) -> np.ndarray:
    """Correlator outputs (complex64, one per tap) for one block of samples.

    The replica carrier is exp(-2*pi*j*(carrier_freq*k/fs + carrier_phase)), with
    `carrier_phase` in cycles at the block's first sample. Tap `d` (chips) uses the
    code resampled with phase `code_phase + d`, so a positive tap is early (replica
    advanced) and a negative tap is late. Outputs are plain sums, not normalized.
    """
    n = len(block)
    k = _ramp(n)
    # phase in cycles, reduced to [0, 1) before the float32 exp to keep it accurate
    phase = (carrier_freq / fs) * k + carrier_phase
    phase -= np.floor(phase)
    carrier = np.exp((-2j * np.pi) * phase.astype(np.float32))
    mixed = block.astype(np.complex64, copy=False) * carrier
    chips = code_phase + (chip_rate / fs) * k
    out = np.empty(len(taps), dtype=np.complex64)
    for i, tap in enumerate(taps):
        idx = np.floor(chips + tap).astype(np.int64)
        replica = np.take(code, idx, mode="wrap").astype(np.float32)
        out[i] = np.dot(mixed, replica)
    return out
