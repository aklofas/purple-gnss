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
    prompt_code: np.ndarray | None = None,
) -> np.ndarray:
    """Correlator outputs (complex64, one per tap) for one block of samples.

    The replica carrier is exp(-2*pi*j*(carrier_freq*k/fs + carrier_phase)), with
    `carrier_phase` in cycles at the block's first sample. Tap `d` (chips) uses the
    code resampled with phase `code_phase + d`, so a positive tap is early (replica
    advanced) and a negative tap is late. Outputs are plain sums, not normalized.
    With `prompt_code` (a second code of the same length on the same carrier and
    timing) one more output follows the taps: that code's sum at tap 0.
    """
    n = len(block)
    k = _ramp(n)
    # phase in cycles, reduced to [0, 1) before the float32 exp to keep it accurate
    phase = (carrier_freq / fs) * k + carrier_phase
    phase -= np.floor(phase)
    carrier = np.exp((-2j * np.pi) * phase.astype(np.float32))
    mixed = block.astype(np.complex64, copy=False) * carrier
    chips = code_phase + (chip_rate / fs) * k
    out = np.empty(len(taps) + (prompt_code is not None), dtype=np.complex64)
    for i, tap in enumerate(taps):
        idx = np.floor(chips + tap).astype(np.int64)
        replica = np.take(code, idx, mode="wrap").astype(np.float32)
        out[i] = np.dot(mixed, replica)
    if prompt_code is not None:
        idx = np.floor(chips).astype(np.int64)
        out[-1] = np.dot(mixed, np.take(prompt_code, idx, mode="wrap").astype(np.float32))
    return out
