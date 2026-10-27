"""Correlation of one coherent interval against carrier and code replicas."""

import numpy as np

from pyrx.codes import sample_code


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
    advanced) and a negative tap is late. Outputs are plain sums, not normalised.
    """
    n = len(block)
    k = np.arange(n, dtype=np.float64)
    carrier = np.exp(-2j * np.pi * (carrier_freq / fs * k + carrier_phase))
    mixed = block.astype(np.complex128) * carrier
    out = np.empty(len(taps), dtype=np.complex64)
    for i, tap in enumerate(taps):
        replica = sample_code(code, chip_rate, fs, n, phase_chips=code_phase + tap)
        out[i] = np.dot(mixed, replica)
    return out
