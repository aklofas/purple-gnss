"""Parallel code-phase search (PCPS) acquisition.

For each Doppler bin the samples are mixed to baseband and circularly correlated
with one code period through the FFT; the squared magnitudes of every whole code
period in the buffer are summed non-coherently. The peak of the resulting
Doppler x code-offset grid gives the signal's code offset and Doppler, and the
ratio of the peak to the grid mean gives a C/N0 estimate.
"""

from dataclasses import dataclass

import numpy as np
from scipy import fft

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca, sample_code

DETECT_THRESHOLD = 38.0  # dB-Hz


@dataclass(frozen=True)
class AcqResult:
    prn: int
    code_offset: float  # seconds from the buffer start to the start of a code period
    doppler: float  # Hz, after quadratic interpolation between bins
    cn0: float  # dB-Hz
    detected: bool
    grid: np.ndarray  # float32, shape (len(dopplers), samples per period), peak normalised to 1
    dopplers: np.ndarray  # Hz, bin centres


def acquire(
    samples: np.ndarray,
    fs: float,
    f_if: float,
    code: np.ndarray,
    chip_rate: float,
    period: float,
    doppler_max: float = 5000.0,
    doppler_step: float | None = None,
    threshold: float = DETECT_THRESHOLD,
    prn: int = 0,
) -> AcqResult:
    """Search `samples` for `code` over Doppler +/- `doppler_max` around `f_if`."""
    n = round(fs * period)
    if len(samples) < n:
        raise ValueError(f"need at least {n} samples (one code period), got {len(samples)}")
    if doppler_step is None:
        doppler_step = 0.5 / period
    dopplers = np.arange(-doppler_max, doppler_max + doppler_step / 2, doppler_step)
    code_fft = np.conj(fft.fft(sample_code(code, chip_rate, fs, n)))
    n_blocks = len(samples) // n
    k = np.arange(n, dtype=np.float64)
    grid = np.zeros((len(dopplers), n), dtype=np.float32)
    for i, fd in enumerate(dopplers):
        carrier = np.exp(-2j * np.pi * (f_if + fd) / fs * k).astype(np.complex64)
        for b in range(n_blocks):
            x = samples[b * n : (b + 1) * n] * carrier
            corr = fft.ifft(fft.fft(x, workers=-1) * code_fft, workers=-1)
            grid[i] += np.abs(corr).astype(np.float32) ** 2
    bin_idx, col = np.unravel_index(np.argmax(grid), grid.shape)
    p_max = grid[bin_idx, col]
    p_mean = float(np.mean(grid))
    cn0 = 10.0 * np.log10((p_max - p_mean) / p_mean / period) if p_mean > 0 else 0.0
    doppler = _fine_doppler(grid[:, col], dopplers, bin_idx)
    return AcqResult(
        prn=prn,
        code_offset=col / fs,
        doppler=float(doppler),
        cn0=float(cn0),
        detected=bool(cn0 >= threshold),
        grid=grid / p_max if p_max > 0 else grid,
        dopplers=dopplers,
    )


def _fine_doppler(column: np.ndarray, dopplers: np.ndarray, i: int) -> float:
    """Vertex of the parabola through the peak bin and its neighbours; edge bins as-is."""
    if i == 0 or i == len(dopplers) - 1:
        return float(dopplers[i])
    a, b, _ = np.polyfit(dopplers[i - 1 : i + 2], column[i - 1 : i + 2].astype(np.float64), 2)
    return float(-b / (2.0 * a)) if a != 0 else float(dopplers[i])


def acquire_l1ca(samples: np.ndarray, fs: float, f_if: float, prn: int, **kwargs) -> AcqResult:
    """PCPS acquisition of GPS L1 C/A PRN `prn`."""
    return acquire(samples, fs, f_if, gps_ca(prn), CA_CHIP_RATE, CA_PERIOD, prn=prn, **kwargs)
