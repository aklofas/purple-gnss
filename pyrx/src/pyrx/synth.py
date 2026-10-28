"""Synthetic GNSS signals for tests."""

import numpy as np

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca, sample_code
from pyrx.signals import L1_FREQ


def synth_l1ca(
    prn: int,
    fs: float,
    f_if: float,
    doppler: float,
    code_offset: int,
    n: int,
    cn0_dbhz: float,
    iq: bool,
    seed: int = 0,
    data_bits: bool = False,
    bits: np.ndarray | None = None,
) -> np.ndarray:
    """`n` complex64 samples of a GPS L1 C/A signal in white noise.

    The code period starts at sample index `code_offset` (wrapping before it); the
    carrier sits at `f_if + doppler` Hz and the code rate is Doppler-scaled to match.
    Signal amplitude is 1. For `iq=False` the result is real (a cosine carrier) with
    the noise variance chosen so the one-sided C/N0 is still `cn0_dbhz`. With
    `data_bits`, a random +/-1 bit multiplies the signal every 20 ms, bit edges
    aligned to code-period starts (counted from `code_offset`). With `bits`
    (0/1, one per 20 ms, bit 0 starting at `code_offset`), a bit value 1
    multiplies the signal by -1 and samples beyond either end hold the nearest bit.
    """
    if data_bits and bits is not None:
        raise ValueError("data_bits and bits cannot both be given")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / fs
    chip_rate = CA_CHIP_RATE * (1.0 + doppler / L1_FREQ)
    code = sample_code(gps_ca(prn), chip_rate, fs, n, phase_chips=-code_offset * chip_rate / fs)
    phase = 2.0 * np.pi * (f_if + doppler) * t
    noise_var = fs / 10 ** (cn0_dbhz / 10.0)
    if iq:
        signal = code * np.exp(1j * phase)
        noise = np.sqrt(noise_var / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    else:
        signal = code * np.cos(phase)
        noise = np.sqrt(noise_var / 4.0) * rng.standard_normal(n)
    if data_bits:
        period = round(fs * CA_PERIOD)
        n_bits = (n - code_offset) // (20 * period) + 2
        bits = rng.choice([-1.0, 1.0], size=n_bits)
        bit_index = (np.arange(n) - code_offset) // (20 * period) + 1  # bit 0 before offset
        signal = signal * bits[bit_index]
    elif bits is not None:
        period = round(fs * CA_PERIOD)
        bit_index = (np.arange(n) - code_offset) // (20 * period)  # bit 0 starts at code_offset
        sym = np.where(np.asarray(bits) == 1, -1.0, 1.0)
        signal = signal * sym[np.clip(bit_index, 0, len(sym) - 1)]
    return (signal + noise).astype(np.complex64)
