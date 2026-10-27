"""Synthetic GNSS signals for tests."""

import numpy as np

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca, sample_code

L1_FREQ = 1575.42e6


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
) -> np.ndarray:
    """`n` complex64 samples of a GPS L1 C/A signal in white noise.

    The code period starts at sample index `code_offset` (wrapping before it); the
    carrier sits at `f_if + doppler` Hz and the code rate is Doppler-scaled to match.
    Signal amplitude is 1. For `iq=False` the result is real (a cosine carrier) with
    the noise variance chosen so the one-sided C/N0 is still `cn0_dbhz`. With
    `data_bits`, a random +/-1 bit multiplies the signal every 20 ms, bit edges
    aligned to code-period starts (counted from `code_offset`).
    """
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
    return (signal + noise).astype(np.complex64)
