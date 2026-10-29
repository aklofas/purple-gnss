"""Synthetic GNSS signals for tests."""

import numpy as np

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca, sample_code
from pyrx.signals import L1_FREQ

BLOCK = 1 << 20  # samples generated at a time, so the float64 intermediates stay small


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
    chip_rate = CA_CHIP_RATE * (1.0 + doppler / L1_FREQ)
    phase_chips = -code_offset * chip_rate / fs
    noise_var = fs / 10 ** (cn0_dbhz / 10.0)
    period = round(fs * CA_PERIOD)
    # drawn in one order whatever the block size: the in-phase noise of every sample,
    # then the quadrature noise, then the data bits
    noise_i = rng.standard_normal(n)
    noise_q = rng.standard_normal(n) if iq and data_bits else None
    sym = None
    if data_bits:
        n_bits = (n - code_offset) // (20 * period) + 2
        sym = rng.choice([-1.0, 1.0], size=n_bits)
    elif bits is not None:
        sym = np.where(np.asarray(bits) == 1, -1.0, 1.0)
    code = gps_ca(prn)
    out = np.empty(n, dtype=np.complex64)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        k = np.arange(start, stop)
        t = k.astype(np.float64) / fs
        chips = sample_code(code, chip_rate, fs, stop - start, phase_chips, start)
        phase = 2.0 * np.pi * (f_if + doppler) * t
        if iq:
            signal = chips * np.exp(1j * phase)
            q = noise_q[start:stop] if noise_q is not None else rng.standard_normal(stop - start)
            noise = np.sqrt(noise_var / 2.0) * (noise_i[start:stop] + 1j * q)
        else:
            signal = chips * np.cos(phase)
            noise = np.sqrt(noise_var / 4.0) * noise_i[start:stop]
        if data_bits:
            signal = signal * sym[(k - code_offset) // (20 * period) + 1]  # bit 0 before offset
        elif bits is not None:
            bit_index = (k - code_offset) // (20 * period)  # bit 0 starts at code_offset
            signal = signal * sym[np.clip(bit_index, 0, len(sym) - 1)]
        out[start:stop] = signal + noise
    return out
