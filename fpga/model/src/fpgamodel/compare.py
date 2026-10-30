"""Post-correlation SNR of the model and of pyrx's float correlator on the same samples.

The loss of the hardware arithmetic (2-bit samples, 3-level LO) is measured as the ratio
of two SNRs computed the same way: the prompt power on the tracked PRN, less the noise
power, over the mean prompt power on PRNs that are not in the signal (the noise power),
summed over `n_ms` windows of one code period with a fixed NCO. The ratio is independent of
the correlators' gains; with eight absent PRNs the noise estimate scatters by about 0.1 dB.
pyrx's windows are sample-aligned 1 ms blocks and the model's are its dumps (the first partial
for a nonzero code offset): a misalignment of up to one code period, measured at 0.03 dB or
less on the recording.
"""

from math import log10

import numpy as np
from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca
from pyrx.correlator import correlate
from pyrx.signals import L1_FREQ

from fpgamodel.channel import Channel, nco_step
from fpgamodel.quantize import code_bits

PROMPT = np.array([0.0])


def start_phase(code_offset_s: float, chip_rate: float) -> tuple[int, int]:
    """Quarter-chip index and 32-bit fraction of the code phase at sample 0 when the code
    period starts `code_offset_s` seconds in."""
    q = (-code_offset_s * 4.0 * chip_rate) % 4092.0
    chip_q = int(np.floor(q))
    frac = round((q - chip_q) * 2**32)
    if frac == 2**32:
        chip_q, frac = (chip_q + 1) % 4092, 0
    return chip_q, frac


def _chip_rate(doppler: float) -> float:
    return CA_CHIP_RATE * (1.0 + doppler / L1_FREQ)


def prompt_snr_pyrx(samples, fs, f_if, doppler, code_offset_s, prn, wrong_prns, n_ms) -> float:
    """pyrx's float prompt correlator with a fixed NCO over `n_ms` one-period windows."""
    x = np.asarray(samples).astype(np.complex64)
    n = round(fs * CA_PERIOD)
    fc = f_if + doppler
    chip_rate = _chip_rate(doppler)
    power = {}
    for code_prn in (prn, *wrong_prns):
        code = gps_ca(code_prn)
        p = 0.0
        for m in range(n_ms):
            start = m * n
            phi = (fc * start / fs) % 1.0
            chips = (start / fs - code_offset_s) * chip_rate
            p += abs(correlate(x[start : start + n], fs, fc, phi, code, chip_rate, chips, PROMPT)[0]) ** 2
        power[code_prn] = p
    return _ratio(power, prn, wrong_prns)


def prompt_snr_model(codes, fs, f_if, doppler, code_offset_s, prn, wrong_prns, n_ms) -> float:
    """The bit-true channel with a fixed NCO over its first `n_ms` dumps."""
    chip_rate = _chip_rate(doppler)
    chip_q, frac = start_phase(code_offset_s, chip_rate)
    power = {}
    for code_prn in (prn, *wrong_prns):
        ch = Channel(code_bits(gps_ca(code_prn)))
        ch.set_steps(nco_step(f_if + doppler, fs), nco_step(4.0 * chip_rate, fs))
        ch.set_phase(0, frac, chip_q)
        ch.enabled = True
        dumps = ch.run(codes)[:n_ms]
        if len(dumps) < n_ms:
            raise ValueError(f"{len(codes)} samples hold only {len(dumps)} code periods")
        power[code_prn] = sum(float(d.ip) ** 2 + float(d.qp) ** 2 for d in dumps)
    return _ratio(power, prn, wrong_prns)


def _ratio(power, prn, wrong_prns) -> float:
    noise = sum(power[w] for w in wrong_prns) / len(wrong_prns)
    snr = power[prn] / noise - 1.0  # the tracked PRN's power holds signal plus noise
    if not noise > 0.0 or not snr > 0.0:
        raise ValueError(f"PRN {prn} is not above the noise of PRNs {wrong_prns}: SNR {snr:.3g}")
    return snr


def loss_db(snr_ref: float, snr_model: float) -> float:
    return 10.0 * log10(snr_ref / snr_model)
