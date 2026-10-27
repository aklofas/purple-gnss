"""Code and carrier tracking of one GPS L1 C/A channel.

Each 1 ms epoch correlates one code period at prompt, early, late and noise taps,
then updates the carrier (FLL during pull-in, Costas PLL after) and the code
offset (carrier aiding every epoch, non-coherent early/late DLL every 10 ms).
C/N0 is estimated from prompt versus noise-tap power over 1 s. The loop
constants follow Pocket SDR so the two trackers can be compared number for
number; they are not tuned for the hardware receiver.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import atan, floor, log10, pi

import numpy as np

from pyrx.acquisition import AcqResult
from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca
from pyrx.correlator import correlate
from pyrx.samples import SampleFile

L1_FREQ = 1575.42e6
B_PLL = 5.0  # Hz
B_FLL = (5.0, 2.0)  # Hz, first and second half of pull-in
B_DLL = 0.25  # Hz
T_FLL_PULLIN = 1.0  # s
T_DLL = 0.010  # s, non-coherent early/late accumulation
T_CN0 = 1.0  # s
CN0_LOST = 25.0  # dB-Hz
EL_SPACING = 0.25  # chips each side of prompt
NOISE_TAP = -10.0  # chips
TAPS = np.array([0.0, EL_SPACING, -EL_SPACING, NOISE_TAP])  # prompt, early, late, noise


@dataclass
class TrackResult:
    prn: int
    t: np.ndarray  # s, start of each epoch's integration window
    corr: np.ndarray  # complex64 (n_epochs, 4): prompt, early, late, noise
    code_offset: np.ndarray  # s, code-period start within [0, CA_PERIOD), after the update
    doppler: np.ndarray  # Hz
    carrier_phase: np.ndarray  # cycles, accumulated Doppler phase
    cn0: np.ndarray  # dB-Hz, held between 1 s updates
    locked: bool
    lost_at: float | None


def track_l1ca(
    source,
    f_if: float,
    prn: int,
    code_offset: float,
    doppler: float,
    cn0: float,
    t_start: float = 0.0,
    duration: float | None = None,
) -> TrackResult:
    """Track PRN `prn` in `source` (a SampleFile or ArraySamples) from an acquisition seed.

    `code_offset` (s) and `doppler` (Hz) are relative to `t_start`; `cn0` seeds the
    estimate. Stops at the end of the data (or of `duration`), or at loss of lock.
    """
    fs = source.fs
    t = CA_PERIOD
    n = round(fs * t)
    code = gps_ca(prn)
    end_sample = source.n_samples
    if duration is not None:
        end_sample = min(end_sample, round((t_start + duration) * fs))
    n_max = max(0, int((end_sample / fs - t_start) / t) + 1)

    out_t = np.empty(n_max)
    out_corr = np.empty((n_max, 4), dtype=np.complex64)
    out_coff = np.empty(n_max)
    out_fd = np.empty(n_max)
    out_adr = np.empty(n_max)
    out_cn0 = np.empty(n_max)

    t_code = t_start + code_offset  # s, start of the code period being integrated
    fd = doppler
    phi = 0.0  # carrier phase (cycles) at the current window start
    adr = 0.0
    err_phase_prev = 0.0
    prompt_prev = 0j
    sum_e = sum_l = sum_p = sum_n = 0.0
    n_dll = round(T_DLL / t)
    n_cn0 = round(T_CN0 / t)
    lost_at = None
    k = 0
    while True:
        start = round(t_code * fs)
        if start < 0 or start + n > end_sample or k >= n_max:
            break
        fc = f_if + fd
        chip_rate = CA_CHIP_RATE * (1.0 + fd / L1_FREQ)
        residual_chips = (start / fs - t_code) * chip_rate
        # float64 throughout the loop state: a float32 leaking into t_code would
        # limit the code time to ~1e-10 s and random-walk the code phase
        c = correlate(
            source.read(start, n), fs, fc, phi, code, chip_rate, residual_chips, TAPS
        ).astype(np.complex128)
        prompt = complex(c[0])
        epoch = k + 1

        # carrier: FLL during pull-in, Costas PLL after
        if epoch * t <= T_FLL_PULLIN:
            if k >= 1:
                dot = prompt.real * prompt_prev.real + prompt.imag * prompt_prev.imag
                cross = prompt.real * prompt_prev.imag - prompt.imag * prompt_prev.real
                if dot != 0.0:
                    b = B_FLL[0] if epoch * t < T_FLL_PULLIN / 2 else B_FLL[1]
                    fd -= b / 0.25 * atan(cross / dot) / (2.0 * pi)
        elif prompt.real != 0.0:
            err_phase = atan(prompt.imag / prompt.real) / (2.0 * pi)
            w = B_PLL / 0.53
            fd += 1.4 * w * (err_phase - err_phase_prev) + w * w * err_phase * t
            err_phase_prev = err_phase
        prompt_prev = prompt

        # code: non-coherent early/late DLL every T_DLL
        sum_e += float(abs(c[1]))
        sum_l += float(abs(c[2]))
        if epoch % n_dll == 0:
            if sum_e + sum_l > 0.0:
                err_code = (sum_e - sum_l) / (sum_e + sum_l) / 2.0 * t / len(code)
                t_code -= B_DLL / 0.25 * err_code * t * n_dll
            sum_e = sum_l = 0.0

        # C/N0 every T_CN0
        sum_p += abs(prompt) ** 2
        sum_n += float(abs(c[3])) ** 2
        if epoch % n_cn0 == 0:
            if sum_p > 0.0 and sum_n > 0.0:
                s = max(sum_p - sum_n, 0.01 * sum_n)
                cn0 += 0.5 * (10.0 * log10(s / sum_n / t) - cn0)
            sum_p = sum_n = 0.0

        # advance one code period with carrier aiding
        adr += fd * t
        t_code_next = t_code + t * (1.0 - fd / L1_FREQ)
        start_next = round(t_code_next * fs)
        phi += fc * (start_next - start) / fs
        phi -= floor(phi)
        t_code = t_code_next

        out_t[k] = start / fs
        out_corr[k] = c
        out_coff[k] = t_code % t
        out_fd[k] = fd
        out_adr[k] = adr
        out_cn0[k] = cn0
        k += 1
        if cn0 < CN0_LOST:
            lost_at = start / fs
            break

    return TrackResult(
        prn=prn,
        t=out_t[:k],
        corr=out_corr[:k],
        code_offset=out_coff[:k],
        doppler=out_fd[:k],
        carrier_phase=out_adr[:k],
        cn0=out_cn0[:k],
        locked=lost_at is None,
        lost_at=lost_at,
    )


def track_from_acq(source, f_if: float, acq: AcqResult, **kwargs) -> TrackResult:
    """Track from an acquisition result."""
    return track_l1ca(source, f_if, acq.prn, acq.code_offset, acq.doppler, acq.cn0, **kwargs)


def _track_job(args):
    path, fs, iq, f_if, prn, code_offset, doppler, cn0, kwargs = args
    return track_l1ca(SampleFile(path, fs, iq), f_if, prn, code_offset, doppler, cn0, **kwargs)


def track_channels(
    path: str | os.PathLike,
    fs: float,
    iq: bool,
    f_if: float,
    seeds: list[tuple[int, float, float, float]],
    workers: int | None = None,
    **kwargs,
) -> list[TrackResult]:
    """Track several PRNs of one recording in parallel processes, one channel each.

    `seeds` holds `(prn, code_offset, doppler, cn0)` per channel; results come back
    in the same order. Each worker reopens the file, so nothing large is pickled in.
    """
    jobs = [(os.fspath(path), fs, iq, f_if, *seed, kwargs) for seed in seeds]
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers <= 1 or len(jobs) == 1:
        return [_track_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_track_job, jobs))
