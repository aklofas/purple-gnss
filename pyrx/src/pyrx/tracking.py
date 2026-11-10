"""Code and carrier tracking of one GPS L1 C/A channel.

Each 1 ms epoch correlates one code period at prompt, early, late and noise taps,
then updates the carrier (FLL during pull-in, its discriminator on coherent sums of
2 then 5 epochs, `FLL_STAGES`, so that a weak channel hands over within a hertz;
third-order Costas PLL after, so that a moving antenna's Doppler rate leaves no
phase lag, its integrator held while the prompt power is down so that an obstruction's
onset does not wind it up) and the code offset (carrier aiding every epoch, non-coherent
early/late DLL every 10 ms).
C/N0 is estimated from prompt versus noise-tap power over 1 s, and over the same
window a carrier phase-lock indicator, cos 2φ of the prompt (1 when phase-locked,
near 0 when the carrier spins or there is only noise). A track ends after two
consecutive windows with C/N0 below `CN0_LOST` or the indicator below `PLI_LOCKED`.
The constants follow Pocket SDR's trackers (the DLL and the C/N0 window its Python
tracker's, the third-order PLL its C tracker's) so the loops can be compared with
them number for number; the FLL on coherent sums, the integrator hold and the
two-window loss rule are this receiver's own. Nothing is tuned for the hardware
receiver yet.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import atan, floor, log10, pi

import numpy as np

from pyrx.codes import CA_CHIP_RATE, CA_LENGTH, CA_PERIOD, gps_ca
from pyrx.correlator import correlate
from pyrx.samples import SampleFile
from pyrx.signals import L1_FREQ

B_PLL = 5.0  # Hz, the third-order Costas loop's noise bandwidth
# The PLL's integrator is held while the prompt power summed over T_DLL is below FADE_HOLD of
# its running mean (exponential, time constant T_FADE_REF): an obstruction's onset otherwise
# winds it up on the faded signal and slips a strong channel.
FADE_HOLD = 0.5
T_FADE_REF = 0.5  # s
# FLL pull-in stages: (seconds, bandwidth Hz, epochs summed coherently before the discriminator).
# The sum length sets the pull-in range, ±1/(4 sum) = ±125 Hz then ±50 Hz, and divides the
# discriminator's noise: a weak satellite's acquisition seed can be 100 Hz off, and at 30 dB-Hz
# one epoch alone hands over 10 Hz off.
FLL_STAGES = ((0.3, 5.0, 2), (0.7, 2.0, 5))
B_DLL = 0.25  # Hz
T_FLL_PULLIN = 1.0  # s
T_DLL = 0.010  # s, non-coherent early/late accumulation
FADE_REF_GAIN = T_DLL / T_FADE_REF  # the running mean's update per 10 ms sum
T_CN0 = 1.0  # s
CN0_LOST = 25.0  # dB-Hz
PLI_LOCKED = 0.25  # cos 2φ of the prompt over a C/N0 window; phase-locked above it
LOST_WINDOWS = 2  # consecutive bad windows that end a track
EL_SPACING = 0.25  # chips each side of prompt
NOISE_TAP = -10.0  # chips
TAPS = np.array([0.0, EL_SPACING, -EL_SPACING, NOISE_TAP])  # prompt, early, late, noise

# Relations the loop relies on: the stages make up the pull-in; the loss rule exempts only the
# first C/N0 window from the phase-lock test because the pull-in lies within it; the first
# fade-power sum after the handover covers a whole DLL period because the pull-in ends on a
# DLL boundary.
_N_PULLIN = round(T_FLL_PULLIN / CA_PERIOD)
_N_DLL = round(T_DLL / CA_PERIOD)
if not (
    sum(round(seconds / CA_PERIOD) for seconds, _, _ in FLL_STAGES) == _N_PULLIN
    and _N_PULLIN <= round(T_CN0 / CA_PERIOD)
    and _N_PULLIN % _N_DLL == 0
):
    raise ValueError("the tracking constants break a relation the loop relies on")


@dataclass
class CarrierLoop:
    """The third-order Costas loop with its integrator hold, from the FLL handover on.

    One instance per channel, updated once per epoch. Every receiver built on these loops
    (the offline tracker here, the model of the hardware channel's loop) calls this update,
    so their carrier loops are the same arithmetic.
    """

    err_phase_prev: float = 0.0
    phase_acc: float = 0.0  # the integrated phase term, zero until the handover
    fade_sum: float = 0.0  # prompt power over the current T_DLL window
    fade_ref: float = 0.0  # its running mean over the windows before
    fading: bool = False  # the integrator is held over the current window

    def update(self, prompt: complex, closes_window: bool) -> tuple[float, float]:
        """Take one epoch's prompt sum; returns (Doppler increment in Hz, phase error in
        cycles). `closes_window` marks the epoch that ends a T_DLL window. A zero in-phase
        prompt leaves the discriminator undefined: (0.0, nan), only the fade window advancing.
        """
        t = CA_PERIOD
        # the integrator is held while the 10 ms prompt power is below FADE_HOLD of its
        # running mean: at an obstruction's onset it would wind up on the faded signal
        # and slip the loop. The window closes on schedule whatever the prompt.
        self.fade_sum += abs(prompt) ** 2
        if closes_window:
            self.fading = self.fade_ref > 0.0 and self.fade_sum < FADE_HOLD * self.fade_ref
            self.fade_ref = (
                self.fade_sum
                if self.fade_ref == 0.0
                else self.fade_ref + FADE_REF_GAIN * (self.fade_sum - self.fade_ref)
            )
            self.fade_sum = 0.0
        if prompt.real == 0.0:
            return 0.0, np.nan
        # third order as Pocket SDR's C tracker (a3 = 1.1, b3 = 2.4, W = Bn / 0.7845): no
        # steady-state lag on a Doppler ramp
        err_phase = atan(prompt.imag / prompt.real) / (2.0 * pi)
        w = B_PLL / 0.7845
        if not self.fading:
            self.phase_acc += w * w * w * err_phase * t
        increment = (
            2.4 * w * (err_phase - self.err_phase_prev)
            + 1.1 * w * w * err_phase * t
            + self.phase_acc * t
        )
        self.err_phase_prev = err_phase
        return increment, err_phase


def dll_update(sum_e: float, sum_l: float) -> tuple[float, float]:
    """The DLL's update from the early and late magnitudes summed over T_DLL: returns (code
    error, code-time advance to apply), both in seconds of code time. A window without
    early or late power moves nothing: (nan, 0.0).
    """
    if not sum_e + sum_l > 0.0:
        return np.nan, 0.0
    t = CA_PERIOD
    err_code = (sum_e - sum_l) / (sum_e + sum_l) / 2.0 * t / CA_LENGTH
    return err_code, B_DLL / 0.25 * err_code * t * _N_DLL


@dataclass
class TrackResult:
    """Per-epoch channel state; epoch k integrates one code period.

    t[k]: s from the source start, the first sample of epoch k's window (the code
        start rounded to the sample grid).
    corr[k]: complex64 prompt, early, late, noise correlator sums of epoch k.
    code_offset[k]: s, the code-period start used for epoch k, reduced into
        [0, CA_PERIOD). The exact start is t[k] + wrap(code_offset[k] - t[k]),
        within half a sample of t[k].
    doppler[k]: Hz, after epoch k's loop update (applies from epoch k+1 on).
    carrier_phase[k]: cycles, accumulated Doppler phase at the end of epoch k:
        the integral over the epoch spacing of the Doppler the NCO applied
        during epoch k, which is doppler[k-1] (doppler[k] takes effect from
        epoch k+1 on). Every channel's
        carrier NCO starts at the receiver's IF phase at sample 0, f_if * t[0]
        modulo one cycle, so the accumulated phases of different channels share
        one origin and are comparable to a fraction of a cycle.
    phase[k]: cycles in [0, 1), the carrier NCO phase at t[k].
    cn0[k]: dB-Hz, a causal estimate: the seed during the first 1 s window, then at
        each window's end the previous value moved halfway to that window's estimate
        and held until the next. The value at an epoch never includes the window
        the epoch is in, so it starts to follow a change in signal strength within
        a second and closes half of the remaining gap every second after. It is what a receiver can
        report in real time, and what the observations carry as S1C.
    phase_lock[k]: carrier phase-lock indicator of the 1 s window containing epoch k,
        the sum over the window of I_P^2 - Q_P^2 over the sum of I_P^2 + Q_P^2 (cos 2φ
        in [-1, 1], 1 when phase-locked of either data polarity, near 0 when the
        carrier spins or there is only noise), written at the window's end; unlike
        cn0 it describes the epoch's own window. nan in an incomplete final window.
    locked, lost_at: a track ends at the epoch completing the LOST_WINDOWS-th consecutive
        bad window, bad meaning cn0 below CN0_LOST or, from the second window on (the
        first is the FLL pull-in), phase_lock below PLI_LOCKED; lost_at is that epoch's
        t, None when the track ran to the end of the data.
    carrier_error[k]: cycles, the carrier discriminator of epoch k: during pull-in the
        FLL's frequency discriminator (minus the phase change over one coherent sum of
        the stage's epochs, FLL_STAGES) on the epochs that complete a sum with a
        predecessor in the same stage, nan on the other pull-in epochs; the Costas phase
        error after it; nan where the discriminator's denominator is zero (dot product of
        consecutive sums in the FLL, prompt in-phase sum in the PLL).
    code_error[k]: s of code time, the early-minus-late discriminator on the epochs
        that update the DLL, nan between them and where the accumulated early and late
        magnitudes are both zero.
    """

    prn: int
    t: np.ndarray
    corr: np.ndarray
    code_offset: np.ndarray
    doppler: np.ndarray
    carrier_phase: np.ndarray
    phase: np.ndarray
    cn0: np.ndarray
    phase_lock: np.ndarray
    locked: bool
    lost_at: float | None
    carrier_error: np.ndarray
    code_error: np.ndarray


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
    out_phase = np.empty(n_max)
    out_cn0 = np.empty(n_max)
    out_pli = np.full(n_max, np.nan)
    out_err_carrier = np.full(n_max, np.nan)
    out_err_code = np.full(n_max, np.nan)

    # plain floats throughout: a numpy float32 seed would otherwise propagate into
    # t_code and limit the code time to ~1e-10 s
    t_start = float(t_start)
    f_if = float(f_if)
    t_code = t_start + float(code_offset)  # s, start of the code period being integrated
    fd = float(doppler)
    cn0 = float(cn0)
    # carrier NCO phase (cycles) at the current window start, referenced to the
    # IF phase at sample 0 so that all channels share one carrier-phase origin
    phi = (f_if * round(t_code * fs) / fs) % 1.0
    adr = 0.0
    pll = CarrierLoop()
    # pull-in stages as (last epoch, bandwidth, epochs per sum); the sums restart at each stage
    fll_stages = []
    n_end = 0
    for seconds, b, n_sum in FLL_STAGES:
        n_end += round(seconds / t)
        fll_stages.append((n_end, b, n_sum))
    fll_stage = 0
    fll_acc = 0j  # the coherent sum of the current stage's epochs so far
    fll_n = 0  # epochs in it
    fll_prev = None  # the previous completed sum of this stage
    sum_e = sum_l = sum_p = sum_n = sum_d = 0.0
    n_dll = _N_DLL
    n_cn0 = round(T_CN0 / t)
    bad_windows = 0
    lost = False
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
        out_t[k] = start / fs
        out_phase[k] = phi
        out_corr[k] = c
        out_coff[k] = t_code % t

        # carrier: FLL during pull-in on coherent sums of prompts, Costas PLL after
        if epoch * t <= T_FLL_PULLIN:
            if epoch > fll_stages[fll_stage][0]:
                fll_stage += 1
                fll_acc, fll_n, fll_prev = 0j, 0, None
            _, b, n_sum = fll_stages[fll_stage]
            fll_acc += prompt
            fll_n += 1
            if fll_n == n_sum:
                if fll_prev is not None:
                    dot = fll_acc.real * fll_prev.real + fll_acc.imag * fll_prev.imag
                    cross = fll_acc.real * fll_prev.imag - fll_acc.imag * fll_prev.real
                    if dot != 0.0:
                        angle = atan(cross / dot)
                        out_err_carrier[k] = angle / (2.0 * pi)
                        fd -= b / 0.25 * angle / (2.0 * pi)  # reassociating moves it by an ulp
                fll_prev, fll_acc, fll_n = fll_acc, 0j, 0
        else:
            increment, out_err_carrier[k] = pll.update(prompt, epoch % n_dll == 0)
            fd += increment

        # code: non-coherent early/late DLL every T_DLL
        sum_e += float(abs(c[1]))
        sum_l += float(abs(c[2]))
        if epoch % n_dll == 0:
            out_err_code[k], advance = dll_update(sum_e, sum_l)
            t_code -= advance
            sum_e = sum_l = 0.0

        # C/N0 and the phase-lock indicator every T_CN0; the loss rule at the window end
        sum_p += abs(prompt) ** 2
        sum_n += float(abs(c[3])) ** 2
        sum_d += prompt.real**2 - prompt.imag**2
        if epoch % n_cn0 == 0:
            if sum_p > 0.0 and sum_n > 0.0:
                s = max(sum_p - sum_n, 0.01 * sum_n)
                cn0 += 0.5 * (10.0 * log10(s / sum_n / t) - cn0)
            pli = sum_d / sum_p if sum_p > 0.0 else np.nan
            out_pli[k - n_cn0 + 1 : k + 1] = pli
            bad = cn0 < CN0_LOST or (epoch > n_cn0 and pli < PLI_LOCKED)
            bad_windows = bad_windows + 1 if bad else 0
            lost = bad_windows >= LOST_WINDOWS
            sum_p = sum_n = sum_d = 0.0

        # advance one code period with carrier aiding; the epoch spacing is the
        # Doppler-scaled code period, and the carrier phase integrates over it
        dt = t * (1.0 - fd / L1_FREQ)
        adr += (fc - f_if) * dt  # the Doppler the NCO ran with during this epoch
        t_code_next = t_code + dt
        start_next = round(t_code_next * fs)
        phi += fc * (start_next - start) / fs
        phi -= floor(phi)
        t_code = t_code_next

        out_fd[k] = fd
        out_adr[k] = adr
        out_cn0[k] = cn0
        k += 1
        if lost:
            lost_at = start / fs
            break

    return TrackResult(
        prn=prn,
        t=out_t[:k],
        corr=out_corr[:k],
        code_offset=out_coff[:k],
        doppler=out_fd[:k],
        carrier_phase=out_adr[:k],
        phase=out_phase[:k],
        cn0=out_cn0[:k],
        phase_lock=out_pli[:k],
        locked=lost_at is None,
        lost_at=lost_at,
        carrier_error=out_err_carrier[:k],
        code_error=out_err_code[:k],
    )


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
    # spawn, not fork: the parent holds BLAS threads and forking them is unsafe
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_track_job, jobs))
