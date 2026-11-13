"""Code and carrier tracking of one channel.

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

A pilot signal (GPS L5, Galileo E5a) carries one chip of a secondary code on every code
period, so its prompt changes sign from epoch to epoch until that code is known. Its
channel pulls in with the FLL on single epochs (`PILOT_FLL_STAGES`), runs the same Costas
loop, and from `T_SEC_SYNC` on tests the last prompts against the secondary code at the
current alignment (Pocket SDR's rule, `SEC_SYNC_RATIO`, here only while the Costas loop is
phase-locked). Once that sync is found the chip is removed from every prompt, and the
carrier loop becomes the same third-order update on the four-quadrant phase of
`T_PILOT_COH` coherent sums (Pocket SDR's pilot loop): whole-cycle phase, and 13 dB more
signal per update. A sum that carries no signal (its coherence below `COAST_COHERENCE`)
is not used: the loop coasts through an outage at the Doppler and Doppler rate it had
instead of updating on noise, which is this receiver's own. The data component of the
same signal is correlated at the pilot's timing and its symbols are kept; it steers
nothing.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import atan, atan2, floor, log10, pi

import numpy as np

from pyrx.codes import CA_LENGTH, CA_PERIOD
from pyrx.correlator import correlate
from pyrx.samples import SampleFile
from pyrx.signalspec import L1CA, SIGNALS, SignalSpec

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
# The same for a pilot before its secondary code is known: single epochs, since a sum over a
# sign change cancels. This is Pocket SDR's pull-in; at 30 dB-Hz it hands over 10 Hz off.
PILOT_FLL_STAGES = ((0.5, 5.0, 1), (0.5, 2.0, 1))
T_SEC_SYNC = 1.5  # s into a channel from which the secondary code is searched
SEC_SYNC_RATIO = 0.8  # the code's correlation with the last in-phase prompts over their mean size
# ... accepted only while the Costas loop is phase-locked: cos 2φ of the last prompts. Without
# this the 20-chip code matches noise within a second.
SEC_SYNC_LOCK = 0.5
SEC_SYNC_LOCK_EPOCHS = 100
T_PILOT_COH = 0.020  # s, the coherent sum of the pilot's carrier loop once the code is synced
# A coherent sum is taken into the pilot's loop when its coherence, |sum of the prompts|^2 over
# their count times their summed power, reaches COAST_COHERENCE: 1 on a clean signal, 1/20 on
# noise, 0.28 at 25 dB-Hz. Below it the sum carries no signal and the loop coasts.
COAST_COHERENCE = 0.20
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
_N_COH = round(T_PILOT_COH / CA_PERIOD)
if not (
    sum(round(seconds / CA_PERIOD) for seconds, _, _ in FLL_STAGES) == _N_PULLIN
    and sum(round(seconds / CA_PERIOD) for seconds, _, _ in PILOT_FLL_STAGES) == _N_PULLIN
    and _N_PULLIN <= round(T_CN0 / CA_PERIOD)
    and _N_PULLIN % _N_DLL == 0
    and T_SEC_SYNC >= T_FLL_PULLIN
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
    coasting: bool = False  # the last coherent sum carried no signal and was not used

    def fade_window(self, prompt: complex, closes_window: bool) -> None:
        """Take one epoch's prompt into the hold's power window; `closes_window` marks the
        epoch that ends a T_DLL window."""
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

    def update(self, prompt: complex, closes_window: bool) -> tuple[float, float]:
        """Take one epoch's prompt sum; returns (Doppler increment in Hz, phase error in
        cycles). `closes_window` marks the epoch that ends a T_DLL window. A zero in-phase
        prompt leaves the discriminator undefined: (0.0, nan), only the fade window advancing.
        """
        t = CA_PERIOD
        self.fade_window(prompt, closes_window)
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

    def update_coherent(self, total: complex, power: float, dt: float) -> tuple[float, float]:
        """The third-order update on the four-quadrant phase of `total`, the coherent sum of
        the prompts of `dt` seconds with their secondary chips removed; `power` is the sum
        of those prompts' squared magnitudes. Returns (Doppler increment in Hz, phase error
        in cycles). The previous error and the integrated term carry over from `update`,
        and so does its integrator hold: the caller keeps `fade_window` running epoch by
        epoch, and a sum completed while that flag is up (the last closed T_DLL window's
        prompt power below FADE_HOLD of its running mean) leaves the integrated term alone.

        A sum whose coherence is below COAST_COHERENCE is noise: its phase is uniform over
        the cycle and would random-walk the Doppler. The loop coasts on it instead: the
        discriminator reads zero, which takes the last error's proportional term back out
        of the Doppler, and the integrated term (the Doppler rate) keeps steering. Returns
        (increment, nan). One noise sum in seventy still reaches the threshold, so the
        first sum taken after a coasted one moves the Doppler by its proportional term
        only; the integrated terms take a sum's error when the sum before it was taken too.

        The two rules cover different fades. Behind an obstruction the signal is mostly
        attenuated and flickering, not gone: its weak sums pass the coherence gate with a
        noisy phase, and the hold is what keeps them out of the rate estimate that the
        loop then coasts on.
        """
        w = B_PLL / 0.7845
        n = round(dt / CA_PERIOD)
        if not (power > 0.0 and abs(total) ** 2 >= COAST_COHERENCE * n * power):
            increment = -2.4 * w * self.err_phase_prev + self.phase_acc * dt
            self.err_phase_prev = 0.0
            self.coasting = True
            return increment, np.nan
        err_phase = atan2(total.imag, total.real) / (2.0 * pi)
        if self.coasting:
            increment = 2.4 * w * (err_phase - self.err_phase_prev) + self.phase_acc * dt
        else:
            if not self.fading:
                self.phase_acc += w * w * w * err_phase * dt
            increment = (
                2.4 * w * (err_phase - self.err_phase_prev)
                + 1.1 * w * w * err_phase * dt
                + self.phase_acc * dt
            )
        self.err_phase_prev = err_phase
        self.coasting = False
        return increment, err_phase


def dll_update(sum_e: float, sum_l: float, code_length: int = CA_LENGTH) -> tuple[float, float]:
    """The DLL's update from the early and late magnitudes summed over T_DLL: returns (code
    error, code-time advance to apply), both in seconds of code time, for a code of
    `code_length` chips per period. A window without early or late power moves nothing:
    (nan, 0.0).
    """
    if not sum_e + sum_l > 0.0:
        return np.nan, 0.0
    t = CA_PERIOD
    err_code = (sum_e - sum_l) / (sum_e + sum_l) / 2.0 * t / code_length
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

    A pilot signal's channel (`signal` names the signal) adds:

    sec_sync: the epoch at which the secondary code was found, None if it never was.
        Up to and including it the channel runs as described above, on prompts that
        still carry the secondary code. From the next epoch on the carrier loop works
        on coherent sums of T_PILOT_COH: carrier_error is the four-quadrant phase of a
        sum, finite on the epoch that completes one the loop takes, nan on one it coasts
        on and between, and phase_lock is
        cos 2φ of the window's sums (1 when locked, -1 with the loop standing half its
        update rate off, near 0 when the carrier spins).
    sec_origin, sec_polarity: epoch k carries chip (k - sec_origin) modulo the code
        length of the secondary code, times sec_polarity (+1 or -1): the sign the
        Costas loop had settled on. With -1 the carrier NCO stands half a cycle from
        the pilot's phase, for the rest of the track. corr holds the prompts as
        correlated, the secondary chips not removed.
    data_prompt[k]: complex64, the data component's prompt sum of epoch k at the pilot's
        carrier and code timing, as correlated.
    symbols, symbol_epoch: the data component's soft symbols from the first symbol that
        starts after the sync, at epoch symbol_epoch: each the sum of its epochs'
        data prompts, their secondary chips and sec_polarity removed, along the signal's
        data axis; positive for a symbol 0. They run to the end of the track, the
        unlocked windows before a loss included. None without a sync.
    held[k]: True on the epoch completing a coherent sum the loop coasted on: the sum's
        coherence was below COAST_COHERENCE, so the signal was gone or too faded to use
        and the loop ran on at the Doppler and Doppler rate it had. After a stretch of
        them the phase is an extrapolation and can be whole cycles away while every
        window still reads phase-locked.
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
    signal: str = "L1CA"
    sec_sync: int | None = None
    sec_origin: int = 0
    sec_polarity: int = 0
    data_prompt: np.ndarray | None = None
    symbols: np.ndarray | None = None
    symbol_epoch: int | None = None
    held: np.ndarray | None = None


def _symbols(spec: SignalSpec, prn: int, data_prompt, sec_sync, sec_origin, sec_polarity):
    """Soft symbols of the data component from the first symbol starting after the sync:
    (symbols, epoch of the first one)."""
    data_sec = spec.data_secondary(prn)
    first = sec_sync + 1
    first += (sec_origin - first) % spec.symbol_epochs
    count = (len(data_prompt) - first) // spec.symbol_epochs
    if count <= 0:
        return np.empty(0, dtype=np.float32), first
    k = np.arange(first, first + count * spec.symbol_epochs)
    wiped = data_prompt[k] * (data_sec[(k - sec_origin) % len(data_sec)] * sec_polarity)
    sums = wiped.reshape(count, spec.symbol_epochs).sum(axis=1)
    return (sums * np.conj(spec.data_axis)).real.astype(np.float32), first


def track(
    source,
    f_if: float,
    spec: SignalSpec,
    prn: int,
    code_offset: float,
    doppler: float,
    cn0: float,
    t_start: float = 0.0,
    duration: float | None = None,
) -> TrackResult:
    """Track PRN `prn` of the signal `spec` in `source` (a SampleFile or ArraySamples) from
    an acquisition seed.

    `code_offset` (s) and `doppler` (Hz) are relative to `t_start`; `cn0` seeds the
    estimate. Stops at the end of the data (or of `duration`), or at loss of lock.
    """
    fs = source.fs
    t = CA_PERIOD
    n = round(fs * t)
    code = spec.code(prn)
    pilot = spec.pilot
    sec = spec.secondary(prn) if pilot else None
    data_code = (spec.data_code(prn),) if pilot else ()  # the data component's prompt too
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
    out_data = np.empty(n_max, dtype=np.complex64) if pilot else None
    out_held = np.zeros(n_max, dtype=bool) if pilot else None

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
    for seconds, b, n_sum in PILOT_FLL_STAGES if pilot else FLL_STAGES:
        n_end += round(seconds / t)
        fll_stages.append((n_end, b, n_sum))
    fll_stage = 0
    fll_acc = 0j  # the coherent sum of the current stage's epochs so far
    fll_n = 0  # epochs in it
    fll_prev = None  # the previous completed sum of this stage
    sum_e = sum_l = sum_p = sum_n = sum_d = 0.0
    # pilot: the secondary code's sync, then the coherent sum the carrier loop works on
    sec_sync = None
    sec_origin = sec_polarity = 0
    coh_acc = 0j
    coh_pow = 0.0
    coh_n = 0
    sum_ps = 0.0  # the power of what the phase-lock indicator is formed from
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
        chip_rate = spec.chip_rate * (1.0 + fd / spec.carrier)
        residual_chips = (start / fs - t_code) * chip_rate
        # float64 throughout the loop state: a float32 leaking into t_code would
        # limit the code time to ~1e-10 s and random-walk the code phase
        c = correlate(
            source.read(start, n), fs, fc, phi, code, chip_rate, residual_chips, TAPS, *data_code
        ).astype(np.complex128)
        prompt = complex(c[0])
        epoch = k + 1
        out_t[k] = start / fs
        out_phase[k] = phi
        out_corr[k] = c[:4]
        out_coff[k] = t_code % t
        if pilot:
            out_data[k] = c[4]
        lock_term = prompt  # what this epoch adds to the phase-lock indicator, None for nothing

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
        elif sec_sync is None:
            increment, out_err_carrier[k] = pll.update(prompt, epoch % n_dll == 0)
            fd += increment
        else:
            # the secondary chip removed, the loop works on coherent sums of T_PILOT_COH
            pll.fade_window(prompt, epoch % n_dll == 0)
            coh_acc += prompt * (sec[(k - sec_origin) % len(sec)] * sec_polarity)
            coh_pow += abs(prompt) ** 2
            coh_n += 1
            lock_term = None
            if coh_n == _N_COH:
                increment, out_err_carrier[k] = pll.update_coherent(coh_acc, coh_pow, T_PILOT_COH)
                fd += increment
                out_held[k] = pll.coasting
                lock_term = coh_acc
                coh_acc, coh_pow, coh_n = 0j, 0.0, 0

        # pilot: the secondary code at the current alignment against the last in-phase
        # prompts, accepted while the Costas loop is phase-locked; the chip is removed from
        # the next epoch on
        if (
            pilot
            and sec_sync is None
            and epoch * t >= T_SEC_SYNC
            and epoch >= max(len(sec), SEC_SYNC_LOCK_EPOCHS)
        ):
            last = out_corr[epoch - len(sec) : epoch, 0].real.astype(np.float64)
            match = float(np.dot(last, sec)) / len(sec)
            recent = out_corr[epoch - SEC_SYNC_LOCK_EPOCHS : epoch, 0].astype(np.complex128)
            coherent = float(np.sum(recent.real**2 - recent.imag**2))
            if abs(match) >= SEC_SYNC_RATIO * float(np.mean(np.abs(last))) and (
                coherent >= SEC_SYNC_LOCK * float(np.sum(np.abs(recent) ** 2))
            ):
                sec_sync = k
                sec_origin = k - (len(sec) - 1)
                sec_polarity = 1 if match > 0.0 else -1

        # code: non-coherent early/late DLL every T_DLL
        sum_e += float(abs(c[1]))
        sum_l += float(abs(c[2]))
        if epoch % n_dll == 0:
            out_err_code[k], advance = dll_update(sum_e, sum_l, spec.code_length)
            t_code -= advance
            sum_e = sum_l = 0.0

        # C/N0 and the phase-lock indicator every T_CN0; the loss rule at the window end
        sum_p += abs(prompt) ** 2
        sum_n += float(abs(c[3])) ** 2
        if lock_term is not None:
            sum_d += lock_term.real**2 - lock_term.imag**2
            if pilot:
                sum_ps += abs(lock_term) ** 2
        if epoch % n_cn0 == 0:
            if sum_p > 0.0 and sum_n > 0.0:
                s = max(sum_p - sum_n, 0.01 * sum_n)
                cn0 += 0.5 * (10.0 * log10(s / sum_n / t) - cn0)
            # a pilot's indicator is normalized by what it was formed from: single prompts
            # before the sync, coherent sums after
            lock_power = sum_ps if pilot else sum_p
            pli = sum_d / lock_power if lock_power > 0.0 else np.nan
            out_pli[k - n_cn0 + 1 : k + 1] = pli
            bad = cn0 < CN0_LOST or (epoch > n_cn0 and pli < PLI_LOCKED)
            bad_windows = bad_windows + 1 if bad else 0
            lost = bad_windows >= LOST_WINDOWS
            sum_p = sum_n = sum_d = sum_ps = 0.0

        # advance one code period with carrier aiding; the epoch spacing is the
        # Doppler-scaled code period, and the carrier phase integrates over it
        dt = t * (1.0 - fd / spec.carrier)
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

    symbols = symbol_epoch = None
    if sec_sync is not None:
        symbols, symbol_epoch = _symbols(
            spec, prn, out_data[:k], sec_sync, sec_origin, sec_polarity
        )
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
        signal=spec.name,
        sec_sync=sec_sync,
        sec_origin=sec_origin,
        sec_polarity=sec_polarity,
        data_prompt=out_data[:k] if pilot else None,
        symbols=symbols,
        symbol_epoch=symbol_epoch,
        held=out_held[:k] if pilot else None,
    )


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
    """Track GPS L1 C/A PRN `prn`: `track` with the L1 C/A signal."""
    return track(source, f_if, L1CA, prn, code_offset, doppler, cn0, t_start, duration)


def _track_job(args):
    path, fs, iq, f_if, signal, prn, code_offset, doppler, cn0, kwargs = args
    source = SampleFile(path, fs, iq)
    return track(source, f_if, SIGNALS[signal], prn, code_offset, doppler, cn0, **kwargs)


def track_channels(
    path: str | os.PathLike,
    fs: float,
    iq: bool,
    f_if: float,
    seeds: list[tuple[int, float, float, float]],
    workers: int | None = None,
    signal: str = "L1CA",
    **kwargs,
) -> list[TrackResult]:
    """Track several PRNs of one signal in one recording in parallel processes, one channel
    each.

    `seeds` holds `(prn, code_offset, doppler, cn0)` per channel; results come back
    in the same order. Each worker reopens the file, so nothing large is pickled in.
    """
    jobs = [(os.fspath(path), fs, iq, f_if, signal, *seed, kwargs) for seed in seeds]
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers <= 1 or len(jobs) == 1:
        return [_track_job(j) for j in jobs]
    # spawn, not fork: the parent holds BLAS threads and forking them is unsafe
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_track_job, jobs))
