"""Tracking loops closed around the channel's dumps, the way the MCU will close them.

`Tracker` reproduces pyrx.tracking's discriminators and loop constants (FLL pull-in then a
third-order Costas PLL for the carrier, its integrator held while the prompt power is down,
a non-coherent early-minus-late DLL every 10 ms for the code) and
turns them into the NCO step words a channel takes. Five deliberate differences from pyrx:

1. pyrx moves its correlation window in time at each DLL update, while a channel can only
   change its code NCO rate, so the DLL output becomes a chip-rate offset applied over the
   following 10 ms, with carrier aiding folded into the code step.
2. Step words written after dump k take effect at dump k + 1, one epoch later than pyrx
   applies its update.
3. pyrx estimates C/N0 from a noise correlator tap the channel does not have; the tracker
   uses the second and fourth moments of the prompt magnitude instead, which data bits and
   the residual carrier of pull-in leave alone, and it does not declare a loss of lock from
   a window that overlaps pull-in.
4. The FLL pull-in is the 1 ms two-stage one (5 Hz, then 2 Hz); pyrx pulls in on 2 ms then
   5 ms coherent sums, so the two Dopplers agree only from the handover on.
5. pyrx ends a track after two consecutive 1 s windows that are bad by C/N0 (below CN0_LOST)
   or by its phase-lock indicator (cos 2φ of the prompt below PLI_LOCKED); the tracker has
   no indicator and ends a track at the first window after pull-in whose smoothed C/N0 is
   below CN0_LOST. The firmware is to take the moments C/N0 of difference 3 together with
   pyrx's indicator and two-window rule, which the dumps' prompt I/Q allow.
"""

from dataclasses import dataclass
from math import atan, log10, pi, sqrt

import numpy as np
from pyrx.codes import CA_CHIP_RATE, CA_LENGTH, CA_PERIOD
from pyrx.signals import L1_FREQ
from pyrx.tracking import (
    B_DLL,
    B_PLL,
    CN0_LOST,
    FADE_HOLD,
    FADE_REF_GAIN,
    T_CN0,
    T_DLL,
    T_FLL_PULLIN,
)

from fpgamodel.channel import Channel, Dump, nco_step

B_FLL = (5.0, 2.0)  # Hz, halves of a 1 ms two-stage pull-in (pyrx sums 2 ms then 5 ms instead)


@dataclass(frozen=True)
class Epoch:
    """Loop state after one dump: the dump's sample count (up to and including its last
    sample), the Doppler the loop then set, and the code fraction (quarter chips) the code
    NCO held at that sample."""

    sample: int
    doppler: float
    code_frac_q: float


class Tracker:
    """`skip_first` drops the first dump: a channel started mid-period by `set_phase` (as
    `compare.start_phase` places it) dumps a partial period first, which the loops must not
    see. The bench and the firmware set it whenever they start a channel that way."""

    def __init__(
        self, fs: float, f_if: float, doppler: float, cn0: float = 45.0, skip_first: bool = False
    ):
        self.fs = fs
        self.f_if = f_if
        self.doppler = float(doppler)
        self.cn0 = float(cn0)
        self.epoch = 0
        self.locked = True
        self.rate_offset = 0.0  # chips per second added to the Doppler-scaled chip rate
        self._skip = skip_first
        self._last_epoch: int | None = None
        self._prompt_prev = 0j
        self._err_phase_prev = 0.0
        self._phase_acc = 0.0  # the PLL's integrated phase term, zero until the handover
        self._fade_sum = self._fade_ref = 0.0  # 10 ms prompt power and its running mean
        self.fading = False  # the integrator is held over the current 10 ms window
        self._sum_e = self._sum_l = 0.0
        self._prompt_mag2: list[float] = []
        self._n_dll = round(T_DLL / CA_PERIOD)
        self._n_cn0 = round(T_CN0 / CA_PERIOD)

    def chip_rate(self) -> float:
        return CA_CHIP_RATE * (1.0 + self.doppler / L1_FREQ) + self.rate_offset

    def steps(self) -> tuple[int, int]:
        return nco_step(self.f_if + self.doppler, self.fs), nco_step(4.0 * self.chip_rate(), self.fs)

    def update(self, dump: Dump) -> tuple[int, int]:
        """Take one dump, update the loops, return the next (carrier step, code step).

        The dumps must come in order with none missing (the schedules count epochs and the
        FLL differences consecutive prompts): a dump whose epoch is not the previous one
        plus one raises ValueError, and the caller ends the track there.
        """
        t = CA_PERIOD
        if self._last_epoch is not None and dump.epoch != self._last_epoch + 1:
            raise ValueError(
                f"dump epoch {dump.epoch} after {self._last_epoch}: the loops need every dump in order"
            )
        self._last_epoch = dump.epoch
        if self._skip:
            self._skip = False  # the partial first period: no update, no epoch
            return self.steps()
        self.epoch += 1
        prompt = complex(dump.ip, dump.qp)
        if self.epoch * t <= T_FLL_PULLIN:
            if self.epoch > 1:
                prev = self._prompt_prev
                dot = prompt.real * prev.real + prompt.imag * prev.imag
                cross = prompt.real * prev.imag - prompt.imag * prev.real
                if dot != 0.0:
                    b = B_FLL[0] if self.epoch * t < T_FLL_PULLIN / 2 else B_FLL[1]
                    self.doppler -= b / 0.25 * atan(cross / dot) / (2.0 * pi)
        else:
            # the integrator is held while the 10 ms prompt power is below FADE_HOLD of its
            # running mean, as pyrx does: an obstruction's onset would wind it up. The window
            # closes on schedule whatever the prompt; integer dumps with ip == 0 are routine.
            self._fade_sum += abs(prompt) ** 2
            if self.epoch % self._n_dll == 0:
                self.fading = self._fade_ref > 0.0 and self._fade_sum < FADE_HOLD * self._fade_ref
                self._fade_ref = (
                    self._fade_sum
                    if self._fade_ref == 0.0
                    else self._fade_ref + FADE_REF_GAIN * (self._fade_sum - self._fade_ref)
                )
                self._fade_sum = 0.0
            if prompt.real != 0.0:  # the discriminator is undefined on a zero in-phase sum
                # third-order Costas loop as pyrx (Pocket SDR's C tracker: a3 = 1.1, b3 = 2.4,
                # W = Bn / 0.7845), the same expression order so the increments match bit for bit
                err_phase = atan(prompt.imag / prompt.real) / (2.0 * pi)
                w = B_PLL / 0.7845
                if not self.fading:
                    self._phase_acc += w * w * w * err_phase * t
                self.doppler += (
                    2.4 * w * (err_phase - self._err_phase_prev)
                    + 1.1 * w * w * err_phase * t
                    + self._phase_acc * t
                )
                self._err_phase_prev = err_phase
        self._prompt_prev = prompt

        self._sum_e += abs(complex(dump.ie, dump.qe))
        self._sum_l += abs(complex(dump.il, dump.ql))
        if self.epoch % self._n_dll == 0:
            self.rate_offset = 0.0  # a window without early or late power moves nothing, as in pyrx
            if self._sum_e + self._sum_l > 0.0:
                err_code = (self._sum_e - self._sum_l) / (self._sum_e + self._sum_l) / 2.0 * t / CA_LENGTH
                advance_s = B_DLL / 0.25 * err_code * t * self._n_dll  # pyrx moves t_code by this
                self.rate_offset = advance_s * CA_CHIP_RATE / T_DLL
            self._sum_e = self._sum_l = 0.0

        # C/N0 every T_CN0 from the moments of the prompt magnitude: for a signal of power S
        # in complex noise of power N, E|z|^2 = S + N and E|z|^4 = S^2 + 4SN + 2N^2, so
        # S = sqrt(2 M2^2 - M4); independent of data bits and of the carrier phase. Noisier
        # than a noise tap near the loss threshold (about 1.6 dB per window at 28 dB-Hz, 0.3 dB
        # at 35), so a satellite near 28 dB-Hz reads below CN0_LOST a few percent of the time;
        # with the half-gain smoothing the estimate itself crosses CN0_LOST only after about
        # two low readings in a row, and one crossing ends the track (difference 5 above).
        self._prompt_mag2.append(abs(prompt) ** 2)
        if self.epoch % self._n_cn0 == 0:
            mag2 = np.array(self._prompt_mag2)
            m2, m4 = mag2.mean(), (mag2 * mag2).mean()
            s = sqrt(max(2.0 * m2 * m2 - m4, 0.0))
            n = m2 - s
            if n > 0.0:
                s = max(s, 0.01 * n)
                self.cn0 += 0.5 * (10.0 * log10(s / n / t) - self.cn0)
            self._prompt_mag2.clear()
            if self.epoch * t > T_FLL_PULLIN and self.cn0 < CN0_LOST:
                self.locked = False
        return self.steps()


def track_model(ch: Channel, tracker: Tracker, codes: np.ndarray, block: int = 12_000) -> list[Epoch]:
    """Close `tracker` around the model channel over `codes`; returns one Epoch per dump.

    Each run stops at the first dump so its feedback is applied before any later sample,
    whatever the block size."""
    if not ch.enabled:
        raise ValueError("the channel is disabled: it would consume no samples")
    if block <= 0:
        raise ValueError(f"block size {block} would consume no samples")
    history: list[Epoch] = []
    pos = 0
    while pos < len(codes):
        before = ch.sample_count
        dumps = ch.run(codes[pos : pos + block], max_dumps=1)
        pos += ch.sample_count - before
        for dump in dumps:
            ch.set_steps(*tracker.update(dump))
            history.append(Epoch(dump.sample, tracker.doppler, ch.dump_code_frac / 2**32))
            if not tracker.locked:
                return history
    return history


def code_start_time(epoch: Epoch, fs: float, chip_rate: float) -> float:
    """Seconds from the data start to the code-period start the channel saw at this dump: the
    quarter-chip fraction left in the NCO places it just before the next sample."""
    return epoch.sample / fs - epoch.code_frac_q / (4.0 * chip_rate)


def compare_with_pyrx(history: list[Epoch], ref, fs: float, t_from: float) -> tuple[float, float]:
    """(Doppler RMS difference in Hz, largest code-start difference in chips) against a pyrx
    TrackResult, over the epochs at or after `t_from` seconds."""
    ref_start = ref.t + ((ref.code_offset - ref.t + 0.5 * CA_PERIOD) % CA_PERIOD) - 0.5 * CA_PERIOD
    dopp, code = [], []
    for e in history:
        t_dump = e.sample / fs
        if t_dump < t_from:
            continue
        j = int(np.argmin(np.abs(ref.t - t_dump)))  # the pyrx window starting at this dump
        if j == 0 or abs(ref.t[j] - t_dump) > CA_PERIOD:
            continue
        # ref.doppler[j] is set after window j; the Doppler in force during it, like the one
        # this dump's update set for the channel's next period, is ref.doppler[j - 1]
        chip_rate = CA_CHIP_RATE * (1.0 + ref.doppler[j - 1] / L1_FREQ)
        t_model = code_start_time(e, fs, chip_rate)
        dt = (t_model - ref_start[j] + 0.5 * CA_PERIOD) % CA_PERIOD - 0.5 * CA_PERIOD
        dopp.append(e.doppler - ref.doppler[j - 1])
        code.append(abs(dt) * chip_rate)
    if not dopp:
        raise ValueError("no epochs to compare")
    return float(np.sqrt(np.mean(np.square(dopp)))), float(max(code))
