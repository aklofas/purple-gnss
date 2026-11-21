"""Bit-true model of one FPGA tracking channel.

The model is the reference the Verilog is held to: every quantity is an integer of the
width the hardware has, and the cocotb bench requires the RTL's dumps to equal this
model's exactly. `Channel.step` processes one sample and is the definition; `Channel.run`
processes a block the same way, vectorized segment by segment between dumps.

Per valid sample the channel mixes the 2-bit sample with a 3-level local oscillator
selected by the top three bits of a 32-bit carrier NCO, correlates the result with the
early, prompt and late chips of a code held as 0/1 bits (1 = chip value -1), and
accumulates six sums that wrap at `acc_w` bits. The code phase is a chip index and a
34-bit fraction of a chip, kept as a quarter-chip index and a 32-bit fraction; the step is
34 bits, so the phase advances by less than one chip per sample. The chip index passing
the code's end ends a code period and applies the shadow step words,
and every `integ` periods (1 to 20) it also dumps the sums, increments the epoch counter
and applies a new integration length. A disabled channel ignores samples altogether.
"""

from dataclasses import dataclass

import numpy as np
from pyrx.synth import synth_l1ca

from fpgamodel.quantize import to_sign_mag, values

MASK32 = (1 << 32) - 1
MASK34 = (1 << 34) - 1

# (cos, sin) of the 8 LO phases, 3-level: phase p covers the octant centered on
# (p + 0.5) * 45 degrees
LO_TABLE = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
_COS = np.array([c for c, _ in LO_TABLE], dtype=np.int64)
_SIN = np.array([s for _, s in LO_TABLE], dtype=np.int64)


def nco_step(freq: float, fs: float) -> int:
    """32-bit carrier NCO step word for `freq` Hz at `fs` samples per second.

    The NCO advances at most one full turn per sample, so |freq| must be below `fs`.
    """
    if abs(freq) >= fs:
        raise ValueError(f"{freq} Hz cannot be stepped at {fs} samples/s")
    return round(freq / fs * 2**32) % 2**32


def code_step(chip_rate: float, fs: float) -> int:
    """34-bit code NCO step word, chips per sample times 2^34, for `chip_rate` chips per
    second at `fs` samples per second.

    The chip index advances at most once per sample, so the chip rate must be below the
    sample rate. `CODE_STEP` holds bits 31:0 of the word and `CODE_STEP_HI` bits 33:32; below
    a quarter chip per sample (L1 C/A at any rate the front end offers) the high bits are 0.
    """
    word = round(chip_rate / fs * 2**34) if 0 <= chip_rate < fs else 1 << 34
    if word >> 34:
        raise ValueError(f"{chip_rate} chips/s cannot be stepped at {fs} samples/s")
    return word


@dataclass(frozen=True)
class Dump:
    """The six correlator sums of one dump (`integ` code periods); `sample` counts the samples this channel
    processed while enabled, up to and including the one that completed the dump (the
    channel's own count, not the core's global sample count)."""

    epoch: int
    ie: int
    qe: int
    ip: int
    qp: int
    il: int
    ql: int
    sample: int


@dataclass(frozen=True)
class Measurement:
    """Channel state before the next sample is processed; `sample` is the channel's own count
    of enabled samples, equal to the core's `MEAS_SAMPLE` only for a channel whose enable was in
    effect before the first valid sample and never dropped since."""

    carr_phase: int
    code_frac: int
    chip_q: int
    epoch: int
    sample: int


class Channel:
    def __init__(self, code_bits: np.ndarray, *, acc_w: int = 25, code_len: int = 1023):
        words = np.asarray(code_bits, dtype=np.uint64)
        shifts = np.arange(32, dtype=np.uint64)
        bits = ((words[:, np.newaxis] >> shifts) & np.uint64(1)).reshape(-1)
        if len(bits) < code_len:
            raise ValueError(f"{len(words)} code words hold fewer than {code_len} chips")
        self.code = bits[:code_len].astype(np.int64)  # 0/1, 1 = chip value -1
        self.code_len = code_len
        self.n_q = 4 * code_len
        self.acc_w = acc_w
        self._enabled = False
        self.spacing_q = 1
        self.carr_acc = 0
        self.code_acc = 0
        self.chip_q = 0
        self.epoch = 0
        self.sample_count = 0
        self.carr_step = 0
        self.code_step = 0
        self._shadow = (0, 0)
        self.integ = 1  # code periods per dump
        self._integ_shadow = 1
        self._periods = 0  # code periods completed since the last dump
        self._acc = [0] * 6  # ie, qe, ip, qp, il, ql
        self.dump_code_frac = 0  # code NCO fraction right after the last dump's wrap

    # ---- configuration ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, on: bool) -> None:
        # the hardware copies the step and integration-length inputs into the active ones
        # while disabled, so values written while enabled take effect at a disable even if
        # no period ended in between
        self._enabled = bool(on)
        if not on:
            self.carr_step, self.code_step = self._shadow
            self.integ = self._integ_shadow

    def set_steps(self, carr_step: int, code_step: int) -> None:
        """New NCO step words: applied at the end of the code period, or at once while disabled."""
        self._shadow = (int(carr_step) & MASK32, int(code_step) & MASK34)  # ints, not numpy
        if not self.enabled:
            self.carr_step, self.code_step = self._shadow

    def set_integ(self, n: int) -> None:
        """Code periods per dump, 1 to 20 (0 acts as 1, more as 20): applied at the next dump,
        or at once while disabled."""
        self._integ_shadow = min(max(int(n), 1), 20)
        if not self.enabled:
            self.integ = self._integ_shadow

    def set_phase(self, carr_phase: int, code_frac: int, chip_q: int) -> None:
        """Set the NCO accumulators directly; only while disabled."""
        if self.enabled:
            raise ValueError("phase can only be set while the channel is disabled")
        carr_phase, code_frac, chip_q = int(carr_phase), int(code_frac), int(chip_q)
        if not 0 <= chip_q < self.n_q:
            raise ValueError(f"quarter-chip index {chip_q} outside 0..{self.n_q - 1}")
        self.carr_acc = carr_phase & MASK32
        self.code_acc = code_frac & MASK32
        self.chip_q = chip_q

    def measurement(self) -> Measurement:
        return Measurement(self.carr_acc, self.code_acc, self.chip_q, self.epoch, self.sample_count)

    def _spacing(self) -> int:
        return self.spacing_q if self.spacing_q in (1, 2) else 1

    def _wrap(self, v: int) -> int:
        half = 1 << (self.acc_w - 1)
        return (v + half) % (2 * half) - half

    def _period_end(self) -> Dump | None:
        """The chip index wrapped: take the shadow steps, and dump if `integ` periods are done."""
        self.carr_step, self.code_step = self._shadow
        self._periods += 1
        if self._periods < self.integ:
            return None
        self._periods = 0
        self.integ = self._integ_shadow
        self.dump_code_frac = self.code_acc
        self.epoch += 1
        d = Dump(self.epoch, *(self._wrap(a) for a in self._acc), self.sample_count)
        self._acc = [0] * 6
        return d

    # ---- one sample: the definition ---------------------------------------------
    def step(self, code: int) -> Dump | None:
        if not self.enabled:
            return None
        cos, sin = LO_TABLE[self.carr_acc >> 29]
        s = (3 if code & 1 else 1) * (-1 if code & 2 else 1)
        mix_i, mix_q = s * cos, -s * sin
        spacing = self._spacing()
        for i, d in enumerate((spacing, 0, -spacing)):
            chip = self.code[((self.chip_q + d) % self.n_q) >> 2]
            sign = -1 if chip else 1
            self._acc[2 * i] += sign * mix_i
            self._acc[2 * i + 1] += sign * mix_q
        self.sample_count += 1
        self.carr_acc = (self.carr_acc + self.carr_step) & MASK32
        total = self.code_acc + self.code_step
        self.code_acc = total & MASK32
        self.chip_q += total >> 32  # up to three quarter chips
        if self.chip_q >= self.n_q:
            self.chip_q -= self.n_q
            return self._period_end()
        return None

    # ---- a block of samples, vectorized ------------------------------------------
    def run(self, codes: np.ndarray, max_dumps: int | None = None) -> list[Dump]:
        """Process `codes`, stopping right after the `max_dumps`-th dump when given; the
        samples consumed show in `sample_count`. Each vectorized segment ends at a code
        period's end, where the steps may change."""
        dumps: list[Dump] = []
        codes = np.asarray(codes, dtype=np.int64)
        if not self.enabled:
            return dumps
        i, n = 0, len(codes)
        while i < n and (max_dumps is None or len(dumps) < max_dumps):
            need = self.n_q - self.chip_q  # quarter chips until the chip index wraps
            if self.code_step == 0:
                k = n - i
            else:
                k_dump = -(-(need * 2**32 - self.code_acc) // self.code_step)
                k = min(k_dump, n - i)
            self._accumulate(codes[i : i + k])
            self.sample_count += k
            self.carr_acc = (self.carr_acc + k * self.carr_step) & MASK32
            total = self.code_acc + k * self.code_step
            self.code_acc = total & MASK32
            self.chip_q += total >> 32
            i += k
            if self.chip_q >= self.n_q:
                self.chip_q -= self.n_q
                assert self.chip_q < 4
                if (d := self._period_end()) is not None:
                    dumps.append(d)
        return dumps

    def _accumulate(self, block: np.ndarray) -> None:
        k = np.arange(len(block), dtype=np.uint64)
        carr = (np.uint64(self.carr_acc) + k * np.uint64(self.carr_step)) & np.uint64(MASK32)
        ph = (carr >> np.uint64(29)).astype(np.int64)
        s = values(block).astype(np.int64)
        mix_i = s * _COS[ph]
        mix_q = -s * _SIN[ph]
        ticks = (np.uint64(self.code_acc) + k * np.uint64(self.code_step)) >> np.uint64(32)
        chip_q = np.int64(self.chip_q) + ticks.astype(np.int64)
        spacing = self._spacing()
        for i, d in enumerate((spacing, 0, -spacing)):
            chips = self.code[((chip_q + d) % self.n_q) >> 2]
            sign = 1 - 2 * chips
            self._acc[2 * i] += int(np.dot(sign, mix_i))
            self._acc[2 * i + 1] += int(np.dot(sign, mix_q))


def synthetic_codes(
    prn: int,
    fs: float,
    f_if: float,
    doppler: float,
    code_offset: int,
    n: int,
    cn0_dbhz: float,
    seed: int = 0,
) -> np.ndarray:
    """2-bit codes of a synthetic real L1 C/A signal (see `pyrx.synth.synth_l1ca`)."""
    x = synth_l1ca(prn, fs, f_if, doppler, code_offset, n, cn0_dbhz, iq=False, seed=seed)
    return to_sign_mag(x.real)
