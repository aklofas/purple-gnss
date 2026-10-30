"""Bit-true model of one FPGA tracking channel.

The model is the reference the Verilog is held to: every quantity is an integer of the
width the hardware has, and the cocotb bench requires the RTL's dumps to equal this
model's exactly. `Channel.step` processes one sample and is the definition; `Channel.run`
processes a block the same way, vectorized segment by segment between dumps.

Per valid sample the channel mixes the 2-bit sample with a 3-level local oscillator
selected by the top three bits of a 32-bit carrier NCO, correlates the result with the
early, prompt and late chips of a code held as 0/1 bits (1 = chip value -1), and
accumulates six sums that wrap at `acc_w` bits. The code NCO counts quarter chips; the
chip index wrapping at 4 * code_len dumps the sums, increments the epoch counter and
applies the shadow step words. A disabled channel ignores samples altogether.
"""

from dataclasses import dataclass

import numpy as np
from pyrx.synth import synth_l1ca

from fpgamodel.quantize import to_sign_mag, values

MASK32 = (1 << 32) - 1

# (cos, sin) of the 8 LO phases, 3-level: phase p covers the octant centered on
# (p + 0.5) * 45 degrees
LO_TABLE = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
_COS = np.array([c for c, _ in LO_TABLE], dtype=np.int64)
_SIN = np.array([s for _, s in LO_TABLE], dtype=np.int64)


def nco_step(freq: float, fs: float) -> int:
    """32-bit NCO step word for `freq` Hz at `fs` samples per second.

    An NCO advances at most one full turn per sample, so |freq| must be below `fs`; for the
    code NCO, which counts quarter chips, that means the sample rate must exceed four times
    the chip rate (4.092 MHz for L1 C/A).
    """
    if abs(freq) >= fs:
        raise ValueError(f"{freq} Hz cannot be stepped at {fs} samples/s")
    return round(freq / fs * 2**32) % 2**32


@dataclass(frozen=True)
class Dump:
    """The six correlator sums of one code period; `sample` counts the valid samples
    processed up to and including the one that completed the period."""

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
    """Channel state before the next sample is processed."""

    carr_phase: int
    code_frac: int
    chip_q: int
    epoch: int
    sample: int


class Channel:
    def __init__(self, code_bits: np.ndarray, *, acc_w: int = 20, code_len: int = 1023):
        words = np.asarray(code_bits, dtype=np.uint64)
        shifts = np.arange(32, dtype=np.uint64)
        bits = ((words[:, np.newaxis] >> shifts) & np.uint64(1)).reshape(-1)
        if len(bits) < code_len:
            raise ValueError(f"{len(words)} code words hold fewer than {code_len} chips")
        self.code = bits[:code_len].astype(np.int64)  # 0/1, 1 = chip value -1
        self.code_len = code_len
        self.n_q = 4 * code_len
        self.acc_w = acc_w
        self.enabled = False
        self.spacing_q = 1
        self.carr_acc = 0
        self.code_acc = 0
        self.chip_q = 0
        self.epoch = 0
        self.sample_count = 0
        self.carr_step = 0
        self.code_step = 0
        self._shadow = (0, 0)
        self._acc = [0] * 6  # ie, qe, ip, qp, il, ql

    # ---- configuration ---------------------------------------------------------
    def set_steps(self, carr_step: int, code_step: int) -> None:
        """New NCO step words: applied at the next dump, or at once while disabled."""
        self._shadow = (carr_step & MASK32, code_step & MASK32)
        if not self.enabled:
            self.carr_step, self.code_step = self._shadow

    def set_phase(self, carr_phase: int, code_frac: int, chip_q: int) -> None:
        """Set the NCO accumulators directly; only while disabled."""
        if self.enabled:
            raise ValueError("phase can only be set while the channel is disabled")
        self.carr_acc = carr_phase & MASK32
        self.code_acc = code_frac & MASK32
        self.chip_q = chip_q % self.n_q

    def measurement(self) -> Measurement:
        return Measurement(self.carr_acc, self.code_acc, self.chip_q, self.epoch, self.sample_count)

    def _spacing(self) -> int:
        return self.spacing_q if self.spacing_q in (1, 2) else 1

    def _wrap(self, v: int) -> int:
        half = 1 << (self.acc_w - 1)
        return (v + half) % (2 * half) - half

    def _dump(self) -> Dump:
        self.epoch += 1
        d = Dump(self.epoch, *(self._wrap(a) for a in self._acc), self.sample_count)
        self._acc = [0] * 6
        self.carr_step, self.code_step = self._shadow
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
        if total >> 32:
            self.chip_q += 1
            if self.chip_q == self.n_q:
                self.chip_q = 0
                return self._dump()
        return None

    # ---- a block of samples, vectorized ------------------------------------------
    def run(self, codes: np.ndarray) -> list[Dump]:
        dumps: list[Dump] = []
        codes = np.asarray(codes, dtype=np.int64)
        if not self.enabled:
            return dumps
        i, n = 0, len(codes)
        while i < n:
            need = self.n_q - self.chip_q  # ticks until the chip index wraps
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
                assert self.chip_q == self.n_q
                self.chip_q = 0
                dumps.append(self._dump())
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
