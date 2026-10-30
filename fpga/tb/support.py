"""Shared pieces of the cocotb benches (imported inside the simulator process)."""

import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, ValueChange
from pyrx.codes import CA_CHIP_RATE
from pyrx.signals import L1_FREQ

from fpgamodel.channel import Dump, nco_step

RECORDING = "L1_20211226_082401_24MHz_I.bin"
FS, F_IF = 24e6, 6.0e6
CLK_NS = 10


def captures_dir() -> Path | None:
    value = os.environ.get("PYRX_CAPTURES")
    if value and (Path(value) / RECORDING).is_file():
        return Path(value)
    return None


def channel_steps(doppler: float, fs: float = FS, f_if: float = F_IF) -> tuple[int, int]:
    """(carrier step, code step) words for a Doppler at a sample rate and IF."""
    return nco_step(f_if + doppler, fs), nco_step(4 * CA_CHIP_RATE * (1 + doppler / L1_FREQ), fs)


def write_hex(path: Path, codes: np.ndarray) -> None:
    """One 2-bit code per line, the format tb_channel.v reads with $readmemh."""
    path.write_text("\n".join(f"{int(c):x}" for c in codes) + "\n")


class ChannelBench:
    """Drives tb_channel.v: clock and reset, code and phase loading, sample streaming with
    pauses, and a monitor that records every dump as a model `Dump`."""

    def __init__(self, dut):
        self.dut = dut
        self.dumps: list[Dump] = []

    async def start(self) -> None:
        dut = self.dut
        cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
        for name in ("load", "n_samples", "gap", "stream_en", "pause_en", "pause_at", "enable",
                     "carr_step", "code_step", "load_phase", "carr_phase_in", "code_frac_in",
                     "chip_q_in", "code_we", "code_waddr", "code_wdata"):
            getattr(dut, name).value = 0
        dut.spacing_q.value = 1
        dut.rst.value = 1
        await ClockCycles(dut.clk, 4)
        dut.rst.value = 0
        await ClockCycles(dut.clk, 2)
        cocotb.start_soon(self._monitor())

    async def _monitor(self) -> None:
        dut = self.dut
        while True:
            await ValueChange(dut.epoch)
            await ReadOnly()
            self.dumps.append(Dump(
                int(dut.epoch.value),
                dut.ie.value.to_signed(), dut.qe.value.to_signed(),
                dut.ip.value.to_signed(), dut.qp.value.to_signed(),
                dut.il.value.to_signed(), dut.ql.value.to_signed(),
                int(dut.dump_sample.value),
            ))

    async def load_samples(self, codes: np.ndarray, gap: int = 0) -> None:
        write_hex(Path.cwd() / "samples.hex", codes)
        self.dut.n_samples.value = len(codes)
        self.dut.gap.value = gap
        self.dut.load.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.load.value = 0
        await RisingEdge(self.dut.clk)

    async def load_code(self, words: np.ndarray) -> None:
        dut = self.dut
        for addr, word in enumerate(words):
            dut.code_we.value = 1
            dut.code_waddr.value = addr
            dut.code_wdata.value = int(word)
            await RisingEdge(dut.clk)
        dut.code_we.value = 0
        await ClockCycles(dut.clk, 8)  # the channel refreshes its chip neighbors

    def set_steps(self, carr_step: int, code_step: int) -> None:
        self.dut.carr_step.value = carr_step
        self.dut.code_step.value = code_step

    async def load_phase(self, carr_phase: int, code_frac: int, chip_q: int) -> None:
        dut = self.dut
        dut.carr_phase_in.value = carr_phase
        dut.code_frac_in.value = code_frac
        dut.chip_q_in.value = chip_q
        dut.load_phase.value = 1
        await RisingEdge(dut.clk)
        dut.load_phase.value = 0
        await ClockCycles(dut.clk, 8)  # the channel refreshes its chip neighbors

    async def set_enable(self, on: bool) -> None:
        self.dut.enable.value = int(on)
        await RisingEdge(self.dut.clk)

    def stream(self, on: bool = True) -> None:
        self.dut.stream_en.value = int(on)

    async def pause_at(self, k: int) -> None:
        """Hold the stream before sample k is presented (consumed == k)."""
        dut = self.dut
        assert int(dut.consumed.value) <= k, "pause point already passed"
        dut.pause_at.value = k
        dut.pause_en.value = 1
        if int(dut.consumed.value) != k:
            await RisingEdge(dut.paused)
        await RisingEdge(dut.clk)

    def resume(self) -> None:
        self.dut.pause_en.value = 0

    async def finish(self) -> None:
        dut = self.dut
        if not int(dut.done.value):
            await RisingEdge(dut.done)
        await ClockCycles(dut.clk, 8)  # drain the pipeline


def assert_dumps_equal(rtl: list[Dump], model: list[Dump], ignore_sample: bool = False) -> None:
    def key(d: Dump):
        return (d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql) if ignore_sample else d

    assert len(rtl) == len(model), f"{len(rtl)} RTL dumps, {len(model)} model dumps"
    for a, b in zip(rtl, model, strict=True):
        assert key(a) == key(b), f"RTL {a}\nmodel {b}"
