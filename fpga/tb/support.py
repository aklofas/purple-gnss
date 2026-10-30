"""Shared pieces of the cocotb benches (imported inside the simulator process)."""

import os
import random
from pathlib import Path
from typing import Protocol

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer, ValueChange
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.signals import L1_FREQ

from fpgamodel import regmap as R
from fpgamodel.channel import Dump, nco_step
from fpgamodel.quantize import code_bits, pack_words

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
                     "chip_q_in", "code_we", "code_waddr", "code_wdata", "meas_strobe"):
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
        await ClockCycles(self.dut.clk, 10)  # enable takes effect eight clocks after it rises

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

    async def strobe(self) -> None:
        """Raise meas_strobe for one clock: with the sample presented on that edge, if any."""
        self.dut.meas_strobe.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.meas_strobe.value = 0

    async def strobe_with_sample(self, k: int) -> None:
        """Strobe with sample k (pausing the stream before it, then releasing it)."""
        await self.pause_at(k)
        self.dut.meas_strobe.value = 1
        self.resume()
        await RisingEdge(self.dut.clk)
        self.dut.meas_strobe.value = 0

    async def latched(self) -> tuple[int, int, int, int]:
        """(carrier phase, code fraction, quarter-chip index, epoch) held by the latch."""
        dut = self.dut
        await ReadOnly()
        values = (dut.meas_carr_phase, dut.meas_code_frac, dut.meas_chip_q, dut.meas_epoch)
        latched = tuple(int(v.value) for v in values)
        await Timer(1, unit="ns")  # leave the read-only phase without passing another edge
        return latched


def assert_dumps_equal(rtl: list[Dump], model: list[Dump], ignore_sample: bool = False) -> None:
    def key(d: Dump):
        return (d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql) if ignore_sample else d

    assert len(rtl) == len(model), f"{len(rtl)} RTL dumps, {len(model)} model dumps"
    for a, b in zip(rtl, model, strict=True):
        assert key(a) == key(b), f"RTL {a}\nmodel {b}"


# ---- register-bus access: the direct bus driver and helpers shared with the SPI master ----


class RegAccess(Protocol):
    dut: object

    async def write(self, addr: int, data: int) -> None: ...

    async def read(self, addr: int) -> int: ...


class BusDriver:
    """Drives rx_core's register bus directly: one write per clock, reads return the word
    registered the cycle after bus_re."""

    def __init__(self, dut):
        self.dut = dut
        dut.bus_addr.value = 0
        dut.bus_wdata.value = 0
        dut.bus_we.value = 0
        dut.bus_re.value = 0

    async def write(self, addr: int, data: int) -> None:
        dut = self.dut
        dut.bus_addr.value = addr
        dut.bus_wdata.value = data & 0xFFFFFFFF
        dut.bus_we.value = 1
        await RisingEdge(dut.clk)
        dut.bus_we.value = 0

    async def read(self, addr: int) -> int:
        dut = self.dut
        dut.bus_addr.value = addr
        dut.bus_re.value = 1
        await RisingEdge(dut.clk)
        dut.bus_re.value = 0
        await ReadOnly()
        value = dut.bus_rdata.value.to_unsigned()
        await RisingEdge(dut.clk)
        return value


def signed32(v: int) -> int:
    return v - (1 << 32) if v & (1 << 31) else v


async def load_code(bus: RegAccess, k: int, prn: int) -> None:
    await bus.write(R.ch(k, R.CODE_WR_ADDR), 0)
    for word in code_bits(gps_ca(prn)):
        await bus.write(R.ch(k, R.CODE_WR_DATA), int(word))


async def load_playback(bus: RegAccess, codes: np.ndarray) -> None:
    await bus.write(R.PB_ADDR, 0)
    for word in pack_words(codes):
        await bus.write(R.PB_DATA, int(word))


async def configure_channel(bus: RegAccess, k: int, carr_step: int, code_step: int,
                            carr_phase: int = 0, code_frac: int = 0, chip_q: int = 0,
                            spacing_q: int = 1) -> None:
    await bus.write(R.ch(k, R.CARR_STEP), carr_step)
    await bus.write(R.ch(k, R.CODE_STEP), code_step)
    await bus.write(R.ch(k, R.CARR_PHASE), carr_phase)
    await bus.write(R.ch(k, R.CODE_PHASE_FRAC), code_frac)
    await bus.write(R.ch(k, R.CODE_CHIP), chip_q)
    await bus.write(R.ch(k, R.SPACING_Q), spacing_q)


async def enable_channel(bus: RegAccess, k: int, on: bool = True) -> None:
    """Write the channel's enable bit; it takes effect eight clocks later."""
    await bus.write(R.ch(k, R.CH_CTRL), int(on))
    await ClockCycles(bus.dut.clk, 10)


async def start_playback(bus: RegAccess, n_samples: int) -> None:
    await bus.write(R.PLAYBACK_LEN, n_samples)
    await bus.write(R.PLAYBACK_CTRL, 1)


async def read_dump(bus: RegAccess, k: int, clear: bool = True) -> tuple[Dump | None, bool]:
    """The channel's holding registers as a Dump (sample = -1: the core does not know it),
    and whether to read again at once.

    A dump that lands during the read replaces the registers the moment it does, EPOCH with
    them. EPOCH is read before and after the sums: a change means the sums could mix two
    periods, so the result is (None, True), nothing is cleared (the new dump is pending, and
    DUMP_MISSED counted the one it replaced) and the caller reads again. Otherwise the clear
    goes out and EPOCH is read a third time: a change there means a dump landed around the
    clear and sits in the registers, acknowledged or not, so the dump read is good and the
    caller must read again at once, whatever DUMP_PENDING says."""
    epoch = await bus.read(R.ch(k, R.EPOCH))
    sums = [signed32(await bus.read(R.ch(k, off))) for off in (R.IE, R.QE, R.IP, R.QP, R.IL, R.QL)]
    if await bus.read(R.ch(k, R.EPOCH)) != epoch:
        return None, True
    if not clear:
        return Dump(epoch, *sums, -1), False
    await bus.write(R.DUMP_CLEAR, 1 << k)
    return Dump(epoch, *sums, -1), await bus.read(R.ch(k, R.EPOCH)) != epoch


async def read_dumps_now(bus: RegAccess, k: int) -> list[Dump]:
    """Every dump channel k holds right now: `read_dump` again while it asks for it."""
    dumps: list[Dump] = []
    again = True
    while again:
        dump, again = await read_dump(bus, k)
        if dump is not None:
            dumps.append(dump)
    return dumps


async def collect_dumps(bus: RegAccess, n_channels: int, poll_cycles: int = 20) -> list[list[Dump]]:
    """Read every dump of every channel until playback is done and nothing is pending."""
    dumps: list[list[Dump]] = [[] for _ in range(n_channels)]
    while True:
        pending = await bus.read(R.DUMP_PENDING)
        for k in range(n_channels):
            if pending & (1 << k):
                dumps[k].extend(await read_dumps_now(bus, k))
        status = await bus.read(R.STATUS)
        if status & R.STATUS_PB_DONE and not pending:
            await ClockCycles(bus.dut.clk, 8)
            if not await bus.read(R.DUMP_PENDING):
                return dumps
        else:
            await ClockCycles(bus.dut.clk, poll_cycles)


class SpiMaster:
    """Bit-banged SPI master (mode 0, MSB first) for rx_spi, satisfying RegAccess.

    Transaction: command byte (bit7 write, bit6 auto-increment), two address bytes, then
    32-bit words; a read has one dummy byte before its data. The master is asynchronous to
    the fabric clock, as the board's is: every transaction starts at a random offset from
    the clock (seeded by cocotb) and the default period, 82.6 ns, is not a multiple of it,
    just under the fabric/8 limit at the bench's 10 ns clock.
    """

    def __init__(self, dut, sck_period_ns: float = 82.6):
        self.dut = dut
        self.half_ns = sck_period_ns / 2
        dut.spi_sck.value = 0
        dut.spi_mosi.value = 0
        dut.spi_cs_n.value = 1

    async def _bit(self, bit: int) -> int:
        dut = self.dut
        dut.spi_mosi.value = bit
        await Timer(self.half_ns, unit="ns")
        inp = int(dut.spi_miso.value)
        dut.spi_sck.value = 1
        await Timer(self.half_ns, unit="ns")
        dut.spi_sck.value = 0
        return inp

    async def _byte(self, out: int) -> int:
        inp = 0
        for i in range(8):
            inp = (inp << 1) | await self._bit((out >> (7 - i)) & 1)
        return inp

    async def _begin(self, command: int, addr: int) -> None:
        await Timer(random.randint(1, 1000 * CLK_NS - 1), unit="ps")  # any phase of the fabric clock
        self.dut.spi_cs_n.value = 0
        await Timer(self.half_ns, unit="ns")
        await self._byte(command)
        await self._byte(addr >> 8)
        await self._byte(addr & 0xFF)

    async def _end(self) -> None:
        await Timer(self.half_ns, unit="ns")
        self.dut.spi_cs_n.value = 1
        await Timer(4 * self.half_ns, unit="ns")
        await RisingEdge(self.dut.clk)

    async def write_words(self, addr: int, words, increment: bool = True) -> None:
        await self._begin(0x80 | (0x40 if increment else 0), addr)
        for w in words:
            for shift in (24, 16, 8, 0):
                await self._byte((int(w) >> shift) & 0xFF)
        await self._end()

    async def read_words(self, addr: int, n: int, increment: bool = True) -> list[int]:
        await self._begin(0x40 if increment else 0, addr)
        await self._byte(0)  # dummy
        words = []
        for _ in range(n):
            v = 0
            for _ in range(4):
                v = (v << 8) | await self._byte(0)
            words.append(v)
        await self._end()
        return words

    async def write(self, addr: int, data: int) -> None:
        await self.write_words(addr, [data])

    async def read(self, addr: int) -> int:
        return (await self.read_words(addr, 1))[0]

    # ---- malformed transactions ----
    async def abort_mid_word(self, addr: int, data: int) -> None:
        """A write cut short by CS after two data bytes."""
        await self._begin(0xC0, addr)
        await self._byte(data >> 24)
        await self._byte((data >> 16) & 0xFF)
        await self._end()

    async def abort_in_header(self, addr: int) -> None:
        """A write whose CS rises after the command byte and the first address byte."""
        await Timer(random.randint(1, 1000 * CLK_NS - 1), unit="ps")
        self.dut.spi_cs_n.value = 0
        await Timer(self.half_ns, unit="ns")
        await self._byte(0xC0)
        await self._byte(addr >> 8)
        await self._end()

    async def read_words_aborted(self, addr: int, n_full: int, extra_bytes: int) -> list[int]:
        """A read burst cut short by CS `extra_bytes` into word n_full + 1."""
        await self._begin(0x40, addr)
        await self._byte(0)
        words = []
        for _ in range(n_full):
            v = 0
            for _ in range(4):
                v = (v << 8) | await self._byte(0)
            words.append(v)
        for _ in range(extra_bytes):
            await self._byte(0)
        await self._end()
        return words

    async def write_words_cs_glitch(self, addr: int, words, glitch_clocks: int) -> None:
        """An auto-increment write burst with CS pulsed high for `glitch_clocks` fabric clocks
        after the first word; the slave then parses the following bytes as a new header."""
        await self._begin(0xC0, addr)
        for shift in (24, 16, 8, 0):
            await self._byte((int(words[0]) >> shift) & 0xFF)
        await Timer(self.half_ns, unit="ns")
        self.dut.spi_cs_n.value = 1
        await Timer(glitch_clocks * CLK_NS, unit="ns")
        self.dut.spi_cs_n.value = 0
        await Timer(self.half_ns, unit="ns")
        for w in words[1:]:
            for shift in (24, 16, 8, 0):
                await self._byte((int(w) >> shift) & 0xFF)
        await self._end()

    async def write_with_cs_hold(self, addr: int, data: int, hold_clocks: int) -> None:
        """A single-word write whose last SCK rise comes right after a fabric clock edge and
        whose CS rises `hold_clocks` fabric clocks after that rise (0: in the same clock)."""
        dut = self.dut
        await self._begin(0x80, addr)
        for i in range(31):
            await self._bit((data >> (31 - i)) & 1)
        dut.spi_mosi.value = data & 1
        await Timer(self.half_ns, unit="ns")
        await RisingEdge(dut.clk)
        dut.spi_sck.value = 1
        if hold_clocks:
            await ClockCycles(dut.clk, hold_clocks)
        dut.spi_cs_n.value = 1  # SCK stays high past it: one timestep keeps only a last deposit
        await Timer(self.half_ns, unit="ns")
        dut.spi_sck.value = 0
        await Timer(round(3 * self.half_ns * 1000), unit="ps")
        await RisingEdge(dut.clk)
