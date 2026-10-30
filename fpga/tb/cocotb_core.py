"""rx_core: register bus, global registers, sample mux and playback RAM around channels.

Synthetic stimulus at 8 MHz (2 MHz IF): the quarter-chip code NCO needs more than 4.092 MHz,
and 12 ms is then 96 000 samples, which fits the board's 147 456-sample playback RAM.
N_CHANNELS = 2, except for the one case the 12-channel build runs.
"""

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.signals import L1_FREQ
from pyrx.synth import synth_l1ca

from fpgamodel import regmap as R
from fpgamodel.channel import Channel, Measurement, nco_step
from fpgamodel.compare import start_phase
from fpgamodel.quantize import code_bits, to_sign_mag
from tb.support import (
    CLK_NS,
    BusDriver,
    assert_dumps_equal,
    collect_dumps,
    configure_channel,
    enable_channel,
    load_code,
    load_playback,
    read_dump,
    start_playback,
)

FS, F_IF = 8e6, 2e6
N_CHANNELS = 2
PB_DEPTH = 147_456
PRN = 7


def steps(doppler):
    return nco_step(F_IF + doppler, FS), nco_step(4 * CA_CHIP_RATE * (1 + doppler / L1_FREQ), FS)


def signal(prn, doppler, code_offset, n_ms, cn0=45.0, seed=0):
    return synth_l1ca(prn, FS, F_IF, doppler, code_offset, round(FS * n_ms * 1e-3), cn0, iq=False,
                      seed=seed).real


def phase_of(code_offset, doppler):
    return start_phase(code_offset / FS, CA_CHIP_RATE * (1 + doppler / L1_FREQ))


def model(prn, doppler, code_offset=0, spacing_q=1) -> Channel:
    chip_q, frac = phase_of(code_offset, doppler)
    ch = Channel(code_bits(gps_ca(prn)))
    ch.spacing_q = spacing_q
    ch.set_steps(*steps(doppler))
    ch.set_phase(0, frac, chip_q)
    ch.enabled = True
    return ch


async def start(dut) -> BusDriver:
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    dut.pin_valid.value = 0
    dut.pin_sample.value = 0
    dut.pll_lock.value = 1
    bus = BusDriver(dut)
    dut.rst.value = 1
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)
    return bus


async def set_up_channel(bus, k, prn, doppler, code_offset=0, spacing_q=1):
    chip_q, frac = phase_of(code_offset, doppler)
    await load_code(bus, k, prn)
    await configure_channel(bus, k, *steps(doppler), 0, frac, chip_q, spacing_q)
    await enable_channel(bus, k)


async def run_playback(bus, codes):
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    return await collect_dumps(bus, N_CHANNELS)


@cocotb.test()
async def id_version_and_constants(dut):
    bus = await start(dut)
    assert await bus.read(R.ID) == R.ID_VALUE
    assert await bus.read(R.VERSION) == R.VERSION_BASE | R.VERSION_LATCH
    assert await bus.read(R.N_CHANNELS) == N_CHANNELS
    assert await bus.read(R.PB_DEPTH) == PB_DEPTH
    assert await bus.read(R.STATUS) == R.STATUS_PLL_LOCK
    dut.pll_lock.value = 0
    await RisingEdge(dut.clk)
    assert await bus.read(R.STATUS) == 0


@cocotb.test()
async def unmapped_reads_zero_writes_ignored(dut):
    bus = await start(dut)
    for addr in (0x0FFF, 0x0E, 0x102, R.ch(N_CHANNELS, 0), R.ch(0, 0xFF), R.ch(0, 0x16), 0x2000, 0xFFFF):
        await bus.write(addr, 0xDEADBEEF)
        assert await bus.read(addr) == 0, hex(addr)
    assert await bus.read(R.ID) == R.ID_VALUE  # nothing hung


@cocotb.test()
async def soft_reset_mid_run_keeps_only_the_rams(dut):
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 6))
    await set_up_channel(bus, 0, PRN, 1000.0)
    await load_playback(bus, codes)
    await bus.write(R.MEAS_PERIOD, 5000)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 9000)  # a dump is pending, a strobe has landed, playback runs
    assert await bus.read(R.DUMP_PENDING) == 0b01
    assert await bus.read(R.STATUS) == R.STATUS_PLL_LOCK | R.STATUS_PB_RUNNING
    await bus.write(R.CTRL, R.CTRL_RESET)
    await ClockCycles(dut.clk, 4)
    zeroed = (R.CTRL, R.PLAYBACK_LEN, R.MEAS_PERIOD, R.DUMP_PENDING, R.SAMPLE_CNT_LO)
    zeroed += tuple(R.ch(0, off) for off in (R.CH_CTRL, R.CARR_STEP, R.EPOCH, R.DUMP_MISSED, R.IP))
    zeroed += tuple(R.ch(0, off) for off in (R.MEAS_SAMPLE_LO, R.MEAS_CODE_CHIP, R.MEAS_EPOCH))
    for addr in zeroed:
        assert await bus.read(addr) == 0, hex(addr)
    assert await bus.read(R.STATUS) == R.STATUS_PLL_LOCK  # neither running nor done
    # both RAMs kept their contents: steps, phase and enable alone track the samples again
    chip_q, frac = phase_of(0, 1000.0)
    await configure_channel(bus, 0, *steps(1000.0), 0, frac, chip_q)
    await enable_channel(bus, 0)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    dumps = await collect_dumps(bus, N_CHANNELS)
    assert_dumps_equal(dumps[0], model(PRN, 1000.0).run(codes), ignore_sample=True)
    assert await bus.read(R.SAMPLE_CNT_LO) == len(codes)


@cocotb.test()
async def playback_dumps_match_model(dut):
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 12))
    await set_up_channel(bus, 0, PRN, 1000.0)
    dumps = await run_playback(bus, codes)
    expected = model(PRN, 1000.0).run(codes)
    assert len(expected) in (11, 12)
    assert_dumps_equal(dumps[0], expected, ignore_sample=True)
    assert dumps[1] == []
    assert await bus.read(R.ch(0, R.DUMP_MISSED)) == 0


@cocotb.test()
async def playback_length_clamped_and_zero_is_done(dut):
    bus = await start(dut)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, 0)
    await ClockCycles(dut.clk, 4)
    status = await bus.read(R.STATUS)
    assert status & R.STATUS_PB_DONE and not status & R.STATUS_PB_RUNNING
    assert await bus.read(R.SAMPLE_CNT_LO) == 0
    await start_playback(bus, PB_DEPTH + 1000)
    await ClockCycles(dut.clk, 2)  # the start pulse reaches the playback counter
    status = await bus.read(R.STATUS)
    assert status & R.STATUS_PB_RUNNING and not status & R.STATUS_PB_DONE
    await ClockCycles(dut.clk, PB_DEPTH + 50)
    status = await bus.read(R.STATUS)
    assert status & R.STATUS_PB_DONE and not status & R.STATUS_PB_RUNNING
    assert await bus.read(R.SAMPLE_CNT_LO) == PB_DEPTH


@cocotb.test()
async def playback_end_freezes_channels(dut):
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 5))
    await set_up_channel(bus, 0, PRN, 1000.0)
    dumps = await run_playback(bus, codes)
    n = len(model(PRN, 1000.0).run(codes))
    assert len(dumps[0]) == n >= 4
    epoch = await bus.read(R.ch(0, R.EPOCH))
    await ClockCycles(dut.clk, 10_000)
    assert await bus.read(R.ch(0, R.EPOCH)) == epoch == n
    assert await bus.read(R.DUMP_PENDING) == 0
    assert await bus.read(R.SAMPLE_CNT_LO) == len(codes)


@cocotb.test()
async def pin_source_dumps_match_model(dut):
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, -1500.0, 777, 6))
    await set_up_channel(bus, 0, PRN, -1500.0, 777)
    await bus.write(R.CTRL, R.CTRL_ENABLE)
    for c in codes:
        dut.pin_sample.value = int(c)
        dut.pin_valid.value = 1
        await RisingEdge(dut.clk)
    dut.pin_valid.value = 0
    await ClockCycles(dut.clk, 8)
    pending = await bus.read(R.DUMP_PENDING)
    assert pending == 0b01  # the last dump still waits; earlier ones were missed on purpose
    assert await bus.read(R.SAMPLE_CNT_LO) == len(codes)
    expected = model(PRN, -1500.0, 777).run(codes)
    latest = await read_dump(bus, 0)
    assert_dumps_equal([latest], expected[-1:], ignore_sample=True)
    assert await bus.read(R.ch(0, R.DUMP_MISSED)) == len(expected) - 1


@cocotb.test()
async def dump_missed_counts_and_values_replace(dut):
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 4))
    await set_up_channel(bus, 0, PRN, 1000.0)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, len(codes) + 20)
    expected = model(PRN, 1000.0).run(codes)
    assert len(expected) >= 3
    assert await bus.read(R.ch(0, R.DUMP_MISSED)) == len(expected) - 1
    assert await bus.read(R.DUMP_PENDING) == 0b01
    assert_dumps_equal([await read_dump(bus, 0)], expected[-1:], ignore_sample=True)
    assert await bus.read(R.DUMP_PENDING) == 0


@cocotb.test()
async def sample_count_stops_with_global_enable(dut):
    bus = await start(dut)
    codes = np.zeros(5000, dtype=np.uint8)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, 5000)
    await ClockCycles(dut.clk, 5100)
    assert await bus.read(R.SAMPLE_CNT_LO) == 5000
    assert await bus.read(R.SAMPLE_CNT_HI) == 0
    # the counter stops with the global enable
    await bus.write(R.CTRL, R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, 5000)
    await ClockCycles(dut.clk, 5100)
    assert await bus.read(R.SAMPLE_CNT_LO) == 5000


@cocotb.test()
async def two_channels_track_two_prns(dut):
    bus = await start(dut)
    x = signal(7, 1000.0, 0, 8, seed=1) + signal(12, -2500.0, 1500, 8, seed=2)
    codes = to_sign_mag(x)
    await set_up_channel(bus, 0, 7, 1000.0)
    await set_up_channel(bus, 1, 12, -2500.0, 1500)
    dumps = await run_playback(bus, codes)
    assert_dumps_equal(dumps[0], model(7, 1000.0).run(codes), ignore_sample=True)
    assert_dumps_equal(dumps[1], model(12, -2500.0, 1500).run(codes), ignore_sample=True)
    assert len(dumps[0]) >= 7 and len(dumps[1]) >= 7


@cocotb.test()
async def last_channel_tracks_like_the_first(dut):
    # run in the 12-channel build: channel decode and readback at an index above 1
    bus = await start(dut)
    n = await bus.read(R.N_CHANNELS)
    assert n > 2
    x = signal(7, 1000.0, 0, 3, seed=1) + signal(12, -2500.0, 1500, 3, seed=2)
    codes = to_sign_mag(x)
    await set_up_channel(bus, 0, 7, 1000.0)
    await set_up_channel(bus, n - 1, 12, -2500.0, 1500)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    dumps = await collect_dumps(bus, n)
    assert_dumps_equal(dumps[0], model(7, 1000.0).run(codes), ignore_sample=True)
    assert_dumps_equal(dumps[n - 1], model(12, -2500.0, 1500).run(codes), ignore_sample=True)
    assert len(dumps[0]) >= 2 and len(dumps[n - 1]) >= 2
    assert all(dumps[k] == [] for k in range(1, n - 1))


@cocotb.test()
async def phase_registers_only_while_disabled(dut):
    bus = await start(dut)
    await bus.write(R.ch(0, R.CARR_PHASE), 0x11111111)
    await bus.write(R.ch(0, R.CODE_PHASE_FRAC), 0x22222222)
    await bus.write(R.ch(0, R.CODE_CHIP), 4091)
    assert await bus.read(R.ch(0, R.CARR_PHASE)) == 0x11111111
    assert await bus.read(R.ch(0, R.CODE_CHIP)) == 4091
    await enable_channel(bus, 0)
    await bus.write(R.ch(0, R.CARR_PHASE), 0x33333333)
    await bus.write(R.ch(0, R.CODE_CHIP), 5)
    assert await bus.read(R.ch(0, R.CARR_PHASE)) == 0x11111111
    assert await bus.read(R.ch(0, R.CODE_PHASE_FRAC)) == 0x22222222
    assert await bus.read(R.ch(0, R.CODE_CHIP)) == 4091
    await enable_channel(bus, 0, False)
    await bus.write(R.ch(0, R.CODE_CHIP), 5)
    assert await bus.read(R.ch(0, R.CODE_CHIP)) == 5


@cocotb.test()
async def code_write_while_enabled_is_ignored(dut):
    # the channel refreshes its cached chips only while disabled, so a code write to an
    # enabled channel is dropped: the word pointer stays and the old code keeps tracking
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 4))
    await set_up_channel(bus, 0, PRN, 1000.0)
    for _ in range(2):
        await bus.write(R.ch(0, R.CODE_WR_DATA), 0xFFFF_FFFF)
    assert await bus.read(R.ch(0, R.CODE_WR_ADDR)) == 0
    dumps = await run_playback(bus, codes)
    assert_dumps_equal(dumps[0], model(PRN, 1000.0).run(codes), ignore_sample=True)


@cocotb.test()
async def code_chip_out_of_range_is_ignored(dut):
    # the quarter-chip index runs 0..4091; the register's 12 bits would hold 4092..4095, which
    # the channel could never wrap from, and anything above is a firmware error: all ignored
    bus = await start(dut)
    await bus.write(R.ch(0, R.CODE_CHIP), 4091)
    for value in (4092, 4095, 4096, 0xFFFF_FFFF):
        await bus.write(R.ch(0, R.CODE_CHIP), value)
        assert await bus.read(R.ch(0, R.CODE_CHIP)) == 4091, value
    await bus.write(R.ch(0, R.CODE_CHIP), 0)
    assert await bus.read(R.ch(0, R.CODE_CHIP)) == 0


async def read_measurement(bus, k):
    frac = await bus.read(R.ch(k, R.MEAS_CODE_PHASE_FRAC))
    chip_q = await bus.read(R.ch(k, R.MEAS_CODE_CHIP))
    carr = await bus.read(R.ch(k, R.MEAS_CARR_PHASE))
    epoch = await bus.read(R.ch(k, R.MEAS_EPOCH))
    lo = await bus.read(R.ch(k, R.MEAS_SAMPLE_LO))
    hi = await bus.read(R.ch(k, R.MEAS_SAMPLE_HI))
    return Measurement(carr, frac, chip_q, epoch, (hi << 32) | lo)


@cocotb.test()
async def measurement_trigger_captures_all_channels_on_one_sample(dut):
    bus = await start(dut)
    x = signal(7, 1000.0, 0, 8, seed=1) + signal(12, -2500.0, 1500, 8, seed=2)
    codes = to_sign_mag(x)
    await set_up_channel(bus, 0, 7, 1000.0)
    await set_up_channel(bus, 1, 12, -2500.0, 1500)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 20_000)
    await bus.write(R.MEAS_TRIG, 1)
    await ReadOnly()
    at_write = int(dut.sample_cnt.value)  # samples consumed once the write has landed
    await ClockCycles(dut.clk, len(codes))
    m0 = await read_measurement(bus, 0)
    m1 = await read_measurement(bus, 1)
    assert m0.sample == m1.sample
    assert m0.sample == at_write, (m0.sample, at_write)  # the next valid sample is strobed
    for k, (prn, doppler, offset) in enumerate(((7, 1000.0, 0), (12, -2500.0, 1500))):
        mdl = model(prn, doppler, offset)
        mdl.run(codes[: m0.sample])
        assert (m0, m1)[k] == mdl.measurement()


@cocotb.test()
async def measurement_periodic(dut):
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 2.5))  # 20 000 samples
    await set_up_channel(bus, 0, PRN, 1000.0)
    await load_playback(bus, codes)
    await bus.write(R.MEAS_PERIOD, 5000)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 7000)
    first = await read_measurement(bus, 0)
    assert first.sample == 5000
    await ClockCycles(dut.clk, len(codes))
    last = await read_measurement(bus, 0)
    assert last.sample == 15_000
    mdl = model(PRN, 1000.0)
    mdl.run(codes[:15_000])
    assert last == mdl.measurement()
    assert last.epoch == mdl.epoch >= 1


@cocotb.test()
async def code_write_address_wraps_at_32_words(dut):
    bus = await start(dut)
    await bus.write(R.ch(0, R.CODE_WR_ADDR), 31)
    await bus.write(R.ch(0, R.CODE_WR_DATA), 0xFFFFFFFF)
    assert await bus.read(R.ch(0, R.CODE_WR_ADDR)) == 0
    await bus.write(R.ch(0, R.CODE_WR_ADDR), 33)
    assert await bus.read(R.ch(0, R.CODE_WR_ADDR)) == 1
