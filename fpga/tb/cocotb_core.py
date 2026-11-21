"""rx_core: register bus, global registers, sample mux and playback RAM around channels.

Synthetic stimulus at 8 MHz (2 MHz IF): the quarter-chip code NCO needs more than 4.092 MHz,
and 12 ms is then 96 000 samples, which fits the board's 147 456-sample playback RAM.
N_CHANNELS = 2, except for the one case the 12-channel build runs.
"""

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.signals import L1_FREQ
from pyrx.synth import synth_l1ca

from fpgamodel import regmap as R
from fpgamodel.channel import Channel, Measurement, code_step, nco_step
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
    read_dumps_now,
    read_event,
    signed32,
    start_playback,
)

FS, F_IF = 8e6, 2e6
N_CHANNELS = 2
PB_DEPTH = 147_456
PRN = 7


def steps(doppler):
    return nco_step(F_IF + doppler, FS), code_step(CA_CHIP_RATE * (1 + doppler / L1_FREQ), FS)


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
    dut.evt_in.value = 0
    dut.link_bad_frames.value = 0
    dut.link_overruns.value = 0
    dut.link_underruns.value = 0
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
    assert await bus.read(R.VERSION) == R.VERSION_ALL == 0x000F0300
    assert await bus.read(R.N_CHANNELS) == N_CHANNELS
    assert await bus.read(R.PB_DEPTH) == PB_DEPTH
    assert await bus.read(R.STATUS) == R.STATUS_PLL_LOCK
    dut.pll_lock.value = 0
    await RisingEdge(dut.clk)
    assert await bus.read(R.STATUS) == 0


@cocotb.test()
async def unmapped_reads_zero_writes_ignored(dut):
    bus = await start(dut)
    for addr in (0x0FFF, 0x10, 0x102, 0x205, R.ch(N_CHANNELS, 0), R.ch(0, 0xFF), R.ch(0, 0x17), 0x2000, 0xFFFF):
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
    zeroed += (R.SAMPLE_CNT_HI, R.PB_ADDR, R.MEAS_SEQ)
    zeroed += tuple(R.ch(0, off) for off in (R.CH_CTRL, R.CODE_WR_ADDR))
    zeroed += tuple(R.ch(0, off) for off in (R.EPOCH, R.DUMP_MISSED, R.IE, R.QE, R.IP, R.QP, R.IL))
    zeroed += tuple(R.ch(0, off) for off in (R.QL, R.MEAS_CODE_PHASE_FRAC, R.MEAS_CODE_CHIP))
    zeroed += tuple(R.ch(0, off) for off in (R.MEAS_CARR_PHASE, R.MEAS_EPOCH, R.MEAS_SAMPLE_LO))
    zeroed += (R.ch(0, R.MEAS_SAMPLE_HI),)
    for addr in zeroed:
        assert await bus.read(addr) == 0, hex(addr)
    assert await bus.read(R.ch(0, R.SPACING_Q)) == 1  # its reset value
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
    latest, again = await read_dump(bus, 0)
    assert_dumps_equal([latest], expected[-1:], ignore_sample=True)
    assert not again
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
    assert_dumps_equal([(await read_dump(bus, 0))[0]], expected[-1:], ignore_sample=True)
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


async def latch_after_strobe(bus, k):
    """Strobe the latch with three live pin samples and return channel k's latched phase
    (carrier phase, code fraction, quarter-chip index): the only way the write-only phase
    registers are read, and the right one, since it shows what the NCOs hold."""
    dut = bus.dut
    await bus.write(R.CTRL, R.CTRL_ENABLE)  # the pins as the source
    await bus.write(R.MEAS_TRIG, 1)
    dut.pin_valid.value = 1
    await ClockCycles(dut.clk, 3)
    dut.pin_valid.value = 0
    await bus.write(R.CTRL, 0)
    m = await read_measurement(bus, k)
    return (m.carr_phase, m.code_frac, m.chip_q)


@cocotb.test()
async def phase_registers_only_while_disabled(dut):
    # the phase words have no readback; the latch shows what landed in the NCOs: a write to
    # any one of them on a disabled channel reloads all three, writes while enabled are dropped
    bus = await start(dut)
    await bus.write(R.ch(0, R.CARR_PHASE), 0x11111111)
    await bus.write(R.ch(0, R.CODE_PHASE_FRAC), 0x22222222)
    await bus.write(R.ch(0, R.CODE_CHIP), 4091)
    assert await bus.read(R.ch(0, R.CARR_PHASE)) == 0  # write-only
    assert await latch_after_strobe(bus, 0) == (0x11111111, 0x22222222, 4091)
    await enable_channel(bus, 0)  # with step words of 0 the NCOs stand still
    await bus.write(R.ch(0, R.CARR_PHASE), 0x33333333)
    await bus.write(R.ch(0, R.CODE_CHIP), 5)
    assert await latch_after_strobe(bus, 0) == (0x11111111, 0x22222222, 4091)
    await enable_channel(bus, 0, False)
    await bus.write(R.ch(0, R.CODE_CHIP), 5)
    assert await latch_after_strobe(bus, 0) == (0x11111111, 0x22222222, 5)


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
async def meas_period_written_mid_run_counts_from_the_write(dut):
    # the period counter holds at 0 while the period is 0 and restarts on a MEAS_PERIOD write,
    # so the first periodic strobe comes MEAS_PERIOD valid samples after the write
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 2.5))  # 20 000 samples
    await set_up_channel(bus, 0, PRN, 1000.0)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 3000)
    await bus.write(R.MEAS_PERIOD, 5000)
    await ReadOnly()
    at_write = int(dut.sample_cnt.value)
    await ClockCycles(dut.clk, 4900)
    assert (await read_measurement(bus, 0)).sample == 0  # nothing yet
    await ClockCycles(dut.clk, 200)
    first = await read_measurement(bus, 0)
    assert first.sample == at_write + 5000, (first.sample, at_write)
    await ClockCycles(dut.clk, len(codes))
    assert (await read_measurement(bus, 0)).sample == at_write + 15_000


@cocotb.test()
async def playback_write_past_the_end_is_dropped(dut):
    # the RAM holds PB_DEPTH / 16 words; a write at or past that is dropped (the address bits
    # would otherwise alias onto word 0) and PB_ADDR stays at the end as the tell
    bus = await start(dut)
    words = PB_DEPTH // 16
    await bus.write(R.PB_ADDR, 0)
    await bus.write(R.PB_DATA, 0x1111_1111)
    await bus.write(R.PB_ADDR, words - 1)
    await bus.write(R.PB_DATA, 0x2222_2222)
    await bus.write(R.PB_DATA, 0x3333_3333)
    assert await bus.read(R.PB_ADDR) == words
    await bus.write(R.PB_ADDR, 1 << 14)  # the address bits of word 0 at this depth
    await bus.write(R.PB_DATA, 0x4444_4444)
    assert await bus.read(R.PB_ADDR) == 1 << 14
    await RisingEdge(dut.clk)
    assert int(dut.pb.mem[0].value) == 0x1111_1111
    assert int(dut.pb.mem[words - 1].value) == 0x2222_2222


@cocotb.test()
async def step_pair_commits_on_the_code_step_write(dut):
    # a CARR_STEP written alone changes nothing; the CODE_STEP write lands both at once, so
    # a dump between the two writes never runs a new carrier step with the old code step
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 12))
    await set_up_channel(bus, 0, PRN, 1000.0)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    carr_new, code_new = steps(3000.0)
    dumps = []
    # two polls per sample played; playback ends after len(codes), so fewer dumps would hang here
    budget = 40 * len(codes)
    while len(dumps) < 12:
        assert budget > 0, f"only {len(dumps)} dumps arrived"
        budget -= 20
        if await bus.read(R.DUMP_PENDING) & 1:
            dumps.extend(await read_dumps_now(bus, 0))
            if len(dumps) == 4:
                await bus.write(R.ch(0, R.CARR_STEP), carr_new)  # alone: nothing changes
            if len(dumps) == 7:
                await bus.write(R.ch(0, R.CODE_STEP), code_new)  # both land, from dump 8 on
        else:
            await ClockCycles(dut.clk, 20)
    m = model(PRN, 1000.0)
    expected = m.run(codes, max_dumps=7)
    m.set_steps(carr_new, code_new)
    expected += m.run(codes[m.sample_count :])
    assert_dumps_equal(dumps, expected, ignore_sample=True)
    assert await bus.read(R.ch(0, R.DUMP_MISSED)) == 0


@cocotb.test()
async def code_chip_out_of_range_is_ignored(dut):
    # the quarter-chip index runs 0..4091; the register's 12 bits would hold 4092..4095, which
    # the channel could never wrap from, and anything above is a firmware error: all ignored
    bus = await start(dut)
    await bus.write(R.ch(0, R.CODE_CHIP), 4091)
    for value in (4092, 4095, 4096, 0xFFFF_FFFF):
        await bus.write(R.ch(0, R.CODE_CHIP), value)
        assert (await latch_after_strobe(bus, 0))[2] == 4091, value
    await bus.write(R.ch(0, R.CODE_CHIP), 0)
    assert (await latch_after_strobe(bus, 0))[2] == 0


async def read_measurement(bus, k):
    """The channel's latch as a Measurement, read again if a strobe landed during the read."""
    while True:
        seq = await bus.read(R.MEAS_SEQ)
        frac = await bus.read(R.ch(k, R.MEAS_CODE_PHASE_FRAC))
        chip_q = await bus.read(R.ch(k, R.MEAS_CODE_CHIP))
        carr = await bus.read(R.ch(k, R.MEAS_CARR_PHASE))
        epoch = await bus.read(R.ch(k, R.MEAS_EPOCH))
        lo = await bus.read(R.ch(k, R.MEAS_SAMPLE_LO))
        hi = await bus.read(R.ch(k, R.MEAS_SAMPLE_HI))
        if await bus.read(R.MEAS_SEQ) == seq:
            return Measurement(carr, frac, chip_q, epoch, (hi << 32) | lo)


@cocotb.test()
async def meas_seq_counts_strobes_and_guards_the_latch_read(dut):
    # MEAS_SEQ counts every strobe, triggered or periodic, so a reader that sees it change
    # across the six latch words knows the words straddle two strobes
    bus = await start(dut)
    codes = to_sign_mag(signal(PRN, 1000.0, 0, 2.5))  # 20 000 samples
    await set_up_channel(bus, 0, PRN, 1000.0)
    await load_playback(bus, codes)
    assert await bus.read(R.MEAS_SEQ) == 0
    await bus.write(R.MEAS_PERIOD, 5000)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 2000)
    await bus.write(R.MEAS_TRIG, 1)
    await ClockCycles(dut.clk, 10)
    assert await bus.read(R.MEAS_SEQ) == 1
    for _ in range(6000):  # the first periodic strobe lands at sample 5000
        if int(dut.sample_cnt.value) >= 4990:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError(f"sample count stuck at {int(dut.sample_cnt.value)}")
    seq_before = await bus.read(R.MEAS_SEQ)
    await ClockCycles(dut.clk, 20)
    seq_after = await bus.read(R.MEAS_SEQ)
    assert (seq_before, seq_after) == (1, 2)  # a read of the words across here must retry
    await ClockCycles(dut.clk, len(codes))
    assert await bus.read(R.MEAS_SEQ) == 4  # 5000, 10 000, 15 000 and the trigger
    last = await read_measurement(bus, 0)
    assert last.sample == 15_000


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


# ---- map 0x0300: integration length, dump burst, first-sample latch, wide code step ------
PERIOD = 2000  # samples per code period in the cases below: 45 periods fit the playback RAM
FAST_STEP = round(1023 / PERIOD * 2**34)
FAST_CARR = round(0.3 * 2**32)


def random_words(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2**32, size=32, dtype=np.uint64)


def random_samples(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 4, size=n, dtype=np.uint8)


def fast_model(words, code_step_word=FAST_STEP, integ=1, carr_step=FAST_CARR, carr_phase=0) -> Channel:
    ch = Channel(words)
    ch.set_steps(carr_step, code_step_word)
    ch.set_phase(carr_phase, 0, 0)
    ch.set_integ(integ)
    ch.enabled = True
    return ch


async def set_up_fast_channel(bus, k, words, code_step_word=FAST_STEP, integ=None,
                              carr_step=FAST_CARR, carr_phase=0, enable=True):
    await bus.write(R.ch(k, R.CODE_WR_ADDR), 0)
    for word in words:
        await bus.write(R.ch(k, R.CODE_WR_DATA), int(word))
    await configure_channel(bus, k, carr_step, code_step_word, carr_phase)
    if integ is not None:
        await bus.write(R.ch(k, R.CH_INTEG), integ)
    if enable:
        await enable_channel(bus, k)


@cocotb.test()
async def dump_burst_brackets_epoch(dut):
    bus = await start(dut)
    words, codes = random_words(1), random_samples(3 * PERIOD + 100)
    await set_up_fast_channel(bus, 0, words)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, len(codes) + 20)
    burst = await bus.read_words(R.ch(0, R.DUMP_BURST), R.DUMP_BURST_WORDS)
    last = fast_model(words).run(codes)[-1]
    assert last.epoch == 3
    assert burst[0] == burst[7] == last.epoch == await bus.read(R.ch(0, R.EPOCH))
    assert [signed32(w) for w in burst[1:7]] == [last.ie, last.qe, last.ip, last.qp, last.il, last.ql]
    assert burst[1:7] == [await bus.read(R.ch(0, off)) for off in (R.IE, R.QE, R.IP, R.QP, R.IL, R.QL)]


@cocotb.test()
async def integ_register_reads_back_and_clamps(dut):
    bus = await start(dut)
    assert await bus.read(R.ch(0, R.CH_INTEG)) == 1  # its reset value
    for written, read in ((7, 7), (0, 1), (20, 20), (21, 20), (99, 20), (0x100, 20), (1, 1)):
        await bus.write(R.ch(0, R.CH_INTEG), written)
        assert await bus.read(R.ch(0, R.CH_INTEG)) == read, written
    assert await bus.read(R.ch(1, R.CH_INTEG)) == 1  # the other channel's is its own


@cocotb.test()
async def meas_epoch_counts_dumps_at_integ_20(dut):
    bus = await start(dut)
    words, codes = random_words(2), random_samples(45 * PERIOD)
    await set_up_fast_channel(bus, 0, words, integ=20)
    dumps = (await run_playback(bus, codes))[0]
    expected = fast_model(words, integ=20).run(codes)
    assert [d.epoch for d in expected] == [1, 2]
    assert_dumps_equal(dumps, expected, ignore_sample=True)
    assert await bus.read(R.ch(0, R.EPOCH)) == 2
    await latch_after_strobe(bus, 0)
    assert await bus.read(R.ch(0, R.MEAS_EPOCH)) == 2  # dumps, not the 45 code periods


@cocotb.test()
async def sums_at_integ_20_are_sign_extended_from_25_bits(dut):
    # all chips +1, every sample +3 and the LO held at cos = -2: 20 periods of 6000 samples
    # sum to -720 000, past the 20 bits one period needs
    bus = await start(dut)
    words = np.zeros(32, dtype=np.uint64)
    step = round(1023 / 6000 * 2**34)
    codes = np.full(20 * 6000 + 50, 0b01, dtype=np.uint8)
    await set_up_fast_channel(bus, 0, words, step, integ=20, carr_step=0, carr_phase=3 << 29)
    dumps = (await run_playback(bus, codes))[0]
    expected = fast_model(words, step, integ=20, carr_step=0, carr_phase=3 << 29).run(codes)
    assert len(expected) == 1 and expected[0].ip == -6 * expected[0].sample < -(2**19)
    assert_dumps_equal(dumps, expected, ignore_sample=True)


async def start_sample(bus, k) -> int:
    lo = await bus.read(R.ch(k, R.CH_START_SAMPLE_LO))
    return (await bus.read(R.ch(k, R.CH_START_SAMPLE_HI)) << 32) | lo


@cocotb.test()
async def ch_start_sample_is_the_first_accepted_sample(dut):
    # a channel enabled while samples flow: its carrier phase at a measurement strobe is the
    # loaded phase plus one step per sample from CH_START_SAMPLE to MEAS_SAMPLE
    bus = await start(dut)
    words, codes = random_words(3), random_samples(20_000)
    carr_phase = 0x1234_5678
    await set_up_fast_channel(bus, 0, words)
    await set_up_fast_channel(bus, 1, words, carr_phase=carr_phase, enable=False)
    assert await start_sample(bus, 0) == 0 and await start_sample(bus, 1) == 0  # nothing yet
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 3000)
    await bus.write(R.ch(1, R.CH_CTRL), 1)
    await ClockCycles(dut.clk, 3000)
    await bus.write(R.MEAS_TRIG, 1)
    await ClockCycles(dut.clk, 20)
    m = await read_measurement(bus, 1)
    n0 = await start_sample(bus, 1)
    assert 3000 < n0 < 3100 and await start_sample(bus, 0) == 0  # channel 0 took sample 0
    assert m.carr_phase == (carr_phase + FAST_CARR * (m.sample - n0)) % 2**32
    reference = fast_model(words, carr_phase=carr_phase)
    reference.run(codes[n0 : m.sample])
    assert (m.code_frac, m.chip_q) == (reference.code_acc, reference.chip_q)


@cocotb.test()
async def ch_start_sample_relatches_on_each_enable(dut):
    bus = await start(dut)
    words, codes = random_words(4), random_samples(20_000)
    await set_up_fast_channel(bus, 0, words, enable=False)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    await ClockCycles(dut.clk, 2000)
    await bus.write(R.ch(0, R.CH_CTRL), 1)
    await ClockCycles(dut.clk, 2000)
    first = await start_sample(bus, 0)
    await bus.write(R.ch(0, R.CH_CTRL), 0)
    await ClockCycles(dut.clk, 2000)
    assert await start_sample(bus, 0) == first  # a disabled channel keeps its last value
    await bus.write(R.ch(0, R.CH_CTRL), 1)
    await ClockCycles(dut.clk, 2000)
    second = await start_sample(bus, 0)
    assert 2000 < first < 2100 and 6000 < second < 6100
    await bus.write(R.ch(0, R.CH_START_SAMPLE_LO), 5)  # read-only
    assert await start_sample(bus, 0) == second


@cocotb.test()
async def code_step_hi_commits_with_the_code_step_write(dut):
    # CODE_STEP_HI holds bits 33:32 of the step and waits, as CARR_STEP does, for the
    # CODE_STEP write: alone it changes nothing, then all three words land together
    bus = await start(dut)
    words, codes = random_words(5), random_samples(60_000)
    l5 = code_step(10.23e6, 38.4e6)  # 0.266 chips per sample: bit 32 set
    assert l5 >> 32 == 1
    await set_up_fast_channel(bus, 0, words)
    await load_playback(bus, codes)
    await bus.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(bus, len(codes))
    dumps = []
    budget = 40 * len(codes)
    while not (await bus.read(R.STATUS) & R.STATUS_PB_DONE and not await bus.read(R.DUMP_PENDING)):
        assert budget > 0
        budget -= 20
        if await bus.read(R.DUMP_PENDING) & 1:
            before = len(dumps)
            dumps.extend(await read_dumps_now(bus, 0))
            if before < 3 <= len(dumps):
                await bus.write(R.ch(0, R.CODE_STEP_HI), 1)  # alone: nothing changes
            if before < 6 <= len(dumps):
                await bus.write(R.ch(0, R.CODE_STEP), l5 & 0xFFFF_FFFF)  # all three land
        else:
            await ClockCycles(dut.clk, 20)
    assert await bus.read(R.ch(0, R.DUMP_MISSED)) == 0
    m = fast_model(words)
    expected = m.run(codes, max_dumps=6)
    m.set_steps(FAST_CARR, l5)
    expected += m.run(codes[m.sample_count :])
    lengths = np.diff([d.sample for d in expected])
    assert (lengths[:6] > 1990).all() and (lengths[7:] > 3830).all()  # 2000, then 3840 samples
    assert_dumps_equal(dumps, expected, ignore_sample=True)
    assert await bus.read(R.ch(0, R.CODE_STEP_HI)) == 0  # write-only, like the other step words


@cocotb.test()
async def soft_reset_clears_the_map_0300_registers(dut):
    bus = await start(dut)
    words, codes = random_words(6), random_samples(16_000)
    slow = round(1023 / 5000 * 2**34)  # under a quarter chip per sample: the high bits are 0
    assert slow >> 32 == 0
    await set_up_fast_channel(bus, 0, words, integ=5)
    await bus.write(R.ch(0, R.CODE_STEP_HI), 1)  # staged, never committed
    await bus.write(R.EVT_CTRL, R.EVT_ENABLE | R.EVT_FALLING)
    await bus.write(R.CTRL, R.CTRL_RESET)
    await ClockCycles(dut.clk, 4)
    assert await bus.read(R.ch(0, R.CH_INTEG)) == 1 and await bus.read(R.EVT_CTRL) == 0
    # the staged high bits are gone: CARR_STEP and CODE_STEP alone give the plain step
    await bus.write(R.ch(0, R.CARR_STEP), FAST_CARR)
    await bus.write(R.ch(0, R.CODE_STEP), slow)
    await enable_channel(bus, 0)
    dumps = (await run_playback(bus, codes))[0]
    expected = fast_model(words, slow).run(codes)
    assert len(expected) == 3
    assert_dumps_equal(dumps, expected, ignore_sample=True)


# ---- event stamp and link status --------------------------------------------------------
SYNC_CLOCKS = 2  # an edge is stamped with the sample going in two clocks after the pin changes


async def start_pins(dut):
    """The pins as the sample source with a sample on every clock."""
    bus = await start(dut)
    await bus.write(R.CTRL, R.CTRL_ENABLE)
    dut.pin_valid.value = 1
    await ClockCycles(dut.clk, 50)
    return bus


async def set_evt(dut, level: int) -> int:
    """Change the event pin right after a clock edge; returns the index of the sample that
    goes in on the next edge."""
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")  # past the edge's updates: the count includes its sample
    n = int(dut.sample_cnt.value)
    dut.evt_in.value = level
    return n


@cocotb.test()
async def event_stamp_latches_sample_count(dut):
    bus = await start_pins(dut)
    assert await read_event(bus) == (0, 0)
    await set_evt(dut, 1)  # not enabled yet: nothing is stamped
    await ClockCycles(dut.clk, 30)
    await set_evt(dut, 0)
    await bus.write(R.EVT_CTRL, R.EVT_ENABLE)
    assert await bus.read(R.EVT_CTRL) == R.EVT_ENABLE and await read_event(bus) == (0, 0)
    n1 = await set_evt(dut, 1)
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (1, n1 + SYNC_CLOCKS)
    await set_evt(dut, 0)  # the other edge: ignored
    await ClockCycles(dut.clk, 40)
    n2 = await set_evt(dut, 1)
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (2, n2 + SYNC_CLOCKS) and n2 > n1 + 16
    assert await bus.read(R.EVT_MISSED) == 0


@cocotb.test()
async def event_stamp_counts_close_events_as_missed(dut):
    # a second edge before 16 samples have passed since the last stamp is dropped and
    # counted, so the stamp words hold still while they are read; the sixteenth sample is
    # far enough
    bus = await start_pins(dut)
    await bus.write(R.EVT_CTRL, R.EVT_ENABLE)
    n1 = await set_evt(dut, 1)
    await ClockCycles(dut.clk, 1)
    await set_evt(dut, 0)
    await ClockCycles(dut.clk, 1)
    await set_evt(dut, 1)  # 4 samples after the first edge
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (1, n1 + SYNC_CLOCKS) and await bus.read(R.EVT_MISSED) == 1
    for spacing, stamped in ((15, False), (16, True)):
        await set_evt(dut, 0)
        await ClockCycles(dut.clk, 40)
        seq, _ = await read_event(bus)
        missed = await bus.read(R.EVT_MISSED)
        n_a = await set_evt(dut, 1)
        await ClockCycles(dut.clk, 3)
        await set_evt(dut, 0)
        await ClockCycles(dut.clk, spacing - 5)
        n_b = await set_evt(dut, 1)
        assert n_b - n_a == spacing
        await ClockCycles(dut.clk, 40)
        if stamped:
            assert await read_event(bus) == (seq + 2, n_b + SYNC_CLOCKS)
            assert await bus.read(R.EVT_MISSED) == missed
        else:
            assert await read_event(bus) == (seq + 1, n_a + SYNC_CLOCKS)
            assert await bus.read(R.EVT_MISSED) == missed + 1


@cocotb.test()
async def event_stamp_falling_edge_and_disable(dut):
    bus = await start_pins(dut)
    await bus.write(R.EVT_CTRL, R.EVT_ENABLE | R.EVT_FALLING)
    await set_evt(dut, 1)
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (0, 0)  # the rising edge is not the chosen one
    n = await set_evt(dut, 0)
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (1, n + SYNC_CLOCKS)
    await bus.write(R.EVT_CTRL, R.EVT_FALLING)  # disabled
    await set_evt(dut, 1)
    await ClockCycles(dut.clk, 40)
    await set_evt(dut, 0)
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (1, n + SYNC_CLOCKS) and await bus.read(R.EVT_MISSED) == 0
    # enabling for the rising edge with the pin already high stamps nothing
    await set_evt(dut, 1)
    await ClockCycles(dut.clk, 40)
    await bus.write(R.EVT_CTRL, R.EVT_ENABLE)
    await ClockCycles(dut.clk, 40)
    assert await read_event(bus) == (1, n + SYNC_CLOCKS)


@cocotb.test()
async def event_stamp_counts_samples_not_clocks(dut):
    # with a sample on every third clock the stamp is still the index of the sample going
    # in, and the minimum spacing is counted in samples
    bus = await start(dut)
    await bus.write(R.CTRL, R.CTRL_ENABLE)
    await bus.write(R.EVT_CTRL, R.EVT_ENABLE)

    async def feed():
        while True:
            dut.pin_valid.value = 1
            await RisingEdge(dut.clk)
            dut.pin_valid.value = 0
            await ClockCycles(dut.clk, 2)

    cocotb.start_soon(feed())
    await ClockCycles(dut.clk, 60)
    await set_evt(dut, 1)
    await ClockCycles(dut.clk, 6)
    seq, stamp = await read_event(bus)
    assert seq == 1 and 18 <= stamp <= 23  # near clock 60 / 3
    await set_evt(dut, 0)
    await ClockCycles(dut.clk, 20)
    await set_evt(dut, 1)  # some 10 samples after the stamp, 30 clocks
    await ClockCycles(dut.clk, 60)
    assert await read_event(bus) == (1, stamp) and await bus.read(R.EVT_MISSED) == 1


@cocotb.test()
async def link_status_reads_the_slave_counters(dut):
    bus = await start(dut)
    assert await bus.read(R.LINK_STATUS) == 0
    dut.link_bad_frames.value = 0x1234
    dut.link_overruns.value = 0x56
    dut.link_underruns.value = 0x78
    await RisingEdge(dut.clk)
    assert await bus.read(R.LINK_STATUS) == 0x7856_1234
    await bus.write(R.LINK_STATUS, 0)  # read-only
    assert await bus.read(R.LINK_STATUS) == 0x7856_1234
