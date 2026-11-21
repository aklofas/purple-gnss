"""rx_spi: the SPI slave in front of rx_core, on one data lane and on four (N_CHANNELS = 1,
default RAM)."""

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
from pyrx.acquisition import acquire_l1ca
from pyrx.codes import gps_ca
from pyrx.samples import read_samples

from fpgamodel import regmap as R
from fpgamodel.channel import Channel, synthetic_codes
from fpgamodel.compare import start_phase
from fpgamodel.quantize import code_bits, pack_words, to_sign_mag
from tb.support import (
    CLK_NS,
    F_IF,
    FS,
    RECORDING,
    SpiMaster,
    assert_dumps_equal,
    captures_dir,
    channel_steps,
    collect_dumps,
    configure_channel,
    enable_channel,
    start_playback,
)


async def start(dut, sck_period_ns: float = 82.6, quad: bool = False) -> SpiMaster:
    clock = cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    dut.pin_valid.value = 0
    dut.pin_sample.value = 0
    dut.pll_lock.value = 1
    dut.evt_in.value = 0
    spi = SpiMaster(dut, sck_period_ns, quad)
    spi.clock = clock  # the overrun case stops the fabric clock
    dut.rst.value = 1
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)
    return spi


@cocotb.test()
async def read_id_and_version(dut):
    spi = await start(dut)
    assert await spi.read(R.ID) == R.ID_VALUE
    version = R.VERSION_ALL
    assert await spi.read(R.VERSION) == version
    assert await spi.read_words(R.ID, 2) == [R.ID_VALUE, version]


@cocotb.test()
async def lanes_are_high_impedance_when_idle(dut):
    # with chip select high the slave drives none of the four data lanes
    spi = await start(dut)
    assert int(dut.spi_io_oe.value) == 0
    assert await spi.read(R.ID) == R.ID_VALUE  # the master checks the lanes around the frame
    assert int(dut.spi_io_oe.value) == 0


@cocotb.test()
async def write_then_read_register(dut):
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0xA5C3F00F)
    assert await spi.read(R.MEAS_PERIOD) == 0xA5C3F00F
    await spi.write(R.PLAYBACK_LEN, 1)
    assert await spi.read(R.PLAYBACK_LEN) == 1


@cocotb.test()
async def burst_auto_increment_write_and_read(dut):
    spi = await start(dut)
    words = [0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666]
    # PB_ADDR, then one PB_DATA word, then unmapped addresses that swallow the rest
    await spi.write_words(R.PB_ADDR, [7] + words)
    assert await spi.read(R.PB_ADDR) == 8
    assert int(dut.core.pb.mem[7].value) == words[0]
    await spi.write_words(R.ch(0, R.CODE_WR_ADDR), [3])
    await spi.write_words(R.ch(0, R.CODE_WR_DATA), words[:2], increment=False)
    assert await spi.read(R.ch(0, R.CODE_WR_ADDR)) == 5
    # MEAS_PERIOD, N_CHANNELS (read-only), PLAYBACK_CTRL (bit 0 clear: no start), PLAYBACK_LEN
    await spi.write_words(R.MEAS_PERIOD, [10, 20, 30, 40])
    assert await spi.read_words(R.MEAS_PERIOD, 4) == [10, 1, 0, 40]


@cocotb.test()
async def fixed_address_mode(dut):
    spi = await start(dut)
    await spi.write_words(R.MEAS_PERIOD, [1, 2, 3], increment=False)
    assert await spi.read(R.MEAS_PERIOD) == 3
    assert await spi.read(R.PLAYBACK_LEN) == 0
    assert await spi.read_words(R.ID, 3, increment=False) == [R.ID_VALUE] * 3


@cocotb.test()
async def aborted_word_is_not_written(dut):
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0x12345678)
    await spi.abort_mid_word(R.MEAS_PERIOD, 0xFFFFFFFF)
    assert await spi.read(R.MEAS_PERIOD) == 0x12345678
    assert await spi.read(R.ID) == R.ID_VALUE


async def link_status(spi) -> tuple[int, int, int]:
    """(frames rejected by signature, write overruns, read underruns), read on a slow clock."""
    word = await SpiMaster(spi.dut).read(R.LINK_STATUS)
    return word & 0xFFFF, (word >> 16) & 0xFF, word >> 24


# SCK periods against the bench's 10 ns fabric clock: the rated 25 MHz against the slowest
# fabric clock a board has, 34.9 MHz, is 13.96 ns here; 40 MHz is 8.7 ns
RATED_NS, HEADROOM_NS = 13.96, 8.7
PATTERN = [0xDEADBEEF, 0x01234567, 0x89ABCDEE, 0x0F0F0F0F]  # the third lands on PLAYBACK_CTRL
PATTERN_BACK = [0xDEADBEEF, 1, 0, 0x0F0F0F0F]  # MEAS_PERIOD, N_CHANNELS, PLAYBACK_CTRL, PLAYBACK_LEN


@cocotb.test()
async def quad_read_and_write_bursts(dut):
    spi = await start(dut, quad=True)
    assert await spi.read_words(R.ID, 2) == [R.ID_VALUE, R.VERSION_ALL]
    assert await spi.read_words(R.ID, 3, increment=False) == [R.ID_VALUE] * 3
    await spi.write_words(R.MEAS_PERIOD, PATTERN)
    assert await spi.read_words(R.MEAS_PERIOD, 4) == PATTERN_BACK
    words = [int(w) for w in np.random.default_rng(1).integers(0, 2**32, size=16)]
    await spi.write(R.PB_ADDR, 0)
    await spi.write_words(R.PB_DATA, words, increment=False)
    assert await spi.read(R.PB_ADDR) == 16
    assert [int(dut.core.pb.mem[i].value) for i in range(16)] == words
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def single_and_quad_frames_interleave(dut):
    # the lane width is each frame's own choice
    spi = await start(dut)
    for i in range(6):
        quad = bool(i & 1)
        await spi.write_words(R.MEAS_PERIOD, [0x1000 + i, 0, 0, 0x2000 + i], quad=quad)
        assert await spi.read_words(R.MEAS_PERIOD, 4, quad=not quad) == [0x1000 + i, 1, 0, 0x2000 + i]
        assert await spi.read(R.ID) == R.ID_VALUE
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def bad_signature_frame_is_ignored_and_counted(dut):
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0x12345678)
    # every bit a clock late: the command byte no longer ends in the signature
    await spi.write_with_slipped_clock(R.MEAS_PERIOD, 0xFFFFFFFF)
    assert await spi.read(R.MEAS_PERIOD) == 0x12345678
    assert await link_status(spi) == (1, 0, 0)
    # the command bytes of the earlier register map, a write and a read: nothing lands,
    # nothing is driven
    await spi.raw_frame(bytes([0x80, 0x00, R.MEAS_PERIOD, 0xFF, 0xFF, 0xFF, 0xFF]))
    await spi.raw_frame(bytes([0x40, 0x00, R.ID, 0, 0, 0, 0, 0]))
    assert await spi.read(R.MEAS_PERIOD) == 0x12345678
    assert await link_status(spi) == (3, 0, 0)
    assert await spi.read(R.ID) == R.ID_VALUE


@cocotb.test()
async def sck_at_1_to_4_and_1_to_1_of_the_fabric_clock(dut):
    spi = await start(dut)
    for period in (4 * CLK_NS, CLK_NS):
        for quad in (False, True):
            fast = SpiMaster(dut, period, quad)
            await fast.write_words(R.MEAS_PERIOD, [0, 0, 0, 0])
            await fast.write_words(R.MEAS_PERIOD, PATTERN)
            assert await fast.read_words(R.MEAS_PERIOD, 4) == PATTERN_BACK, (period, quad)
            assert await fast.read(R.ID) == R.ID_VALUE
    assert await link_status(spi) == (0, 0, 0)


async def long_bursts(dut, period_ns: float, quad: bool) -> bool:
    """A 64-word read and a 64-word write at `period_ns`; whether every word was right."""
    fast = SpiMaster(dut, period_ns, quad)
    words = [int(w) for w in np.random.default_rng(2).integers(0, 2**32, size=64)]
    await fast.write(R.PB_ADDR, 0)
    await fast.write_words(R.PB_DATA, words, increment=False)
    slow = SpiMaster(dut)
    good = await slow.read(R.PB_ADDR) == 64
    good &= [int(dut.core.pb.mem[i].value) for i in range(64)] == words
    good &= await fast.read_words(R.ID, 64, increment=False) == [R.ID_VALUE] * 64
    burst = await fast.read_words(R.CTRL, 12)  # CTRL to PB_DEPTH, auto-increment
    good &= burst == [await slow.read(R.CTRL + i) for i in range(12)]
    return good


@cocotb.test()
async def sck_at_the_rated_rate_against_the_slow_fabric(dut):
    spi = await start(dut)
    for quad in (False, True):
        assert await long_bursts(dut, RATED_NS, quad), quad
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def first_read_word_is_in_time_at_any_phase_up_to_the_fabric_rate(dut):
    # the first word of a read has eight dummy clocks to cross both ways; whether it makes
    # it depends on where the frame falls against the fabric clock, so many frames at each
    # rate, each at its own random phase. A simulation has no setup time, so SCK equal to
    # the fabric clock is only shown to have margin by a clock a tenth faster
    spi = await start(dut)
    for period in (RATED_NS, CLK_NS, 0.9 * CLK_NS):
        for i in range(150):
            fast = SpiMaster(dut, period, quad=bool(i & 1))
            assert await fast.read(R.ID) == R.ID_VALUE, (period, i)
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def sck_headroom_at_40_mhz(dut):
    # a measurement, not a requirement: 40 MHz against the slowest fabric clock. Whatever
    # the outcome, a wrong word must come with a counted underrun or overrun
    spi = await start(dut)
    results = {quad: await long_bursts(dut, HEADROOM_NS, quad) for quad in (False, True)}
    bad, overruns, underruns = await link_status(spi)
    dut._log.info(f"40 MHz against 34.9 MHz: words right {results}, overruns {overruns}, underruns {underruns}")
    assert bad == 0
    assert all(results.values()) or overruns + underruns > 0


@cocotb.test()
async def cs_rise_mid_word_drops_it_and_clears(dut):
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0x12345678)
    await spi.abort_mid_word(R.MEAS_PERIOD, 0xFFFFFFFF)
    assert await spi.read(R.MEAS_PERIOD) == 0x12345678
    # in a quad burst from DUMP_CLEAR: two whole words, then five nibbles of a third, which
    # would be MEAS_PERIOD's
    quad = SpiMaster(dut, quad=True)
    await quad._begin(R.CMD_WRITE | R.CMD_INCR | R.CMD_QUAD, R.DUMP_CLEAR)
    for word in (0, 0):
        await quad._word_out(word, True)
    for nibble in (0xF, 0xF, 0xF, 0xF, 0xF):
        await quad._clock(nibble)
    await quad._end()
    assert await spi.read(R.MEAS_PERIOD) == 0x12345678
    await quad.write_words(R.DUMP_CLEAR, [0, 0, 0xA5A5A5A5])  # the same burst whole: it lands
    assert await spi.read(R.MEAS_PERIOD) == 0xA5A5A5A5
    await spi.write(R.PLAYBACK_LEN, 77)  # the next frame parses cleanly
    assert await spi.read(R.PLAYBACK_LEN) == 77
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def write_overrun_is_counted_not_corrupted(dut):
    # the fabric clock stops in the middle of a write burst: the words the request queue
    # has no room for are counted and dropped, and the ones it took land in order
    spi = await start(dut)
    words = [int(w) for w in np.random.default_rng(3).integers(0, 2**32, size=12)]
    await spi.write(R.PB_ADDR, 0)
    await spi._begin(R.CMD_WRITE, R.PB_DATA)
    for word in words[:2]:
        await spi._word_out(word, False)
    await Timer(5 * CLK_NS, unit="ns")  # the first two are through
    spi.clock.cancel()
    for word in words[2:]:
        await spi._word_out(word, False)
    spi.clock = cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await spi._end()
    await ClockCycles(dut.clk, 20)
    accepted = await spi.read(R.PB_ADDR)
    assert accepted == 6  # two, then a queue of four
    assert await link_status(spi) == (0, 1, 0)  # one frame lost words
    assert [int(dut.core.pb.mem[i].value) for i in range(accepted)] == words[:accepted]
    assert await spi.read(R.ID) == R.ID_VALUE


async def watch_bus_writes(dut, log: list) -> None:
    """Append (address, data) of every write the slave puts on the register bus."""
    while True:
        await RisingEdge(dut.clk)
        if int(dut.bus_we.value):
            log.append((int(dut.bus_addr.value), int(dut.bus_wdata.value)))


def fabric_clock(dut):
    return cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())


@cocotb.test()
async def write_overrun_drops_the_rest_of_the_frame(dut):
    # the fabric clock stalls inside an auto-increment burst and comes back before the burst
    # ends: once a word was dropped, a later word of that frame would land on an earlier
    # address, so none of them may be written
    spi = await start(dut)
    writes: list = []
    cocotb.start_soon(watch_bus_writes(dut, writes))
    base = 0x4000  # unmapped: the bus sees the writes, no register takes them
    words = [int(w) for w in np.random.default_rng(4).integers(0, 2**32, size=12)]
    await spi._begin(R.CMD_WRITE | R.CMD_INCR, base)
    for word in words[:2]:
        await spi._word_out(word, False)
    await Timer(5 * CLK_NS, unit="ns")
    spi.clock.cancel()
    for word in words[2:8]:  # four queue up, two find the queue full
        await spi._word_out(word, False)
    spi.clock = fabric_clock(dut)
    await ClockCycles(dut.clk, 12)  # the queue has room again
    for word in words[8:]:
        await spi._word_out(word, False)
    await spi._end()
    await ClockCycles(dut.clk, 20)
    assert writes == [(base + i, words[i]) for i in range(6)]
    assert await link_status(spi) == (0, 1, 0)
    # the next frame is whole again
    writes.clear()
    await spi.write_words(base, words[:3])
    await ClockCycles(dut.clk, 20)
    assert writes == [(base + i, words[i]) for i in range(3)]


@cocotb.test()
async def frame_whose_header_finds_the_queue_full_is_dropped_whole(dut):
    # with the fabric clock stalled, one frame fills the queue and the next one's command and
    # address are refused: its words must not be written at the first frame's address
    spi = await start(dut)
    writes: list = []
    cocotb.start_soon(watch_bus_writes(dut, writes))
    await ClockCycles(dut.clk, 4)
    spi.clock.cancel()
    await spi.write_words(0x4000, [0x11, 0x22, 0x33])  # header and three words: the queue is full
    await spi.write_words(0x5000, [0x44, 0x55])
    spi.clock = fabric_clock(dut)
    await ClockCycles(dut.clk, 20)
    assert writes == [(0x4000, 0x11), (0x4001, 0x22), (0x4002, 0x33)]
    assert await link_status(spi) == (0, 1, 0)
    writes.clear()
    await spi.write_words(0x5000, [0x44, 0x55])
    await ClockCycles(dut.clk, 20)
    assert writes == [(0x5000, 0x44), (0x5001, 0x55)]


@cocotb.test()
async def events_during_a_fabric_stall_are_all_counted(dut):
    # the counts cross from the SCK domain, so they must survive a stretch in which the
    # fabric clock does not run: 40 rejected frames and one long dropped burst
    spi = await start(dut)
    await ClockCycles(dut.clk, 4)
    spi.clock.cancel()
    for _ in range(40):
        await spi.raw_frame(bytes([0x80, 0x00, R.MEAS_PERIOD, 0, 0, 0, 1]))
    await spi.write_words(R.PB_DATA, range(300), increment=False, quad=True)
    spi.clock = fabric_clock(dut)
    await ClockCycles(dut.clk, 20)
    assert await link_status(spi) == (40, 1, 0)
    assert await spi.read(R.PB_ADDR) == 3  # the three words the queue held


@cocotb.test()
async def read_underrun_is_counted_and_realigns(dut):
    # a clock so fast that the first word cannot be there after eight dummy clocks: the
    # frame returns zeros from the missing word on and is counted once; the next frame is
    # whole again
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0x12345678)
    for quad in (False, True):
        hasty = SpiMaster(dut, 2.0, quad)
        assert await hasty.read_words(R.MEAS_PERIOD, 4, increment=False) == [0, 0, 0, 0], quad
        assert await spi.read(R.MEAS_PERIOD) == 0x12345678
        assert await spi.read(R.ID) == R.ID_VALUE
    assert await link_status(spi) == (0, 0, 2)


@cocotb.test()
async def read_cut_short_after_the_header_leaves_the_next_read_clean(dut):
    # chip select rising right after a read's address, at any phase of the fabric clock: the
    # word that read was about to fetch must not show up in the next frame
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0x0BADF00D)
    for i in range(24):
        cut = SpiMaster(dut, (RATED_NS, 4 * CLK_NS, 82.6)[i % 3], quad=bool(i & 1))
        await cut.read_header_only(R.MEAS_PERIOD)
        assert await cut.read(R.ID) == R.ID_VALUE, i
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def three_fabric_clocks_of_cs_high_separate_frames(dut):
    # the documented minimum between frames, at any phase: reads of different registers back
    # to back, a read cut short among them, each returning its own word
    spi = await start(dut)
    await spi.write(R.MEAS_PERIOD, 0x0BADF00D)
    await spi.write(R.PLAYBACK_LEN, 77)
    for i in range(120):
        tight = SpiMaster(dut, RATED_NS, quad=bool(i & 1), cs_high_ns=3 * CLK_NS)
        if i % 5 == 0:
            await tight.read_header_only(R.PLAYBACK_LEN)
        assert await tight.read(R.MEAS_PERIOD) == 0x0BADF00D, i
        assert await tight.read_words(R.ID, 2) == [R.ID_VALUE, R.VERSION_ALL], i
        await tight.write(R.PLAYBACK_LEN, 77 + i)
        assert await tight.read(R.PLAYBACK_LEN) == 77 + i, i
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def reads_fetch_ahead_without_side_effects(dut):
    # a read burst fetches past its last word (here into DUMP_CLEAR and MEAS_TRIG): nothing
    # is cleared and nothing is triggered
    spi = await start(dut)
    codes = synthetic_codes(7, FS, F_IF, 1000.0, 0, round(FS * 2e-3), 45.0)
    await board_sequence_setup(spi, 7, 1000.0, 0.0, codes)
    await ClockCycles(dut.clk, len(codes) + 50)
    for quad in (False, True):
        assert await spi.read_words(R.SAMPLE_CNT_HI, 2, quad=quad) == [0, 1]  # then DUMP_PENDING
        assert await spi.read(R.DUMP_PENDING) == 1
        assert await spi.read(R.MEAS_SEQ) == 0


@cocotb.test()
async def soft_reset_clears_link_status(dut):
    spi = await start(dut)
    await spi.raw_frame(bytes([0x80, 0x00, R.MEAS_PERIOD, 0, 0, 0, 1]))
    assert await link_status(spi) == (1, 0, 0)
    await spi.write(R.CTRL, R.CTRL_RESET)  # the frame that carries the reset ends normally
    assert await link_status(spi) == (0, 0, 0)
    await spi.write(R.MEAS_PERIOD, 5)
    assert await spi.read(R.MEAS_PERIOD) == 5


@cocotb.test()
async def aborted_header_and_read_leave_no_trace(dut):
    spi = await start(dut)
    await spi.write_words(R.MEAS_PERIOD, [11, 22, 34, 44])
    await spi.abort_in_header(R.MEAS_PERIOD)
    assert await spi.read_words(R.MEAS_PERIOD, 4) == [11, 1, 0, 44]
    assert await spi.read_words_aborted(R.MEAS_PERIOD, 2, 3) == [11, 1]
    assert await spi.read_words(R.MEAS_PERIOD, 4) == [11, 1, 0, 44]
    assert await spi.read(R.ID) == R.ID_VALUE


@cocotb.test()
async def cs_glitch_restarts_the_framing(dut):
    # CS high for three fabric clocks inside a burst ends it: the word before the glitch is
    # written, the bytes after it are taken as a new header (here one without the signature:
    # nothing lands)
    spi = await start(dut)
    await spi.write_words_cs_glitch(R.MEAS_PERIOD, [0xA5A5A5A5, 0x00123456, 0x789ABCDE], 3)
    assert await spi.read(R.MEAS_PERIOD) == 0xA5A5A5A5
    assert await spi.read(R.PLAYBACK_LEN) == 0
    assert await spi.read(R.ID) == R.ID_VALUE
    assert await link_status(spi) == (1, 0, 0)  # that header's command byte, 0x00, has no signature


@cocotb.test()
async def sample_count_words_are_live(dut):
    # neither count word reacts to a read (every read fetches past its end, so a
    # read side effect would fire from reads of other registers); a coherent 48-bit value
    # comes from the measurement latch, taken on a write
    spi = await start(dut)
    codes = np.zeros(60_000, dtype=np.uint8)
    await spi.write(R.PB_ADDR, 0)
    await spi.write_words(R.PB_DATA, pack_words(codes), increment=False)
    await spi.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(spi, len(codes))
    await ClockCycles(dut.clk, 10)
    dut.core.sample_cnt.value = 2**32 - 600  # crosses the word boundary during the reads
    lo = await spi.read(R.SAMPLE_CNT_LO)
    assert lo > 2**32 - 600
    assert await spi.read(R.SAMPLE_CNT_HI) == 1  # live: the count has crossed by now
    await spi.read(R.STATUS)
    assert await spi.read(R.SAMPLE_CNT_HI) == 1
    await spi.write(R.MEAS_TRIG, 1)
    lo, hi = await spi.read_words(R.ch(0, R.MEAS_SAMPLE_LO), 2)
    assert hi == 1 and 0 < lo < 20_000  # the latched pair is coherent


async def board_sequence_setup(spi, prn: int, doppler: float, code_offset_s: float, codes):
    """The board test's register sequence on one channel, up to the write that starts playback."""
    chip_q, frac = start_phase(code_offset_s, 1.023e6 * (1 + doppler / 1575.42e6))
    carr_step, code_step = channel_steps(doppler)

    assert await spi.read(R.ID) == R.ID_VALUE
    assert await spi.read(R.VERSION) == R.VERSION_ALL
    words = [int(w) for w in code_bits(gps_ca(prn))]
    await spi.write(R.ch(0, R.CODE_WR_ADDR), 0)
    await spi.write_words(R.ch(0, R.CODE_WR_DATA), words[:16], increment=False)
    assert await spi.read(R.ch(0, R.CODE_WR_ADDR)) == 16
    await spi.write_words(R.ch(0, R.CODE_WR_DATA), words[16:], increment=False)
    packed = pack_words(codes)
    await spi.write(R.PB_ADDR, 0)
    await spi.write_words(R.PB_DATA, packed, increment=False)
    assert await spi.read(R.PB_ADDR) == len(packed)
    await configure_channel(spi, 0, carr_step, code_step, 0, frac, chip_q)
    await enable_channel(spi, 0)
    await spi.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(spi, len(codes))
    return carr_step, code_step, frac, chip_q


async def board_sequence(spi, prn: int, doppler: float, code_offset_s: float, codes) -> None:
    """The board test's register sequence on one channel, every dump equal to the model's."""
    carr_step, code_step, frac, chip_q = await board_sequence_setup(spi, prn, doppler, code_offset_s, codes)
    dumps = await collect_dumps(spi, 1)

    m = Channel(code_bits(gps_ca(prn)))
    m.set_steps(carr_step, code_step)
    m.set_phase(0, frac, chip_q)
    m.enabled = True
    expected = m.run(codes)
    assert len(expected) >= 2
    assert_dumps_equal(dumps[0], expected, ignore_sample=True)
    assert await spi.read(R.ch(0, R.DUMP_MISSED)) == 0
    assert await spi.read(R.SAMPLE_CNT_LO) == len(codes)
    assert np.all(np.array([d.ip for d in expected]) != 0)
    assert await link_status(spi) == (0, 0, 0)


@cocotb.test()
async def board_sequence_over_spi(dut):
    spi = await start(dut)
    prn, doppler = 7, 1000.0
    codes = synthetic_codes(prn, FS, F_IF, doppler, 0, round(FS * 3e-3), 45.0)
    await board_sequence(spi, prn, doppler, 0.0, codes)


@cocotb.test()
async def board_sequence_over_spi_quad(dut):
    spi = await start(dut, quad=True)
    prn, doppler = 7, 1000.0
    codes = synthetic_codes(prn, FS, F_IF, doppler, 0, round(FS * 3e-3), 45.0)
    await board_sequence(spi, prn, doppler, 0.0, codes)


@cocotb.test()
async def board_sequence_over_spi_recording(dut):
    # the first 3 ms of the recording, on the strongest PRN acquisition finds in them
    spi = await start(dut)
    x = read_samples(captures_dir() / RECORDING, FS, iq=False, duration=3e-3).real
    results = [acquire_l1ca(x, FS, F_IF, prn) for prn in range(1, 33)]
    best = max(results, key=lambda a: a.cn0 if a.detected else -1.0)
    assert best.detected
    await board_sequence(spi, best.prn, best.doppler, best.code_offset, to_sign_mag(x))
