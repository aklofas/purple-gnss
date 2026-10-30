"""rx_spi: the single-lane SPI slave in front of rx_core (N_CHANNELS = 1, default RAM)."""

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
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
    load_code,
    start_playback,
)


async def start(dut, sck_period_ns: float = 82.6) -> SpiMaster:
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    dut.pin_valid.value = 0
    dut.pin_sample.value = 0
    dut.pll_lock.value = 1
    spi = SpiMaster(dut, sck_period_ns)
    dut.rst.value = 1
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)
    return spi


@cocotb.test()
async def read_id_and_version(dut):
    spi = await start(dut)
    assert await spi.read(R.ID) == R.ID_VALUE
    version = R.VERSION_BASE | R.VERSION_LATCH
    assert await spi.read(R.VERSION) == version
    assert await spi.read_words(R.ID, 2) == [R.ID_VALUE, version]


@cocotb.test()
async def write_then_read_register(dut):
    spi = await start(dut)
    await spi.write(R.ch(0, R.CARR_STEP), 0xA5C3F00F)
    assert await spi.read(R.ch(0, R.CARR_STEP)) == 0xA5C3F00F
    await spi.write(R.ch(0, R.CODE_STEP), 1)
    assert await spi.read(R.ch(0, R.CODE_STEP)) == 1


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
    await spi.write_words(R.ch(0, R.CARR_STEP), [10, 20, 30, 40, 50, 60])  # CARR_STEP..SPACING_Q
    assert await spi.read_words(R.ch(0, R.CARR_STEP), 6) == [10, 20, 30, 40, 50, 60 & 3]


@cocotb.test()
async def fixed_address_mode(dut):
    spi = await start(dut)
    await spi.write_words(R.ch(0, R.CARR_STEP), [1, 2, 3], increment=False)
    assert await spi.read(R.ch(0, R.CARR_STEP)) == 3
    assert await spi.read(R.ch(0, R.CODE_STEP)) == 0
    assert await spi.read_words(R.ID, 3, increment=False) == [R.ID_VALUE] * 3


@cocotb.test()
async def aborted_word_is_not_written(dut):
    spi = await start(dut)
    await spi.write(R.ch(0, R.CARR_STEP), 0x12345678)
    await spi.abort_mid_word(R.ch(0, R.CARR_STEP), 0xFFFFFFFF)
    assert await spi.read(R.ch(0, R.CARR_STEP)) == 0x12345678
    assert await spi.read(R.ID) == R.ID_VALUE


@cocotb.test()
async def sck_at_fabric_over_8(dut):
    spi = await start(dut, sck_period_ns=8 * CLK_NS)  # the documented limit, exactly
    words = [0xDEADBEEF, 0x01234567, 0x89ABCDEF]
    await spi.write_words(R.ch(0, R.CARR_STEP), words)
    assert await spi.read_words(R.ch(0, R.CARR_STEP), 3) == words
    slow = SpiMaster(dut, sck_period_ns=32 * CLK_NS)
    assert await slow.read_words(R.ch(0, R.CARR_STEP), 3) == words


@cocotb.test()
async def cs_hold_boundary(dut):
    # the word is lost only when CS rises in the same fabric clock as the last SCK rise
    spi = await start(dut)
    for hold_clocks, landed in ((0, False), (1, True), (2, True)):
        await spi.write(R.ch(0, R.CARR_STEP), 0)
        await spi.write_with_cs_hold(R.ch(0, R.CARR_STEP), 0xA5C3F00F, hold_clocks)
        value = await spi.read(R.ch(0, R.CARR_STEP))
        assert value == (0xA5C3F00F if landed else 0), (hold_clocks, hex(value))
        assert await spi.read(R.ID) == R.ID_VALUE


@cocotb.test()
async def aborted_header_and_read_leave_no_trace(dut):
    spi = await start(dut)
    await spi.write_words(R.ch(0, R.CARR_STEP), [11, 22, 33])
    await spi.abort_in_header(R.ch(0, R.CARR_STEP))
    assert await spi.read_words(R.ch(0, R.CARR_STEP), 3) == [11, 22, 33]
    assert await spi.read_words_aborted(R.ch(0, R.CARR_STEP), 2, 3) == [11, 22]
    assert await spi.read_words(R.ch(0, R.CARR_STEP), 3) == [11, 22, 33]
    assert await spi.read(R.ID) == R.ID_VALUE


@cocotb.test()
async def cs_glitch_restarts_the_framing(dut):
    # CS high for three fabric clocks inside a burst ends it: the word before the glitch is
    # written, the bytes after it are taken as a new header (here a read: nothing lands)
    spi = await start(dut)
    await spi.write_words_cs_glitch(R.ch(0, R.CARR_STEP), [0xA5A5A5A5, 0x00123456, 0x789ABCDE], 3)
    assert await spi.read(R.ch(0, R.CARR_STEP)) == 0xA5A5A5A5
    assert await spi.read(R.ch(0, R.CODE_STEP)) == 0
    assert await spi.read(R.ID) == R.ID_VALUE


@cocotb.test()
async def status_read_relatches_sample_count_high_word(dut):
    # every read fetches the word after its last one, so a STATUS read fetches SAMPLE_CNT_LO
    # and re-latches SAMPLE_CNT_HI: read the pair as one burst, never LO, STATUS, HI
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
    assert await spi.read(R.SAMPLE_CNT_HI) == 0  # latched with LO: coherent
    await spi.read(R.STATUS)
    assert await spi.read(R.SAMPLE_CNT_HI) == 1  # re-latched by the read-ahead, LO unread
    lo, hi = await spi.read_words(R.SAMPLE_CNT_LO, 2)
    assert hi == 1 and lo < 10_000


async def board_sequence(spi, prn: int, doppler: float, code_offset_s: float, codes) -> None:
    """The board test's register sequence on one channel, every dump equal to the model's."""
    chip_q, frac = start_phase(code_offset_s, 1.023e6 * (1 + doppler / 1575.42e6))
    carr_step, code_step = channel_steps(doppler)

    assert await spi.read(R.ID) == R.ID_VALUE
    assert await spi.read(R.VERSION) == R.VERSION_BASE | R.VERSION_LATCH
    await load_code(spi, 0, prn)
    await spi.write(R.PB_ADDR, 0)
    await spi.write_words(R.PB_DATA, pack_words(codes), increment=False)
    await configure_channel(spi, 0, carr_step, code_step, 0, frac, chip_q)
    await enable_channel(spi, 0)
    await spi.write(R.CTRL, R.CTRL_ENABLE | R.CTRL_SOURCE_PLAYBACK)
    await start_playback(spi, len(codes))
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


@cocotb.test()
async def board_sequence_over_spi(dut):
    spi = await start(dut)
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
