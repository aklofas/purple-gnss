"""The tracking loop closed around the RTL through SPI, as the MCU will close it.

Three seconds of the recording stream into rx_spi's live sample pins; after every dump the
bench reads the six sums over SPI, runs the tracker and writes the next step words. The RTL's
dumps must equal a model run fed the same samples and the same step schedule, and the loop
must track like pyrx (`fpgamodel.loop.compare_with_pyrx`).
"""

from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, First, RisingEdge, Timer
from pyrx.acquisition import acquire_l1ca
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.samples import SampleFile, read_samples
from pyrx.signals import L1_FREQ
from pyrx.tracking import track_l1ca

from fpgamodel import regmap as R
from fpgamodel.channel import Channel
from fpgamodel.compare import start_phase
from fpgamodel.loop import Epoch, Tracker, compare_with_pyrx
from fpgamodel.quantize import code_bits, pack_words, to_sign_mag
from tb.support import (
    CLK_NS,
    F_IF,
    FS,
    RECORDING,
    SpiMaster,
    assert_dumps_equal,
    captures_dir,
    configure_channel,
    enable_channel,
    load_code,
    read_dump_retrying,
)

DURATION = 3.0


@cocotb.test()
async def closed_loop_rtl(dut):
    path = captures_dir() / RECORDING
    head = read_samples(path, FS, iq=False, duration=0.01)
    results = [acquire_l1ca(head, FS, F_IF, prn) for prn in range(1, 33)]
    best = max(results, key=lambda a: a.cn0 if a.detected else -1.0)
    assert best.detected
    x = read_samples(path, FS, iq=False, duration=DURATION).real
    codes = to_sign_mag(x)
    chip_rate = CA_CHIP_RATE * (1 + best.doppler / L1_FREQ)
    chip_q, frac = start_phase(best.code_offset, chip_rate)
    np.savetxt(Path.cwd() / "samples.hex", pack_words(codes), fmt="%08x")

    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    dut.load.value = 0
    dut.stream_en.value = 0
    dut.n_samples.value = len(codes)
    spi = SpiMaster(dut)
    dut.rst.value = 1
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    dut.load.value = 1
    await ClockCycles(dut.clk, 2)
    dut.load.value = 0

    tracker = Tracker(FS, F_IF, best.doppler, best.cn0)
    await load_code(spi, 0, best.prn)
    await configure_channel(spi, 0, *tracker.steps(), 0, frac, chip_q)
    await enable_channel(spi, 0)
    await spi.write(R.CTRL, R.CTRL_ENABLE)
    dut.stream_en.value = 1

    dumps, schedule = [], []
    while True:
        # wait on the levels, not the edges: playback can end while a dump is being served
        if not int(dut.dump_irq.value) and not int(dut.done.value):
            await First(RisingEdge(dut.dump_irq), RisingEdge(dut.done), Timer(5, unit="ms"))
            assert int(dut.dump_irq.value) or int(dut.done.value), "no dump for 5 ms"
        if not int(dut.dump_irq.value):
            break  # done, and nothing pending
        d = await read_dump_retrying(spi, 0)
        steps = tracker.update(d)
        await spi.write_words(R.ch(0, R.CARR_STEP), list(steps))
        dumps.append(d)
        schedule.append(steps)
        if not tracker.locked:
            break
    assert tracker.locked
    assert len(dumps) >= 2990
    assert await spi.read(R.ch(0, R.DUMP_MISSED)) == 0

    # the same samples and step schedule through the model: bit-exact closed loop
    m = Channel(code_bits(gps_ca(best.prn)))
    m.set_steps(*Tracker(FS, F_IF, best.doppler).steps())
    m.set_phase(0, frac, chip_q)
    m.enabled = True
    expected, history = [], []
    k = 0
    for start in range(0, len(codes), 12_000):
        for dump in m.run(codes[start : start + 12_000]):
            if k < len(schedule):
                m.set_steps(*schedule[k])
                history.append(Epoch(dump.sample, tracker_doppler(schedule[k]), m.dump_code_frac / 2**32))
            expected.append(dump)
            k += 1
    assert_dumps_equal(dumps, expected, ignore_sample=True)

    ref = track_l1ca(SampleFile(path, FS, iq=False), F_IF, best.prn, best.code_offset, best.doppler,
                     best.cn0, duration=DURATION)
    assert ref.locked
    dopp_rms, code_max = compare_with_pyrx(history, ref, FS, t_from=1.0)
    dut._log.info(f"PRN {best.prn}: Doppler RMS {dopp_rms:.3f} Hz, code max {code_max:.4f} chips")
    assert dopp_rms < 0.5  # the same bounds as the model loop test
    assert code_max < 0.002


def tracker_doppler(steps: tuple[int, int]) -> float:
    """Doppler implied by a carrier step word."""
    carr = steps[0]
    f = carr / 2**32 * FS
    return f - F_IF
