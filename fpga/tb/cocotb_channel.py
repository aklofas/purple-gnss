"""Channel RTL against the bit-true model: every dump must match exactly."""

import cocotb
import numpy as np
from pyrx.acquisition import acquire_l1ca
from pyrx.codes import gps_ca
from pyrx.samples import read_samples

from fpgamodel.channel import Channel, Dump, synthetic_codes
from fpgamodel.compare import start_phase
from fpgamodel.quantize import code_bits, to_sign_mag
from tb.support import (
    F_IF,
    FS,
    RECORDING,
    ChannelBench,
    assert_dumps_equal,
    captures_dir,
    channel_steps,
)

PRN = 7


def model(prn=PRN, doppler=0.0, spacing_q=1, phase=(0, 0, 0)) -> Channel:
    ch = Channel(code_bits(gps_ca(prn)))
    ch.spacing_q = spacing_q
    ch.set_steps(*channel_steps(doppler))
    ch.set_phase(*phase)
    ch.enabled = True
    return ch


async def setup(dut, prn=PRN, doppler=0.0, spacing_q=1, phase=(0, 0, 0)) -> ChannelBench:
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(code_bits(gps_ca(prn)))
    bench.dut.spacing_q.value = spacing_q
    bench.set_steps(*channel_steps(doppler))
    await bench.load_phase(*phase)
    await bench.set_enable(True)
    return bench


async def run_all(bench: ChannelBench, codes, gap=0) -> None:
    await bench.load_samples(codes, gap)
    bench.stream()
    await bench.finish()


@cocotb.test()
async def aligned_clean(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 200.0)
    bench = await setup(dut, doppler=doppler)
    await run_all(bench, codes)
    expected = model(doppler=doppler).run(codes)
    assert len(expected) >= 2 and expected[1].ip > 2 * 24_000
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def doppler_and_noise(dut):
    doppler = -2345.6
    from pyrx.synth import synth_l1ca
    x = synth_l1ca(PRN, FS, F_IF, doppler, 5000, 24_000 * 50, 40.0, iq=False, data_bits=True)
    codes = to_sign_mag(x.real)
    bench = await setup(dut, doppler=doppler)
    await run_all(bench, codes)
    assert_dumps_equal(bench.dumps, model(doppler=doppler).run(codes))


@cocotb.test()
async def valid_gaps_do_not_change_dumps(dut):
    doppler = 750.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 60_000, 45.0)
    bench = await setup(dut, doppler=doppler)
    await run_all(bench, codes, gap=3)
    assert_dumps_equal(bench.dumps, model(doppler=doppler).run(codes))


@cocotb.test()
async def step_update_one_sample_before_dump(dut):
    doppler, new_doppler = 1000.0, 4000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 200.0)
    ref = model(doppler=doppler)
    ref_dumps = ref.run(codes)
    k_wrap = ref_dumps[1].sample - 1  # index of the sample that completes period 2
    bench = await setup(dut, doppler=doppler)
    await bench.load_samples(codes)
    bench.stream()
    await bench.pause_at(k_wrap)
    bench.set_steps(*channel_steps(new_doppler))
    bench.resume()
    await bench.finish()
    m = model(doppler=doppler)
    expected = m.run(codes[:k_wrap])
    m.set_steps(*channel_steps(new_doppler))
    expected += m.run(codes[k_wrap:])
    assert expected[:2] == ref_dumps[:2] and expected[2] != ref_dumps[2]
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def accumulator_wrap(dut):
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(np.zeros(32, dtype=np.uint32))
    code_step = round(4092 / 100_000 * 2**32)
    bench.set_steps(0, code_step)
    await bench.load_phase(0, 0, 0)
    await bench.set_enable(True)
    codes = np.full(120_000, 0b01, dtype=np.uint8)
    await run_all(bench, codes)
    m = Channel(np.zeros(32, dtype=np.uint32))
    m.set_steps(0, code_step)
    m.enabled = True
    expected = m.run(codes)
    assert expected[0].ip == 6 * expected[0].sample - 2**20
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def disable_mid_period_then_resume(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 45.0)
    k, skipped = 30_000, 1000
    bench = await setup(dut, doppler=doppler)
    await bench.load_samples(codes)
    bench.stream()
    await bench.pause_at(k)
    await bench.set_enable(False)
    bench.resume()
    await bench.pause_at(k + skipped)
    await bench.set_enable(True)
    bench.resume()
    await bench.finish()
    m = model(doppler=doppler)
    expected = m.run(codes[:k]) + m.run(codes[k + skipped :])
    for d in bench.dumps:
        assert d.sample < k or d.sample > k + skipped
    assert_dumps_equal(bench.dumps, expected, ignore_sample=True)
    assert [d.sample for d in bench.dumps] == [
        d.sample if d.sample <= k else d.sample + skipped for d in expected
    ]


@cocotb.test()
async def load_phase_ignored_while_enabled(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 48_000, 200.0)
    bench = await setup(dut, doppler=doppler)
    await bench.load_samples(codes)
    bench.stream()
    await bench.pause_at(10_000)
    dut.carr_phase_in.value = 0x12345678
    dut.code_frac_in.value = 0x9ABCDEF0
    dut.chip_q_in.value = 2000
    dut.load_phase.value = 1
    bench.resume()
    await bench.pause_at(10_010)
    dut.load_phase.value = 0
    bench.resume()
    await bench.finish()
    assert_dumps_equal(bench.dumps, model(doppler=doppler).run(codes))


@cocotb.test()
async def spacing_q_half_chip(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 60_000, 200.0)
    bench = await setup(dut, doppler=doppler, spacing_q=2)
    await run_all(bench, codes)
    expected = model(doppler=doppler, spacing_q=2).run(codes)
    assert expected[1].ie < 0.6 * expected[1].ip
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test(skip=captures_dir() is None)
async def recording_200ms(dut):
    x = read_samples(captures_dir() / RECORDING, FS, iq=False, duration=0.2).real
    results = [acquire_l1ca(x[: round(FS * 0.01)], FS, F_IF, prn) for prn in range(1, 33)]
    best = max(results, key=lambda a: a.cn0 if a.detected else -1.0)
    assert best.detected
    chip_q, frac = start_phase(best.code_offset, 1.023e6 * (1 + best.doppler / 1575.42e6))
    codes = to_sign_mag(x)
    bench = await setup(dut, prn=best.prn, doppler=best.doppler, phase=(0, frac, chip_q))
    await run_all(bench, codes)
    expected = model(prn=best.prn, doppler=best.doppler, phase=(0, frac, chip_q)).run(codes)
    assert len(expected) >= 199
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def enable_right_after_phase_load(dut):
    # samples keep streaming while the channel is disabled; a phase load and an enable in
    # consecutive clocks must not start on stale neighbor chips: the channel starts a few
    # clocks later and from then on matches the model run from the first consumed sample
    doppler, chip_q0 = 1000.0, 2000
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 45.0)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(code_bits(gps_ca(PRN)))
    bench.set_steps(*channel_steps(doppler))
    await bench.load_samples(codes)
    bench.stream()
    await bench.pause_at(5000)
    dut.chip_q_in.value = chip_q0
    dut.load_phase.value = 1
    bench.resume()
    await cocotb.triggers.RisingEdge(dut.clk)
    dut.load_phase.value = 0
    dut.enable.value = 1  # no wait for the chip-neighbor refresh
    await bench.finish()
    assert len(bench.dumps) >= 2
    probe = model(doppler=doppler, phase=(0, 0, chip_q0))
    s1 = probe.run(codes)[0].sample  # samples from the start to the first wrap: NCO only
    k0 = bench.dumps[0].sample - s1
    assert 5000 <= k0 <= 5000 + 12, k0  # the channel started within a few clocks of enable
    expected = model(doppler=doppler, phase=(0, 0, chip_q0)).run(codes[k0:])
    shifted = [Dump(d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql, d.sample + k0) for d in expected]
    assert_dumps_equal(bench.dumps, shifted)


@cocotb.test()
async def code_len_1024_dumps(dut):
    # built with CODE_LEN = 1024: the quarter-chip count 4096 needs one more bit than the index
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(np.zeros(32, dtype=np.uint32))
    code_step = round(4096 / 20_000 * 2**32)  # one period per 20 000 samples
    bench.set_steps(0, code_step)
    await bench.load_phase(0, 0, 0)
    await bench.set_enable(True)
    codes = np.full(50_000, 0b01, dtype=np.uint8)
    await run_all(bench, codes)
    m = Channel(np.zeros(32, dtype=np.uint32), code_len=1024)
    m.set_steps(0, code_step)
    m.enabled = True
    expected = m.run(codes)
    assert len(expected) == 2
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def disable_for_one_and_two_clocks(dut):
    # a disable of a clock or two drops that sample and the eight that follow (the enable
    # guard), nothing else: the dumps equal the model with exactly those samples removed
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 45.0)
    bench = await setup(dut, doppler=doppler)
    await bench.load_samples(codes)
    bench.stream()
    gaps = ((20_000, 1), (40_000, 2))
    for k, low_clocks in gaps:
        await bench.pause_at(k)
        dut.enable.value = 0
        bench.resume()
        await bench.pause_at(k + low_clocks)
        dut.enable.value = 1
        bench.resume()
    await bench.finish()
    skipped = [(k, low_clocks + 8) for k, low_clocks in gaps]
    kept = np.concatenate([codes[:20_000], codes[20_000 + skipped[0][1] : 40_000],
                           codes[40_000 + skipped[1][1] :]])
    expected = model(doppler=doppler).run(kept)
    assert_dumps_equal(bench.dumps, expected, ignore_sample=True)
    assert len(bench.dumps) >= 2
