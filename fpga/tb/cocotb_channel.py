"""Channel RTL against the bit-true model: every dump must match exactly."""

import cocotb
import numpy as np
from pyrx.acquisition import acquire_l1ca
from pyrx.codes import gps_ca
from pyrx.samples import read_samples

from fpgamodel.channel import Channel, Dump, Measurement, code_step, synthetic_codes
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


def model_integ(n: int, **kwargs) -> Channel:
    """`model` with the integration length written before the enable."""
    ch = model(**kwargs)
    ch.enabled = False
    ch.set_integ(n)
    ch.enabled = True
    return ch


async def setup(dut, prn=PRN, doppler=0.0, spacing_q=1, phase=(0, 0, 0), integ=1) -> ChannelBench:
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(code_bits(gps_ca(prn)))
    bench.dut.spacing_q.value = spacing_q
    bench.set_integ(integ)
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
async def integ_20_dumps_match_model(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 24_000 * 45, 45.0)
    bench = await setup(dut, doppler=doppler, integ=20)
    await run_all(bench, codes)
    expected = model_integ(20, doppler=doppler).run(codes)
    assert [d.epoch for d in expected] == [1, 2]  # epochs count dumps, not code periods
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def integ_0_acts_as_1_and_above_20_as_20(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 24_000 * 23, 45.0)
    bench = await setup(dut, doppler=doppler, integ=0)
    await bench.load_samples(codes)
    bench.stream()
    await bench.pause_at(24_000 * 2 + 100)  # two dumps of one period each
    await bench.set_enable(False)
    bench.set_integ(31)
    await bench.set_enable(True)
    bench.resume()
    await bench.finish()
    m = model(doppler=doppler)
    expected = m.run(codes[: 24_000 * 2 + 100])
    m.enabled = False
    m.set_integ(31)
    m.enabled = True
    expected += m.run(codes[24_000 * 2 + 100 :])
    lengths = np.diff([0] + [d.sample for d in expected])
    assert len(expected) == 3 and lengths[1] < 24_010 and lengths[2] > 20 * 23_990
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def integ_change_applies_at_next_dump(dut):
    # written while enabled, a new length waits for the dump: the one in flight keeps the
    # old length
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 24_000 * 9, 45.0)
    k = 24_000 + 100  # just after the first dump
    bench = await setup(dut, doppler=doppler)
    await bench.load_samples(codes)
    bench.stream()
    await bench.pause_at(k)
    bench.set_integ(3)
    bench.resume()
    await bench.finish()
    m = model(doppler=doppler)
    expected = m.run(codes[:k])
    m.set_integ(3)
    expected += m.run(codes[k:])
    lengths = np.diff([d.sample for d in expected])
    assert len(expected) == 4 and lengths[0] < 24_010 and (lengths[1:] > 3 * 23_990).all()
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def started_flags_the_first_accepted_sample_of_each_enable(dut):
    doppler = 1000.0
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 45.0)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(code_bits(gps_ca(PRN)))
    bench.set_steps(*channel_steps(doppler))
    await bench.load_samples(codes)
    bench.stream()
    assert bench.started() == (0, 0)  # samples pass a disabled channel unflagged
    await bench.pause_at(5000)
    dut.enable.value = 1
    bench.resume()
    await bench.pause_at(30_000)
    # the eight-clock guard, then the first sample the channel takes (as the dumps show in
    # enable_right_after_phase_load)
    assert bench.started() == (1, 5008)
    dut.enable.value = 0
    bench.resume()
    await bench.pause_at(40_000)
    dut.enable.value = 1
    bench.resume()
    await bench.finish()
    assert bench.started() == (2, 40_008)
    m = model(doppler=doppler)
    expected = m.run(codes[5008:30_000])
    m.enabled = False
    m.enabled = True
    expected += m.run(codes[40_008:])
    shifted = [
        Dump(d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql, d.sample + (5008 if d.sample <= 24_992 else 15_016))
        for d in expected
    ]
    assert_dumps_equal(bench.dumps, shifted)


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
    m = Channel(np.zeros(32, dtype=np.uint32), acc_w=20)
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


@cocotb.test()
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
async def recording_200ms_integ_20(dut):
    x = read_samples(captures_dir() / RECORDING, FS, iq=False, duration=0.2).real
    results = [acquire_l1ca(x[: round(FS * 0.01)], FS, F_IF, prn) for prn in range(1, 33)]
    best = max(results, key=lambda a: a.cn0 if a.detected else -1.0)
    assert best.detected
    chip_q, frac = start_phase(best.code_offset, 1.023e6 * (1 + best.doppler / 1575.42e6))
    codes = to_sign_mag(x)
    bench = await setup(dut, prn=best.prn, doppler=best.doppler, phase=(0, frac, chip_q), integ=20)
    await run_all(bench, codes)
    expected = model_integ(20, prn=best.prn, doppler=best.doppler, phase=(0, frac, chip_q)).run(codes)
    assert len(expected) == 10  # the sums span data-bit edges wherever they fall
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
    assert k0 == 5009, k0  # the eight-clock guard, then the first sample it accepts
    expected = model(doppler=doppler, phase=(0, 0, chip_q0)).run(codes[k0:])
    shifted = [Dump(d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql, d.sample + k0) for d in expected]
    assert_dumps_equal(bench.dumps, shifted)


def _latch_of(m: Measurement) -> tuple[int, int, int, int]:
    return (m.carr_phase, m.code_frac, m.chip_q, m.epoch)


@cocotb.test()
async def measurement_latch_at_load_guard_and_wrap(dut):
    # the latch holds the channel's state before the strobed sample: a disabled channel
    # gives its loaded phase (a strobe in the load's own clock still sees the old one), the
    # enable guard gives the same frozen state, the wrap sample the state before the wrap,
    # and the sample after it the wrapped index with the new epoch
    doppler, phase = 1000.0, (0x1234_5678, 0x9ABC_DEF0, 2000)
    phase2 = (0x0F0F_0F0F, 0x1357_9BDF, 2600)  # loaded before enable: the guard strobe's tell
    codes = synthetic_codes(PRN, FS, F_IF, doppler, 0, 72_000, 45.0)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(code_bits(gps_ca(PRN)))
    bench.set_steps(*channel_steps(doppler))
    await bench.load_samples(codes)
    bench.stream()
    m = Channel(code_bits(gps_ca(PRN)))
    m.set_steps(*channel_steps(doppler))

    dut.carr_phase_in.value, dut.code_frac_in.value, dut.chip_q_in.value = phase
    dut.load_phase.value = 1
    await bench.strobe()  # the load lands on this edge: the latch still sees the reset state
    dut.load_phase.value = 0
    assert await bench.latched() == (0, 0, 0, 0)
    await bench.strobe()
    m.set_phase(*phase)
    assert await bench.latched() == _latch_of(m.measurement())
    await bench.load_phase(*phase2)
    m.set_phase(*phase2)

    await bench.pause_at(5000)
    bench.resume()
    await cocotb.triggers.RisingEdge(dut.clk)  # sample 5000 goes in, the channel disabled
    dut.enable.value = 1  # the same timing as enable_right_after_phase_load
    await cocotb.triggers.ClockCycles(dut.clk, 3)  # inside the eight-clock guard
    await bench.strobe()
    assert await bench.latched() == _latch_of(m.measurement())
    await cocotb.triggers.ClockCycles(dut.clk, 10)  # the guard runs out

    m.enabled = True
    probe = Channel(code_bits(gps_ca(PRN)))
    probe.set_steps(*channel_steps(doppler))
    probe.set_phase(*phase2)
    probe.enabled = True
    k0 = 5009  # the first sample the channel accepts (enable_right_after_phase_load)
    wraps = [k0 + d.sample - 1 for d in probe.run(codes[k0:])[:2]]  # the samples that wrap
    for k in (wraps[0], wraps[0] + 1, wraps[1] + 1):
        await bench.strobe_with_sample(k)
        m.run(codes[k0 + m.sample_count : k])  # up to the strobed sample
        assert await bench.latched() == _latch_of(m.measurement()), k
    assert m.epoch == 2
    await bench.finish()


async def _random_code_high_rate(dut, code_len: int) -> None:
    # a random code at a quarter chip per sample less one percent (a quarter-chip boundary on
    # 99 % of the samples) and half-chip spacing
    rng = np.random.default_rng(code_len)
    words = rng.integers(0, 2**32, size=32, dtype=np.uint64)
    codes = rng.integers(0, 4, size=60_000, dtype=np.uint8)
    carr_step, code_step = round(0.3 * 2**32), round(0.99 * 2**32)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(words)
    dut.spacing_q.value = 2
    bench.set_steps(carr_step, code_step)
    await bench.load_phase(0, 0, 0)
    await bench.set_enable(True)
    await run_all(bench, codes)
    m = Channel(words, code_len=code_len)
    m.spacing_q = 2
    m.set_steps(carr_step, code_step)
    m.enabled = True
    expected = m.run(codes)
    assert len(expected) >= 14
    assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def random_code_high_rate_half_chip(dut):
    await _random_code_high_rate(dut, 1023)


@cocotb.test()
async def random_code_high_rate_half_chip_1024(dut):
    # built with CODE_LEN = 1024
    await _random_code_high_rate(dut, 1024)


JUST_UNDER_A_CHIP = 2**34 - 1  # the chip index advances on every sample but one in 2^34
L5_AT_38_4_MSPS = code_step(10.23e6, 38.4e6)  # 0.266 chips per sample
L5_AT_24_MSPS = code_step(10.23e6, 24e6)  # 0.426 chips per sample
CARR_STEP = round(0.3 * 2**32)


def _code_len(dut) -> int:
    return int(dut.CODE_LEN.value)


def _random_code(dut) -> np.ndarray:
    return np.random.default_rng(_code_len(dut)).integers(0, 2**32, size=32, dtype=np.uint64)


def _random_code_model(dut, step: int, spacing_q: int, phase=(0, 0, 0)) -> Channel:
    m = Channel(_random_code(dut), code_len=_code_len(dut))
    m.spacing_q = spacing_q
    m.set_steps(CARR_STEP, step)
    m.set_phase(*phase)
    m.enabled = True
    return m


async def _high_code_rate(dut, step: int, n: int, gaps=(0,)) -> None:
    # a random code and random samples at both tap spacings: the RTL's dumps are the model's
    # (the same in a build with CODE_LEN = 1024)
    codes = np.random.default_rng(step % 1000).integers(0, 4, size=n, dtype=np.uint8)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(_random_code(dut))
    for spacing_q in (1, 2):
        for gap in gaps:
            await bench.restart()
            dut.spacing_q.value = spacing_q
            bench.set_steps(CARR_STEP, step)
            await bench.load_phase(0, 0, 0)
            await bench.set_enable(True)
            await run_all(bench, codes, gap)
            expected = _random_code_model(dut, step, spacing_q).run(codes)
            assert len(expected) >= 3
            assert_dumps_equal(bench.dumps, expected)


@cocotb.test()
async def code_step_just_under_one_chip_per_sample(dut):
    await _high_code_rate(dut, JUST_UNDER_A_CHIP, 6000)


@cocotb.test()
async def l5_ratio_at_38_4_msps(dut):
    await _high_code_rate(dut, L5_AT_38_4_MSPS, 16_000)


@cocotb.test()
async def l5_ratio_at_24_msps(dut):
    await _high_code_rate(dut, L5_AT_24_MSPS, 10_000)


@cocotb.test()
async def l5_ratio_with_valid_gaps(dut):
    await _high_code_rate(dut, L5_AT_24_MSPS, 10_000, gaps=(1, 3))


@cocotb.test()
async def enable_right_after_phase_load_at_high_code_rates(dut):
    # the phase lands two chips before the code's end and the enable follows on the next
    # clock: the first samples after the guard already advance the chip index across the
    # end of the code, and every tap must hold the right chip
    codes = np.random.default_rng(5).integers(0, 4, size=14_000, dtype=np.uint8)
    phase = (0x1234_5678, 0xF000_0000, 4 * (_code_len(dut) - 2) + 3)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(_random_code(dut))
    for step in (JUST_UNDER_A_CHIP, L5_AT_24_MSPS):
        await bench.restart()
        dut.spacing_q.value = 2
        bench.set_steps(CARR_STEP, step)
        await bench.load_samples(codes)
        bench.stream()
        await bench.pause_at(5000)
        dut.carr_phase_in.value, dut.code_frac_in.value, dut.chip_q_in.value = phase
        dut.load_phase.value = 1
        bench.resume()
        await cocotb.triggers.RisingEdge(dut.clk)
        dut.load_phase.value = 0
        dut.enable.value = 1  # no wait for the chips around the new phase
        await bench.finish()
        k0 = bench.started()[1]
        assert k0 == 5009, k0  # the eight-clock guard, then the first sample it accepts
        expected = _random_code_model(dut, step, 2, phase).run(codes[k0:])
        assert len(expected) >= 3
        shifted = [Dump(d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql, d.sample + k0) for d in expected]
        assert_dumps_equal(bench.dumps, shifted)


@cocotb.test()
async def measurement_latch_at_l5_ratio(dut):
    # the latch splits the code phase as the register map does: the quarter-chip index (chip
    # index and the top two fraction bits) and the low 32 fraction bits; strobes on the
    # sample that ends a code period, the one after it and a later one
    codes = np.random.default_rng(6).integers(0, 4, size=16_000, dtype=np.uint8)
    phase = (0x0F0F_0F0F, 0x1357_9BDF, 2601)
    bench = ChannelBench(dut)
    await bench.start()
    await bench.load_code(_random_code(dut))
    dut.spacing_q.value = 1
    bench.set_steps(CARR_STEP, L5_AT_38_4_MSPS)
    await bench.load_phase(*phase)
    await bench.set_enable(True)
    await bench.load_samples(codes)
    bench.stream()
    wrap = _random_code_model(dut, L5_AT_38_4_MSPS, 1, phase).run(codes)[0].sample - 1
    m = _random_code_model(dut, L5_AT_38_4_MSPS, 1, phase)
    for k in (wrap, wrap + 1, wrap + 1001):
        await bench.strobe_with_sample(k)
        m.run(codes[m.sample_count : k])  # up to the strobed sample
        assert await bench.latched() == _latch_of(m.measurement()), k
    assert m.epoch == 1 and m.chip_q != 0
    await bench.finish()


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
        bench.resume()  # sample k goes in on the next edge with enable low
        await cocotb.triggers.ClockCycles(dut.clk, low_clocks)
        dut.enable.value = 1  # exactly low_clocks edges, each with a valid sample
    await bench.finish()
    skipped = [(k, low_clocks + 8) for k, low_clocks in gaps]
    kept = np.concatenate([codes[:20_000], codes[20_000 + skipped[0][1] : 40_000],
                           codes[40_000 + skipped[1][1] :]])
    expected = model(doppler=doppler).run(kept)
    assert_dumps_equal(bench.dumps, expected, ignore_sample=True)
    assert len(bench.dumps) >= 2
