import numpy as np
import pytest
from pyrx.codes import CA_CHIP_RATE, gps_ca
from pyrx.signals import L1_FREQ

from fpgamodel.channel import LO_TABLE, Channel, Dump, Measurement, nco_step, synthetic_codes
from fpgamodel.quantize import code_bits

FS, F_IF = 24e6, 6.0e6
# at exactly fs/4 with no Doppler the carrier is sampled at four phases only and the
# 3-level LO's quadrature error never averages out, so the default stimulus has Doppler
DOPPLER = 1000.0
PHASE_ALIGNED = 0  # the synthetic carrier is a cosine with phase 0 at sample 0


def _steps(doppler):
    return nco_step(F_IF + doppler, FS), nco_step(4 * CA_CHIP_RATE * (1 + doppler / L1_FREQ), FS)


def _channel(prn=7, doppler=DOPPLER, spacing_q=1):
    ch = Channel(code_bits(gps_ca(prn)))
    ch.spacing_q = spacing_q
    ch.set_steps(*_steps(doppler))
    ch.set_phase(PHASE_ALIGNED, 0, 0)
    ch.enabled = True
    return ch


def _aligned(prn=7, doppler=DOPPLER, n=24_000 * 3, cn0=200.0):  # cn0 200: effectively no noise
    codes = synthetic_codes(prn, FS, F_IF, doppler, 0, n, cn0)
    return _channel(prn, doppler), codes


def test_nco_step():
    assert nco_step(FS / 4, FS) == 1 << 30
    assert nco_step(-FS / 4, FS) == 3 << 30
    assert nco_step(6.0e6, 24e6) == 1 << 30


def test_nco_step_rejects_a_rate_the_nco_cannot_tick():
    # the code NCO ticks at most once per sample: the sample rate must exceed 4 x chip rate
    with pytest.raises(ValueError):
        nco_step(4 * CA_CHIP_RATE, 4e6)
    with pytest.raises(ValueError):
        nco_step(-4e6, 4e6)


def test_lo_table_is_a_3_level_circle():
    assert len(LO_TABLE) == 8
    assert LO_TABLE[0] == (2, 1) and LO_TABLE[2] == (-1, 2) and LO_TABLE[6] == (1, -2)
    assert all(c * c + s * s == 5 for c, s in LO_TABLE)


def test_aligned_prompt_peaks_and_early_late_balance():
    ch, codes = _aligned()
    dumps = ch.run(codes)
    assert [d.epoch for d in dumps] == list(range(1, len(dumps) + 1))
    assert len(dumps) in (2, 3)  # 3 ms; the third dump lands on sample 72000 or just after
    d = dumps[1]
    assert d.ip > 2 * 24_000  # hard-limited cosine x 3-level LO: about 3 per sample
    assert abs(d.ie - d.il) < 0.05 * d.ip  # symmetric taps
    assert 0.65 * d.ip < d.ie < 0.85 * d.ip  # 0.25 chip off: about 0.75
    assert abs(d.qp) < 0.1 * d.ip
    assert 23_999 <= d.sample - dumps[0].sample <= 24_001


def test_prompt_angle_is_minus_the_carrier_phase():
    # the replica is exp(-j phase), as in pyrx.correlator: advancing the NCO by an eighth
    # of a cycle turns the prompt by -45 degrees
    _, codes = _aligned()
    for eighths in (1, -1, 2):
        ch = Channel(code_bits(gps_ca(7)))
        ch.set_steps(*_steps(DOPPLER))
        ch.set_phase(eighths * (1 << 29), 0, 0)
        ch.enabled = True
        d = ch.run(codes)[1]
        angle = np.degrees(np.arctan2(d.qp, d.ip))
        assert abs(angle + 45.0 * eighths) < 2.0


def test_wrong_code_correlates_to_nothing():
    ch, codes = _aligned(prn=7)
    other = _channel(prn=9)
    d = other.run(codes)[1]
    assert abs(d.ip) < 0.05 * ch.run(codes)[1].ip


def test_run_matches_step_sample_by_sample():
    ch_a, codes = _aligned(doppler=1234.5, n=50_000, cn0=40.0)
    ch_b = _channel(doppler=1234.5)
    assert ch_a.run(codes) == [d for c in codes if (d := ch_b.step(int(c))) is not None]
    assert (ch_a.carr_acc, ch_a.code_acc, ch_a.chip_q) == (ch_b.carr_acc, ch_b.code_acc, ch_b.chip_q)
    assert ch_a.sample_count == ch_b.sample_count == 50_000


def test_run_in_blocks_matches_run_whole():
    ch_a, codes = _aligned(doppler=-800.0, n=50_000, cn0=45.0)
    whole = ch_a.run(codes)
    first_dump = whole[0].sample
    for split in (first_dump - 1, first_dump, first_dump + 1, 7):
        ch_b = _channel(doppler=-800.0)
        assert ch_b.run(codes[:split]) + ch_b.run(codes[split:]) == whole


def test_shadow_steps_apply_at_dump():
    _, codes = _aligned(n=72_000)
    steady = _channel()
    ref = steady.run(codes)
    changed = _channel()
    early = changed.run(codes[:30_000])
    changed.set_steps(*_steps(3000.0))  # mid period 2: applies at dump 2, used from period 3
    late = changed.run(codes[30_000:])
    assert early + late[:1] == ref[:2]
    at_dump = _channel()  # written one sample before the wrap: still applied at dump 2
    exact = at_dump.run(codes[: ref[1].sample - 1])
    at_dump.set_steps(*_steps(3000.0))
    assert exact + at_dump.run(codes[ref[1].sample - 1 :]) == early + late
    assert (early + late)[2] != ref[2]


def test_accumulator_wraps_mod_2_20():
    ch = Channel(np.zeros(32, dtype=np.uint32))  # all chips +1
    ch.set_steps(0, nco_step(4092 / 100_000 * FS, FS))  # one period per 100 000 samples
    ch.enabled = True
    dumps = ch.run(np.full(120_000, 0b01, dtype=np.uint8))  # constant +3, LO cos = 2: +6 each
    assert len(dumps) == 1
    d = dumps[0]
    assert d.sample > 87_382  # 6 * 87_382 overflows 20 bits
    assert d.ip == 6 * d.sample - 2**20
    assert d.ie == d.il == d.ip
    assert d.qp == (-3 * d.sample + 2**19) % 2**20 - 2**19  # sin = 1: -3 each, wrapped


def test_disabled_channel_ignores_samples():
    ch, codes = _aligned()
    ch.enabled = False
    before = ch.measurement()
    assert ch.run(codes) == []
    assert ch.step(1) is None
    assert ch.measurement() == before


def test_set_phase_requires_disabled():
    ch, _ = _aligned()
    with pytest.raises(ValueError):
        ch.set_phase(0, 0, 0)


def test_steps_written_while_disabled_take_effect_at_once():
    ch, codes = _aligned()
    ch.enabled = False
    ch.set_steps(*_steps(500.0))
    ch.enabled = True
    ref = _channel(doppler=500.0)
    assert ch.run(codes[:30_000]) == ref.run(codes[:30_000])


def test_spacing_q_0_and_3_behave_as_1():
    _, codes = _aligned()
    ref = _channel(spacing_q=1).run(codes)
    assert _channel(spacing_q=0).run(codes) == ref
    assert _channel(spacing_q=3).run(codes) == ref
    assert _channel(spacing_q=2).run(codes) != ref


def test_half_chip_spacing_lowers_early_and_late():
    _, codes = _aligned()
    quarter = _channel(spacing_q=1).run(codes)[1]
    half = _channel(spacing_q=2).run(codes)[1]
    assert half.ip == quarter.ip
    assert 0.4 * half.ip < half.ie < 0.6 * half.ip


def test_measurement_is_state_before_next_sample():
    ch_a, codes = _aligned(doppler=250.0, n=30_000)
    ch_a.run(codes[:12_345])
    ch_b = _channel(doppler=250.0)
    for c in codes[:12_345]:
        ch_b.step(int(c))
    m = ch_a.measurement()
    assert m == ch_b.measurement()
    assert m == Measurement(ch_b.carr_acc, ch_b.code_acc, ch_b.chip_q, 0, 12_345)
    ch_a.run(codes[12_345:])
    assert ch_a.measurement().epoch == 1


def test_dump_fields():
    d = Dump(1, 2, 3, 4, 5, 6, 7, 8)
    assert (d.epoch, d.ie, d.qe, d.ip, d.qp, d.il, d.ql, d.sample) == (1, 2, 3, 4, 5, 6, 7, 8)


def test_dump_code_frac_is_the_fraction_right_after_the_wrap():
    ch_a, codes = _aligned(doppler=333.0, n=30_000)
    ch_a.run(codes)
    ch_b = _channel(doppler=333.0)
    frac = None
    for c in codes:
        if ch_b.step(int(c)) is not None:
            frac = ch_b.code_acc
    assert frac is not None
    assert ch_a.dump_code_frac == frac == ch_b.dump_code_frac
    assert ch_a.dump_code_frac != ch_a.code_acc  # the block ran on past the dump


def test_run_can_stop_at_the_first_dump():
    ch_a, codes = _aligned(n=72_000)
    ref = _channel().run(codes)
    first = ch_a.run(codes, max_dumps=1)
    assert first == ref[:1]
    assert ch_a.sample_count == ref[0].sample
    rest = ch_a.run(codes[ch_a.sample_count :])
    assert first + rest == ref


def test_disabling_promotes_steps_written_while_enabled():
    # the RTL copies the step inputs into the active NCO steps while disabled; steps written
    # while enabled and then a disable before the next dump must do the same in the model
    _, codes = _aligned(n=72_000)
    ch = _channel()
    ch.run(codes[:10_000])
    ch.set_steps(*_steps(3000.0))  # pending until the next dump
    ch.enabled = False
    ch.enabled = True
    ref = _channel()
    ref.run(codes[:10_000])
    ref.enabled = False
    ref.set_steps(*_steps(3000.0))  # written while disabled: active at once
    ref.enabled = True
    assert ch.run(codes[10_000:]) == ref.run(codes[10_000:])


def test_set_phase_rejects_a_quarter_chip_index_past_the_code():
    ch = Channel(code_bits(gps_ca(7)))
    ch.set_phase(0, 0, 4091)
    with pytest.raises(ValueError):
        ch.set_phase(0, 0, 4092)
    with pytest.raises(ValueError):
        ch.set_phase(0, 0, -1)


def _state(ch: Channel) -> tuple:
    return (
        ch.carr_acc, ch.code_acc, ch.chip_q, ch.epoch, ch.sample_count, ch.carr_step,
        ch.code_step, ch.dump_code_frac, ch.measurement(), tuple(ch._acc), ch._shadow,
    )  # fmt: skip


@pytest.mark.parametrize("seed", range(8))
def test_run_matches_step_under_random_schedules(seed):
    # `run` must equal `step` sample for sample whatever the code length, spacing, phase,
    # step words (including a code step of 0), block boundaries, mid-run step writes,
    # disable/phase/enable sequences and `max_dumps` stops: dumps and the whole state
    rng = np.random.default_rng(seed)
    code_len = int(rng.choice([1023, 1024, 37]))
    words = rng.integers(0, 2**32, size=32, dtype=np.uint64)
    codes = rng.integers(0, 4, size=20_000, dtype=np.uint8)
    n_q = 4 * code_len

    def random_steps():
        code_step = 0 if rng.random() < 0.1 else int(rng.integers(1, 2**32))
        return int(rng.integers(0, 2**32)), code_step

    def random_phase():
        return int(rng.integers(0, 2**32)), int(rng.integers(0, 2**32)), int(rng.integers(0, n_q))

    a, b = Channel(words, code_len=code_len), Channel(words, code_len=code_len)
    spacing, steps, phase = int(rng.integers(0, 4)), random_steps(), random_phase()
    for ch in (a, b):
        ch.spacing_q = spacing
        ch.set_steps(*steps)
        ch.set_phase(*phase)
        ch.enabled = True
    i = 0
    while i < len(codes):
        block = codes[i : i + int(rng.integers(1, 3000))]
        max_dumps = int(rng.integers(1, 3)) if rng.random() < 0.3 else None
        got = a.run(block, max_dumps=max_dumps)
        consumed = a.sample_count - b.sample_count
        expected = [d for c in block[:consumed] if (d := b.step(int(c))) is not None]
        assert got == expected
        assert _state(a) == _state(b)
        i += consumed
        action = rng.integers(0, 4)
        if action == 0:
            steps = random_steps()
            a.set_steps(*steps)
            b.set_steps(*steps)
        elif action == 1:
            phase = random_phase()
            for ch in (a, b):
                ch.enabled = False
                ch.set_phase(*phase)
                ch.enabled = True
        assert _state(a) == _state(b)
    assert a.sample_count == len(codes)
