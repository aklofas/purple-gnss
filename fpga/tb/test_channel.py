import pytest

from tb.conftest import RTL, TB, build, needs_captures, run_cocotb

CASES = [
    "aligned_clean",
    "doppler_and_noise",
    "valid_gaps_do_not_change_dumps",
    "step_update_one_sample_before_dump",
    "integ_20_dumps_match_model",
    "integ_0_acts_as_1_and_above_20_as_20",
    "integ_change_applies_at_next_dump",
    "started_flags_the_first_accepted_sample_of_each_enable",
    "disable_mid_period_then_resume",
    "load_phase_ignored_while_enabled",
    "spacing_q_half_chip",
    pytest.param("recording_200ms", marks=needs_captures),
    pytest.param("recording_200ms_integ_20", marks=needs_captures),
    "enable_right_after_phase_load",
    "disable_for_one_and_two_clocks",
    "measurement_latch_at_load_guard_and_wrap",
    "random_code_high_rate_half_chip",
]


@pytest.fixture(scope="module")
def channel_build():
    return build("tb_channel", [RTL / "channel.v", TB / "tb_channel.v"])


@pytest.mark.parametrize("case", CASES)
def test_channel(channel_build, case):
    run_cocotb(channel_build, "tb_channel", "tb.cocotb_channel", case)


def test_channel_20_bit_accumulators_wrap():
    runner = build("tb_channel", [RTL / "channel.v", TB / "tb_channel.v"], {"ACC_W": 20})
    run_cocotb(runner, "tb_channel", "tb.cocotb_channel", "accumulator_wrap")


@pytest.mark.parametrize("case", ["code_len_1024_dumps", "random_code_high_rate_half_chip_1024"])
def test_channel_code_len_1024(case):
    runner = build("tb_channel", [RTL / "channel.v", TB / "tb_channel.v"], {"CODE_LEN": 1024})
    run_cocotb(runner, "tb_channel", "tb.cocotb_channel", case)
