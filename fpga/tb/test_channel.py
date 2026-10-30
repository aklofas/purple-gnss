import pytest

from tb.conftest import RTL, TB, build, run_cocotb

CASES = [
    "aligned_clean",
    "doppler_and_noise",
    "valid_gaps_do_not_change_dumps",
    "step_update_one_sample_before_dump",
    "accumulator_wrap",
    "disable_mid_period_then_resume",
    "load_phase_ignored_while_enabled",
    "spacing_q_half_chip",
    "recording_200ms",
    "enable_right_after_phase_load",
]


@pytest.fixture(scope="module")
def channel_build():
    return build("tb_channel", [RTL / "channel.v", TB / "tb_channel.v"])


@pytest.mark.parametrize("case", CASES)
def test_channel(channel_build, case):
    run_cocotb(channel_build, "tb_channel", "tb.cocotb_channel", case)


def test_channel_code_len_1024():
    runner = build("tb_channel", [RTL / "channel.v", TB / "tb_channel.v"], {"CODE_LEN": 1024, "CODE_AW": 10})
    run_cocotb(runner, "tb_channel", "tb.cocotb_channel", "code_len_1024_dumps")
