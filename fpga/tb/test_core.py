import pytest

from tb.conftest import RTL, build, run_cocotb

CASES = [
    "id_version_and_constants",
    "unmapped_reads_zero_writes_ignored",
    "soft_reset_clears_state_not_playback_ram",
    "playback_dumps_match_model",
    "playback_length_clamped_and_zero_is_done",
    "playback_end_freezes_channels",
    "pin_source_dumps_match_model",
    "dump_missed_counts_and_values_replace",
    "sample_count_reads_coherently",
    "two_channels_track_two_prns",
    "phase_registers_only_while_disabled",
    "measurement_trigger_captures_all_channels_on_one_sample",
    "measurement_periodic",
    "version_reports_latch",
]


@pytest.fixture(scope="module")
def core_build():
    return build(
        "rx_core",
        [RTL / "channel.v", RTL / "playback.v", RTL / "rx_core.v"],
        {"N_CHANNELS": 2, "PB_DEPTH": 131072},
    )


@pytest.mark.parametrize("case", CASES)
def test_core(core_build, case):
    run_cocotb(core_build, "rx_core", "tb.cocotb_core", case)
