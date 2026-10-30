import pytest

from tb.conftest import RTL, build, run_cocotb

CASES = [
    "id_version_and_constants",
    "unmapped_reads_zero_writes_ignored",
    "soft_reset_mid_run_keeps_only_the_rams",
    "playback_dumps_match_model",
    "playback_length_clamped_and_zero_is_done",
    "playback_write_past_the_end_is_dropped",
    "playback_end_freezes_channels",
    "pin_source_dumps_match_model",
    "dump_missed_counts_and_values_replace",
    "sample_count_stops_with_global_enable",
    "two_channels_track_two_prns",
    "phase_registers_only_while_disabled",
    "code_chip_out_of_range_is_ignored",
    "code_write_while_enabled_is_ignored",
    "measurement_trigger_captures_all_channels_on_one_sample",
    "measurement_periodic",
    "meas_period_written_mid_run_counts_from_the_write",
    "code_write_address_wraps_at_32_words",
]


SOURCES = [RTL / "channel.v", RTL / "playback.v", RTL / "rx_core.v"]


@pytest.fixture(scope="module")
def core_build():
    return build("rx_core", SOURCES, {"N_CHANNELS": 2})  # the board's playback RAM depth


@pytest.mark.parametrize("case", CASES)
def test_core(core_build, case):
    run_cocotb(core_build, "rx_core", "tb.cocotb_core", case)


def test_core_12_channels():
    runner = build("rx_core", SOURCES, {"N_CHANNELS": 12})
    run_cocotb(runner, "rx_core", "tb.cocotb_core", "last_channel_tracks_like_the_first")


def test_core_16_channels_elaborates():
    build("rx_core", SOURCES, {"N_CHANNELS": 16})  # the map's ceiling: four channel-index bits
