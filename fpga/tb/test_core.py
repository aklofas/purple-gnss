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
    "step_pair_commits_on_the_code_step_write",
    "code_chip_out_of_range_is_ignored",
    "code_write_while_enabled_is_ignored",
    "measurement_trigger_captures_all_channels_on_one_sample",
    "measurement_periodic",
    "meas_period_written_mid_run_counts_from_the_write",
    "meas_seq_counts_strobes_and_guards_the_latch_read",
    "code_write_address_wraps_at_32_words",
    "dump_burst_brackets_epoch",
    "integ_register_reads_back_and_clamps",
    "meas_epoch_counts_dumps_at_integ_20",
    "sums_at_integ_20_are_sign_extended_from_25_bits",
    "ch_start_sample_is_the_first_accepted_sample",
    "ch_start_sample_relatches_on_each_enable",
    "code_step_hi_commits_with_the_code_step_write",
    "soft_reset_clears_the_map_0300_registers",
    "event_stamp_latches_sample_count",
    "event_stamp_counts_close_events_as_missed",
    "event_stamp_falling_edge_and_disable",
    "event_stamp_counts_samples_not_clocks",
    "link_status_reads_the_slave_counters",
]


SOURCES = [RTL / "channel.v", RTL / "playback.v", RTL / "event_stamp.v", RTL / "rx_core.v"]


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
