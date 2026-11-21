import pytest

from tb.conftest import RTL, build, needs_captures, run_cocotb

CASES = [
    "read_id_and_version",
    "lanes_are_high_impedance_when_idle",
    "write_then_read_register",
    "burst_auto_increment_write_and_read",
    "fixed_address_mode",
    "aborted_word_is_not_written",
    "quad_read_and_write_bursts",
    "single_and_quad_frames_interleave",
    "bad_signature_frame_is_ignored_and_counted",
    "sck_at_1_to_4_and_1_to_1_of_the_fabric_clock",
    "sck_at_the_rated_rate_against_the_slow_fabric",
    "first_read_word_is_in_time_at_any_phase_up_to_the_fabric_rate",
    "sck_headroom_at_40_mhz",
    "cs_rise_mid_word_drops_it_and_clears",
    "write_overrun_is_counted_not_corrupted",
    "read_underrun_is_counted_and_realigns",
    "read_cut_short_after_the_header_leaves_the_next_read_clean",
    "three_fabric_clocks_of_cs_high_separate_frames",
    "reads_fetch_ahead_without_side_effects",
    "soft_reset_clears_link_status",
    "aborted_header_and_read_leave_no_trace",
    "cs_glitch_restarts_the_framing",
    "sample_count_words_are_live",
    "board_sequence_over_spi",
    "board_sequence_over_spi_quad",
    pytest.param("board_sequence_over_spi_recording", marks=needs_captures),
]


SOURCES = ("channel.v", "playback.v", "event_stamp.v", "rx_core.v", "cdc_fifo.v", "spi_slave.v", "rx_spi.v")


@pytest.fixture(scope="module")
def spi_build():
    return build(
        "rx_spi",
        [RTL / name for name in SOURCES],
        {"N_CHANNELS": 1},
    )


@pytest.mark.parametrize("case", CASES)
def test_spi(spi_build, case):
    run_cocotb(spi_build, "rx_spi", "tb.cocotb_spi", case)
