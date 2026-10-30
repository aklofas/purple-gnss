import pytest

from tb.conftest import RTL, build, needs_captures, run_cocotb

CASES = [
    "read_id_and_version",
    "write_then_read_register",
    "burst_auto_increment_write_and_read",
    "fixed_address_mode",
    "aborted_word_is_not_written",
    "sck_at_fabric_over_8",
    "cs_hold_boundary",
    "aborted_header_and_read_leave_no_trace",
    "cs_glitch_restarts_the_framing",
    "sample_count_words_are_live",
    "board_sequence_over_spi",
    pytest.param("board_sequence_over_spi_recording", marks=needs_captures),
]


@pytest.fixture(scope="module")
def spi_build():
    return build(
        "rx_spi",
        [RTL / "channel.v", RTL / "playback.v", RTL / "rx_core.v", RTL / "spi_slave.v", RTL / "rx_spi.v"],
        {"N_CHANNELS": 1},
    )


@pytest.mark.parametrize("case", CASES)
def test_spi(spi_build, case):
    run_cocotb(spi_build, "rx_spi", "tb.cocotb_spi", case)
