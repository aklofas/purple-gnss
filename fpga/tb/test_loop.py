import os

import pytest

from tb.conftest import RTL, TB, build, run_cocotb


@pytest.mark.skipif(not os.environ.get("FPGA_SLOW"), reason="set FPGA_SLOW=1 for the closed-loop RTL run")
def test_closed_loop_rtl():
    runner = build(
        "tb_rx_spi",
        [RTL / "channel.v", RTL / "playback.v", RTL / "rx_core.v", RTL / "spi_slave.v", RTL / "rx_spi.v",
         TB / "tb_rx_spi.v"],
        {"N_CHANNELS": 1},
    )
    run_cocotb(runner, "tb_rx_spi", "tb.cocotb_loop", "closed_loop_rtl")
