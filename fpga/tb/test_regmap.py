"""rx_core.v's register offsets against fpgamodel.regmap, name for name and value for value."""

import re

from fpgamodel import regmap as R
from tb.conftest import RTL

GLOBAL = [
    "ID", "VERSION", "CTRL", "STATUS", "SAMPLE_CNT_LO", "SAMPLE_CNT_HI", "DUMP_PENDING",
    "DUMP_CLEAR", "MEAS_TRIG", "MEAS_PERIOD", "N_CHANNELS", "PLAYBACK_CTRL", "PLAYBACK_LEN",
    "PB_DEPTH", "PB_ADDR", "PB_DATA",
]  # fmt: skip
CHANNEL = [
    "CH_CTRL", "CH_START_SAMPLE_LO", "CH_START_SAMPLE_HI", "CARR_STEP", "CODE_STEP", "CARR_PHASE",
    "CODE_PHASE_FRAC", "CODE_CHIP", "SPACING_Q", "CODE_WR_ADDR", "CODE_WR_DATA", "EPOCH",
    "DUMP_MISSED", "IE", "QE", "IP", "QP", "IL", "QL", "MEAS_CODE_PHASE_FRAC", "MEAS_CODE_CHIP",
    "MEAS_CARR_PHASE", "MEAS_EPOCH", "MEAS_SAMPLE_LO", "MEAS_SAMPLE_HI",
]  # fmt: skip

SOURCE = (RTL / "rx_core.v").read_text()


def _localparams(prefix: str, width: int) -> dict[str, int]:
    """{name: value} of every `<prefix>_<name> = <width>'h<hex>` localparam in rx_core.v."""
    pattern = rf"\b{prefix}_(\w+)\s*=\s*{width}'h([0-9A-Fa-f]+)"
    return {name: int(value, 16) for name, value in re.findall(pattern, SOURCE)}


def test_global_offsets_match_regmap():
    assert _localparams("A", 12) == {name: getattr(R, name) for name in GLOBAL}


def test_channel_offsets_match_regmap():
    assert _localparams("C", 8) == {name: getattr(R, name) for name in CHANNEL}


def test_id_and_version_match_regmap():
    assert int(re.search(r"ID_VALUE\s*=\s*32'h([0-9A-Fa-f]+)", SOURCE)[1], 16) == R.ID_VALUE
    version = int(re.search(r"VERSION\s*=\s*32'h([0-9A-Fa-f]+)", SOURCE)[1], 16)
    assert version == R.VERSION_BASE | R.VERSION_LATCH
