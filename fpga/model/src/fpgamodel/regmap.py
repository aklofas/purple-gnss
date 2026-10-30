"""Register map of the FPGA receiver core, shared by the model, the bench and the firmware.

Word addresses. The global block starts at 0, channel k at CH_BASE + k * CH_STRIDE.
"""

ID_VALUE = 0x50474E53  # "PGNS"
VERSION_BASE = 0x0100
VERSION_LATCH = 1 << 16  # measurement latch implemented
VERSION_START_SAMPLE = 1 << 17  # CH_START_SAMPLE implemented

# global block
ID = 0x00
VERSION = 0x01
CTRL = 0x02  # bit0 enable, bit1 soft reset, bit2 source (0 pins, 1 playback)
STATUS = 0x03  # bit0 pll_lock, bit1 playback running, bit2 playback done
SAMPLE_CNT_LO = 0x04
SAMPLE_CNT_HI = 0x05
DUMP_PENDING = 0x06
DUMP_CLEAR = 0x07
MEAS_TRIG = 0x08
MEAS_PERIOD = 0x09
N_CHANNELS = 0x0A
PLAYBACK_CTRL = 0x0B
PLAYBACK_LEN = 0x0C
PB_DEPTH = 0x0D
PB_ADDR = 0x100
PB_DATA = 0x101

CTRL_ENABLE = 1 << 0
CTRL_RESET = 1 << 1
CTRL_SOURCE_PLAYBACK = 1 << 2
STATUS_PLL_LOCK = 1 << 0
STATUS_PB_RUNNING = 1 << 1
STATUS_PB_DONE = 1 << 2

# channel block
CH_BASE = 0x1000
CH_STRIDE = 0x100
CH_CTRL = 0x00  # bit0 enable
CH_START_SAMPLE_LO = 0x01
CH_START_SAMPLE_HI = 0x02
CARR_STEP = 0x03
CODE_STEP = 0x04
CARR_PHASE = 0x05
CODE_PHASE_FRAC = 0x06
CODE_CHIP = 0x07
SPACING_Q = 0x08
CODE_WR_ADDR = 0x09
CODE_WR_DATA = 0x0A
EPOCH = 0x0B
DUMP_MISSED = 0x0C
IE = 0x10
QE = 0x11
IP = 0x12
QP = 0x13
IL = 0x14
QL = 0x15
MEAS_CODE_PHASE_FRAC = 0x18
MEAS_CODE_CHIP = 0x19
MEAS_CARR_PHASE = 0x1A
MEAS_EPOCH = 0x1B
MEAS_SAMPLE_LO = 0x1C
MEAS_SAMPLE_HI = 0x1D


def ch(k: int, offset: int) -> int:
    """Address of register `offset` of channel `k`."""
    return CH_BASE + k * CH_STRIDE + offset
