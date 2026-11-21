"""Register map of the FPGA receiver core, the bench's and the firmware's view of rx_core.v.

Word addresses. The global block starts at 0, channel k at CH_BASE + k * CH_STRIDE. The
bench checks rx_core.v's offsets against these constants name for name; the model itself
does not use them.
"""

ID_VALUE = 0x50474E53  # "PGNS"
# 0x0100 had readable step and phase words, no MEAS_SEQ, a latched SAMPLE_CNT_HI; 0x0200 an
# oversampled single-lane slave with unsigned command bytes, a quarter-chip code NCO limited to
# a quarter chip per sample, and one code period per dump
VERSION_BASE = 0x0300
VERSION_LATCH = 1 << 16  # measurement latch implemented
VERSION_START_SAMPLE = 1 << 17  # CH_START_SAMPLE implemented
VERSION_QUAD_LINK = 1 << 18  # SCK-clocked SPI slave with a four-lane data phase
VERSION_EVENT_STAMP = 1 << 19  # event-stamp block present
VERSION_ALL = (
    VERSION_BASE | VERSION_LATCH | VERSION_START_SAMPLE | VERSION_QUAD_LINK | VERSION_EVENT_STAMP
)

# global block
ID = 0x00
VERSION = 0x01
CTRL = 0x02  # bit0 enable, bit1 soft reset (keeps RAM contents), bit2 source (0 pins, 1 playback)
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
MEAS_SEQ = 0x0E  # measurement strobes since reset: read before and after the MEAS_* words
LINK_STATUS = 0x0F  # 15:0 frames rejected by signature, 23:16 write overruns, 31:24 read underruns
PB_ADDR = 0x100
PB_DATA = 0x101
EVT_CTRL = 0x200  # bit0 enable, bit1 edge (0 rising, 1 falling)
EVT_SEQ = 0x201  # events stamped since reset: read before and after the stamp words
EVT_SAMPLE_LO = 0x202  # global sample count at the latest event's edge
EVT_SAMPLE_HI = 0x203
EVT_MISSED = 0x204  # events dropped for arriving within 16 samples of the last stamp

CTRL_ENABLE = 1 << 0
CTRL_RESET = 1 << 1
CTRL_SOURCE_PLAYBACK = 1 << 2
STATUS_PLL_LOCK = 1 << 0
STATUS_PB_RUNNING = 1 << 1
STATUS_PB_DONE = 1 << 2
EVT_ENABLE = 1 << 0
EVT_FALLING = 1 << 1

# channel block
CH_BASE = 0x1000
CH_STRIDE = 0x100
CH_CTRL = 0x00  # bit0 enable
CH_START_SAMPLE_LO = 0x01  # global sample count of the first sample accepted after the enable
CH_START_SAMPLE_HI = 0x02
CARR_STEP = 0x03
CODE_STEP = 0x04  # bits 31:0 of the code step, chips per sample x 2^34; commits the step words
CARR_PHASE = 0x05
CODE_PHASE_FRAC = 0x06
CODE_CHIP = 0x07
SPACING_Q = 0x08
CODE_WR_ADDR = 0x09
CODE_WR_DATA = 0x0A
EPOCH = 0x0B
DUMP_MISSED = 0x0C
CH_INTEG = 0x0D  # code periods per dump, 1 to 20
CODE_STEP_HI = 0x0E  # bits 33:32 of the code step, staged until the CODE_STEP write
EPOCH_HEAD = 0x0F  # EPOCH again, right before the sums
IE = 0x10
QE = 0x11
IP = 0x12
QP = 0x13
IL = 0x14
QL = 0x15
EPOCH_TAIL = 0x16  # EPOCH again, right after the sums
DUMP_BURST = EPOCH_HEAD  # an auto-increment read from here brackets the six sums with EPOCH
DUMP_BURST_WORDS = 8
MEAS_CODE_PHASE_FRAC = 0x18
MEAS_CODE_CHIP = 0x19
MEAS_CARR_PHASE = 0x1A
MEAS_EPOCH = 0x1B
MEAS_SAMPLE_LO = 0x1C
MEAS_SAMPLE_HI = 0x1D


def ch(k: int, offset: int) -> int:
    """Address of register `offset` of channel `k`."""
    return CH_BASE + k * CH_STRIDE + offset


# SPI command byte: bit 7 write, bit 6 auto-increment, bit 5 data on four lanes, bits 4:0 a
# fixed signature that a frame shifted by a clock does not carry
CMD_SIGNATURE = 0b10110
CMD_SIGNATURE_MASK = 0x1F
CMD_READ = CMD_SIGNATURE
CMD_WRITE = 0x80 | CMD_SIGNATURE
CMD_INCR = 0x40
CMD_QUAD = 0x20
