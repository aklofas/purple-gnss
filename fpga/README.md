# FPGA

The hardware receiver's FPGA side: shared Verilog cores, a bit-true Python model of them, a
cocotb bench that holds the two to exact agreement, and one top level per board.

```
fpga/
  rtl/        shared Verilog modules
  model/      fpgamodel, the bit-true Python model (installable package)
  tb/         cocotb benches, run as pytest tests
  targets/    one directory per board: top level, pin file, build script
```

## Boards

| Target | Board | Notes |
|---|---|---|
| [`targets/evn`](targets/evn/README.md) | Lattice LIFCL-40-EVN | SPI and sample pins on the PMOD headers, fabric clock from the internal oscillator, GPLL as a lock check |

Synthesis of the `evn` target with Yosys, nextpnr-nexus and prjoxide (LIFCL-40, 31,680 LUT4,
84 EBR):

| `N_CHANNELS` | LUT4 | FF | EBR | fmax | Bitstream (compressed) |
|---|---|---|---|---|---|
| 1 | 2,209 | 1,281 | 19 | 138.9 MHz | 245 KB |
| 12 | 15,905 | 9,124 | 30 | 105.2 MHz | 516 KB |

The playback RAM takes 16 EBR of that. Each channel adds about 1,250 LUT4 and 700
flip-flops, three times the correlator alone: the per-channel control registers (32-bit step,
phase, code-address and miss-counter words) and the readback mux are most of it, and are the
first thing to trim when channel count matters.

## Running the bench

From the repository root, with the project venv:

```
uv pip install --python .venv/bin/python -e fpga/model
.venv/bin/pytest fpga
```

Tests on the recordings need `PYRX_CAPTURES` set to the directory holding them, exactly as the
`pyrx` suite does; without it they skip. Lint: `verilator --lint-only -Wall fpga/rtl/*.v
--top-module rx_spi` is clean; the board top adds the Lattice primitives, see the target README.

The cocotb benches build the RTL with Icarus by default (`SIM=verilator` selects Verilator,
`WAVES=1` records traces) under `fpga/tb/sim_build/`. Each bench streams its samples from a
hex file through a small Verilog wrapper, so simulation time is spent in the simulator, not in
Python: the nine channel cases, including 200 ms of the recording, take about three minutes.

## Register map

`rx_core` exposes everything on a 16-bit word-address, 32-bit data bus (`fpgamodel.regmap`
holds the same constants). A write lands on the clock where `bus_we` is high; a read returns
its word the clock after `bus_re`. Unmapped addresses read zero and ignore writes.

Global block at `0x0000`:

| Offset | Name | Access | Notes |
|---|---|---|---|
| 0x00 | `ID` | R | `0x50474E53` |
| 0x01 | `VERSION` | R | bits 15:0 = `0x0100`; bit 16 = measurement latch implemented; bit 17 = `CH_START_SAMPLE` implemented |
| 0x02 | `CTRL` | RW | bit0 global enable, bit1 soft reset (self-clearing, resets everything but the playback RAM), bit2 source (0 pins, 1 playback) |
| 0x03 | `STATUS` | R | bit0 pll_lock, bit1 playback running, bit2 playback done |
| 0x04 | `SAMPLE_CNT_LO` | R | valid samples since enable; reading it latches `SAMPLE_CNT_HI` |
| 0x05 | `SAMPLE_CNT_HI` | R | bits 15:0, as latched by the last `SAMPLE_CNT_LO` read |
| 0x06 | `DUMP_PENDING` | R | bit k = channel k has an unread dump |
| 0x07 | `DUMP_CLEAR` | W | write bits to clear |
| 0x08 | `MEAS_TRIG` | W | write 1: strobe on the next valid sample (not built yet) |
| 0x09 | `MEAS_PERIOD` | RW | samples between periodic strobes, 0 = off (not built yet) |
| 0x0A | `N_CHANNELS` | R | |
| 0x0B | `PLAYBACK_CTRL` | W | bit0 start, one-shot from sample 0 |
| 0x0C | `PLAYBACK_LEN` | RW | samples; clamped to `PB_DEPTH`, 0 is done at once |
| 0x0D | `PB_DEPTH` | R | playback RAM size in samples |
| 0x100 | `PB_ADDR` | RW | playback word index |
| 0x101 | `PB_DATA` | W | 16 samples, bits [2i+1:2i] = sample 16·addr+i; auto-increments `PB_ADDR` |

Channel k at `0x1000 + k·0x100` (up to 16 channels):

| Offset | Name | Access | Notes |
|---|---|---|---|
| 0x00 | `CH_CTRL` | RW | bit0 enable |
| 0x01, 0x02 | `CH_START_SAMPLE_LO/HI` | RW | placeholder, reads 0 |
| 0x03 | `CARR_STEP` | RW | carrier NCO step, cycles per sample × 2^32; takes effect at the next dump, at once while disabled |
| 0x04 | `CODE_STEP` | RW | code NCO step, quarter chips per sample × 2^32; same timing |
| 0x05 | `CARR_PHASE` | RW | carrier NCO phase; written only while disabled |
| 0x06 | `CODE_PHASE_FRAC` | RW | code NCO fraction; written only while disabled |
| 0x07 | `CODE_CHIP` | RW | quarter-chip index 0..4091 of the prompt tap; written only while disabled |
| 0x08 | `SPACING_Q` | RW | early/late offset in quarter chips, 1 or 2 (reset 1; 0 and 3 act as 1) |
| 0x09 | `CODE_WR_ADDR` | RW | code word index 0..31 |
| 0x0A | `CODE_WR_DATA` | W | 32 chips, bit i = chip 32·addr+i, set = chip value −1; auto-increments |
| 0x0B | `EPOCH` | R | dumps so far |
| 0x0C | `DUMP_MISSED` | R | dumps that landed while the pending bit was still set |
| 0x10..0x15 | `IE, QE, IP, QP, IL, QL` | R | the last dump, sign-extended 20-bit sums |
| 0x18..0x1D | `MEAS_CODE_PHASE_FRAC, MEAS_CODE_CHIP, MEAS_CARR_PHASE, MEAS_EPOCH, MEAS_SAMPLE_LO, MEAS_SAMPLE_HI` | R | zero until the measurement latch is built |

Two timing rules for whoever drives the bus: enable a channel at least eight clocks after its
last code or phase write (the channel refreshes the chips around its code phase while
disabled; any SPI transaction is longer than that), and the sample rate must exceed four
times the chip rate, 4.092 MHz for L1 C/A, because the code NCO ticks at most once per
sample.

## SPI transaction

`rx_spi` puts `rx_core` behind a single-lane SPI slave (`spi_slave.v`), the bring-up interface
for the evaluation board. Mode 0 (clock idle low, data sampled on the rising edge), MSB first,
CS active low, SCK at most the fabric clock divided by 8 (4.8 MHz at 38.4 MHz).

| Bytes | Content |
|---|---|
| 0 | command: bit 7 = 1 write / 0 read, bit 6 = 1 auto-increment the word address / 0 fixed, bits 5:0 zero |
| 1, 2 | 16-bit word address, high byte first |
| write: 3.. | 32-bit data words, high byte first; each word is committed when its last bit arrives |
| read: 3 | one dummy byte while the slave fetches the first word |
| read: 4.. | 32-bit data words, high byte first |

Words keep coming for as long as CS is held low; the address advances by one per word in
auto-increment mode. Raising CS ends the transaction and drops a partial word. The shape
(command, address, dummy cycles, data, burst) is what the STM32 OCTOSPI does in indirect mode,
so a driver written against this interface moves to quad SPI with only the lane count changing.

## Quantization loss

The channel works on 2-bit samples with a 3-level local oscillator. Its cost against
`pyrx`'s float correlator, measured by `fpgamodel.compare` as the ratio of post-correlation
SNRs on the same samples (prompt power on the tracked PRN over the mean prompt power on
eight PRNs absent from the signal, 100 one-period windows, fixed NCO):

| Signal | What the number measures | Loss |
|---|---|---|
| Synthetic L1 C/A at 45 dB-Hz (`pyrx.synth`), float samples for `pyrx`, 2-bit for the model | 2-bit quantization plus the 3-level LO | 0.86 dB |
| Recording `L1_20211226_082401_24MHz_I.bin`, PRN 26 at 48.7 dB-Hz | The 3-level LO alone: the recording already holds 2-bit samples (values ±1 and ±3), so `pyrx` sees the same quantization | 0.27 dB |

The bench bounds are 1.5 dB total on synthetic signal and 0.8 dB on the recording
(`fpga/model/tests/test_compare.py`).
