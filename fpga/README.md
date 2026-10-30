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
| 1 | 2,415 | 1,465 | 19 | 141.6 MHz | 248 KB |
| 12 | 16,241 | 10,012 | 30 | 94.7 MHz | 538 KB |

The playback RAM takes 18 EBR of that: a 32-bit word only fits the block's 512 × 36 geometry,
so 9,216 words need 18 blocks (packing 18 samples into 36-bit words would hold 165,888 samples
in the same 18 blocks; 16 blocks hold 131,072 samples as 32-bit words). Each channel adds about 1,260 LUT4 and 780
flip-flops, three times the correlator alone: the per-channel control and measurement
registers (32-bit step, phase and latch words) and the readback mux are most of it. Holding
the phase words only in the channel's NCOs and reading them back from there, and a single
readback register file indexed by channel, are the remaining trims when channel count
matters.
The fabric clock on the EVN is 37.5 MHz; the target design's is 38.4 MHz.

## Running the bench

From the repository root, with the project venv:

```
uv pip install --python .venv/bin/python -e 'fpga/model[dev]'
.venv/bin/pytest fpga
```

Install `pyrx` first (the repository's quick start): `fpgamodel` depends on it, and the name
`pyrx` on PyPI is an unrelated package.

Tests on the recordings need `PYRX_CAPTURES` set to the directory holding them, exactly as the
`pyrx` suite does; without it they skip. Lint: `verilator --lint-only -Wall fpga/rtl/*.v
--top-module rx_spi` is clean; the board top adds the Lattice primitives, see the target README.

The cocotb benches build the RTL with Icarus by default (`SIM=verilator` selects Verilator,
`WAVES=1` records traces) under `fpga/tb/sim_build/`. Each bench streams its samples from a
hex file through a small Verilog wrapper, so simulation time is spent in the simulator, not in
Python: the nine channel cases, including 200 ms of the recording, take about three minutes.

## Sample encoding and channel arithmetic

A sample is two bits, `{sign, mag}`: bit 1 set means negative, bit 0 set means magnitude 3,
clear means magnitude 1, the MAX2771's 2-bit output. Samples count only while valid, so a
channel fed from the playback RAM freezes when the RAM ends.

Per sample, a channel (`rtl/channel.v`, `fpgamodel.channel.Channel`) does the following, all
in integers; the model's `step` method is the definition and the cocotb benches hold the RTL to
it exactly.

1. The top three bits of the 32-bit carrier NCO select one of eight local-oscillator phases.
   The oscillator is 3-level in each component: (cos, sin) runs through (2, 1), (1, 2),
   (−1, 2), (−2, 1), (−2, −1), (−1, −2), (1, −2), (2, −1). The product of a ±1/±3 sample and
   a ±1/±2 oscillator value is one of ±1, ±2, ±3, ±6: a lookup, no multiplier. The in-phase
   mix is sample × cos, the quadrature mix is −sample × sin, the replica being exp(−j·phase)
   as in `pyrx.correlator`.
2. The code NCO counts quarter chips in a 32-bit fraction plus an index 0..4091. The prompt
   chip is the code bit at index ÷ 4; early and late are the chips `SPACING_Q` quarter chips
   ahead and behind (0.25 chip by default, pyrx's spacing; 0.5 chip as the alternative). A set
   code bit is a chip of value −1 and negates the mix.
3. Six accumulators, early/prompt/late × I/Q, 20 bits wide, wrap on overflow (a period of
   38 400 samples at ±6 reaches 230 400, so they never do in use).
4. The two NCOs advance by their step words. When the quarter-chip index wraps past 4091 the
   six sums are dumped into the holding registers, the accumulators restart with the next
   sample, the epoch counter increments, and the step words written since the last dump
   take effect, so a step update never splits an integration.

Step words are `round(f / f_sample × 2^32)`: the carrier's `f` is IF plus Doppler, the code's
is four times the Doppler-scaled chip rate. The sample rate therefore has to exceed four times
the chip rate, 4.092 MHz for L1 C/A. The cost of the 2-bit samples and 3-level oscillator
against a float correlator is measured below.

## Register map

`rx_core` exposes everything on a 16-bit word-address, 32-bit data bus (`fpgamodel.regmap`
holds the same constants). A write lands on the clock where `bus_we` is high; a read returns
its word the clock after `bus_re`. Unmapped addresses read zero and ignore writes.

Global block at `0x0000`:

| Offset | Name | Access | Notes |
|---|---|---|---|
| 0x00 | `ID` | R | `0x50474E53` |
| 0x01 | `VERSION` | R | bits 15:0 = `0x0100`; bit 16 = measurement latch implemented; bit 17 = `CH_START_SAMPLE` implemented |
| 0x02 | `CTRL` | RW | bit0 global enable, bit1 soft reset (self-clearing, resets everything but the RAM contents: playback samples and code chips), bit2 source (0 pins, 1 playback) |
| 0x03 | `STATUS` | R | bit0 pll_lock, bit1 playback running, bit2 playback done |
| 0x04 | `SAMPLE_CNT_LO` | R | valid samples since reset, counted while `CTRL` bit 0 is set; reading it latches `SAMPLE_CNT_HI` |
| 0x05 | `SAMPLE_CNT_HI` | R | bits 15:0, as latched by the last `SAMPLE_CNT_LO` read |
| 0x06 | `DUMP_PENDING` | R | bit k = channel k has an unread dump |
| 0x07 | `DUMP_CLEAR` | W | write bits to clear |
| 0x08 | `MEAS_TRIG` | W | write 1: measurement strobe on the next valid sample |
| 0x09 | `MEAS_PERIOD` | RW | valid samples between periodic strobes, first at sample `MEAS_PERIOD`; 0 = off; counted from reset, so set it before enabling |
| 0x0A | `N_CHANNELS` | R | |
| 0x0B | `PLAYBACK_CTRL` | W | bit0 start, one-shot from sample 0 |
| 0x0C | `PLAYBACK_LEN` | RW | samples; clamped to `PB_DEPTH`, 0 is done at once |
| 0x0D | `PB_DEPTH` | R | playback RAM size in samples |
| 0x100 | `PB_ADDR` | RW | playback word index |
| 0x101 | `PB_DATA` | W | 16 samples, bits [2i+1:2i] = sample 16·addr+i; auto-increments `PB_ADDR`; burst it with a fixed address (command bit 6 = 0) |

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
| 0x09 | `CODE_WR_ADDR` | RW | code word index 0..31, wraps |
| 0x0A | `CODE_WR_DATA` | W | 32 chips, bit i = chip 32·addr+i, set = chip value −1; auto-increments; burst it with a fixed address (command bit 6 = 0) |
| 0x0B | `EPOCH` | R | dumps so far |
| 0x0C | `DUMP_MISSED` | R | dumps that landed while the pending bit was still set (16 bits) |
| 0x10..0x15 | `IE, QE, IP, QP, IL, QL` | R | the last dump, sign-extended 20-bit sums |
| 0x18..0x1D | `MEAS_CODE_PHASE_FRAC, MEAS_CODE_CHIP, MEAS_CARR_PHASE, MEAS_EPOCH, MEAS_SAMPLE_LO, MEAS_SAMPLE_HI` | R | the measurement latch: the channel's code NCO fraction, quarter-chip index, carrier NCO phase and dump count as they stood before the strobed sample, and that sample's count (48 bits, the same in every channel) |

The measurement latch is what pseudoranges are built from: one strobe, by `MEAS_TRIG` or
every `MEAS_PERIOD` samples, captures every channel's state on the same sample, so the code
phases of different channels are comparable to the sample. `fpgamodel.channel.Channel.measurement`
returns the same five numbers for the model.

Two things to know when driving the bus: a channel's enable takes effect eight clocks after it
is written (the channel spends them re-reading the chips around its code phase, so a phase or
code write followed at once by enable is still exact), and the sample rate must exceed four
times the chip rate, 4.092 MHz for L1 C/A, because the code NCO ticks at most once per
sample.

## SPI transaction

`rx_spi` puts `rx_core` behind a single-lane SPI slave (`spi_slave.v`), the bring-up interface
for the evaluation board. Mode 0 (clock idle low, data sampled on the rising edge), MSB first,
CS active low, SCK at most the fabric clock divided by 8 (4.8 MHz at 38.4 MHz; 4.3 MHz on the
EVN, whose fabric clock can be as low as 34.9 MHz).

| Bytes | Content |
|---|---|
| 0 | command: bit 7 = 1 write / 0 read, bit 6 = 1 auto-increment the word address / 0 fixed, bits 5:0 zero |
| 1, 2 | 16-bit word address, high byte first |
| write: 3.. | 32-bit data words, high byte first; each word is committed when its last bit arrives |
| read: 3 | one dummy byte while the slave fetches the first word |
| read: 4.. | 32-bit data words, high byte first |

Words keep coming for as long as CS is held low; the address advances by one per word in
auto-increment mode. Raising CS ends the transaction and drops a partial word.

What the slave needs, in fabric clocks (29 ns at the EVN's slowest 34.9 MHz): SCK high and SCK
low each at least 2; MOSI held at least 2 after the SCK rise that samples it; CS low at least 2
before the first SCK rise and at least 2 after the last one (a CS rise within one clock of the
last SCK rise drops that word); CS high at least 2 between transactions; and one CS assertion
per transaction. SCK at an eighth of the fabric clock with equal high and low times meets all
of these; the limits matter for a master that stretches, truncates or releases its clock
between calls. MISO takes its next bit within a few fabric clocks after each SCK rise, to be
sampled on the following rise, and is driven low while CS is high (it is never high impedance),
so give it its own line rather than sharing it with another slave. A read fetches the next word
while bit 24 of the current one shifts out, so a burst also reads the address after its last
word (the same address in fixed mode); reading `SAMPLE_CNT_LO` latches `SAMPLE_CNT_HI`, so read
the pair as one two-word burst.

The shape
(command, address, dummy cycles, data, burst) is what the STM32 OCTOSPI does in indirect mode,
so a driver written against this interface moves to quad SPI with only the lane count changing.

## Closed-loop tracking

`fpgamodel.loop.Tracker` closes pyrx's tracking loops around a channel's dumps the way the MCU
will: FLL pull-in then a Costas PLL on the carrier, a non-coherent early-minus-late DLL every
10 ms, the same constants as `pyrx.tracking`. pyrx moves its correlation window in time at each
DLL update; a channel can only change its code NCO rate, so the DLL output becomes a chip-rate
offset spread over the next 10 ms, with carrier aiding folded into the code step. Step words
written after dump k take effect at dump k + 1.

On the first 3 s of `L1_20211226_082401_24MHz_I.bin` (PRN 26, 48.7 dB-Hz), over the 2 s after
pull-in, against `pyrx.tracking.track_l1ca` from the same acquisition seed:

| Loop around | Doppler, RMS difference | Code-period start, largest difference |
|---|---|---|
| the model (`fpga/model/tests/test_loop.py`) | 0.37 Hz | 0.0004 chips |
| the RTL over SPI (`fpga/tb/test_loop.py`, `FPGA_SLOW=1 SIM=verilator`, 2 min) | 0.37 Hz | 0.0004 chips |

Bounds: 1 Hz RMS and one quarter chip. The RTL run also replays the same samples and the same
step schedule through the model and requires every dump to be equal: the closed loop is
bit-exact too, which is why the two rows agree to the digit.

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
