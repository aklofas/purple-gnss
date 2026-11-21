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
| [`targets/evn`](targets/evn/README.md) | Lattice LIFCL-40-EVN | SPI (one or four data lanes) and the event input on the Raspberry Pi header, sample pins on a PMOD header, fabric clock from the internal oscillator, GPLL as a lock check |

Synthesis of the `evn` target with Yosys, nextpnr-nexus and prjoxide (LIFCL-40, 31,680 LUT4,
84 EBR):

| `N_CHANNELS` | LUT4 | FF | EBR | fmax, fabric clock | fmax, SCK | Bitstream (compressed) |
|---|---|---|---|---|---|---|
| 1 | 3,177 | 1,956 | 19 | 142.0 MHz | 239.3 MHz | 264 KB |
| 12 | 17,214 | 12,065 | 30 | 92.7 MHz | 161.4 MHz | 561 KB |

The playback RAM takes 18 EBR of that: a 32-bit word only fits the block's 512 × 36 geometry,
so 9,216 words need 18 blocks (packing 18 samples into 36-bit words would hold 165,888 samples
in the same 18 blocks; 16 blocks hold 131,072 samples as 32-bit words). Each channel adds about 1,280 LUT4 and 920
flip-flops, several times the correlator alone: the per-channel control and measurement
registers (step, phase, latch and first-sample words) and the readback mux of the dump and
latch words are most of it. Holding the phase words only in the channel's NCOs (the stored copies
serve the three-way reload) is the remaining trim when channel count matters.
The fabric clock on the EVN is 37.5 MHz; the target design's is 38.4 MHz. The SPI slave's frame
logic is a second clock domain, clocked by SCK and constrained at its rated 25 MHz.

## Running the bench

From the repository root, in the virtual environment that holds `pyrx`, with Icarus Verilog on
the path (Verilator too for lint and the closed-loop run):

```
uv pip install --python .venv/bin/python -e 'fpga/model[dev]'
.venv/bin/pytest fpga
```

or, with plain pip, `pip install -e 'fpga/model[dev]'` and then `pytest fpga`. Install `pyrx`
first (the repository's quick start): `fpgamodel` depends on it, and the name `pyrx` on PyPI is
an unrelated package.

Tests on the recordings need `PYRX_CAPTURES` set to the directory holding them, exactly as the
`pyrx` suite does; without it they skip (the SPI board sequence still runs on synthetic
samples). With `PYRX_REQUIRE_CAPTURES=1` as well, the tests on the recording fail without
it, so a gate run, `PYRX_CAPTURES=… PYRX_REQUIRE_CAPTURES=1 .venv/bin/pytest fpga`, cannot
pass by skipping them. The closed-loop RTL run is skipped unless `FPGA_SLOW=1` is set as well; run it with
`PYRX_CAPTURES=… FPGA_SLOW=1 SIM=verilator .venv/bin/pytest fpga/tb/test_loop.py` (about 3
minutes). Lint: `verilator --lint-only -Wall fpga/rtl/*.v --top-module rx_spi` is clean; the
board top adds the Lattice primitives, see the target README.

The cocotb benches build the RTL with Icarus by default (`SIM=verilator` selects Verilator,
`WAVES=1` records traces) under `fpga/tb/sim_build/`. The channel and closed-loop benches
stream their samples from a hex file through a small Verilog wrapper, so simulation time is
spent in the simulator, not in Python; the core and SPI benches load the playback RAM through
register writes. The whole `fpga` run takes about twelve minutes; the two channel cases on
200 ms of the recording are 100 s each.

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
2. The code phase is a chip index, 0 to 1022, and a 34-bit fraction of a chip. The prompt
   chip is the code bit at the index; early and late are the chips `SPACING_Q` quarter chips
   ahead and behind (0.25 chip by default, pyrx's spacing; 0.5 chip as the alternative),
   decided on the top two bits of the fraction. A set code bit is a chip of value −1 and
   negates the mix.
3. Six accumulators, early/prompt/late × I/Q, 25 bits wide, wrap on overflow (20 code periods
   of 38 400 samples at ±6 reach 4 608 000, so they never do in use).
4. The two NCOs advance by their step words; a carry out of the code fraction advances the
   chip index. When the index passes the end of the code, a code period ends and the step
   words written since the last one take effect. Every `CH_INTEG` code periods (1 to 20,
   1 by default) the six sums are also dumped into the holding registers, the accumulators
   restart with the next sample and the epoch counter increments, so a dump is 1 to 20 ms of
   L1 C/A and a step update never waits longer than one code period.

The carrier step word is `round(f / f_sample × 2^32)` with `f` the IF plus Doppler
(`fpgamodel.channel.nco_step`). The code step word is `round(chip rate / f_sample × 2^34)`,
chips per sample with 34 fraction bits (`fpgamodel.channel.code_step`), so the chip index
advances at most once per sample and the sample rate only has to exceed the chip rate: L1 C/A
at 38.4 MHz is 0.027 chips per sample, a 10.23 Mchip/s code 0.266. The cost of the 2-bit
samples and 3-level oscillator against a float correlator is measured below.

## Register map

`rx_core` exposes everything on a 16-bit word-address, 32-bit data bus (`fpgamodel.regmap`
holds the same constants). A write lands on the clock where `bus_we` is high; a read returns
its word the clock after `bus_re`. Unmapped addresses read zero and ignore writes. Reads have
no side effects: the SPI slave fetches up to three words past every read, so no register
snapshots, latches or clears on a read; anything that needs a snapshot takes it on a write.

Global block at `0x0000`:

| Offset | Name | Access | Notes |
|---|---|---|---|
| 0x00 | `ID` | R | `0x50474E53` |
| 0x01 | `VERSION` | R | `0x000F0300`: bits 15:0 = `0x0300`, the register map's version (`0x0200` had an oversampled single-lane SPI slave whose command bytes carried no signature, a code NCO limited to a quarter chip per sample, one code period per dump and no `CH_START_SAMPLE`); bit 16 = measurement latch, bit 17 = `CH_START_SAMPLE`, bit 18 = SCK-clocked slave with a four-lane data phase, bit 19 = event stamp |
| 0x02 | `CTRL` | RW | bit0 global enable, bit1 soft reset (self-clearing, resets everything but the RAM contents: playback samples and code chips), bit2 source (0 pins, 1 playback) |
| 0x03 | `STATUS` | R | bit0 pll_lock, bit1 playback running, bit2 playback done |
| 0x04 | `SAMPLE_CNT_LO` | R | valid samples since reset, counted while `CTRL` bit 0 is set; live |
| 0x05 | `SAMPLE_CNT_HI` | R | bits 47:32 of the same count, live: for a coherent 48-bit value use `MEAS_SAMPLE_LO/HI` after a `MEAS_TRIG`, or read `LO`, `HI`, `LO` and retry when `LO` wrapped |
| 0x06 | `DUMP_PENDING` | R | bit k = channel k has an unread dump |
| 0x07 | `DUMP_CLEAR` | W | write bits to clear |
| 0x08 | `MEAS_TRIG` | W | write 1: measurement strobe on the next valid sample |
| 0x09 | `MEAS_PERIOD` | RW | valid samples between periodic strobes, counted from the write: the first strobe comes `MEAS_PERIOD` valid samples after it; 0 = off |
| 0x0A | `N_CHANNELS` | R | |
| 0x0B | `PLAYBACK_CTRL` | W | bit0 start, one-shot from sample 0 |
| 0x0C | `PLAYBACK_LEN` | RW | samples; clamped to `PB_DEPTH`, 0 is done at once |
| 0x0D | `PB_DEPTH` | R | playback RAM size in samples |
| 0x0E | `MEAS_SEQ` | R | measurement strobes since reset; read it before and after a channel's `MEAS_*` words and read again if it changed |
| 0x0F | `LINK_STATUS` | R | the SPI slave's error counters, saturating, zeroed by a soft reset: bits 15:0 frames rejected for a command byte without the signature, bits 23:16 write frames that lost words because the fabric side had not taken the ones before (only a stalled fabric clock does this; from the first lost word on, the rest of the frame is dropped too), bits 31:24 read frames whose data was clocked out before it arrived (such a frame returns zeros from the missing word on) |
| 0x100 | `PB_ADDR` | RW | playback word index; a `PB_DATA` write at or past the end (`PB_DEPTH` / 16 words) is dropped and leaves it there |
| 0x101 | `PB_DATA` | W | 16 samples, bits [2i+1:2i] = sample 16·addr+i; auto-increments `PB_ADDR`; burst it with a fixed address (command bit 6 = 0) |

| 0x200 | `EVT_CTRL` | RW | event stamp: bit0 enable, bit1 edge (0 rising, 1 falling) |
| 0x201 | `EVT_SEQ` | R | events stamped since reset |
| 0x202, 0x203 | `EVT_SAMPLE_LO/HI` | R | the 48-bit sample count at the latest event, the count `MEAS_SAMPLE` uses |
| 0x204 | `EVT_MISSED` | R | events dropped because they came before 16 samples had passed since the last stamp |

Channel k at `0x1000 + k·0x100` (up to 16 channels):

| Offset | Name | Access | Notes |
|---|---|---|---|
| 0x00 | `CH_CTRL` | RW | bit0 enable |
| 0x01, 0x02 | `CH_START_SAMPLE_LO/HI` | R | the 48-bit sample count of the first sample the channel accepted after its latest enable (the n0 of the carrier-phase formula below); 0 until then, and a disabled channel keeps its last value |
| 0x03 | `CARR_STEP` | W | carrier NCO step, cycles per sample × 2^32; staged until `CODE_STEP` is written |
| 0x04 | `CODE_STEP` | W | bits 31:0 of the code NCO step, chips per sample × 2^34; its write commits `CARR_STEP`, `CODE_STEP_HI` and itself together, and they take effect at the end of the code period, or at once while disabled and in the seven clocks after the enable is written |
| 0x05 | `CARR_PHASE` | W | carrier NCO phase; written only while disabled |
| 0x06 | `CODE_PHASE_FRAC` | W | bits 31:0 of the code phase's 34-bit chip fraction; written only while disabled |
| 0x07 | `CODE_CHIP` | W | the code phase in quarter chips, 0..4091: the chip index of the prompt tap times four, plus the top two bits of the chip fraction; written only while disabled; a value of 4092 or more is ignored |
| 0x08 | `SPACING_Q` | RW | early/late offset in quarter chips, 1 or 2 (reset 1; 0 and 3 act as 1) |
| 0x09 | `CODE_WR_ADDR` | RW | code word index 0..31, wraps |
| 0x0A | `CODE_WR_DATA` | W | 32 chips, bit i = chip 32·addr+i, set = chip value −1; auto-increments; burst it with a fixed address (command bit 6 = 0); written only while the channel is disabled |
| 0x0B | `EPOCH` | R | dumps so far, counted when the dump lands two clocks after its last sample |
| 0x0C | `DUMP_MISSED` | R | dumps that landed while the pending bit was still set (16 bits) |
| 0x0D | `CH_INTEG` | RW | code periods per dump, 1 to 20 (reset 1; a write of 0 stores 1, of more than 20 stores 20); takes effect at the next dump, or at once while disabled |
| 0x0E | `CODE_STEP_HI` | W | bits 33:32 of the code NCO step (reset 0): 0 for any code below a quarter chip per sample, L1 C/A included; 1 for a 10.23 Mchip/s code at 38.4 MHz. Staged like `CARR_STEP` until `CODE_STEP` is written |
| 0x0F | `EPOCH` again | R | the same word as 0x0B |
| 0x10..0x15 | `IE, QE, IP, QP, IL, QL` | R | the last dump, sign-extended 25-bit sums |
| 0x16 | `EPOCH` again | R | the same word as 0x0B: an auto-increment read of eight words from 0x0F is `EPOCH`, the six sums, `EPOCH` |
| 0x18..0x1D | `MEAS_CODE_PHASE_FRAC, MEAS_CODE_CHIP, MEAS_CARR_PHASE, MEAS_EPOCH, MEAS_SAMPLE_LO, MEAS_SAMPLE_HI` | R | the measurement latch: the channel's code phase (low 32 fraction bits, and the quarter-chip index as in `CODE_CHIP`), carrier NCO phase and dump count as they stood before the strobed sample (the count includes a dump completed the sample before, which `EPOCH` shows two clocks later), and that sample's count (48 bits, the same in every channel) |

A write to any of `CARR_PHASE`, `CODE_PHASE_FRAC` or `CODE_CHIP` on a disabled channel loads all
three into the NCOs, so write all three. The code phase in chips is `CODE_CHIP` / 4 +
`CODE_PHASE_FRAC` / 2^34; `fpgamodel.compare.start_phase` gives the two words for a code
offset. The step and phase words are write-only (they read
zero): the live phase is read through the measurement latch, and reading written values back
would cost a third of each channel's readback mux. A phase load leaves the six running sums alone, so on a channel
that has run before, the first dump after it carries the earlier partial period: discard that
dump.

A dump that lands while the previous one is being read replaces the holding registers at once,
so a read can mix two dumps, and a `DUMP_CLEAR` after it would acknowledge the new dump
unread. `EPOCH` changes on the same clock as the sums, so read the eight words from 0x0F in
one auto-increment burst: `EPOCH`, the six sums, and `EPOCH` again; if the two differ, discard
what was read and read again without clearing (the new dump is pending, and `DUMP_MISSED`
counted the one it replaced). Otherwise write `DUMP_CLEAR`
and read `EPOCH` a third time; if that changed, a dump landed around the clear and sits in the
registers whether or not the clear caught it, so keep the dump just read and read again at once,
whatever `DUMP_PENDING` says (the bench's `read_dump` and `read_dumps_now` are the reference).

The measurement latch is what pseudoranges are built from: one strobe, by `MEAS_TRIG` or
every `MEAS_PERIOD` samples, captures every channel's state on the same sample, so the code
phases of different channels are comparable to the sample. A periodic strobe can land while
the six words are being read, so read `MEAS_SEQ` before and after them and read again when it
changed; for pseudoranges, which pair several channels on one strobe, put the one check around
the whole set of channel reads. A disabled channel, or one still in the eight clocks after its enable, latches its
frozen state on every strobe. The carrier NCO advances only on the samples a channel accepts,
so the latched `MEAS_CARR_PHASE` is the phase loaded at enable plus f_IF × (accepted samples)
/ f_sample plus the Doppler integral, all modulo one cycle. The Doppler integral an observable
needs is that value less the loaded phase less f_IF × (`MEAS_SAMPLE` − n0) / f_sample, modulo
one cycle, with n0 the channel's first accepted sample, which `CH_START_SAMPLE` reports (0 for a
channel enabled before the first valid sample). Referenced that way, every channel shares
pyrx's origin, the IF phase at sample 0, with no per-channel start phase to load. `fpgamodel.channel.Channel.measurement`
returns the first four of those numbers for the model; its `sample`, like `Dump.sample`, is the
channel's own count of the samples it processed while enabled, which equals `MEAS_SAMPLE` only
for a channel whose enable was in effect before the first valid sample and never dropped since.

The event stamp ties an outside edge to the sample count: an IMU's data-ready line, a pulse
per second, on the bench a timer output of the MCU. The `evt_in` pin passes a two-flop
synchronizer, and the chosen edge latches the count of the sample then going in, two clocks
after the pin changed; `EVT_SEQ` counts the stamps. Read `EVT_SEQ`, the two sample words and
`EVT_SEQ` again, and read again when it changed. An edge that comes before 16 samples have
passed since the last stamp is not stamped but counted in `EVT_MISSED`; the spacing is counted
in samples, so with no samples flowing only the first edge is stamped.

Three things to know when driving the bus. A channel's enable takes effect eight clocks after
it is written (the channel spends them re-reading the chips around its code phase, so a phase
or code write followed at once by enable is still exact). The sample rate must exceed the
chip rate, because the chip index advances at most once per sample. And playback runs from the clock `PLAYBACK_CTRL` is written, whatever the global enable, the
source and the channels are doing, so write it last: code and samples, steps and phases,
`CH_CTRL`, `CTRL` (enable and the playback source), then `PLAYBACK_CTRL`, at least eight clocks
after the last `CH_CTRL` write on the direct bus (any SPI transaction is far longer). A channel
enabled or a source selected after the start misses samples and nothing flags it, and
`PLAYBACK_CTRL` written while a run is in progress is unsupported.

## SPI transaction

`rx_spi` puts `rx_core` behind an SPI slave (`spi_slave.v`) with four data lanes. Mode 0 (clock
idle low, data sampled on the rising edge), MSB first, CS active low. A single-lane master uses
lane 0 as MOSI and lane 1 as MISO; the data phase alone can use all four lanes.

| Clocks | Content |
|---|---|
| 8 | command byte, on lane 0 |
| 16 | word address, high byte first, on lane 0 |
| read: 8 | dummy clocks while the slave fetches the first word; nobody drives a lane |
| 32 or 8 per word | 32-bit data words, high byte first: on one lane (lane 0 in for a write, lane 1 out for a read), or on four lanes with the high nibble of each byte first and lane 3 its most significant bit |

| Command bit | Meaning |
|---|---|
| 7 | 1 write, 0 read |
| 6 | 1 auto-increment the word address per word, 0 fixed |
| 5 | 1 data on four lanes, 0 on one |
| 4:0 | the signature `10110` |

So the commands are 0x16 and 0x56 (read, fixed and auto-increment), 0x36 and 0x76 the same on
four lanes, 0x96 and 0xD6 the writes, 0xB6 and 0xF6 the four-lane writes
(`fpgamodel.regmap.CMD_*`). A frame whose command byte lacks the signature is ignored up to
its CS rise and counted in `LINK_STATUS`; a frame shifted by a stray clock fails the signature
31 times in 32. Words keep coming for as long as CS is held low, the address advancing by one
per word in auto-increment mode. Raising CS ends the frame and drops a partial word.

The slave drives lane 1, or all four in a four-lane read, only while it sends read data; at
every other time, and whenever CS is high, all four lanes are high impedance, so a top level
or a board pulls them to a level. Its outputs change on the falling edge of SCK.

The frame logic is clocked by SCK itself and crosses to the fabric clock through two small
FIFOs, so SCK is not tied to the fabric clock by a ratio. What bounds it is the first word of
a read: it has the eight dummy clocks to arrive, of which the way there and back takes up to
five fabric clocks and two SCK clocks, so six SCK periods must exceed five fabric clock
periods. The rated SCK is 25 MHz, well inside that at any fabric clock a board has (34.9 MHz at
the slowest on the EVN); in simulation the first word is in time at every phase for SCK up to
1.1 times the fabric clock, and at 40 MHz against 34.9 MHz every word of a long burst was
right. A read that outruns its data returns zeros from the missing word on and is counted in
`LINK_STATUS`. Later words have a whole word time to arrive. Writes are taken at one word per
fabric clock, far faster than a link can send them; if the fabric clock stalls and a word finds
no room, that word and the rest of its frame are dropped and the frame is counted. CS must
stay high for at least three fabric clocks between frames: after a shorter gap a read can
return words fetched for the frame before it, with nothing counted. A read fetches ahead of
what it sends, up to three words past its last one (the same address again in fixed mode),
which is why no register reacts to a read.

The shape (command, address, dummy cycles, data, burst) is the STM32 OCTOSPI's indirect mode:
instruction 8 bits on one line, address 16 bits on one line, 8 dummy cycles on reads and 0 on
writes, data on one line or four, DLR = 4n − 1 for n words, CKMODE 0, DLYBYP 1, DEVSIZE large
enough never to bind. Words go high byte first, so a little-endian MCU swaps the bytes of each
word.

## Closed-loop tracking

`fpgamodel.loop.Tracker` closes pyrx's tracking loops around a channel's dumps the way the MCU
will: FLL pull-in then a third-order Costas PLL on the carrier (its integrator held while the prompt
power is down), a non-coherent early-minus-late DLL every
10 ms, the constants of `pyrx.tracking`; the Costas loop, its hold and the DLL discriminator
are pyrx's own functions, called by both. Fed pyrx's own correlator sums, it
reproduces pyrx's Doppler increments exactly from the FLL-to-PLL handover on
(`fpga/model/tests/test_loop.py`); the pull-ins differ, the tracker keeping the 1 ms two-stage
FLL (5 Hz, then 2 Hz) where pyrx pulls in on 2 ms then 5 ms coherent sums. Closed around a channel it
differs from pyrx in three further deliberate ways: pyrx moves its correlation window in time at each DLL
update, while a channel can only change its code NCO rate, so the DLL output becomes a
chip-rate offset spread over the next 10 ms, with carrier aiding folded into the code step;
step words written after dump k take effect at dump k + 1, one epoch later than pyrx applies
its update; and C/N0 comes from the second and fourth moments of the prompt magnitude rather
than from pyrx's noise tap, which the channel does not have (data bits and the residual
carrier of pull-in leave the moments alone). The loss rule is pyrx's on that C/N0: two
consecutive 1 s windows bad by C/N0 or, after the pull-in window, by the phase-lock indicator
(the same sums of the prompt I/Q); it is the firmware's rule. A channel started mid-period, as `fpgamodel.compare.start_phase` places it
from an acquisition, dumps a partial period first; the tracker's `skip_first` drops that dump,
and the firmware should do the same. pyrx's fade slip flag on an L1 phase reads the loop's
integrator hold epoch by epoch (`TrackResult.held`); `Tracker.fading` is that state at the
current dump, and `track_model` does not keep it in its history, so a receiver that wants
the flag records it per dump.

On the first 3 s of `L1_20211226_082401_24MHz_I.bin` (PRN 26, 48.7 dB-Hz), against
`pyrx.tracking.track_l1ca` from the same acquisition seed, pairing each dump with the pyrx
epoch its update governs, from 2 s on: the pull-ins differ by design, the FLL hands over to the
PLL at 1 s, and the two loops enter that handover from different carrier phases (pyrx starts
its NCO at the IF phase of its first window, the channel's has run from sample 0), so the first
second after it is a transient that says nothing about tracking.

| Loop around | Doppler compared with pyrx's | Doppler, RMS difference | Code-period start, largest difference |
|---|---|---|---|
| the model (`fpga/model/tests/test_loop.py`) | the loop's own value, `Tracker.doppler` | 0.072 Hz | 0.0005 chips |
| the RTL over SPI (`fpga/tb/test_loop.py`, `FPGA_SLOW=1 SIM=verilator`, 2 min) | the value the carrier step word written after each dump implies (one step is 0.0056 Hz at 24 Msps) | 0.072 Hz | 0.0005 chips |

Both rows were measured with the integrator hold in the loop. Bounds: 0.1 Hz RMS and 0.001
chips, set from measured failures (a PLL gain of twice or half
the value reaches 0.20 or 0.10 Hz, a doubled FLL gain 0.10 Hz, a doubled DLL gain 0.0015
chips, a dead DLL 0.0057 chips). The RTL run also replays the same samples and the same step
schedule through the model and requires every dump to be equal: the closed loop is bit-exact
too, and the step word's quantization is far below the difference, which is why the two rows
agree to the digit.

## Quantization loss

The channel works on 2-bit samples with a 3-level local oscillator. Its cost against
`pyrx`'s float correlator, measured by `fpgamodel.compare` as the ratio of post-correlation
SNRs on the same samples (prompt power on the tracked PRN less the noise power, over the noise
power: the mean prompt power on eight PRNs absent from the signal; 100 windows of one code
period, fixed NCO; pyrx's windows are sample-aligned 1 ms blocks and the model's are its dumps,
the first of them partial for a nonzero code offset, a misalignment measured at 0.03 dB or less):

| Signal | What the number measures | Loss |
|---|---|---|
| Synthetic L1 C/A at 45 dB-Hz (`pyrx.synth`), float samples for `pyrx`, 2-bit for the model | 2-bit quantization plus the 3-level LO | 0.89 dB |
| Recording `L1_20211226_082401_24MHz_I.bin`, PRN 26 at 48.7 dB-Hz | The 3-level LO alone: the recording already holds 2-bit samples (values ±1 and ±3), so `pyrx` sees the same quantization | 0.27 dB |

The bench bounds are 1.5 dB total on synthetic signal and 0.8 dB on the recording
(`fpga/model/tests/test_compare.py`).
