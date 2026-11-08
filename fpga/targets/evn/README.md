# LIFCL-40-EVN target

`top.v` wraps `rx_spi` for the Lattice CrossLink-NX evaluation board (FPGA-EB-02028): SPI and
the dump interrupt on the Raspberry Pi header JP5, two sample pins on PMOD header J17, five
LEDs, SW2 as reset.

## Build and load

```
./build.sh            # one channel: build/n1/top.bit, report.txt
./build.sh 12         # twelve channels: build/n12/
openFPGALoader -b crosslinknx_evn build/n1/top.bit
```

Yosys `synth_nexus`, nextpnr-nexus for `LIFCL-40-9BG400C` with `evn.pdc`, `prjoxide pack
--compress`. The script clears the output directory first and stops on any tool failure.
nextpnr can abort while writing the PLL parameters and leave a truncated FASM that prjoxide
packs without complaint, so the script also requires nextpnr's closing log line and the FASM's
PLL and oscillator sections; and because nextpnr drops a port constraint whose selector matches
nothing without a warning, it counts the pull-ups and pull-downs in the FASM against the ones
`evn.pdc` asks for. `report.py` prints the utilization and the fabric clock's fmax; `report.txt`
also records the source commit.

Lint, with `lint_stubs.v` standing in for the two Lattice primitives and allowing their many
unused ports and the registers that start from an initial value at configuration:

```
verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
    -Wno-PROCASSINIT --top-module top ../../rtl/*.v top.v lint_stubs.v
```

## Clocking

The board's only fabric clock input is the 12 MHz oscillator on L13, below the GPLL's 18 MHz
minimum reference (data sheet fIN), and the 125 and 200 MHz oscillators reach only the SERDES
reference pins. The fabric therefore runs from the device's internal oscillator, 450 MHz / 12
= 37.5 MHz nominal; the oscillator is specified at 418.5 to 481.5 MHz, so the fabric runs
between 34.9 and 40.1 MHz (the receiver does not depend on the rate: NCO step words absorb it
and SPI is asynchronous). A GPLL referenced from the same oscillator runs
alongside, 37.5 MHz × 3 = 112.5 MHz on CLKOP with a 1125 MHz VCO, only to check that a
bitstream from the open toolchain configures a PLL that locks. For a precise clock later,
J17 pins 1 and 2 (D10, D9) are the GPLL0 dedicated inputs and are left free.

### PLL settings

With the open toolchain's defaults for the `PLL` primitive the GPLL did not lock on the board:
LED0 stayed off and LED4 dark, so CLKOP was not running at all. The settings below were found
one bitstream at a time (locked: LED0 on and LED4 blinking at 1.6 to 1.8 Hz, 34.9 to 40.1 MHz
× 3 / 2^26; LED1 blinks either way, since the fabric runs from the oscillator):

| Setting | Default | `top.v` | Why |
|---|---|---|---|
| `PLLPD_N` | `UNUSED` | `USED` | the PLL power bit (TN-02095 `lmmi_pllpd_n`, "0: PLL is not used"); with the default the PLL stays powered down and CLKOP never runs, whatever the other settings. Adding this alone to the as-built configuration made the PLL lock |
| `FBK_MASK` | 0b00001000 | 0b00000000 | the feedback divider's minimum-ratio mask (TN-02095 Table C.3); it binds only for ratios below 8, and with the default the feedback divides by 8 instead of `FBK_MMD_DIG` = 3. With `PLLPD_N` alone the PLL locked with LED4 visibly faster than with the mask cleared; the rate was not measured, and a ratio of 8 would put CLKOP at 300 MHz and the VCO at 3 GHz, above its 800 to 1600 MHz range, so what the PLL settled on is not known. `FBK_MMD_PULS_CTL` 0b0001 is the pulse width the table gives for ratios above 2 |
| `V2I_KVCO_SEL`, `KP_VCO`, `V2I_1V_EN` | 85, 0b11001, off | 60, 0b00011, on | the values LiteX's `NXPLL` (`litex/soc/cores/clock/lattice_nx.py`) uses for a 1.0 V core, which it says are the defaults in Lattice's reference designs though not the primitive's; the EVN's core runs at 1.0 V. TN-02095's own register table gives other `lmmi_v2i_kvco_sel` defaults, 4'b1001 for 0.9 V and 4'b0100 for 1 V, 55 and 30 by its "10 + Dec × 5" formula; `KP_VCO` and `V2I_1V_EN` agree with it. Not needed for lock: the variant with `PLLPD_N` alone kept the primitive's 0.9 V defaults and locked too; the set is kept because the EVN's core runs at 1.0 V, and whether it changes anything measurable (jitter, lock range) is untested |

`PLLRESET_ENA` is on as well, so the power-on reset `top.v` drives reaches the PLL. What was
shown is that the toolchain's default leaves the PLL powered down, so any LIFCL design built with
it needs `PLLPD_N` set; `FBK_MASK` matters only when the feedback ratio is below 8. `SEL_FBK` is not in
prjoxide's database and is dropped from the bitstream without a warning; the feedback path is
set by `CLKMUX_FB` alone. The loop-filter constants (`CSET`, `CRIPPLE`, the charge-pump
currents) are the toolchain defaults, not values computed for this frequency plan as Radiant
and LiteX's `NXPLL` do; the PLL locks with them.

## Wiring to the Nucleo

JP5 is the Raspberry Pi header (bank 6; 3.3 V with J33 on 1-2, the default; EB-02028's Table 3.1
calls that jumper J42, a typo: J42 is the core-current header across R162). The board ships with
JP5 unpopulated (its 2 × 20 receptacle is "DNL" in the bill of materials), so solder a header on
first. The pins are numbered as on the Pi: pin 1 is the 3.3 V supply and pin 2 the 5 V supply
(Table 8.3), odd pins in one row and even in the other, so a meter on those two pins of the powered board tells the
orientation before anything is wired. The SPI lines sit
on the Pi's SPI0 positions. The board test drives them from the NUCLEO-H723ZG's OCTOSPI1 in
single-line indirect mode, the peripheral the later quad-SPI link uses, so the register driver
written for the board test carries over. Every solder bridge on the Nucleo stays at its default
(SB59 and SB63 route NCS and CLK to CN10).

| JP5 pin | Ball | Signal | Direction | Nucleo CN10 pin | STM32 pin | OCTOSPI1 signal |
|---|---|---|---|---|---|---|
| 23 | N7 | SPI SCK (pulled down) | in | 15 | PB2 | CLK |
| 19 | N4 | SPI MOSI (pulled down) | in | 23 | PD11 | IO0 |
| 21 | K6 | SPI MISO | out | 21 | PD12 | IO1 |
| 24 | P6 | SPI CS (active low, pulled up) | in | 13 | PG6 | NCS |
| 22 | K5 | DUMP_IRQ (high while any dump is unread) | out | any free GPIO | | |
| 25 | | GND | | 17 | GND | |

Join the grounds; do not join the two boards' 3.3 V supplies. The fallback, if the OCTOSPI does
not come up, is SPI1 on the CN7 Arduino header (SCK D13 / PA5, MISO D12 / PA6, MOSI D11 / PB5
or PA7 by SB33/SB35, CS D10 / PD14 as a GPIO). Its HAL driver releases SCK and MOSI between
calls and can shorten the last clock pulse when it disables the peripheral (ES0491 2.21.5), so
with it send each transaction as one call and check on a logic analyzer that the last SCK high
time and the CS hold after it are both at least two fabric clocks (57 ns at 34.9 MHz).

J17 (PMOD0) pins 3 and 4 (D7, D8) are the live sample input `{sign, mag}`; nothing drives them
on the EVN, so they are pulled down, and `pin_valid` is tied high, so select the playback source
before enabling.

| LED | Ball | Meaning (lit = active) |
|---|---|---|
| LED0 | E17 | PLL lock |
| LED1 | F13 | fabric heartbeat, about 1.1 Hz |
| LED2 | G13 | a dump is pending (held about 56 ms) |
| LED3 | F14 | SPI chip select active (held about 56 ms) |
| LED4 | L16 | PLL heartbeat, about 1.7 Hz |

SW2 (G14) resets everything but the RAM contents (playback samples and code chips).

## First test on the board

1. Load `build/n1/top.bit` over the board's FTDI JTAG. LED1 blinking shows the fabric running;
   LED0 on with LED4 blinking shows the PLL locked. If LED0 stays off, check that the PLL
   settings above reached the bitstream (`grep IP_PLL_ build/n1/top.fasm` shows `PLLPD_N` and
   `FBK_MASK`) and whether LED4 runs at all, and note what was seen.
2. Configure OCTOSPI1 for indirect mode on one line (IMODE, ADMODE and DMODE = 1 line):
   instruction 8 bits, address 16 bits, 8 dummy cycles on reads and none on writes, DLR =
   4n − 1 for n words, CKMODE 0 (clock idle low), SSHIFT 0, DLYBYP 1 (no delay block), DEVSIZE
   31 (the address is a word index, so the OCTOSPI's byte-based size check must never bind),
   and a prescaler that puts CLK just under 4.3 MHz, an eighth of the slowest fabric clock
   (kernel clock / (PRESCALER + 1); 275 MHz / 64 = 4.3 MHz). Do not go much slower: a dump's
   service in step 4 is nine reads and a write, about 630 CLK cycles plus the host's turnaround,
   which is 150 µs of the 0.64 ms between dumps at 4.3 MHz and no longer fits below about 1 MHz
   (`DUMP_MISSED` then counts up). The OCTOSPI keeps CLK low between
   commands and holds NCS a full CLK cycle after the last edge, which meets the slave's timing
   ([SPI transaction](../../README.md#spi-transaction)) at any such prescaler. Words go out
   high byte first, so byte-swap each 32-bit word of a little-endian buffer. Read `ID`
   (0x0000) and expect `0x50474E53`, `VERSION` (0x0001) = `0x00010200` (register map 0x0200,
   measurement latch present).
3. On the host, acquire the first 3 ms of the recording (`pyrx.acquisition.acquire_l1ca`,
   PRN 1 to 32, keep the strongest) and convert the samples with
   `fpgamodel.quantize.to_sign_mag`. With chip rate = 1.023 MHz × (1 + Doppler / 1575.42 MHz),
   `fpgamodel.compare.start_phase(code_offset, chip rate)` gives `CODE_CHIP` and
   `CODE_PHASE_FRAC`; `fpgamodel.channel.nco_step(6 MHz + Doppler, 24 MHz)` gives `CARR_STEP`
   and `nco_step(4 × chip rate, 24 MHz)` gives `CODE_STEP`. Use 24 MHz in `nco_step` even
   though playback runs at the EVN fabric clock: the step words are per sample. Then, over
   SPI and in this order: `CODE_WR_ADDR` (0x1009) = 0 and the PRN's 32 code words
   (`fpgamodel.quantize.code_bits(pyrx.codes.gps_ca(prn))`, also what `Channel` takes) to
   `CODE_WR_DATA` (0x100A) as two bursts of 16, reading `CODE_WR_ADDR` between them (expect
   16); `PB_ADDR` (0x0100) = 0 and the 4,500 sample words (`fpgamodel.quantize.pack_words`) to
   `PB_DATA` (0x0101), then read `PB_ADDR` (expect 4500); both data bursts use a fixed address
   (command bit 6 = 0), and the read-backs show the link delivered every word. Then
   `CARR_STEP`, `CODE_STEP`, `CARR_PHASE` = 0, `CODE_PHASE_FRAC`, `CODE_CHIP` (0x1003 to
   0x1007); `CH_CTRL` (0x1000) = 1; `CTRL` = 5 (enable, playback source); `PLAYBACK_LEN` =
   72000 (samples); and `PLAYBACK_CTRL` = 1 last, since playback runs from that write.
4. Poll `DUMP_PENDING` or watch DUMP_IRQ; read `EPOCH` (0x100B), the signed 32-bit sums
   `IE..QL` (0x1010 to 0x1015) and `EPOCH` again: if it changed, a dump landed during the
   read, so discard what was read and read again without clearing. Otherwise write
   `DUMP_CLEAR` = 1 and read `EPOCH` a third time: if that changed, keep the dump and read
   again at once. Playback runs one sample per fabric clock, so a dump arrives
   about every 0.64 ms. When `STATUS` bit 2 (done) is set and nothing
   is pending, `DUMP_MISSED` (0x100C) must read 0 and `SAMPLE_CNT_LO` 72000 (a new run does
   not clear the count: soft-reset first if anything ran before, a previous playback run or
   the pins source).
5. Run `fpgamodel.channel.Channel` on the same code words and 2-bit samples with the same steps and phase
   (`set_steps`, `set_phase(0, CODE_PHASE_FRAC, CODE_CHIP)`, `enabled = True`, `run`): the
   board returns as many dumps as the model, and each dump's epoch and six sums equal the
   model's (the board reports no sample index, so the model's `sample` field is not compared).
   `fpga/tb/cocotb_spi.py::board_sequence_over_spi` is this sequence in simulation.
