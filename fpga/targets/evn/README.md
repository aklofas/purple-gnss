# LIFCL-40-EVN target

`top.v` wraps `rx_spi` for the Lattice CrossLink-NX evaluation board (FPGA-EB-02028): SPI and
the dump interrupt on PMOD header J18, two sample pins on J17, five LEDs, SW2 as reset.

## Build and load

```
./build.sh            # one channel: build/n1/top.bit, report.txt
./build.sh 12         # twelve channels: build/n12/
./build.sh 1 fbkdiv   # a PLL variant (below): build/n1-fbkdiv/, likewise analog
openFPGALoader -b crosslinknx_evn build/n1/top.bit
```

Yosys `synth_nexus`, nextpnr-nexus for `LIFCL-40-9BG400C` with `evn.pdc`, `prjoxide pack
--compress`. The script clears the output directory first and stops on any tool failure.
nextpnr can abort while writing the PLL parameters and leave a truncated FASM that prjoxide
packs without complaint, so the script also requires nextpnr's closing log line and the FASM's
PLL and oscillator sections; and because nextpnr drops a port constraint whose selector matches
nothing without a warning, it counts the pull-ups and pull-downs in the FASM against the ones
`evn.pdc` asks for. `report.py` prints the utilization and the fabric clock's fmax; `report.txt`
also records the source commit and the PLL variant.

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

### PLL bitstreams

Whether that GPLL locks is the one thing here only the board can tell, and two of its settings
are in doubt. The feedback divider's minimum-ratio mask (`FBK_MASK`) defaults to 8 while the
ratio is 3; TN-02095 Table C.3 says the mask then sets the ratio, which would ask for a 3 GHz
VCO and never lock. And the analog settings the open toolchain defaults to (`V2I_KVCO_SEL` 85,
`KP_VCO` 0b11001, `V2I_1V_EN` off) are the values Lattice's reference files use for a 0.9 V
core; the EVN's core runs at 1.0 V, where those files use 60, 0b00011 and on. `build.sh`
builds three variants of `top.v`, selected with `-D`; load them in this order and stop at the
first that locks:

| Variant | Changes | Bitstream |
|---|---|---|
| `asbuilt` | none | `build/n1/top.bit` |
| `fbkdiv` | `FBK_MASK` cleared; `FBK_MMD_PULS_CTL` 0b0001, the pulse width Table C.3 gives for ratios above 2; `PLLRESET_ENA` on, so the power-on reset `top.v` drives reaches the PLL | `build/n1-fbkdiv/top.bit` |
| `analog` | `fbkdiv` plus the 1.0 V analog set | `build/n1-analog/top.bit` |

Locked: LED0 on and LED4 blinking at 1.6 to 1.8 Hz (34.9 to 40.1 MHz × 3 / 2^26). Not locked:
LED0 off and LED4 dark or at another rate. LED1 blinks either way, since the fabric runs from
the oscillator. Once a variant locks, make its settings the defaults in `top.v`. If none does,
the next things to try are the remaining difference from the reference files (`V2I_PP_ICTRL`
0b11111) and a ratio of 8 (`FBK_MMD_DIG` 8, `DIVA` 2), which stays above the mask without
clearing it and quadruples the LED4 rate.

## Wiring to the Nucleo

J18 is PMOD1 (bank 7, fixed 3.3 V). Standard PMOD numbering: 5 and 11 ground, 6 and 12 3.3 V.

| J18 pin | Ball | Signal | Direction |
|---|---|---|---|
| 1 | E10 | SPI SCK (pulled down) | in |
| 2 | E9 | SPI MOSI (pulled down) | in |
| 3 | E7 | SPI MISO | out |
| 4 | E8 | SPI CS (active low, pulled up) | in |
| 7 | E4 | DUMP_IRQ (high while any dump is unread) | out |
| 5 | | GND | |

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
   LED0 on with LED4 blinking shows the PLL locked. If LED0 stays off, load the next PLL
   variant (above) and note which one locks.
2. From the Nucleo's SPI master (mode 0, SCK ≤ 4.3 MHz: an eighth of the slowest fabric
   clock): read
   `ID` (0x0000) and expect `0x50474E53`, `VERSION` (0x0001) = `0x0100`.
3. Write the PRN's 32 code words to `CODE_WR_ADDR`/`CODE_WR_DATA`, then the recording's 2-bit
   samples 16 per word to `PB_ADDR`/`PB_DATA` (3 ms of the 24 MHz recording is 4500 words),
   then `CARR_STEP`, `CODE_STEP`, `CARR_PHASE`, `CODE_PHASE_FRAC`, `CODE_CHIP` from the same
   acquisition result the bench used (`fpgamodel.compare.start_phase`,
   `fpgamodel.channel.nco_step`), `CH_CTRL` = 1, `CTRL` = 5 (enable, playback source),
   `PLAYBACK_LEN`, `PLAYBACK_CTRL` = 1.
4. Poll `DUMP_PENDING` or watch DUMP_IRQ; read `EPOCH` and `IE..QL`, write `DUMP_CLEAR`.
5. Compare each dump with `fpgamodel.channel.Channel.run` on the same samples: they are equal.
   `fpga/tb/cocotb_spi.py::monday_sequence_over_spi` is this exact sequence in simulation.
