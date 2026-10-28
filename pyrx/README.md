# pyrx

An offline GNSS receiver in Python. It reads recorded IF samples and will grow stage
by stage into a full receiver: acquisition, tracking, navigation-message decode,
observables, a standalone position fix, and finally an RTK fix. Acquisition,
tracking and navigation-message decode exist so far. Each stage is checked against
an external reference before the next one is started: Pocket SDR for acquisition
and tracking, broadcast ephemeris files for the navigation decode, RTKLIB for
positioning.

## Why Python

This code exists to prove the receiver's algorithms on real signals before they are
ported to the FPGA correlators and the MCU firmware, and then to serve as the
floating-point reference those are checked against. Expressiveness and plotting
matter more than speed here. numpy and scipy supply FFTs, filters and linear
algebra; where profiling shows an inner loop dominating, numba is the escape hatch
that keeps everything in one language. Real-time operation is not a goal.

## Install

```sh
pip install -e '.[dev]'
```

## Tests

```sh
pytest
```

The reference tests compare acquisition and tracking with Pocket SDR's results on
its sample recordings, and the navigation decode with the day's broadcast
ephemeris. They need the environment variable `PYRX_CAPTURES` pointing at a
directory holding these files, and skip when it is unset:

| File | Source | SHA-256 |
|---|---|---|
| `L1_20211202_084700_4MHz_IQ.bin` | Pocket SDR sample set, GPS L1, 4 MHz I/Q, 100 ms | `0a8335d2f099e388b474d2afcca1ff91f61cde550dd32bf82fdf199d8a5b8033` |
| `L1_20211201_054600_24MHz_I.bin` | Pocket SDR sample set, GPS L1, 24 MHz real, 6 MHz IF, 100 ms | `05c771f0c152e2bd11ccba0715198afc0aa56e5d420bec0993fa720ac8d508d0` |
| `L1_20211226_082401_24MHz_I.bin` | Pocket SDR sample set, L1+L5 recording of 2021-12-26, GPS L1, 24 MHz real, 6 MHz IF, 30 s | `16c83d58716ae9ba577aac29344766373accb98b34b729afdac8cc2e1352252c` |
| `L1_20220114_004440_8MHz_I.bin` | Pocket SDR sample set, L1+L2 recording of 2022-01-14, GPS L1, 8 MHz real, 2 MHz IF, 30 s | `388c550e2694067566c24fddae61abd1b316eb4bbd0820e0e8fff1818f164de7` |
| `brdc/BRDC00WRD_R_20213600000_01D_GN.rnx` | BKG (`igs.bkg.bund.de/root_ftp/IGS/BRDC/2021/360/`), worldwide GPS broadcast ephemeris, RINEX 3.05, 2021-12-26 | `92fe0754c16b3408691facabad44390ebfa2031ad641334ebe6f8660f8e554c2` |
| `brdc/BRDC00WRD_R_20220140000_01D_GN.rnx` | BKG (`.../BRDC/2022/014/`), worldwide GPS broadcast ephemeris, RINEX 3.05, 2022-01-14 | `ac1488e8b577ddd9b942de2ca8033fa82f8ce56abfe9d6e1781288cfa590d7e2` |
| `brdc/BRDC00IGS_R_20220140000_01D_MN.rnx` | BKG (`.../BRDC/2022/014/`), IGS combined multi-GNSS broadcast file, RINEX 3.04, 2022-01-14; only its GPSA/GPSB header lines are used | `5e366723dcda6bcfb049b5bc0459284d27581855567c1ff977de7cf76e1f701d` |

The 100 ms 24 MHz recording has sample gaps near 40 ms and 90 ms; the acquisition
tests use its first 10 ms, which is also what Pocket SDR integrates by default.

The tracking tests also need Pocket SDR's own tracking logs for the two 30 s
recordings, in a `pocketsdr-trk/` subdirectory, produced with Pocket SDR's
`pocket_trk.py` (its library built with `USE_SOAPY=0 USE_FFTW=1`):

```sh
pocket_trk.py -sig L1CA -prn 4,16,18,26,27,29,31 -f 24 -fi 6.0 -q -log L1_20211226_082401_24MHz_I.trk.log L1_20211226_082401_24MHz_I.bin
pocket_trk.py -sig L1CA -prn 10,12,15,23,24,32  -f 8  -fi 2.0 -q -log L1_20220114_004440_8MHz_I.trk.log  L1_20220114_004440_8MHz_I.bin
```

The PRN lists are the satellites `pyrx-acq` reports at or above 38 dB-Hz in the
first 10 ms of each file.

## Sample format

Signed 8-bit samples. Real recordings hold one byte per sample. I/Q recordings
interleave I and Q bytes and the complex sample is `I - jQ`, the Q sign convention
of the Pocket SDR front-end.

## Acquisition from the command line

```sh
pyrx-acq FILE --fs 4e6 --iq [--f-if 0] [--prn 1-32] [--duration 0.01] [--plot DIR]

By default the first 10 ms of the file are integrated (`--duration 0` for the whole file).
```

Prints one line per PRN in the same format as Pocket SDR's `pocket_acq.py`, so the
two outputs can be diffed. With `--plot`, writes `cn0.png` and one search-grid
image per detected PRN.

## Tracking from the command line

```sh
pyrx-trk FILE --fs 8e6 --f-if 2e6 [--iq] [--prn 1-32] [--duration 0] [--log trk.csv] [--plot DIR]
```

Acquires each PRN on the first 10 ms, tracks the detected ones through the file
(`--duration 0`, the default, means all of it), each channel in its own process,
and prints the final C/N0, Doppler and lock state per PRN. `--log` writes a CSV
with columns `time_s,prn,cn0_dbhz,code_offset_ms,doppler_hz,carrier_cycles`, one
row per channel every 0.1 s. `--plot` writes `trk_prnNN.png` per channel with four
panels: prompt I and Q over the first 2 s (the navigation data bits are visible
once the phase lock settles), Doppler, C/N0, and the early, prompt and late
correlator magnitudes.

## Navigation message decode from the command line

```sh
pyrx-nav FILE --fs 8e6 --f-if 2e6 [--iq] [--prn 1-32] [--duration 0] [--json nav.json]
```

Acquires and tracks as `pyrx-trk` does, then decodes each channel's navigation
message and prints one line per PRN: whether bit sync was found, the subframes
decoded as `ID@TOW` (the GPS seconds of week at the subframe's first bit), and the
ephemeris issue of data and `toe` when subframes 1, 2 and 3 with the same issue
were seen. `--json` writes `{"ephemeris": {"<prn>": {field: value, ...}},
"ionosphere": {"alpha": [...], "beta": [...]} | null}` with the complete
ephemerides only.

## Navigation message decode

GPS L1 C/A carries the 50 bps LNAV message. Once the phase lock has settled, bit
sync finds the data-bit edge among the 20 one-millisecond epochs of a bit by
counting prompt sign changes at each epoch phase; frame sync then looks for the
TLM preamble repeating one subframe (300 bits, 6 s) later with both words of each
candidate passing parity, a valid subframe ID and consecutive TOW counts, trying
both data polarities. From there every 300-bit block whose ten words all pass
parity is a subframe; a parity failure drops that block only. Subframes 1, 2 and 3
decode to the clock and ephemeris parameters, subframe 4 page 18 to the
ionosphere parameters, all scaled per IS-GPS-200 with semi-circles converted to
radians. Each subframe is time-tagged with the tracking epoch of its first bit and
the GPS seconds of week of that bit (the HOW's TOW count refers to the start of
the following subframe). Almanac pages and the UTC parameters are not decoded.

The reference test decodes the two 30 s recordings and requires every field of
every subframe 1, 2 and 3 to equal the broadcast record with the same issue of
data to within half a transmitted LSB (some records in the files are written to
only seven significant digits), every channel to agree on the file start time to
within 30 ms, and the 2021-12-26 recording to yield a complete ephemeris on every
channel. The 2022-01-14 recording starts inside subframe 1, so its window holds
subframes 2 to 5 and no complete ephemeris. Neither window contains subframe 4
page 18, so the ionosphere parameters are checked only by the synthetic
round-trip tests.

## Tracking loops

Each channel integrates 1 ms (one C/A code period) with early and late taps at
±0.25 chip and a noise tap 10 chips away. A frequency-locked loop pulls in for
the first second (5 Hz bandwidth, then 2 Hz), after which a second-order Costas
phase-locked loop with 5 Hz bandwidth holds the carrier. The code loop is a
0.25 Hz delay-locked loop on 10 ms non-coherent early/late sums, with the code
rate aided by the carrier Doppler. C/N0 is estimated from prompt versus noise-tap
power over 1 s, and a channel is dropped below 25 dB-Hz. Every channel's carrier
NCO starts at the receiver's IF phase at sample 0 (modulo one cycle), so the
accumulated carrier phases of different channels share one origin. These constants match
Pocket SDR's so the two trackers can be compared number for number on the same
recording; they are not the final choices for the hardware receiver.
