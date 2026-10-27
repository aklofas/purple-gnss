# pyrx

An offline GNSS receiver in Python. It reads recorded IF samples and will grow stage
by stage into a full receiver: acquisition, tracking, navigation-message decode,
observables, a standalone position fix, and finally an RTK fix. Acquisition and
tracking exist so far. Each stage is checked against an external reference before
the next one is started: Pocket SDR for the signal-processing stages, RTKLIB for
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
its sample recordings. They need the environment variable `PYRX_CAPTURES` pointing
at a directory holding these files, and skip when it is unset:

| File | Source | SHA-256 |
|---|---|---|
| `L1_20211202_084700_4MHz_IQ.bin` | Pocket SDR sample set, GPS L1, 4 MHz I/Q, 100 ms | `0a8335d2f099e388b474d2afcca1ff91f61cde550dd32bf82fdf199d8a5b8033` |
| `L1_20211201_054600_24MHz_I.bin` | Pocket SDR sample set, GPS L1, 24 MHz real, 6 MHz IF, 100 ms | `05c771f0c152e2bd11ccba0715198afc0aa56e5d420bec0993fa720ac8d508d0` |
| `L1_20211226_082401_24MHz_I.bin` | Pocket SDR sample set, L1+L5 recording of 2021-12-26, GPS L1, 24 MHz real, 6 MHz IF, 30 s | `16c83d58716ae9ba577aac29344766373accb98b34b729afdac8cc2e1352252c` |
| `L1_20220114_004440_8MHz_I.bin` | Pocket SDR sample set, L1+L2 recording of 2022-01-14, GPS L1, 8 MHz real, 2 MHz IF, 30 s | `388c550e2694067566c24fddae61abd1b316eb4bbd0820e0e8fff1818f164de7` |

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

## Tracking loops

Each channel integrates 1 ms (one C/A code period) with early and late taps at
±0.25 chip and a noise tap 10 chips away. A frequency-locked loop pulls in for
the first second (5 Hz bandwidth, then 2 Hz), after which a second-order Costas
phase-locked loop with 5 Hz bandwidth holds the carrier. The code loop is a
0.25 Hz delay-locked loop on 10 ms non-coherent early/late sums, with the code
rate aided by the carrier Doppler. C/N0 is estimated from prompt versus noise-tap
power over 1 s, and a channel is dropped below 25 dB-Hz. These constants match
Pocket SDR's so the two trackers can be compared number for number on the same
recording; they are not the final choices for the hardware receiver.
