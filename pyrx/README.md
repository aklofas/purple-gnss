# pyrx

An offline GNSS receiver in Python. It reads recorded IF samples and will grow stage
by stage into a full receiver: acquisition, tracking, navigation-message decode,
observables, a standalone position fix, and finally an RTK fix. Only acquisition
exists so far. Each stage is checked against an external reference before the next
one is started: Pocket SDR for the signal-processing stages, RTKLIB for positioning.

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

The reference tests compare acquisition with Pocket SDR's results on two of its
sample recordings. They need the environment variable `PYRX_CAPTURES` pointing at
a directory holding those files, and skip when it is unset:

| File | Source | SHA-256 |
|---|---|---|
| `L1_20211202_084700_4MHz_IQ.bin` | Pocket SDR sample set, GPS L1, 4 MHz I/Q, 100 ms | `0a8335d2f099e388b474d2afcca1ff91f61cde550dd32bf82fdf199d8a5b8033` |
| `L1_20211201_054600_24MHz_I.bin` | Pocket SDR sample set, GPS L1, 24 MHz real, 6 MHz IF, 100 ms | `05c771f0c152e2bd11ccba0715198afc0aa56e5d420bec0993fa720ac8d508d0` |

The 24 MHz recording has sample gaps near 40 ms and 90 ms; the tests use the
first 10 ms, which is also what Pocket SDR integrates by default.

## Sample format

Signed 8-bit samples. Real recordings hold one byte per sample. I/Q recordings
interleave I and Q bytes and the complex sample is `I - jQ`, the Q sign convention
of the Pocket SDR front-end.

## Acquisition from the command line

```sh
pyrx-acq FILE --fs 4e6 --iq [--f-if 0] [--prn 1-32] [--duration 0.01] [--plot DIR]
```

Prints one line per PRN in the same format as Pocket SDR's `pocket_acq.py`, so the
two outputs can be diffed. With `--plot`, writes `cn0.png` and one search-grid
image per detected PRN.
