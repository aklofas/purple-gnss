# Purple GNSS

**An Open Source RTK GNSS Receiver.**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Where we are today

This project is early. The hardware is in design and the firmware has not
started. What exists today is the reference receiver:

| Part | Status | Start here |
|---|---|---|
| **`pyrx`**, offline receiver in Python | Working: recorded GPS L1 C/A samples to RINEX, and RINEX to a standalone or RTK fix; each stage checked against Pocket SDR and RTKLIB | [`pyrx/README.md`](pyrx/README.md) |
| **Hardware** (RF front-end, FPGA correlators, MCU board) | In design | |
| **Firmware** | Not started | |

## See it in 30 seconds

Take a 30-second recording of GPS L1 and get a position out of it:

```sh
pyrx-rinex L1_20211226_082401_24MHz_I.bin --fs 24e6 --f-if 6e6 --obs out.obs --nav out.nav
pyrx-pvt out.obs out.nav
```

The first command acquires and tracks every GPS satellite in the recording,
decodes their navigation messages and writes the observations and ephemerides
as RINEX files. The second solves a position from them. Add a base station's
RINEX file and `pyrx-rtk` gives the RTK fix.

## Documentation

→ **[`docs/index.md`](docs/index.md)** routes readers by what they want to do. Or jump straight in:

| If you are… | Start here |
|---|---|
| New to GNSS and want to understand every step from satellite to fix | [`docs/fundamentals/`](docs/fundamentals/01-what-gnss-is.md), a ten-page series written for technical readers without a radio background |
| Running the receiver on a recording | [`pyrx/README.md`](pyrx/README.md), every command and the recordings it was checked on |
| Looking up a term | [`docs/fundamentals/glossary.md`](docs/fundamentals/glossary.md) |

## Status

Pre-release. The Python receiver is complete for GPS L1 C/A from samples and
for GPS and Galileo from RINEX. Nothing is released as a package and the
interfaces will change.

## License

Licensed under **MIT** ([`LICENSE-MIT`](LICENSE-MIT)) or **Apache-2.0** ([`LICENSE-APACHE`](LICENSE-APACHE)),
at your option.
