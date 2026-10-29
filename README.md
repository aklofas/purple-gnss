# Purple GNSS

**An Open Source RTK GNSS Receiver.**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Where we are today

This project is early and pre-release. The hardware is in design. What exists
today is the reference receiver, and its interfaces will change:

| Part                                   | Status                                                                                                                                | Start here                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **`pyrx`**, offline receiver in Python | Working: recorded GPS L1 C/A samples to RINEX, and RINEX to a standalone or RTK fix; each stage checked against Pocket SDR and RTKLIB | [`pyrx/README.md`](pyrx/README.md) |
| **Hardware**                           | In design                                                                                                                             |                                    |

## Quick start

Install the receiver (Python 3.12 or later, in a virtual environment) from the
repository root, download Pocket SDR's
[sample recording](https://gpspp.sakura.ne.jp/pocketsdr/L1L5_20211226_082401_24MHz.zip)
(590 MB) and extract `L1_20211226_082401_24MHz_I.bin` from it, 30 seconds of
GPS L1, then get a position out of it:

```sh
pip install -e pyrx
pyrx-rinex L1_20211226_082401_24MHz_I.bin --fs 24e6 --f-if 6e6 --obs out.obs --nav out.nav
pyrx-pvt out.obs out.nav
```

The first command acquires and tracks GPS satellites in a recording,
decodes their navigation messages and writes the observations and ephemerides
as RINEX files. The second solves a position from them. `pyrx-rtk` gives an
RTK fix from rover and base observation files, a navigation file and the base's
coordinates (`--base-xyz`); see
[RTK from the command line](pyrx/README.md#rtk-from-the-command-line).

## Documentation

→ **[`docs/index.md`](docs/index.md)** routes readers by what they want to do. Or jump straight in:

| If you are…                                                        | Start here                                                                                           |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| New to GNSS and want to understand the steps from satellite to fix | [`docs/fundamentals/`](docs/fundamentals/01-what-gnss-is.md), a series written for technical readers |
| Running the receiver on a recording                                | [`pyrx/README.md`](pyrx/README.md), processing commands and recordings                               |
| Looking up a term                                                  | [`docs/fundamentals/glossary.md`](docs/fundamentals/glossary.md)                                     |

## License

Licensed under **MIT** ([`LICENSE-MIT`](LICENSE-MIT)) or **Apache-2.0** ([`LICENSE-APACHE`](LICENSE-APACHE)),
at your option.
