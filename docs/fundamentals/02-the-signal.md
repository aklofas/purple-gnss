# The signal

> **Who this is for:** You know GNSS satellites broadcast their time and orbit. You want to understand what the radio signal actually looks like and why a receiver can hear something that is far weaker than the background noise.

> **You will learn:**
> - The three layers of a GNSS signal: carrier, spreading code, and data
> - What a PRN code is and why every satellite can use the same frequency
> - How a signal 1000 times weaker than the noise is still usable
> - What the different frequency bands are for, and the data and pilot split on modern signals

This page is conceptual. The series continues with [from antenna to samples](03-antenna-to-samples.md).

---

## Three layers

A GNSS signal is built from three things multiplied together.

**The carrier** is a pure radio sine wave. GPS L1 is at 1575.42 MHz, a wavelength of 19.0 cm. On its own it carries no information, but its phase can be measured very precisely, which is what RTK exploits later.

**The spreading code** is a fixed, pseudo-random sequence of +1 and -1 values, called chips, that flips the carrier's sign at a steady rate. For the GPS L1 C/A signal the sequence is 1023 chips long, runs at 1.023 million chips per second, and so repeats every millisecond. Each satellite has its own sequence, identified by its PRN number (pseudo-random noise), which is the number you see when a receiver lists satellites.

**The data message** is a slow stream of bits, 50 per second on L1 C/A, that flips the sign of the whole thing once more. It carries the clock and orbit information. It is so slow that a single bit lasts 20 repetitions of the code.

```text
carrier   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   1575.42 MHz
code      +1 -1 -1 +1 -1 +1 +1 +1 -1 ...          1.023 Mchip/s, repeats every 1 ms
data      +1 ............ -1 ............ +1       50 bit/s, one bit per 20 code periods
```

The code is the clever part, and the next two sections are about it.

## Why every satellite can share one frequency

All GPS satellites transmit on the same frequencies at the same time. They do not interfere with each other because their codes are chosen to be nearly uncorrelated: multiply one satellite's code by another's, chip by chip, and sum over a full period, and the result is close to zero. Multiply a code by a copy of itself, correctly aligned, and the sum is the full 1023. Shift the copy by one chip or more and it drops back to near zero (not exactly zero: the worst misalignment of a C/A code against itself or another leaves about 6 percent of the peak, which is why detection thresholds are set a few times above the floor rather than just above it).

So a receiver generates its own copy of satellite 7's code, multiplies it against whatever arrives at the antenna, and sums. If satellite 7 is present and the copy is aligned, the sum is large. Every other satellite, and the noise, averages out. This operation is called correlation, and it is the single most important thing a receiver does. Each tracking channel performs it continuously, and the FPGA in a hardware receiver is mostly correlators.

Finding the alignment is the timing measurement. The code is a ruler with 1023 marks, each 293 meters long (one chip at the speed of light), repeating every 300 km. Knowing where the code boundary falls in the received signal tells the receiver the transmit time to within a fraction of a chip. The ambiguity of which millisecond it is gets resolved from the data message, as covered in [reading the message](06-navigation-message.md).

## Hearing a signal below the noise

A GPS satellite transmits about 25 watts from 20,000 km away. At the ground the signal power is around 10^-16 watts, roughly 20 dB (100 times) below the thermal noise that any antenna picks up in the same bandwidth. You cannot see a GPS signal on a spectrum analyzer. It is simply not there to the eye.

Correlation is what makes it usable. Summing over 1023 chips adds the signal coherently and the noise incoherently, which improves the signal-to-noise ratio by a factor of 1023, about 30 dB. The 20 dB deficit becomes a 10 dB surplus. Summing over more code periods (10 ms, 20 ms, or longer) buys more, at the cost of a finer search over frequency. This is why a receiver can use a 1-bit analog-to-digital converter: the noise is so dominant that the sign of each sample is almost all the information there is.

The ratio that matters is called C/N0, the carrier power over the noise density, in dB-Hz. Typical values are 35 to 50 dB-Hz in the open sky. You will see this number in every receiver's status display and in every page of this series.

## Bands and modern signals

GPS transmits on three frequencies, and the other systems use the same or neighboring ones so that a single antenna and front-end can cover several.

| Band | Frequency | Wavelength | Signals |
|---|---|---|---|
| L1 / E1 | 1575.42 MHz | 19.0 cm | GPS L1 C/A, L1C; Galileo E1; BeiDou B1C |
| L2 | 1227.60 MHz | 24.4 cm | GPS L2C (and the encrypted P(Y)) |
| L5 / E5a | 1176.45 MHz | 25.5 cm | GPS L5; Galileo E5a; BeiDou B2a |
| E5b | 1207.14 MHz | 24.8 cm | Galileo E5b; BeiDou B2b |
| E6 | 1278.75 MHz | 23.4 cm | Galileo E6, which carries the free HAS corrections |

Two frequencies matter for two reasons. The ionosphere delays the two by different amounts, so a receiver with both can compute and remove the delay instead of guessing it. And in RTK, two wavelengths can be combined into a longer effective wavelength that makes the integer-cycle problem much easier.

The modern signals (L5, E1, E5a) are also better designed than the original L1 C/A. L5 and E5a run their codes ten times faster (10.23 million chips per second), so each chip is 29 m instead of 293 m and timing is correspondingly sharper. And they are split into two components: a data channel carrying the message, and a pilot channel with no data at all. A pilot has no unknown navigation-bit flips, so it supports longer coherent sums and makes weak signals easier to track. Many pilots still change sign according to a known secondary code, which the receiver must align with and remove. Oscillator stability and motion limit how long the sum can remain coherent. Galileo E1 keeps the 1.023 million chip per second rate of L1 C/A and gets its sharper timing a different way, from a modulation called BOC that moves the signal power away from the center of the band. The details differ, the structure of carrier, code, and data does not.

---

**In pyrx:** `codes.py` generates GPS L1 C/A codes and defines their chip rate and period, `signals.py` defines carrier frequencies and wavelengths for the supported observation bands, and `correlator.py` performs the correlation that every later stage is built on.

Next: [From antenna to samples](03-antenna-to-samples.md).
