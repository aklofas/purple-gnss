# The signal

> **Who this is for:** You know GNSS satellites broadcast their time and orbit. You want to understand what the radio signal actually looks like and why a receiver can hear something that is far weaker than the background noise.

> **You will learn:**
> - The three layers of a GNSS signal: carrier, spreading code, and data
> - What a PRN code is, what cross-correlation does with it, and why every satellite can use the same frequency
> - How a signal 100 times weaker than the noise is still usable, and what C/N0 measures
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

As an equation, with $A$ the signal's amplitude, $d(t)$ the data bit and $g(t)$ the code chip at time $t$ (each +1 or -1), $f_c$ the carrier frequency and $\theta$ the carrier's phase, the transmitted signal is

$$
s(t) = A d(t) g(t) \cos(2\pi f_c t + \theta) .
$$

Multiplying a carrier by +1 or -1 is binary phase-shift keying, BPSK: a sign flip is a jump of half a cycle in the carrier's phase. The code flips about 20,000 times faster than the data, and those fast flips spread the signal's power over about 2 MHz around the carrier, where the 50 bit/s data alone would need about 100 Hz. Using a fast code to spread a slow message this way is called direct-sequence spread spectrum. The code is the clever part, and the next two sections are about it.

## Why every satellite can share one frequency

All GPS satellites transmit on the same frequencies at the same time and are told apart by their codes alone, a scheme called code-division multiple access, CDMA. (GLONASS's original signals are the exception: each satellite has its own frequency.) What makes CDMA work is the way the codes correlate with each other.

The cross-correlation of two codes measures how alike they are at a given alignment: multiply one code by the other, chip by chip, and add the products over a full period. With $a_n$ and $b_n$ the chips of two codes and $k$ a shift in chips, it is

$$
R_{ab}(k) = \sum_{n=0}^{1022} a_n b_{(n+k) \bmod 1023} .
$$

Each product is +1 where the chips agree and -1 where they differ, so $R$ counts agreements minus disagreements. The shift wraps around because the code repeats. The correlation of a code with itself, its autocorrelation, is the full 1023 when the shift is zero, since every chip then agrees with itself. At every other shift, and against another satellite's code at any shift, the C/A codes are designed to give one of only three values: -65, -1 or +63. (That holds when the two signals arrive at the same frequency; a Doppler difference between two satellites can make their cross-correlation somewhat larger.) They are Gold codes, built so that this holds for the whole family. Each is the chip-by-chip product of two 1023-chip sequences from 10-stage shift registers, and each satellite uses a different delay between the two. The worst of those values is about 6 percent of the peak (−24 dB), not zero. Detection thresholds are set above the highest peak that noise alone produces across the search; the 6 percent off-peak value also matters, because a strong satellite's code can raise a false peak about 24 dB below its own.

So a receiver generates its own copy of satellite 7's code, called a replica, multiplies it against whatever arrives at the antenna, and sums. If satellite 7 is present and the copy is aligned, the sum is large. Every other satellite, and the noise, averages out. This operation is correlation, and it is the single most important thing a receiver does. A filter that correlates the input against a known copy of the wanted signal is called a matched filter; in white noise, no other linear filter gives a higher signal-to-noise ratio. Each tracking channel performs the correlation continuously, and the FPGA (field-programmable gate array, the programmable logic) of a hardware receiver is mostly correlators.

Between whole-chip shifts the correlation changes smoothly. If the replica is off by a fraction $\delta$ of a chip, then for that fraction of every chip it already shows the neighboring chip, which differs about half the time, and the sum falls to about $1023(1 - \lvert\delta\rvert)$. Plotted against the offset, the peak is a triangle two chips wide at its base:

```text
 correlation
    1023 |               /\
         |              /  \
     512 |             /    \
         |            /      \
      ~0 |-----------/        \-----------
         +-----------+---+---+-----------  replica offset (chips)
                    -1   0  +1
```

A replica half a chip off still gets half the peak; one chip off gets almost nothing. The search for a satellite, on the [finding a satellite](04-acquisition.md) page, relies on the wide part of this triangle to tell whether a signal is there at all. Tracking, on the [staying locked](05-tracking.md) page, relies on its slopes: comparing a replica slightly early with one slightly late shows which way the peak lies.

Finding the alignment is the timing measurement. The code is a ruler with 1023 marks, each 293 meters long (one chip at the speed of light), repeating every 300 km. Knowing where the code boundary falls in the received signal tells the receiver the transmit time to within a fraction of a chip. The ambiguity of which millisecond it is gets resolved from the data message, as covered in [reading the message](06-navigation-message.md).

## Hearing a signal below the noise

A GPS satellite transmits about 25 watts from 20,000 km away. At the ground the signal power is around 10^-16 watts (the GPS interface specification guarantees at least -158.5 dBW for L1 C/A, about 1.4 × 10^-16 W), roughly 20 dB (100 times) below the thermal noise that any antenna picks up in the same bandwidth. (Decibels, dBW and dB-Hz are explained on [the next page](03-antenna-to-samples.md#decibels-in-one-paragraph); for now, 10 dB is a factor of 10 in power and 20 dB a factor of 100.) You cannot see a GPS signal on a spectrum analyzer. It is simply not there to the eye.

Correlation is what makes it usable. When the replica is aligned and the carrier removed, the signal part of each product has the same sign, so over $M$ products the signal's amplitude grows as $M$ and its power as $M^2$. The noise products have random signs, so their sum grows only as $\sqrt{M}$ in amplitude and $M$ in power. The ratio of the two powers improves by a factor of $M$. Summing over 1023 chips improves the signal-to-noise ratio by a factor of 1023, about 30 dB, the processing gain of the code. That factor applies to the signal-to-noise ratio measured in a band as wide as the chip rate. Measured in the full 2 MHz band, the gain is about 33 dB, so the deficit of 18 to 20 dB becomes a surplus of about 15 dB, as the worked example below shows. Seen in frequency, the multiplication by the replica undoes the spreading, so the signal's power collapses from 2 MHz back into a narrow band while the noise stays spread out; this step is called despreading.

Adding products with their signs and phases intact like this is coherent integration. Summing over more code periods (10 ms, 20 ms, or longer) buys more, at the cost of a finer search over frequency. On L1 C/A the data bit sets a limit: a bit lasts 20 ms, and a sum that spans a bit change partly cancels. Because the noise is so dominant, a receiver can use a 1-bit analog-to-digital converter: the sign of each sample is almost all the information there is.

The ratio that matters is called C/N0, the carrier-to-noise density ratio. With $C$ the received signal power in watts and $N_0$ the noise power in one hertz of bandwidth, in watts per hertz, it is $C/N_0$, quoted in dB-Hz. It does not depend on any particular bandwidth. To get the ratio in a bandwidth $B$, divide by $B$; for a coherent sum over a time $T_{\rm int}$, the signal-to-noise ratio of the result is roughly $C/N_0$ times $T_{\rm int}$. A worked example at 45 dB-Hz, a strong open-sky signal: in the 2 MHz of the signal's spread, $10\log_{10}(2 \times 10^6) \approx 63$ dB, so the signal sits 18 dB below the noise. After 1 ms of coherent integration, $10\log_{10}(10^{-3}) = -30$ dB, so it is 15 dB above it. The gain here is 33 dB, a little more than the 30 dB above, because the 2 MHz band is about twice the chip rate. Typical values are 35 to 50 dB-Hz in the open sky. You will see this number in every receiver's status display and in every page of this series.

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

The modern signals (L5, E1, E5a) are also better designed than the original L1 C/A. L5 and E5a run their codes ten times faster (10.23 million chips per second), so each chip is 29 m instead of 293 m and the correlation triangle is ten times narrower, which makes timing correspondingly sharper. And they are split into two components: a data channel carrying the message, and a pilot channel with no data at all. A pilot has no unknown navigation-bit flips, so it supports longer coherent sums and makes weak signals easier to track. Many pilots still change sign according to a known secondary code, which the receiver must align with and remove. Oscillator stability and motion limit how long the sum can remain coherent. Galileo E1 keeps the 1.023 million chip per second rate of L1 C/A and gets its sharper timing a different way, from a modulation called BOC that moves the signal power away from the center of the band. The details differ, the structure of carrier, code, and data does not.

---

**In pyrx:** `codes.py` generates the GPS L1 C/A codes from the two shift registers of IS-GPS-200 and defines their chip rate and period; its tests check the first ten chips of all 32 codes against the specification's table and check that one code's correlation with its own shifted copies never exceeds 65 at any shift other than zero, against 1023 at zero. `correlator.py` computes the correlation sums the tracking loops use, and its tests check that a replica a quarter chip early or late gets three quarters of the aligned sum, as the triangle above predicts. Acquisition computes the same correlation with FFTs. `signals.py` defines carrier frequencies and wavelengths for the supported observation bands. Section 2 of the [signal-chain notebook](../../pyrx/notebooks/01-signal-chain.ipynb) generates these codes and correlates them against a real recording.

Next: [From antenna to samples](03-antenna-to-samples.md).
