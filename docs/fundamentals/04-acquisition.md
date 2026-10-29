# Finding a satellite

> **Who this is for:** You know a GNSS signal is a carrier with a spreading code on it, and you have a stream of samples from the front-end. You want to know how the receiver finds a satellite in that stream.

> **You will learn:**
> - What acquisition has to find: which satellites, their code phase, and their Doppler
> - How cross-correlation against a code replica reveals a satellite buried in noise
> - Why the search is two-dimensional, how big it is, and how an FFT makes it fast
> - What a good detection looks like and what weakens it

This page is conceptual. The series continues with [staying locked](05-tracking.md).

---

## The three unknowns

A receiver that starts cold knows nothing: which satellites are overhead, where their codes begin in the sample stream, or what frequency they arrive at. Acquisition is the search that answers all three for each satellite. It only has to answer them coarsely, because the next stage, tracking, refines the answers and holds them.

**Which satellite.** Each satellite has its own code, identified by its PRN number, so the receiver tries each code in turn. For GPS L1 there are 32 to try.

**Code phase.** The received code is a pattern that repeats every 1 ms. Its code phase is where in that pattern the signal is at a given sample. The receiver's copy has to line up with it to within about half a chip, roughly 500 nanoseconds, for a match to stand out; one chip away the match fades to nothing. A period holds 1023 chips, so testing two positions per chip means about 2000 alignments.

**Doppler.** The satellites move at nearly 4 km/s. The part of that motion along the line of sight shifts the received frequency by up to about 5 kHz for a user standing still, the Doppler shift, and the receiver's own oscillator error adds a few kHz more. The search has to get the frequency roughly right too, for a reason the next section makes clear.

## Cross-correlation against a replica

The tool for finding a known pattern in noise is cross-correlation. The receiver generates a replica of one satellite's code, shifted to a trial code phase and multiplied by a carrier at a trial frequency. It multiplies the incoming samples by the replica, sample by sample, and adds up the products over one code period. When the replica matches the signal, every product has the same sign and the sum grows with every sample. Noise has random signs and mostly cancels, and a wrong shift or another satellite's code gives a far smaller sum, because the codes are chosen to be nearly uncorrelated with each other and with shifted copies of themselves.

Written out, with $x_n$ the samples, $g$ the code, $\tau$ the trial code phase, $f$ the trial frequency, $t_n$ the time of sample $n$ and $M$ the number of samples in one period, the correlation is

$$
R(\tau, f) = \sum_{n=0}^{M-1} x_n g(t_n - \tau) e^{-j 2\pi f t_n} .
$$

The trial frequency is where the carrier sits in the samples, so on samples at an intermediate frequency it is that frequency plus the trial Doppler, $f = f_{\rm IF} + f_D$. The complex exponential strips the carrier off. Its result is a complex number because the receiver does not yet know the carrier's phase, so the search uses the power $\lvert R \rvert^2$, which does not depend on that phase. Adding samples up in phase like this over one period is called coherent integration, and 1 ms of it lifts a satellite that sits far below the noise in any single sample to a clear peak.

The frequency matters because a mismatch makes the products rotate. With the trial frequency 1 kHz off, the phase of the products turns through a full cycle during the 1 ms sum and the sum cancels entirely; at 500 Hz off, about 4 dB is lost. The frequency range is therefore searched in steps of 500 Hz, the Doppler bins, so that no satellite is ever more than 250 Hz from a bin's center. Covering ±5 kHz plus the oscillator error takes 20 to 40 bins.

## The search grid

The search space for one satellite is a grid of roughly 2000 code phases by 30 Doppler bins, about 60,000 cells, each a full 1 ms correlation. Done one cell at a time on the hardware of the 1990s, that took minutes.

```text
                 Doppler bins (Hz)
             -5000 ....... 0 ....... +5000
code       +-------------------------------+
phase  0   |  .   .   .   .   .   .   .   .|
       .   |  .   .   .   .   .   .   .   .|
       .   |  .   .   .   .   .   #   .   .|   <- one cell stands out:
       .   |  .   .   .   .   .   .   .   .|      satellite present
     1022  |  .   .   .   .   .   .   .   .|
           +-------------------------------+
```

The fast method is parallel code phase search. The code repeats every period, so correlating against every shift of it is a circular correlation, and the convolution theorem turns circular correlation into a multiplication in the frequency domain. Take the FFT of 1 ms of samples with the trial carrier removed, multiply it by the complex conjugate of the code's FFT, and take the inverse FFT. The result is the correlation at every code phase at once, a whole Doppler row of the grid, for the cost of two FFTs per row, a forward and an inverse (the code's FFT is computed once and reused). Repeating that for each bin covers the grid in milliseconds.

A software receiver does exactly this, and as a bonus it tests every sample shift rather than two per chip. A hardware receiver on an FPGA may do the same, or may run a bank of parallel correlators that each test one cell, because an FPGA has room for thousands of them and the search then runs continuously in the background.

## Making the decision

The result of the search is a grid of correlation powers. If the satellite is present, one cell stands well above the rest. The receiver turns that into a detection by comparing the peak with a threshold, usually through the ratio of the highest peak to the mean of the grid, or to the second-highest peak away from the main one. The same ratio gives an estimate of the carrier-to-noise density ratio, C/N0, the standard measure of signal strength. The threshold sets the trade between missing a weak satellite and reporting a false alarm, a peak that is only noise; it is set above the highest peak that noise alone produces across the grid. The 6 percent off-peak value of the C/A codes also matters, because a strong satellite's code can raise a false peak about 24 dB below its own. The position of the peak gives the code phase to within half a chip and the Doppler to within half a bin.

Two things make a peak weaker than it should be. If a 1 ms L1 C/A window straddles a data-bit transition, the sign flips partway through the sum and it partly cancels; trying the next 1 ms window avoids that transition. A pilot signal carries no navigation bits, but its known secondary-code sign changes must still be accounted for. And the peak only has full height when the Doppler bin is exactly right, since the correlation falls off smoothly with frequency error, so the true value is estimated from the neighboring bins too.

To reach weaker signals the receiver adds the powers of several consecutive 1 ms correlations. This is non-coherent integration: it keeps the gain from more data without needing the carrier phase or the data bits to stay constant, at the cost of a longer search and a smaller gain per millisecond than coherent integration would give. Receivers acquire indoors and under foliage this way, down to around 25 dB-Hz with enough patience. In the open sky at 45 dB-Hz a single millisecond is enough.

## From cold to warm

A cold start searches the whole grid for every satellite, and after the search it still waits 30 s or more for each satellite's ephemeris to arrive. A warm start uses what the receiver already has: a rough position, the time, and an almanac, a coarse orbit description for the whole constellation stored from the last run. From those it predicts which satellites are up and their Doppler to within a bin or two, and searches only there. The search then takes well under a second, though the first fix still waits for each satellite's own orbit description, its ephemeris, to arrive, as the [navigation message](06-navigation-message.md) page explains. A hot start also kept the ephemerides from a run within the last few hours, skips that wait as well, and is the one that fixes in a second. All three are bookkeeping on top of the same search; the time each takes to produce a position is the time to first fix, TTFF.

Once a satellite is acquired, its coarse code phase and Doppler are handed to a tracking channel, the subject of the next page. Acquisition moves on to the next PRN, and in a running receiver keeps cycling through the satellites that are not yet tracked, so that ones rising over the horizon are picked up.

---

**In pyrx:** `acquisition.py` implements the parallel code phase search described here on a recording, and `pyrx-acq` runs it from the command line; adding `--plot DIR` also writes a C/N0 chart and a search-grid image for each detected satellite. It is checked against Pocket SDR's acquisition on the same recordings. Section 2 of the [signal-chain notebook](../../pyrx/notebooks/01-signal-chain.ipynb) runs this search on a real recording and draws its grid.

Next: [Staying locked](05-tracking.md).
