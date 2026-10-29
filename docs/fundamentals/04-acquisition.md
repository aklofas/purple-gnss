# Finding a satellite

> **Who this is for:** You know a GNSS signal is a carrier with a spreading code on it, and you have a stream of samples from the front-end. You want to know how the receiver finds a satellite in that stream.

> **You will learn:**
> - What acquisition has to find: which satellites, their code phase, and their Doppler
> - Why the search is two-dimensional and how big it is
> - How an FFT turns a million-trial search into a few milliseconds of work
> - What a good detection looks like and what weakens it

This page is conceptual. The series continues with [staying locked](05-tracking.md).

---

## The three unknowns

When a receiver starts cold it knows nothing: not which satellites are overhead, not where their codes begin in the sample stream, not what frequency they arrive at. Acquisition answers all three for each satellite, coarsely, so that the tracking stage can refine and hold them.

**Which satellite.** Try each PRN's code in turn. For GPS L1 there are 32 to try.

**Code phase.** The received code is a repeating 1 ms pattern. The receiver's copy has to be aligned to it to within about half a chip, roughly 500 nanoseconds, for the correlation peak to stand out; it fades to nothing one chip away. There are 1023 chips in a period, so at two trial positions per chip that is about 2000 alignments to test.

**Doppler.** Satellites move at nearly 4 km/s and the component along the line of sight changes the received frequency by up to about 5 kHz for a stationary user. The receiver's own oscillator error adds a few kHz more. If the receiver's replica is off in frequency by 1 kHz, the mismatch rotates the sum through a full cycle over the 1 ms correlation and it cancels entirely; at half that offset about 4 dB is lost. So the frequency range is searched in steps of 500 Hz, so that no satellite is ever more than 250 Hz from a bin, which is 20 to 40 bins.

The search space for one satellite is therefore a grid of roughly 2000 code phases by 30 Doppler bins, 60,000 cells, each of which requires a full 1 ms correlation. Done one at a time on the hardware of the 1990s, that took minutes. Done with an FFT it takes milliseconds.

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

## The FFT trick

Correlating the samples against every possible shift of the code is, mathematically, a circular correlation, and circular correlation is a multiplication in the frequency domain. Take the FFT of 1 ms of samples, multiply by the conjugate of the FFT of the code replica, and take the inverse FFT. The result is the correlation at every code phase at once, an entire column of the grid, for the cost of three FFTs. Repeating this per Doppler bin, with the samples first shifted in frequency by the bin's offset, covers the grid.

A receiver running in software does exactly this. A hardware receiver on an FPGA may do the same, or may use a bank of parallel correlators that each test one cell, because an FPGA has the area for thousands of them and the search then runs continuously in the background.

## Making the decision

The result of the search is a grid of correlation powers. If the satellite is present, one cell is much higher than the rest. The usual test is the ratio of the highest peak to the mean of the grid, or to the second-highest peak outside the main one. A ratio of a few times the noise floor is a detection; below that the receiver moves on. The position of the peak gives the code phase to half a chip and the Doppler to half a bin.

Two things make a peak weaker than it should be. If a 1 ms L1 C/A window straddles a data-bit transition, the correlation partly cancels; trying the next 1 ms window avoids that transition. A pilot avoids navigation-bit transitions, but its known secondary-code sign changes must still be accounted for. And the peak only has full height when the Doppler bin is exactly right, since the correlation falls off smoothly with frequency error, so the true value is estimated by looking at the neighboring bins too.

Summing the power from several consecutive 1 ms windows, non-coherently, lets the receiver find weaker signals at the cost of a longer search. This is how receivers acquire indoors and under foliage, down to around 25 dB-Hz with enough patience. In the open sky at 45 dB-Hz a single millisecond is enough.

## From cold to warm

A cold start searches the whole grid for every satellite. A receiver that knows roughly where it is, what time it is, and has an almanac (a coarse orbit description for the whole constellation, stored from the last run) can predict which satellites are up and their Doppler to within a bin or two, and only search those. The search then takes well under a second, though the fix still has to wait for each satellite's orbit description to arrive, as the [navigation message](06-navigation-message.md) page explains. A receiver that also kept its ephemerides from a run minutes earlier skips that wait too, and is the one that fixes in a second. All of this is bookkeeping on top of the same search.

Once a satellite is acquired, the coarse code phase and Doppler are handed to a tracking channel, which is the subject of the next page. Acquisition then moves on to the next PRN, and in a running receiver keeps cycling through the satellites that are not yet tracked, so that ones rising over the horizon are picked up.

---

**In pyrx:** `acquisition.py` implements the FFT search described here on a recording, and `pyrx-acq` runs it from the command line; adding `--plot DIR` also writes a C/N0 chart and a search-grid image for each detected satellite. It is checked against Pocket SDR's acquisition on the same recordings.

Next: [Staying locked](05-tracking.md).
