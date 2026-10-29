# Staying locked

> **Who this is for:** You know acquisition hands over a rough code phase and Doppler for each satellite. You want to know how the receiver keeps hold of the signal from then on and turns it into precise timing.

> **You will learn:**
> - Why tracking is a pair of feedback loops, one for the code and one for the carrier
> - What early, prompt, and late correlators are
> - How the carrier loop recovers the data bits and the carrier phase
> - What loop bandwidth trades off and what losing lock looks like

This page is conceptual. The series continues with [reading the message](06-navigation-message.md).

---

## Two things to follow

Acquisition gives a code phase good to half a chip (150 m) and a frequency good to a few hundred Hz. Both drift continuously: the satellite moves, the receiver moves, the oscillator wanders. Tracking is the process of continuously re-estimating both and steering the receiver's replica to follow. It runs for as long as the satellite is in view, which can be hours, and every measurement in the rest of this series comes out of it.

The receiver does this with two coupled feedback loops. Each one generates a replica, correlates it with the incoming samples over one code period (1 ms for L1 C/A), looks at the result to see which way the replica is off, and nudges it. The structure is the same as a phase-locked loop in any radio, and the vocabulary is borrowed from there.

## The code loop

The code loop keeps the replica code aligned with the received code. It cannot tell from a single correlation whether the replica is early or late, because correlation power is symmetric around the peak. So it runs three correlators at once: the prompt replica, one shifted half a chip early, and one shifted half a chip late.

```text
 correlation
   power          prompt
     ^              *
     |           *     *
     |        *           *
     |     *                 *
     |  *                       *
     +--+--------+--------+--------+--> replica offset
      early    prompt    late
             -1/2    0   +1/2 chip
```

If the replica is aligned, the early and late correlators are equal. If the replica is late, early is larger than late. The difference between them, normalised, is the error signal, and a filter turns it into a small correction to the replica's code rate. This is the delay-locked loop, the DLL, and the prompt correlator's alignment is the receiver's measurement of when the code arrived, which becomes the pseudorange. A well-designed DLL in the open sky tracks the code to about one hundredth of a chip, a few metres of raw precision before any averaging.

Narrowing the early-late spacing below half a chip sharpens the tracking and reduces the effect of multipath, if the front-end bandwidth is wide enough to preserve the sharp peak. This is one reason survey receivers want wide front-ends.

## The carrier loop

The carrier loop keeps the replica carrier in phase with the received one. Its error signal is the phase angle of the prompt correlator output: if the replica is exactly on frequency and phase, the correlation lands entirely on the I axis, and any Q component means a phase error. The filter adjusts the replica frequency to drive Q back to zero. This is a phase-locked loop, a PLL.

The data bits complicate it. Every 20 ms the data message may flip the sign of the signal, which looks like a sudden 180 degree phase jump. A plain PLL would try to chase it. The Costas loop, the standard answer, uses an error signal that is insensitive to sign, so a flip is ignored, and the sign of the I output after locking is the data bit. Tracking and demodulation come out of the same loop. On pilot signals, after removing any secondary code, a plain PLL with its stronger error signal can be used instead.

When the carrier loop is locked, two things are true. The I correlator output is a clean stream of data bits, which the next page decodes. And the accumulated phase of the replica carrier is a measurement of how the distance to the satellite has changed, cycle by cycle, with a precision of a few millimetres. That is the carrier-phase observable, and it is the basis of RTK.

Before the PLL can lock the frequency has to be within a few tens of Hz, which acquisition does not guarantee. Receivers therefore pull in with a frequency-locked loop first, which measures the change in phase between successive correlations and only needs to be in the right neighbourhood, and switch to the PLL once the frequency error is small.

## Loop bandwidth

Each loop's filter has a bandwidth, which sets how quickly it responds. A wide bandwidth follows fast changes, which a moving receiver or a poor oscillator needs, but lets more noise through into the estimate. A narrow bandwidth averages more noise away but loses lock if the signal moves faster than it can follow. Typical values are 1 Hz or less for the code loop, which can afford to be slow because the carrier loop can aid it, and 10 to 20 Hz for the carrier loop on a moving receiver. Choosing these is the main tuning decision in a receiver, and the right choice depends on the oscillator from the previous page.

## Losing lock

Under a bridge the signal disappears, the correlator outputs become pure noise, and the loops wander. The receiver watches the prompt correlator power against the noise estimate (its C/N0 estimate) and a lock detector built from the ratio of I and Q. When they fall below thresholds for long enough, the channel declares loss of lock and goes back to acquisition, now with a narrow search around its last known state.

The quiet version of this problem is the cycle slip. If the signal fades briefly or the receiver jerks, the carrier loop can lose a cycle or a half-cycle and relock on the next one, with the code loop barely noticing. Every measurement afterwards is off by one wavelength. For a metre-level fix it does not matter. For RTK it is the main enemy, and every RTK engine spends effort detecting slips so that it can restart the integer estimate for that satellite.

---

**In pyrx:** `tracking.py` holds the DLL (with early and late taps each a quarter chip from prompt, half a chip apart), the FLL pull-in and the Costas PLL, `bitsync.py` finds the data-bit boundaries (and later the frame) in the prompt output, and `pyrx-trk` runs tracking on a recording; adding `--plot DIR` writes images of each channel's correlator outputs and loop state. The tracker's C/N0, Doppler and code phase are checked against Pocket SDR's tracker on the same recordings.

Next: [Reading the message](06-navigation-message.md).
