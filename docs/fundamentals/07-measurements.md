# The measurements

> **Who this is for:** You know how the receiver tracks each satellite and reads its message. You want to know exactly what numbers come out of a receiver, what each one means physically, and what is good and bad about each.

> **You will learn:**
> - What a pseudorange is, how it is formed, and what "pseudo" means
> - What the carrier phase measures and why it is precise but carries an integer ambiguity
> - What Doppler gives and what C/N0 is for
> - How the measurements are flagged and shared between receivers, which is where RINEX comes in

This page is conceptual. The series continues with [a standalone fix](08-standalone-fix.md).

---

## Observables

Receiver people call the measurements observables. For each satellite and each signal, at each measurement epoch (typically once a second, sometimes ten times), a receiver produces up to four of them: a pseudorange, a carrier phase, a Doppler, and a signal strength. The epoch is a single instant on the receiver's clock at which every tracking channel is read out together, so that all the measurements of one epoch share that clock. The first three observables are all measurements of the same thing, the receiver's distance to the satellite and how it changes, made in different ways with different strengths. The rest of this series is about combining them.

The equations below use one set of symbols. For a receiver $r$ and a satellite $s$, $\rho$ is the true geometric distance between them, $c$ the speed of light, $\mathrm{dt}_r$ and $\mathrm{dt}^s$ the receiver's and the satellite's clock errors in seconds, $I$ and $T$ the delays the ionosphere and the troposphere add, in meters, and $\eta$ whatever the model leaves out: noise, multipath and small biases. Every equation on this page is for one receiver and one satellite, so the indices $r$ and $s$ are dropped from $P$, $\Phi$, $\rho$, $I$, $T$ and $N$.

## Pseudorange

The pseudorange is the apparent distance to the satellite: the signal's travel time multiplied by the speed of light. The travel time is the difference between two clock readings, the satellite's at transmission and the receiver's at reception:

$$
P = c (t_{\rm rx} - t_{\rm tx}) .
$$

The transmit time $t_{\rm tx}$ comes from the previous page. For a chosen receiver epoch, the receiver finds, for each satellite, the point in the code it was receiving at that instant, and knows from the decoded time of week when, by the satellite's clock, it transmitted that point (the next page corrects that clock to GPS time). The receive time $t_{\rm rx}$ is the receiver's own clock, which is the problem. Its clock is wrong by an unknown amount, so every pseudorange is wrong by the same unknown amount, and the travel time of 65 to 85 milliseconds comes out as something like 65 to 85 milliseconds plus or minus the receiver's error. A clock error of one microsecond is already 300 m. That shared error is why it is called a pseudo-range, and the next page solves for it.

Collecting everything that goes into it, the pseudorange is modeled as

$$
P = \rho + c (\mathrm{dt}_r - \mathrm{dt}^s) + I + T + \eta_P .
$$

Each term is a distance in meters. The position is hidden in $\rho$, the receiver clock term is the unknown just described, the satellite clock term is known from the message, and the two atmospheric delays are estimated from models, as the next page explains.

A pseudorange from the L1 C/A code is noisy at the level of a few decimeters to a meter, set by the 293 m chip and the DLL's precision. It is unambiguous: the receiver knows the full distance, with no missing multiples of anything. It is also corrupted by every delay on the path: the ionosphere adds meters, the troposphere adds meters, multipath adds up to tens of meters. The pseudorange is the honest workhorse. Every fix starts from it.

## Carrier phase

The carrier phase is the number of carrier cycles between the satellite and the receiver, as measured by the tracking loop's accumulated replica phase. It is recorded so that it grows as the distance grows. It is measured to about one hundredth of a cycle, two millimeters on L1, a hundred times or more better than the pseudorange. The ionosphere affects it by the same amount as the pseudorange but with the opposite sign (it advances the phase while delaying the code), which turns out to be useful. And multipath affects it far less, a few centimeters at worst.

The price is that the integer part is unknown. When the tracking loop locks on, it starts counting cycles from zero, and the true count at that moment is not observable. With the L1 wavelength $\lambda = c/f_{L1} \approx 0.19$ m and a satellite some 20,000 km away, that count is around 100 million cycles. From then on the receiver counts every cycle gained and lost, so the phase measures changes in distance perfectly while the absolute distance is off by an unknown integer number of wavelengths. That integer $N$ is the integer ambiguity. It is constant for as long as lock holds, and it changes if the loop slips a cycle.

Multiplying the phase $\varphi$, in cycles, by the wavelength gives a length $\Phi = \lambda\varphi$ in meters, with a model that mirrors the pseudorange's:

$$
\Phi = \rho + c (\mathrm{dt}_r - \mathrm{dt}^s) - I + T + \lambda N + \eta_\Phi .
$$

The ionosphere term has flipped sign, the ambiguity $\lambda N$ has appeared, and $\eta_\Phi$ is a hundred times or more smaller than $\eta_P$. RTK is the business of finding the integers $N$. Until they are found, the carrier phase is a very precise measurement of how far the receiver has moved, not of where it is.

```text
                    pseudorange              carrier phase
  precision         0.3 to 1 m               2 mm
  ambiguity         none                     unknown integer cycles
  ionosphere        delays                   advances, same size
  multipath         up to tens of meters     a few centimeters
  survives a fade   yes                      ambiguity has to be re-solved
```

## Doppler

The Doppler observable $f_D$ is the frequency offset the carrier loop is currently applying, in Hz, positive while the satellite approaches. Since the phase grows with distance, the Doppler is minus the rate of change of the phase, and multiplied by minus the carrier wavelength it becomes the rate of change of distance to the satellite, plus the difference of the receiver's and the satellite's clock drifts, in meters per second:

$$
-\lambda f_D = \dot\rho + c (\dot{\mathrm{dt}}_r - \dot{\mathrm{dt}}^s) + \eta_D ,
$$

where a dot means the rate of change. One hertz of Doppler on L1 is 0.19 m/s, so the 5 kHz of the largest Doppler shifts is about 950 m/s along the line of sight.

The range rate $\dot\rho$ depends on the receiver's velocity in the same way the range depends on its position. With four satellites the solution is three velocity components and the clock drift, by exactly the same argument as for position. Velocity from Doppler is accurate to a few centimeters per second and needs no ambiguity resolution, so it is available instantly and is what navigation displays show as speed and heading.

## C/N0

The signal strength is reported as the carrier-to-noise-density ratio in dB-Hz, estimated from the correlator outputs. It is not used in the position solution directly, but many receivers use it in weighting: a solver that weights its measurements gives each one a weight according to how much it trusts it, and a measurement at 30 dB-Hz is noisier and more likely to be multipath than one at 48 dB-Hz, a signal 18 dB, or 63 times, weaker against the same noise. Such a solver down-weights it. C/N0 is also the first thing a person looks at when a receiver misbehaves.

## Loss-of-lock flags and RINEX

Along with each carrier phase the receiver reports whether lock was continuous since the previous epoch, in the loss-of-lock indicator, LLI. A slip or a reacquisition is flagged, and any user of the carrier phase has to reset its ambiguity for that satellite when it sees the flag. A Costas loop can also be half a cycle out while the data polarity is unknown, or slip by half a cycle; a separate flag marks those phases, and an RTK engine keeps them out of integer fixing until the flag clears, and then restarts that ambiguity, since a change of the half-cycle flag counts as a slip. RINEX stores the two flags as bits of one digit beside the phase: 1 for lost lock, 2 for a possible half cycle, and 3 for both.

Observables are exchanged between receivers and programs in a text format called RINEX (Receiver Independent Exchange Format). An observation file holds, per epoch, one line per satellite with its pseudorange, carrier phase, Doppler and signal strength per signal, with the lock flags. Each column is named by a three-character code: the observable (C for code pseudorange, L for carrier phase, D for Doppler, S for signal strength), the band (1 for L1) and the signal (C for C/A), so an L1 C/A receiver writes `C1C L1C D1C S1C`. A separate navigation file holds the decoded ephemerides. Public reference stations publish both, every day, for the whole world, which is what makes it possible to develop and test RTK without owning two receivers. Everything in the next two pages is computed from RINEX files, and the `pyrx` receiver produces them from recordings.

---

**In pyrx:** `observables.py` forms pseudoranges and carrier phases from the tracker's state at each epoch and `rinex.py` reads and writes the files. `pyrx-rinex` runs the whole signal chain on a recording and writes observation and navigation files with `C1C L1C D1C S1C` for every channel. On two recordings, a test checks the transmit times against Pocket SDR's C tracker, to within 20 ns, and checks RTKLIB's fix from these files against RTKLIB's fix from Pocket SDR's observations of the 2021 recording: within 5 m horizontally and 10 m vertically at every epoch the two share, and in the mean for the 2022 recording, made at the same site, for which Pocket SDR forms no observations. Sections 5 and 6 of the [signal-chain notebook](../../pyrx/notebooks/01-signal-chain.ipynb) form these measurements from a real recording and write them to RINEX.

Next: [A standalone fix](08-standalone-fix.md).
