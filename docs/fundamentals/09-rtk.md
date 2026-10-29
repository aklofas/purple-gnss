# RTK

> **Who this is for:** You know how a standalone fix works and why it is limited to about a metre. You want to understand how two receivers and the carrier phase get to one centimetre.

> **You will learn:**
> - How differencing between receivers and satellites cancels the shared errors
> - Why the carrier-phase integer ambiguity is the central problem, and how it is resolved
> - What a float solution, a fixed solution and a cycle slip mean in practice
> - How RTK relates to PPK and PPP, and where the project's own engine fits

This page is conceptual and closes the series. The [glossary](glossary.md) follows.

---

## The idea

Put a second receiver, the base, at a point whose coordinates are known to the centimetre. It tracks the same satellites as the rover, through almost the same ionosphere and troposphere, using the same broadcast orbits and clocks. Every error that is common to the two is cancelled by taking the difference of their measurements. What survives is the difference in geometry, which is exactly the vector from base to rover, the baseline, plus the two receivers' own noise and their multipath.

Over a baseline of a few kilometres the atmospheric errors are nearly identical at both ends and cancel to centimetres. At 20 km the residual ionosphere is a few centimetres; at 50 km it starts to limit the result and the integer ambiguities become hard to resolve. This is why RTK has a range, and why networks of base stations exist to interpolate the corrections between them.

Real Time Kinematic means the base's measurements are sent to the rover as it moves, over a radio or the internet (where the streaming protocol is called NTRIP), in a format called RTCM, and the rover solves for the baseline at each epoch. Post-processed kinematic, PPK, is the same computation done afterwards from logged files. The maths does not know the difference, and the engine in this project is written against files for exactly that reason.

## Differencing

The receiver forms differences in two steps.

**Between receivers.** Rover minus base, for the same satellite and signal at the same epoch. The satellite's clock error cancels exactly. Its orbit error cancels almost entirely, since both receivers see the satellite from nearly the same direction. The atmospheric delays cancel to the extent they are the same at both ends.

**Between satellites.** The single difference for satellite j minus the single difference for a chosen reference satellite, usually the highest one. Now the receivers' clock errors cancel too, exactly, because each receiver's clock affects all its satellites equally. What is left of each double-differenced pseudorange is the double-differenced geometry, a function of the baseline only, plus noise.

```text
 single difference (rover - base)          removes satellite clock, most of orbit,
                                           most of atmosphere
 double difference (sat j - reference)     removes both receiver clocks
```

Double-differenced pseudoranges alone give the baseline to a few decimetres. The carrier phase is where the centimetres come from.

## The integer ambiguity

Double differencing the carrier phase cancels the same errors and leaves a double-differenced range, precise to millimetres, plus an unknown integer number of wavelengths per satellite pair. The integers are the double-differenced ambiguities. Everything in the single-receiver measurement that made the ambiguity awkward, the receiver clock, the satellite clock, the initial phase of either oscillator, has cancelled. What is left really is an integer, and that is the property RTK exploits.

The solution proceeds in two stages.

**Float.** Treat the ambiguities as unknown real numbers and solve for the baseline and all of them together, by least squares or a Kalman filter that accumulates epochs. The pseudoranges constrain the baseline coarsely; the carrier phases constrain its changes precisely; and as the satellites move across the sky over minutes the geometry changes enough to separate the baseline from the ambiguities. The float solution converges from decimetres towards centimetres over a few minutes, with real-valued ambiguities that are close to integers but not on them.

**Fixed.** The float ambiguities and their covariance matrix define a cloud of likely integer vectors. The task is to find the integer vector closest to the float one, measured in the metric of the covariance, which is a lattice search. The ambiguities are highly correlated, which makes the search badly conditioned, and the LAMBDA method's contribution is a decorrelating transformation that makes it fast. The search returns the best integer vector and the second best. If the best fits the data much better than the second, judged by the ratio of their residuals, usually a threshold of 3, the integers are accepted. The baseline is then recomputed with the ambiguities held at their integers, and it is good to a centimetre or two. If the ratio test fails the receiver reports a float solution and keeps trying next epoch.

Fixing is what the status display means by "RTK fixed" against "RTK float", and the time from startup to a fix is the main performance number of an RTK receiver. With dual frequency it is seconds; the difference of the L1 and L5 phases, called the wide lane, behaves like a signal of 75 cm wavelength, whose integer is easy to find and which then pins down the L1 and L5 integers in turn.

## What goes wrong

A cycle slip on one satellite invalidates its ambiguity, and the engine has to detect it (from the receiver's lock flag, or from a jump in the phase that the pseudorange does not show) and restart that ambiguity while keeping the rest. A satellite setting or a new one rising is handled the same way. A wrong fix, an integer vector accepted that was not the true one, is the worst outcome, because it gives a confidently wrong position by a multiple of a wavelength, and the ratio test and validation of the fixed residuals are the defences against it. Long baselines, low satellites, multipath and a poor oscillator all make fixing slower and wrong fixes likelier.

## Beyond RTK

Precise point positioning, PPP, drops the base station and instead uses precise orbits and clocks, good to centimetres, computed by a global network and either downloaded or, for Galileo, broadcast on E6 for free. Without a base there is nothing to difference the atmosphere against, so it has to be estimated, and convergence takes many minutes, to a decimetre or two. It is the planned fallback in this project for when no base is reachable. Satellite-based augmentation, SBAS, is the metre-level version of the same idea broadcast from geostationary satellites, usable with a single frequency.

The RTK engine in this project follows the single-baseline model of RTKLIB, the open-source reference, and is checked against it epoch by epoch on public station pairs. The intent is for that engine to run on the receiver's own processor on observations from the receiver's own correlators, with the base's observations arriving over a link, which closes the loop that this series opened with the signal leaving the satellite.

---

**In pyrx:** `rtk.py` is the engine (double differences, the Kalman float filter, cycle-slip handling and the ratio test) and `ambiguity.py` is the LAMBDA search. `pyrx-rtk` runs it on rover and base RINEX files. On six hours of two public station pairs, with GPS it fixes exactly the same epochs as RTKLIB and its fixed positions agree with RTKLIB's within 3 mm; they lie within a centimetre of the published marker coordinates on average and within 3 cm epoch by epoch.

Back to the [index](../index.md), or on to the [glossary](glossary.md).
