# RTK

> **Who this is for:** You know how a standalone fix works and why it is limited to about a meter. You want to understand how two receivers and the carrier phase get to one centimeter.

> **You will learn:**
> - How single and double differences between receivers and satellites cancel the shared errors
> - Why the carrier-phase integer ambiguity is the central problem
> - How a Kalman filter gives the float solution, and how LAMBDA and the ratio test turn it into a fixed one
> - What a float solution, a fixed solution and a cycle slip mean in practice
> - How RTK relates to PPK and PPP, and where the project's own engine fits

This page is conceptual and closes the series. The [glossary](glossary.md) follows.

---

## The idea

RTK is the technique that takes a receiver from meters to centimeters. It needs two things the standalone fix does not use: a second receiver nearby, and the carrier phase.

Put a second receiver, the base, at a point whose coordinates are known to the centimeter. It tracks the same satellites as the rover, the receiver whose position is wanted, through almost the same ionosphere and troposphere, using the same broadcast orbits and clocks. Every error that is common to the two is canceled by taking the difference of their measurements. What survives is the difference in geometry, which depends on the vector from base to rover, the baseline, plus the two receivers' own noise and their multipath.

Over a baseline of a few kilometers the atmospheric errors are nearly identical at both ends and cancel to centimeters. At 20 km the residual ionosphere is a few centimeters; at 50 km it starts to limit the result and the integer ambiguities become hard to resolve. This is why RTK has a range, and why networks of base stations exist to interpolate the corrections between them.

Real Time Kinematic means the base's measurements are sent to the rover as it moves, over a radio or the internet (where the streaming protocol is called NTRIP), in a format called RTCM, and the rover solves for the baseline at each epoch. Post-processed kinematic, PPK, is the same computation done afterwards from logged files. The math does not know the difference, and the engine in this project is written against files for exactly that reason.

## Differencing

Start from the two measurement equations of the previous pages, for receiver $r$, satellite $s$ and one signal of wavelength $\lambda$. The pseudorange $P$ and the carrier phase expressed in meters, $\Phi$, are

$$
P_r^s = \rho_r^s + c(\mathrm{dt}_r - \mathrm{dt}^s) + I_r^s + T_r^s + \eta_P ,
$$

$$
\Phi_r^s = \rho_r^s + c(\mathrm{dt}_r - \mathrm{dt}^s) - I_r^s + T_r^s + \lambda N_r^s + \eta_\Phi .
$$

$\rho$ is the geometric range, $`\mathrm{dt}_r`$ and $\mathrm{dt}^s$ the receiver and satellite clock errors, $I$ and $T$ the ionospheric and tropospheric delays, and $\eta$ the noise and multipath. The ionosphere advances the carrier phase by the amount it delays the code, hence its minus sign in the second line. $N$ is the unknown whole number of cycles in the phase measurement, the ambiguity. The phase noise $`\eta_\Phi`$ is a millimeter or two, against a meter or so for the code.

The receiver forms differences in two steps, using the [notation](../../pyrx/notebooks/README.md#notation) of the project's notebooks: $\Delta$ for rover minus base, $\nabla\Delta$ for that, satellite minus reference satellite.

**Between receivers.** The single difference is rover minus base, for the same satellite and signal at the same epoch, $\Delta\Phi^s = \Phi_r^s - \Phi_b^s$. The satellite's clock error cancels exactly. Its orbit error cancels almost entirely, since both receivers see the satellite from nearly the same direction. The atmospheric delays cancel to the extent they are the same at both ends. Leaving out the small atmospheric difference that remains on a short baseline, what is left is

$$
\Delta\Phi^s = \Delta\rho^s + c(\mathrm{dt}_r - \mathrm{dt}_b) + \lambda\Delta N^s + \Delta\eta_\Phi ,
$$

with the two receivers' clock difference still in it.

**Between satellites.** The double difference is the single difference for satellite $s$ minus the single difference for a chosen reference satellite $q$, usually the highest one, $\nabla\Delta\Phi^{s,q} = \Delta\Phi^s - \Delta\Phi^q$. Now the receivers' clock errors cancel too, exactly, because each receiver's clock affects all its satellites equally:

$$
\nabla\Delta\Phi^{s,q} = \nabla\Delta\rho^{s,q} + \lambda\nabla\Delta N^{s,q} + \nabla\Delta\eta_\Phi .
$$

Set the ambiguity term aside until the next section. What is left, in a double-differenced pseudorange as in this phase equation, is the double-differenced geometry, a function of the baseline only, plus noise. For a short baseline $\mathbf{d}$, with $\mathbf{e}^s$ the unit vector toward satellite $s$, that geometry is close to $(\mathbf{e}^q - \mathbf{e}^s) \cdot \mathbf{d}$: the baseline projected onto the difference of two directions in the sky. As the satellites move, those directions turn, and that is what lets the measurements pin down all three components of $\mathbf{d}$.

```text
 single difference (rover - base)          removes satellite clock, most of orbit,
                                           most of atmosphere
 double difference (sat s - reference)     removes both receiver clocks
```

Each double difference combines four measurements, so its noise is larger than that of one: four equal, independent errors of 3 mm add up to 6 mm. Double-differenced pseudoranges alone give the baseline to a few decimeters. The carrier phase is where the centimeters come from.

## The integer ambiguity

Double differencing the carrier phase cancels the same errors and leaves a double-differenced range, precise to millimeters, plus an unknown integer number of wavelengths per satellite pair, the term $\lambda\nabla\Delta N^{s,q}$ above. The integers $\nabla\Delta N^{s,q}$ are the double-differenced integer ambiguities. Everything in the single-receiver measurement that made the ambiguity awkward, the receiver clock, the satellite clock, the initial phase of either oscillator, has canceled. What is left really is an integer, and that is the property RTK exploits. On L1 one cycle is about 19 cm, so getting the integer right is the difference between a centimeter answer and one off by decimeters.

The solution proceeds in two stages.

**Float.** Treat the ambiguities as unknown real numbers and solve for the baseline and all of them together. The tool for this is a Kalman filter, a recursive form of least squares that accumulates epochs. It carries a state vector $\mathbf{x}$, here the rover position and one ambiguity per satellite and signal, together with its covariance $\mathbf{P}$, set in bold to keep it apart from the pseudorange: the matrix whose diagonal holds each state's variance and whose other entries say how the errors of different states move together. At each epoch it does two things. The prediction step carries the state forward and grows $\mathbf{P}$ by the process noise, the uncertainty the model allows to creep in between epochs: a little for the ambiguities, which should stay constant, and a lot for the position of a moving rover. The update step then corrects the state with the epoch's double differences. With $H$ the design matrix of the double differences, $R$ their covariance and $\mathbf{v}$ the innovations, measured minus predicted,

$$
K = \mathbf{P} H^\mathsf{T} (H \mathbf{P} H^\mathsf{T} + R)^{-1}, \qquad \mathbf{x} \leftarrow \mathbf{x} + K\mathbf{v}, \qquad \mathbf{P} \leftarrow (\mathbf{I} - K H) \mathbf{P} .
$$

Here $\mathbf{I}$ is the identity matrix, not the ionosphere. $K$, the Kalman gain, says how far each innovation should move each state, given how uncertain the state and the measurement are. The pseudoranges constrain the baseline coarsely; the carrier phases constrain its changes precisely; and as the satellites move across the sky over minutes the geometry changes enough to separate the baseline from the ambiguities. The float solution converges from decimeters toward centimeters over a few minutes, with real-valued ambiguities that are close to integers but not on them.

**Fixed.** The float ambiguities $\hat{\mathbf{a}}$ and their covariance matrix $Q$ define a cloud of likely integer vectors. The task is to find the integer vector $\mathbf{n}$ closest to the float one, measured in the metric of the covariance, which means minimizing

$$
s(\mathbf{n}) = (\hat{\mathbf{a}} - \mathbf{n})^\mathsf{T} Q^{-1} (\hat{\mathbf{a}} - \mathbf{n})
$$

over all integer vectors. This is integer least squares, a search over a lattice. Rounding each ambiguity on its own ignores $Q$, and the ambiguities are highly correlated (they share the reference satellite and the position), which makes the search badly conditioned: the cloud is a long, thin, tilted ellipsoid. The LAMBDA method, the least-squares ambiguity decorrelation adjustment, contributes a decorrelating transformation, an integer matrix $Z$ whose inverse is also an integer matrix, so that integer vectors map to integer vectors and $s$ is unchanged. In the new coordinates the ellipsoid is much rounder and a depth-first search finds the best candidates quickly; the answer is transformed back.

The search returns the best integer vector and the second best, with distances $s_1 \le s_2$. If the best fits the data much better than the second, judged by the ratio test, $s_2/s_1$, usually against a threshold of 3, the integers are accepted. For example, with $s_1 = 2$ and $s_2 = 9$ the ratio is 4.5 and the fix is accepted; with $s_2 = 5$ it is 2.5 and the epoch stays float. The ratio measures how clearly the best candidate beats its nearest rival under the assumed noise; it is not a probability that the fix is right. The baseline is then recomputed with the ambiguities held at their integers, and it is good to a centimeter or two. If the ratio test fails the receiver reports a float solution and keeps trying next epoch.

Fixing is what the status display means by "RTK fixed" against "RTK float", and the time from startup to a fix is the main performance number of an RTK receiver. With dual frequency it is seconds; the difference of the L1 and L5 phases, called the wide lane, behaves like a signal of wavelength $c/(f_{L1} - f_{L5})$, which is 75 cm for 1575.42 and 1176.45 MHz. Its integer is easy to find, and with it fixed the L1 integer follows from one further combination, such as the ionosphere-free one.

## What goes wrong

A cycle slip on one satellite invalidates its ambiguity, and the engine has to detect it and restart that ambiguity while keeping the rest. It can detect a slip from the receiver's lock flag, from a jump in the phase that the pseudorange does not show, or, with two frequencies, from a jump in the geometry-free combination, the difference of the two phases in meters, in which the range and clocks cancel and only the slowly changing ionosphere and the ambiguities remain. A satellite setting or a new one rising is handled the same way. A wrong fix, an integer vector accepted that was not the true one, is the worst outcome, because it gives a confidently wrong position, off by an amount set by the wrong integers and the satellite geometry, and the ratio test and validation of the fixed residuals are the defenses against it. Long baselines, low satellites, multipath and a poor oscillator all make fixing slower and wrong fixes likelier.

## Beyond RTK

Precise point positioning, PPP, drops the base station and instead uses precise orbits and clocks, good to centimeters, computed by a global network and either downloaded or, for Galileo, broadcast on E6 for free. Without a base there is nothing to difference the atmosphere against, so it has to be estimated, and convergence takes many minutes, to a decimeter or two. Satellite-based augmentation, SBAS, is the meter-level version of the same idea broadcast from geostationary satellites, usable with a single frequency.

The RTK engine in this project follows the single-baseline model of RTKLIB, the open-source reference, and is checked against it epoch by epoch on public station pairs.

---

**In pyrx:** `rtk.py` is the engine (double differences, the Kalman float filter, cycle-slip handling and the ratio test) and `ambiguity.py` is the LAMBDA search. Each kinematic epoch starts from the standalone fix of the previous page; the filter's ambiguities are single differences, and the double differences are formed from them for the search. All signals go into one joint search with a threshold of 3, with no separate wide-lane stage and no feeding of the integers back into the filter. `pyrx-rtk` runs it on rover and base RINEX files. On six hours of two public station pairs, with GPS it fixes exactly the same epochs as RTKLIB and its fixed positions agree with RTKLIB's within 3 mm; they lie within a centimeter of the published marker coordinates on average and within 3 cm epoch by epoch. The [RTK notebook](../../pyrx/notebooks/03-rtk.ipynb) works through differencing, the float filter and the integer search on a real station pair.

Back to the [index](../index.md), or on to the [glossary](glossary.md).
