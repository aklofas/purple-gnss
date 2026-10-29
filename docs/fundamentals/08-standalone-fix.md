# A standalone fix

> **Who this is for:** You know what a pseudorange is. You want to see how a set of them, and the broadcast message, becomes a latitude, longitude and height, and where the meters of error come from.

> **You will learn:**
> - The pseudorange equation, and the four unknowns a fix solves for
> - How satellite positions and clocks are computed at the moment of transmission
> - The corrections applied to every pseudorange before solving
> - How linearization and weighted least squares turn four or more pseudoranges into a position, and how the residuals are checked
> - What dilution of precision says about geometry, and the error budget of a plain fix

This page is conceptual. The series continues with [RTK](09-rtk.md).

---

## The equation

A standalone fix is the position one receiver gets on its own, from its pseudoranges and the broadcast message, with no base station and no correction service. It is what a phone or a car navigation unit computes, and in an RTK receiver it is also the starting point for the precise solution of the next page.

The idea is trilateration. If the receiver's clock were perfect and the signal crossed a vacuum, each pseudorange would be the exact distance to a satellite whose position is known, so the receiver would sit on a sphere around each satellite. Three spheres meet in two points, and only one of them is anywhere near the Earth. Two things spoil that picture. The receiver's clock is off by an unknown amount, which lengthens or shortens every pseudorange by the same distance, and the signal is delayed on its way down. The first is handled by solving for the clock as a fourth unknown, the second by modeling the delays.

Write $P^s$ for the pseudorange to satellite $s$, $\rho^s$ for the geometric distance from the satellite's position at transmission to the receiver's position at reception, $\mathrm{dt}_r$ for the receiver clock error and $\mathrm{dt}^s$ for the satellite's, both in seconds, $c$ for the speed of light, $I^s$ and $T^s$ for the ionospheric and tropospheric delays in meters, and $\eta^s$ for whatever is left: noise, multipath and model error. Then

$$
P^s = \rho^s + c(\mathrm{dt}_r - \mathrm{dt}^s) + I^s + T^s + \eta^s .
$$

A receiver clock that runs ahead makes every pseudorange longer, and a satellite clock that runs ahead makes its own pseudorange shorter, hence the opposite signs. The receiver knows $P^s$, computes the satellite position and $\mathrm{dt}^s$ from the message, models $I^s$ and $T^s$, and is left with four unknowns: the three coordinates hidden in $\rho^s$ and the clock term $c\mathrm{dt}_r$, kept in meters. The clock term is large: a receiver clock 1 µs off adds about 300 m to every pseudorange. With four satellites the system is solvable; with more it is over-determined, and least squares finds the position that fits all of them best.

The rest of this page evaluates each term of that equation honestly and then solves it.

## Where was the satellite

The satellite position is needed at the instant the signal left it, which is the transmit time from [reading the message](06-navigation-message.md#time-of-week), about 70 ms before reception. The receiver gets it from the reception time minus the pseudorange divided by $c$, corrected by the satellite clock. The ephemeris formulas then give the position in the Earth-centered, Earth-fixed frame, ECEF, whose axes turn with the Earth.

One subtlety follows from that rotation. The Earth turned during those 70 ms, by about 5 microradians (some 30 m at the equator), and the frame turned with it, so the satellite's position has to be rotated back by that angle to put it in the frame of the receiver at reception time. At the satellite's distance that moves its coordinates by up to about 135 m, and forgetting it moves the fix by tens of meters. The same correction is often written as a small extra term in the range, the Sagnac correction, which is equivalent.

The satellite clock correction $\mathrm{dt}^s$ is evaluated at the same time. It is the broadcast clock polynomial plus a relativistic term that depends on where the satellite is on its slightly elliptical orbit, and it is adjusted by the broadcast group delay for the signal in use, because the polynomial is defined for a combination of two frequencies and a single-frequency receiver sees a slightly different satellite delay.

## What is on the path

**The ionosphere**, from 60 to 1000 km up, is ionized by the Sun and delays the code by an amount proportional to the electron count along the path and inversely proportional to the frequency squared. On L1 it is 2 to 10 m at the zenith by day in mid-latitudes, more at low elevation, more near the equator, more in years of high solar activity, and smaller at night, though still around a meter. A single-frequency receiver uses the Klobuchar model, whose eight coefficients come in the message: it treats the ionosphere as a thin shell with a daily cycle of delay, and is specified to cut the root-mean-square ionospheric error by at least half; on any one path the improvement varies. A dual-frequency receiver measures the delay from the difference of the two pseudoranges and removes almost all of it, and this is the main reason dual-frequency receivers exist.

**The troposphere**, the lowest 10 km of air, delays all frequencies equally, about 2.3 m at the zenith, ten times that at 5 degrees elevation. Ninety percent of it comes from the dry air and is predictable from pressure and height to a centimeter. The rest is water vapor and is not. A model gives the delay at the zenith, and a mapping function stretches it to the satellite's elevation, the simplest one being $1/\sin\varepsilon$ for elevation $\varepsilon$: at 30 degrees the path through the air is twice as long as overhead. In this project the standalone fix uses the Saastamoinen model with a standard atmosphere and the $1/\sin\varepsilon$ mapping, good to a decimeter or two at the zenith and worse at low elevation; the RTK engine maps with the Niell function.

**Multipath** is not modeled. It is reduced by the antenna, by the correlator design, by weighting low satellites less, and by excluding satellites below an elevation mask, typically 10 to 15 degrees, altogether.

## Solving

The distance from receiver to satellite is not linear in the receiver's position, so the solver cannot solve the equations in one step. It uses the standard trick for nonlinear fitting: start from a guess (the center of the Earth works, from cold), replace each equation by its straight-line approximation around the guess, solve the resulting linear problem for a correction, apply it and repeat. This is linearization, and the repeated solve is the Gauss-Newton method.

Collect the unknowns in $\mathbf{q} = (x, y, z, c\mathrm{dt}_r)$. At the current guess the solver computes, for each satellite, the residual $v^s$, the measured pseudorange minus the one the equation predicts at the guess. It also needs to know how each predicted pseudorange changes when the unknowns change. With $\mathbf{e}^s$ the unit vector from the receiver toward satellite $s$, moving the receiver a small step $\delta\mathbf{x}$ changes the range by $-\mathbf{e}^s \cdot \delta\mathbf{x}$ (moving toward a satellite shortens its range), and raising the clock term by one meter lengthens every pseudorange by one meter. Each satellite therefore contributes one row

$$
H^s = \begin{bmatrix} -e^s_x & -e^s_y & -e^s_z & 1 \end{bmatrix}
$$

to a matrix $H$ with one row per satellite and one column per unknown. $H$ is the design matrix; since its entries are the partial derivatives of the predicted pseudoranges with respect to the unknowns, it is also the Jacobian of the measurement model. The small correction $\delta\mathbf{q}$ to the guess then satisfies, approximately, $\mathbf{v} = H\delta\mathbf{q}$.

With more than four satellites there are more equations than unknowns, and noise means no $\delta\mathbf{q}$ satisfies them all. Least squares picks the one that makes the sum of squared mismatches smallest. Weighted least squares divides each mismatch by $\sigma^s$, the expected error of that pseudorange, so a doubtful measurement counts for less:

$$
\delta\mathbf{q} = \arg\min_{\mathbf{d}} \sum_s \left(\frac{v^s - H^s\mathbf{d}}{\sigma^s}\right)^2
= (H^\mathsf{T} W H)^{-1} H^\mathsf{T} W \mathbf{v}, \qquad W = \mathrm{diag}\left(\frac{1}{(\sigma^s)^2}\right).
$$

The middle expression says what is minimized; the right one is its solution, from the normal equations. The weights are usually a function of elevation, because low satellites suffer more atmosphere and multipath, and in some receivers of C/N0. Four or five iterations converge from anywhere on the planet.

The result is a position in Earth-centered coordinates, which is converted to latitude, longitude and ellipsoidal height, and a receiver clock error. The clock error is discarded by most users and is the whole point for timing users.

A solver also checks the residuals left at the solution. If the measurements fit as well as their assumed errors predict, the sum of the squared normalized residuals $\sum_s (v^s/\sigma^s)^2$ follows a chi-square distribution with $n - 4$ degrees of freedom for $n$ satellites, and a sum far above that distribution's typical range means something is wrong: a bad satellite, a multipath reflection, a misdecoded ephemeris. The usual approach is to drop the worst measurement and solve again. A receiver that runs this kind of consistency check on its own is said to do receiver autonomous integrity monitoring, RAIM.

The same machinery gives velocity and clock drift from the Doppler observables. Each Doppler $f_D$ is first turned into a range rate in meters per second by multiplying by $-\lambda$, as on the [measurements](07-measurements.md) page; the model then includes the satellite's own velocity and clock drift, and the unknowns are the receiver's three velocity components and its clock drift, with the same design matrix.

Each epoch can be solved on its own like this, from its own measurements alone. Many receivers instead run a Kalman filter, which carries the position and clock from one epoch to the next with a model of how they change and blends that prediction with each new set of measurements; the [RTK](09-rtk.md) page describes one. It smooths the output, at the cost of trusting the motion model.

## Geometry

Where the satellites are in the sky matters as much as how good the measurements are. If they are bunched together, the spheres cross at shallow angles and a small error in each range moves the solution a long way; if they are spread across the sky, the errors partly cancel. The dilution of precision, DOP, is the factor by which the geometry multiplies the range error into position error.

It falls out of the least-squares solution. If every pseudorange has the same error $\sigma$, independent of the others, the covariance of the estimate, the matrix whose diagonal holds the variance of each unknown and whose other entries say how their errors move together, is

$$
\mathrm{Cov}(\mathbf{q}) = \sigma^2 (H^\mathsf{T} H)^{-1} .
$$

$(H^\mathsf{T} H)^{-1}$ depends only on the directions to the satellites. The square root of its trace, the sum of its diagonal, is the geometric DOP, GDOP, which counts the clock as well as the position. Its parts have their own names: PDOP for the three coordinates, HDOP and VDOP for the horizontal and vertical parts once the position is turned into local east, north and up, and TDOP for the clock. A position error is then roughly the DOP times the range error: with 1 m of range error and a PDOP of 2, the position is uncertain by about 2 m.

In the open an HDOP near 1 and a PDOP of 1.5 to 3 are typical; a PDOP of 6 or more means a street canyon or a bad hour. Height is usually worse than horizontal position, VDOP larger than HDOP, because all the satellites are above the receiver and none below, so there is nothing to average the vertical error against, and a change of height looks partly like a change of the clock term.

## The error budget

With a single frequency, the broadcast models, and an open sky, after the solver has done its best:

| Source | Typical remaining error |
|---|---|
| Broadcast orbit and clock | 1 m |
| Ionosphere, after the Klobuchar model | 2 to 5 m |
| Troposphere, after the model | 0.2 m |
| Multipath, open sky | 0.5 m |
| Receiver noise | 0.3 m |

Independent errors add as the square root of the sum of their squares, so the lines combine to between about 2.3 and 5 m of range error, the user equivalent range error, UERE. Multiplied by an HDOP near 1, typical of an open sky, that is a few meters horizontally, which is what a phone or a car gets, and height is worse by the larger VDOP. Dual frequency takes the ionosphere line out and the range error drops to one or two meters, since the ionosphere-free combination amplifies noise and multipath about threefold. Of what remains, the orbit, clock and troposphere errors are shared by receivers tens of kilometers apart, and that is the opening the next page exploits.

---

**In pyrx:** `orbits.py` evaluates the ephemeris, `models.py` holds the Klobuchar and Saastamoinen corrections, and `pvt.py` is the weighted least-squares solver, with the same models, weights, 15 degree elevation mask and chi-square test as RTKLIB's single-point mode so the two can be compared; it applies the Earth's rotation as the Sagnac term. An epoch that fails the chi-square test gets no fix, with no second solve without the worst satellite, and so does one with a GDOP above 30. `pyrx-pvt` runs it on RINEX files, and on a day of a public reference station it agrees with RTKLIB to about a millimeter on average, with a handful of epochs up to 0.5 m apart. Sections 3 and 4 of the [standalone-positioning notebook](../../pyrx/notebooks/02-standalone-pvt.ipynb) work through the models and the solve on three hours of station data.

Next: [RTK](09-rtk.md).
