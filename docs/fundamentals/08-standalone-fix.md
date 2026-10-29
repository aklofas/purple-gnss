# A standalone fix

> **Who this is for:** You know what a pseudorange is. You want to see how a set of them, and the broadcast message, becomes a latitude, longitude and height, and where the metres of error come from.

> **You will learn:**
> - How satellite positions and clocks are computed at the moment of transmission
> - The corrections applied to every pseudorange before solving
> - How four or more corrected pseudoranges become a position, by least squares
> - What geometry does to the result, and the error budget of a plain fix

This page is conceptual. The series continues with [RTK](09-rtk.md).

---

## The equation

For each satellite, the pseudorange equals the geometric distance from the satellite's position at transmission to the receiver's position at reception, plus the speed of light times the receiver's clock error, plus the delays on the path, minus the satellite's own clock error. The receiver knows the pseudorange, computes the satellite position and clock from the message, models the path delays, and is left with four unknowns: three coordinates and its clock error. With four satellites the system is solvable; with more it is over-determined and least squares finds the best position.

Everything on this page is about evaluating that equation honestly.

## Where was the satellite

The satellite position is needed at the instant the signal left it, which is the transmit time from the previous page, about 70 ms before reception. The ephemeris formulas give it in the Earth-centred, Earth-fixed frame. One subtlety: the Earth turned during those 70 ms, by about 5 microradians (some 30 m at the equator), and the frame turned with it, so the satellite's position has to be rotated back by that angle to put it in the frame of the receiver at reception time. At the satellite's distance that moves its coordinates by up to about 135 m, and forgetting it moves the fix by tens of metres.

The satellite clock correction is evaluated at the same time, with the relativistic eccentricity term, and the group delay for the signal in use.

## What is on the path

**The ionosphere**, from 60 to 1000 km up, is ionised by the Sun and delays the code by an amount proportional to the electron count along the path and inversely proportional to the frequency squared. On L1 it is 2 to 10 m at the zenith by day in mid-latitudes, more at low elevation, more near the equator, more in years of high solar activity, and almost nothing at night. A single-frequency receiver uses the Klobuchar model from the message, which removes about half. A dual-frequency receiver measures the delay from the difference of the two pseudoranges and removes almost all of it, and this is the main reason dual-frequency receivers exist.

**The troposphere**, the lowest 10 km of air, delays all frequencies equally, about 2.3 m at the zenith, ten times that at 5 degrees elevation. Ninety percent of it comes from the dry air and is predictable from pressure and height to a centimetre. The rest is water vapour and is not. The standard models (Saastamoinen with a mapping function, in this project) get the total to a few centimetres at high elevation and a few decimetres at low.

**Multipath** is not modelled. It is reduced by the antenna, by the correlator design, by weighting low satellites less, and by excluding satellites below an elevation mask, typically 10 to 15 degrees, altogether.

## Solving

The distance from receiver to satellite is not linear in the receiver's position, so the solver starts from a guess (the centre of the Earth works, from cold), linearises around it, solves the linear least-squares problem for the correction to the guess and the clock, and repeats. Three or four iterations converge from anywhere on the planet. Each pseudorange is weighted by its expected quality, usually a function of elevation and, in some receivers, C/N0.

The result is a position in Earth-centred coordinates, which is converted to latitude, longitude and ellipsoidal height, and a receiver clock error. The clock error is discarded by most users and is the whole point for timing users. A solver also checks the residuals: if the measurements do not fit the solution as well as they should, something is wrong (a bad satellite, a multipath reflection, a misdecoded ephemeris), and the usual approach is to drop the worst measurement and solve again.

The same solution, run on the Doppler observables, gives velocity and clock drift.

## Geometry

Where the satellites are in the sky matters as much as how good the measurements are. If they are bunched together, a small error in each range moves the solution a long way; if they are spread across the sky, the errors partly cancel. The dilution of precision, DOP, is the factor by which the geometry multiplies the range error into position error. A DOP of 1 is excellent, 2 to 3 is typical in the open, and 6 or more means a street canyon or a bad hour. Height is always worse than horizontal position, because all the satellites are above the receiver and none below, so there is nothing to average the vertical error against.

## The error budget

With a single frequency, the broadcast models, and an open sky, after the solver has done its best:

| Source | Typical remaining error |
|---|---|
| Broadcast orbit and clock | 1 m |
| Ionosphere, after the Klobuchar model | 2 to 5 m |
| Troposphere, after the model | 0.2 m |
| Multipath, open sky | 0.5 m |
| Receiver noise | 0.3 m |

Combined and multiplied by a DOP of 2, that is a few metres horizontally, which is what a phone or a car gets. Dual frequency takes the ionosphere line out and the result is about a metre. The remaining metre is mostly the orbit and clock and the troposphere, which are the same for any receiver within tens of kilometres, which is the opening the next page exploits.

---

**In pyrx:** `orbits.py` evaluates the ephemeris, `models.py` holds the Klobuchar and Saastamoinen corrections, and `pvt.py` is the weighted least-squares solver, with the same models and weights as RTKLIB's single-point mode so the two can be compared. `pyrx-pvt` runs it on RINEX files, and on a day of a public reference station it agrees with RTKLIB to the millimetre.

Next: [RTK](09-rtk.md).
