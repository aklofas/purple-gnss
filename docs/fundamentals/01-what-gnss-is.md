# What a GNSS is

> **Who this is for:** You know what GPS does from using it, and you want to understand what is actually happening between the satellites and your position.

> **You will learn:**
> - What the constellations are and where the satellites fly
> - Why the whole system is really a very precise clock
> - What a pseudorange is, and why a fix needs four satellites, not three
> - What limits the accuracy of a plain fix, and why RTK can do a hundred times better

This page is conceptual. There is no code here. The series continues with [the signal](02-the-signal.md).

---

## A system of clocks in orbit

GNSS stands for Global Navigation Satellite System. GPS is the American one. Galileo is the European one, GLONASS the Russian one, BeiDou the Chinese one. They work the same way, and a modern receiver uses all of them at once, so "GNSS" is the general term and "GPS" is one member of the family.

Each satellite carries an atomic clock and continuously broadcasts two things: the exact time by its clock, and a description of its own orbit precise enough to compute where it is at any instant. That is the whole broadcast. The satellites do not know where you are, and they never hear from you.

A receiver listens to several satellites and measures each signal's time of flight, the time it took to travel from the satellite to the antenna. Multiplying by the speed of light gives the distance to each one. A GPS satellite overhead is about 20,200 km away, so its signal arrives after about 67 ms; one near the horizon is about 25,800 km away and its signal takes about 86 ms. Distances to points whose positions are known give a position, the same geometry a surveyor uses, called trilateration. Everything else in this series is about doing those two steps well: measuring the travel time, and knowing where the satellite was when it sent the signal.

## Where the satellites are

| System  | Design constellation                                              | Altitude  | Orbit period |
| ------- | ----------------------------------------------------------------- | --------- | ------------ |
| GPS     | 24 or more, in six planes                                         | 20,200 km | 11 h 58 min  |
| Galileo | 24 plus 6 spares, in three planes                                 | 23,200 km | 14 h 05 min  |
| GLONASS | 24, in three planes                                               | 19,100 km | 11 h 16 min  |
| BeiDou  | medium Earth orbit plus geostationary and inclined geosynchronous | 21,500 km | 12 h 53 min  |

The table gives the size each constellation is designed around. The number actually in service changes as satellites are launched and retired, and each operator publishes a constellation status page with the current count.

The altitudes listed are for medium Earth orbits, far above the International Space Station (400 km) and far below geostationary satellites (36,000 km). BeiDou also has geostationary and inclined geosynchronous satellites at about 36,000 km. A GPS satellite crosses the sky in a few hours and the pattern repeats every sidereal day. From any open spot on Earth you can usually see 8 to 12 GPS satellites, and 25 to 40 satellites across all four systems.

The medium Earth orbits are nearly circular and spread over several planes for even coverage. Their inclinations to the equator are about 55 degrees for GPS and BeiDou, 56 degrees for Galileo, and 65 degrees for GLONASS. A satellite's orbit is predictable to within a meter or two over a few hours from the parameters it broadcasts, which is what the receiver relies on.

## Time is the product

A radio signal travels 30 cm in a nanosecond. To measure a distance to a meter, the receiver has to measure a travel time to about 3 nanoseconds. The satellite clocks are good enough for that. The receiver's clock is not: it is an ordinary crystal oscillator, wrong by microseconds and drifting all the time. A clock error of 1 microsecond, small by everyday standards, is 300 m of range.

The trick is that the receiver's clock error is the same for every satellite it listens to at a given instant. The receiver therefore accepts distances that are all wrong by the same unknown amount, and solves for that amount alongside its position. These distances before the clock error is removed are called pseudoranges, and they are the basic measurement of the whole system.

Written as an equation, with $P^s$ the pseudorange to satellite $s$, $\mathbf{x}^s$ the satellite's position when it sent the signal, $\mathbf{x}_r$ the receiver's position, $c$ the speed of light and $\mathrm{dt}_r$ the receiver's clock error in seconds,

$$
P^s = \lVert \mathbf{x}^s - \mathbf{x}_r \rVert + c \mathrm{dt}_r .
$$

The first term is the true distance and the second is the same for every satellite. The receiver knows $P^s$ from its measurement and $\mathbf{x}^s$ from the broadcast orbit, which leaves four unknowns: the three coordinates of $\mathbf{x}_r$ and $\mathrm{dt}_r$. Each satellite gives one equation, so a fix needs four satellites: three for position, one for the receiver clock. With more than four the equations disagree slightly because each measurement carries some error, and the receiver picks the position that fits them all best by least squares, as the [standalone fix](08-standalone-fix.md) page shows. That page also adds the smaller terms this equation leaves out, such as the satellite's own clock error, the atmospheric delays and the Earth's rotation during the signal's travel.

A side effect is that every GNSS receiver is also a clock accurate to tens of nanoseconds, which is why cell towers, power grids, and financial exchanges all have GNSS antennas on the roof.

## What limits a plain fix

With one antenna and no outside help, a receiver gets a position good to a few meters. The errors come from:

- The satellite's broadcast orbit and clock are only accurate to about a meter.
- The ionosphere, the charged upper atmosphere, slows the signal by a variable amount worth a few meters to tens of meters. It is the largest single error and the reason modern receivers use two frequencies.
- The lower atmosphere adds a couple of meters, mostly predictable.
- Reflections off buildings and the ground, called multipath, add meters in bad places.
- The receiver's own measurement noise is a fraction of a meter.

Together these make up the error budget of a fix. The position error is the range error scaled by a factor that depends on where the satellites sit in the sky, the dilution of precision (DOP), which is small when the satellites are spread out and large when they are bunched together.

The next pages explain where each of these errors comes from. Most of them are the same for two receivers a few kilometers apart, because both see the same satellites through nearly the same atmosphere. A second receiver at a known location can measure the errors and the first receiver can subtract them. That is differential GNSS, and its precise form, RTK (real-time kinematic), is what this project is for.

## Why RTK reaches centimeters

Subtracting a nearby reference receiver's errors brings a fix down to under a meter. Getting to one or two centimeters takes a second idea: measuring the signal's carrier wave itself, not just the timing marks on it. The carrier wavelength is 19 cm at the main GPS frequency, and with a strong signal and steady tracking a receiver can measure its phase to a millimeter or two, about a hundredth of a cycle. The catch is that it cannot tell which cycle it is looking at: the measurement is the fraction of a cycle plus an unknown whole number of cycles, the integer ambiguity. Working out that integer for every satellite, called ambiguity resolution, is the central problem of RTK, and it is covered in the [RTK](09-rtk.md) page.

---

**In pyrx:** `gpstime.py` holds the time scales (GPS week and time of week, the GPS epoch), and `orbits.py` computes satellite positions from broadcast orbits. The whole chain from recorded samples to an RTK fix is described in [`pyrx/README.md`](../../pyrx/README.md). Three notebooks run the whole chain on real data: the [signal-chain notebook](../../pyrx/notebooks/01-signal-chain.ipynb) from samples to measurements, the [standalone-positioning notebook](../../pyrx/notebooks/02-standalone-pvt.ipynb) from measurements to a position, and the [RTK notebook](../../pyrx/notebooks/03-rtk.ipynb) from a base and a rover to a centimeter-level fix.

Next: [The signal](02-the-signal.md).
