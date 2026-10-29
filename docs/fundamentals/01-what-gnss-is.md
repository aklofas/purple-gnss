# What a GNSS is

> **Who this is for:** You know what GPS does from using it, and you want to understand what is actually happening between the satellites and your position.

> **You will learn:**
> - What the constellations are and where the satellites fly
> - Why the whole system is really a very precise clock
> - Why a fix needs four satellites, not three
> - What limits the accuracy of a plain fix, and why RTK can do a hundred times better

This page is conceptual. There is no code here. The series continues with [the signal](02-the-signal.md).

---

## A system of clocks in orbit

GNSS stands for Global Navigation Satellite System. GPS is the American one. Galileo is the European one, GLONASS the Russian one, BeiDou the Chinese one. They work the same way, and a modern receiver uses all of them at once, so "GNSS" is the general term and "GPS" is one member of the family.

Each satellite carries an atomic clock and continuously broadcasts two things: the exact time by its clock, and a description of its own orbit precise enough to compute where it is at any instant. That is the whole broadcast. The satellites do not know where you are, and they never hear from you.

A receiver listens to several satellites, works out how long each signal took to arrive, and multiplies by the speed of light to get the distance to each one. Distances to known points give a position. Everything else in this series is about doing those two steps well: measuring the travel time, and knowing where the satellite was.

## Where the satellites are

| System | Satellites in use | Altitude | Orbit period |
|---|---|---|---|
| GPS | about 31 | 20,200 km | 11 h 58 min |
| Galileo | about 28 | 23,200 km | 14 h 05 min |
| GLONASS | about 24 | 19,100 km | 11 h 16 min |
| BeiDou | about 45, in three orbit types | 21,500 km for the main group | 12 h 53 min |

The altitudes listed are for medium Earth orbits, far above the International Space Station (400 km) and far below geostationary satellites (36,000 km). BeiDou also has geostationary and inclined geosynchronous satellites at about 36,000 km. A GPS satellite crosses the sky in a few hours and the pattern repeats every sidereal day. From any open spot on Earth you can usually see 8 to 12 GPS satellites, and 25 to 40 satellites across all four systems.

The medium Earth orbits are nearly circular and spread over several planes for even coverage. Their inclinations to the equator are about 55 degrees for GPS and BeiDou, 56 degrees for Galileo, and 65 degrees for GLONASS. A satellite's orbit is predictable to within a metre or two over a few hours from the parameters it broadcasts, which is what the receiver relies on.

## Time is the product

A radio signal travels 30 cm in a nanosecond. To measure a distance to a metre, the receiver has to measure a travel time to about 3 nanoseconds. The satellite clocks are good enough for that. The receiver's clock is not: it is an ordinary crystal oscillator, wrong by microseconds and drifting all the time.

The trick is that the receiver's clock error is the same for every satellite it listens to at a given instant. So instead of measuring four true distances, the receiver measures four distances that are all wrong by the same unknown amount, and solves for that amount alongside its position. This is why a fix needs four satellites: three unknowns for position, one for the receiver clock. The distances before the clock error is removed are called pseudoranges, and they are the basic measurement of the whole system.

A side effect is that every GNSS receiver is also a clock accurate to tens of nanoseconds, which is why cell towers, power grids, and financial exchanges all have GNSS antennas on the roof.

## What limits a plain fix

With one antenna and no outside help, a receiver gets a position good to a few metres. The errors come from:

- The satellite's broadcast orbit and clock are only accurate to about a metre.
- The ionosphere, the charged upper atmosphere, slows the signal by a variable amount worth a few metres to tens of metres. It is the largest single error and the reason modern receivers use two frequencies.
- The lower atmosphere adds a couple of metres, mostly predictable.
- Reflections off buildings and the ground, called multipath, add metres in bad places.
- The receiver's own measurement noise is a fraction of a metre.

The next pages explain where each of these comes from. The important thing to notice now is that most of them are the same for two receivers a few kilometres apart. A second receiver at a known location can measure the errors and the first receiver can subtract them. That is differential GNSS, and its precise form, RTK, is what this project is for.

## Why RTK reaches centimetres

Subtracting a nearby reference receiver's errors brings a fix down to tens of centimetres. Getting to one or two centimetres takes a second idea: measuring the signal's carrier wave itself, not just the timing marks on it. The carrier wavelength is 19 cm at the main GPS frequency, and a receiver can measure its phase to a millimetre or two. The catch is that it cannot tell which cycle it is looking at. Working out that integer count of cycles, for every satellite, is the central problem of RTK, and it is covered in the [RTK](09-rtk.md) page.

---

**In pyrx:** `gpstime.py` holds the time scales (GPS week and time of week, the GPS epoch), and `orbits.py` computes satellite positions from broadcast orbits. The whole chain from recorded samples to an RTK fix is described in [`pyrx/README.md`](../../pyrx/README.md).

Next: [The signal](02-the-signal.md).
