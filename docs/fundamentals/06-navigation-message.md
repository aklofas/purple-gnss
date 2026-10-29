# Reading the message

> **Who this is for:** You know the tracking loop produces a stream of data bits from each satellite. You want to know what is in them and how the receiver gets from bits to "where was this satellite, and when did it send this".

> **You will learn:**
> - How the GPS message is framed and how the receiver finds the frame boundaries
> - What the ephemeris is and why it is only good for a few hours
> - How the time of week turns a code-phase measurement into a transmit time
> - What the almanac, clock corrections and ionosphere parameters are for

This page is conceptual. The series continues with [the measurements](07-measurements.md).

---

## 50 bits per second

The GPS L1 C/A message, called LNAV, arrives at 50 bits per second. A tracking loop that is locked delivers one bit every 20 ms, with the sign ambiguity of the Costas loop still unresolved: the receiver has a stream of bits that might all be inverted.

The message is organised into 30-bit words, 10 words to a 300-bit subframe lasting 6 seconds, and 5 subframes to a 1500-bit frame lasting 30 seconds. Every subframe starts with the same 8-bit preamble, at the head of its first word, the telemetry word; the second word, the handover word, starts with a 17-bit time-of-week count; and every word ends with 6 parity bits. The receiver finds the frame by searching the bit stream for the preamble or its inverse, checking that the parity of the word holds, and confirming that another preamble appears exactly 300 bits later. The preamble is what fixes the sign: the parity scheme is built so that an inverted word still passes and decodes to the same bits, so it cannot tell, but the preamble pattern can. Once synchronised, every subframe is checked word by word and a parity failure throws the subframe away.

```text
 one frame = 30 s
 +------------+------------+------------+------------+------------+
 | subframe 1 | subframe 2 | subframe 3 | subframe 4 | subframe 5 |
 | clock, week| ephemeris  | ephemeris  | iono, UTC, | almanac    |
 |            |            |            | almanac    |            |
 +------------+------------+------------+------------+------------+
   6 s each; subframes 4 and 5 cycle through 25 pages over 12.5 min
```

Modern signals (L2C, L5, Galileo E1 and E5a) use different framing, with forward error correction and a cyclic redundancy check instead of parity words, and the content is organised differently. The information is the same.

## The ephemeris

Subframes 2 and 3 carry the ephemeris: the satellite's own orbit, as a set of 16 parameters. Six of them describe an ideal Kepler ellipse (size, shape, tilt, and where the satellite is on it at a reference time). The rest are small periodic and secular corrections that absorb the effects of the Earth's non-spherical gravity, the Moon, the Sun and solar radiation pressure. Evaluating the formulas in the GPS interface specification with these parameters gives the satellite's position in an Earth-fixed frame to within a metre or two, at any instant within a couple of hours of the reference time.

The parameters are a curve fit, not physics, and the fit degrades outside its window. Each ephemeris is stamped with an issue number and a reference time, and the receiver treats one as valid for four hours by convention. The satellites broadcast a fresh set every two hours, uploaded from the control segment's tracking of the real orbits. This is why a receiver that has been off for a day cannot use its stored ephemerides and has to wait for new ones, and why, even once it has found the satellites and synchronised to their bits, it needs at least 18 seconds before its first fix: the three subframes that carry the ephemeris and the clock take 18 seconds to arrive, and a receiver that starts partway through them waits for the missing ones in the next frame, up to about 36 seconds.

## The clock

Subframe 1 carries the satellite clock correction: three polynomial coefficients (offset, drift, and drift rate) that describe how far the satellite's atomic clock is from GPS system time. Even an atomic clock in orbit is off by up to a millisecond, which would be 300 km of range error. The correction brings it to a few nanoseconds. Subframe 1 also carries the week number, a health flag, an accuracy estimate for the orbit and clock, and a group-delay term that accounts for the different signal paths of L1 and L2 inside the satellite.

A relativistic effect is handled here too. The satellite clock runs faster than one on the ground, by 38 microseconds a day, because it is higher in the gravity well and moving fast; the clocks are built to tick slow by that amount so that on average they match. What remains is a periodic term from the orbit's slight eccentricity, up to 45 nanoseconds (14 m), which the receiver computes and removes from every measurement.

## Time of week

The 17-bit count in the second word of every subframe gives the time of week on the satellite's own clock, in units of 6 seconds, at which the next subframe begins. This is the number that turns tracking into ranging. From the tracking loop the receiver knows, to a fraction of a chip, where the code period boundaries fall in its sample stream. From bit synchronisation it knows which code period is the first of a bit, and from frame synchronisation it knows which bit is the first of a subframe. Chaining these together, it knows that the code period starting at a particular sample was transmitted at a particular time of week on the satellite's clock, plus or minus the code-phase fraction; the clock correction in subframe 1 then turns satellite time into GPS time. That transmit time is half of the pseudorange measurement on the next page, and the whole chain from code phase through bit and frame to time of week is what makes it unambiguous.

GPS time itself is a continuous count of weeks and seconds since midnight on 6 January 1980, with no leap seconds. It is currently 18 seconds ahead of UTC, and the message carries the offset. The LNAV week number is only 10 bits, so it rolls over every 1024 weeks (about 19.6 years), and the receiver has to know roughly what era it is in to disambiguate. Modern messages use 13 bits.

## The rest of the message

Subframes 4 and 5 cycle through 25 pages over 12.5 minutes and carry the almanac, a coarse orbit and clock for every satellite in the constellation, accurate to a kilometre or so and valid for months. It is what lets a warm receiver predict which satellites are up. They also carry the eight Klobuchar parameters of a simple ionosphere model that removes about half of the ionospheric delay for a single-frequency receiver, and the parameters relating GPS time to UTC.

Galileo's message adds a few things of its own, notably the free high-accuracy corrections on E6, which bring the orbit and clock from a metre to a few centimetres and are the basis of the planned PPP fallback in this project.

---

**In pyrx:** `bitsync.py` finds the bit edge and the frame in the tracker's output, `lnav.py` holds the word layout, the parity check and the ephemeris assembly, `navdecode.py` runs them on a tracked channel, and `ephemeris.py` holds the resulting records. `pyrx-nav` runs the chain on a recording and prints what it decoded, checked against the day's published broadcast ephemeris.

Next: [The measurements](07-measurements.md).
