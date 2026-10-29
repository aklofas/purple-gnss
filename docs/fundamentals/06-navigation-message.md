# Reading the message

> **Who this is for:** You know the tracking loop produces a stream of data bits from each satellite. You want to know what is in them and how the receiver gets from bits to "where was this satellite, and when did it send this".

> **You will learn:**
> - How the receiver groups correlator outputs into data bits (bit synchronization) and finds the subframe boundaries (frame synchronization)
> - How the parity code catches bad words, and how the preamble settles the sign the Costas loop left open
> - What the ephemeris is, how its Kepler elements describe an orbit, and why it is only good for a few hours
> - How the clock correction and the time of week turn a code-phase measurement into a transmit time
> - What the almanac, ionosphere parameters and week number are for

This page is conceptual. The series continues with [the measurements](07-measurements.md).

---

## 50 bits per second

The GPS L1 C/A message, called LNAV, arrives at 50 bits per second. Its job is to tell the receiver two things it cannot measure: where each satellite was, and when, on GPS time, each part of the signal was sent. Without it the tracking loops have precise timing with nothing to anchor it to.

A tracking loop that is locked delivers one correlator output per 1 ms code period, and one bit lasts 20 of them. The first step is bit synchronization: finding which of the 20 code periods starts a bit. The receiver counts, over a few seconds, how often the sign of the prompt $I$ output changes at each of the 20 positions. Real bit edges all fall at one position; noise flips the sign at random elsewhere. Once one position clearly wins, the receiver adds the 20 values of $I$ in each bit, a 20 ms coherent integration, and takes the sign of the sum as the bit. The bit stream still carries the sign ambiguity of the Costas loop: it might all be inverted.

The message is organized into 30-bit words, 10 words to a 300-bit subframe lasting 6 seconds, and 5 subframes to a 1500-bit frame lasting 30 seconds. Every subframe starts with the same 8-bit preamble, 10001011, at the head of its first word, the telemetry word; the second word, the handover word, starts with a 17-bit time-of-week count; and every word ends with 6 parity bits.

```text
 one frame = 30 s
 +------------+------------+------------+------------+------------+
 | subframe 1 | subframe 2 | subframe 3 | subframe 4 | subframe 5 |
 | clock, week| ephemeris  | ephemeris  | iono, UTC, | almanac    |
 |            |            |            | almanac    |            |
 +------------+------------+------------+------------+------------+
   6 s each; subframes 4 and 5 cycle through 25 pages over 12.5 min
```

The parity bits form an error-detecting code, a (32,26) Hamming code in the interface specification's terms: the 6 parity bits of a word are computed from its 24 data bits and from the last 2 bits of the previous word, so any single wrong bit makes the check fail. Frame synchronization is the search for the subframe boundaries. The receiver searches the bit stream for the preamble or its inverse, checks that the parity of the word holds, and confirms that another preamble appears exactly 300 bits later. The preamble is what fixes the sign: the parity scheme is built so that an inverted word still passes and decodes to the same bits, so it cannot tell, but the preamble pattern can. Once synchronized, every subframe is checked word by word and a parity failure throws the subframe away.

Modern signals (L2C, L5, Galileo E1 and E5a) use different framing, with forward error correction and a cyclic redundancy check instead of parity words, and the content is organized differently. The information is the same.

## The ephemeris

Subframes 2 and 3 carry the ephemeris: the satellite's own orbit, as a set of 16 parameters. Six of them are Kepler elements, which describe an ideal ellipse: its size (the semi-major axis, broadcast as its square root, about 5154 $\sqrt{\text{m}}$ for a GPS orbit of 26,560 km radius), its shape (the eccentricity), the orientation of its plane (inclination and the longitude of the ascending node), the orientation of the ellipse within that plane (the argument of perigee), and where the satellite is on it at the reference time (the mean anomaly). The rest are small periodic and secular corrections that absorb the effects of the Earth's non-spherical gravity, the Moon, the Sun and solar radiation pressure.

To find the satellite at time $t$, the receiver advances the mean anomaly $M$ from its value at the reference time $t_{oe}$ at the orbit's mean motion $n$, solves Kepler's equation for the eccentric anomaly $E$,

$$
M = M_0 + n (t - t_{oe}), \qquad M = E - e \sin E ,
$$

with $e$ the eccentricity, and goes from there to a position in the orbital plane, applies the corrections, and rotates into an Earth-fixed frame. Kepler's equation has no closed-form solution, but a few iterations of $E \leftarrow M + e \sin E$ converge quickly because $e$ is small, at most a few hundredths for GPS. The interface specification gives every step; the result is the satellite's position to within a meter or two, at any instant within a couple of hours of the reference time.

The parameters are a curve fit, not physics, and the fit degrades outside its window. Each ephemeris is stamped with an issue number (the issue of data, IODE) and a reference time, and the receiver treats one as valid for four hours by convention, the curve-fit interval of normal operations. The satellites broadcast a fresh set every two hours, uploaded from the control segment's tracking of the real orbits. This is why a receiver that has been off for a day cannot use its stored ephemerides and has to wait for new ones, and why, even once it has found the satellites and synchronized to their bits, its time to first fix (TTFF) is at least 18 seconds: the three subframes that carry the ephemeris and the clock take 18 seconds to arrive, and a receiver that starts partway through them waits for the missing ones in the next frame, up to about 36 seconds.

## The clock

Subframe 1 carries the satellite clock correction: three polynomial coefficients (offset $a_{f0}$, drift $a_{f1}$, and drift rate $a_{f2}$) that describe how far the satellite's atomic clock is from GPS system time, around a clock reference time $t_{oc}$. The receiver reads the time the satellite sent a signal, $t_{sv}$, on the satellite's clock, and corrects it with

$$
\Delta t_{sv} = a_{f0} + a_{f1}(t - t_{oc}) + a_{f2}(t - t_{oc})^2 + \Delta t_r , \qquad t = t_{sv} - \Delta t_{sv} ,
$$

where $t$ is GPS system time and $\Delta t_r$ a relativistic term described below. Even an atomic clock in orbit is off by up to a millisecond, which would be 300 km of range error. The correction brings it to a few nanoseconds. Subframe 1 also carries the week number, a health flag, an accuracy estimate for the orbit and clock, and a group-delay term that accounts for the different signal paths of L1 and L2 inside the satellite.

The relativistic effect is handled here too. The satellite clock runs faster than one on the ground by a net 38 microseconds a day: the weaker gravity at its altitude speeds it up by about 45 microseconds, and its orbital speed slows it by about 7; the clocks are built to tick slow by that amount so that on average they match. What remains is a periodic term from the orbit's slight eccentricity, which the receiver computes from the ephemeris and removes from every measurement:

$$
\Delta t_r = F e \sqrt{A} \sin E , \qquad F = -4.442807633\times10^{-10} \text{ s/m}^{1/2} ,
$$

with $\sqrt{A}$ the broadcast square root of the semi-major axis and $E$ the eccentric anomaly from the orbit computation. For $e$ = 0.02 and $\sqrt{A}$ = 5154 $\sqrt{\text{m}}$ the term ranges between about −46 and +46 nanoseconds (±14 m) over each orbit.

## Time of week

The 17-bit count in the second word of every subframe gives the time of week on the satellite's own clock, in units of 6 seconds, at which the next subframe begins. This is the number that turns tracking into ranging. With $z$ that count, the subframe that carries it began at

$$
t_{\rm start} = (6z - 6) \bmod 604800 \quad \text{seconds of the week}.
$$

A count of 5046, for example, says the next subframe starts at 30,276 s into the week, so this one started at 30,270 s. The modulo keeps the result inside the week: the last subframe of a week carries $z = 0$, and it started at 604,794 s, six seconds before the next week begins.

From the tracking loop the receiver knows, to a fraction of a chip, where the code period boundaries fall in its sample stream. From bit synchronization it knows which code period is the first of a bit, and from frame synchronization it knows which bit is the first of a subframe. Chaining these together, if code period $k_0$ is the first of the subframe and each period lasts $T_{\rm code}$ = 1 ms, code period $k$ was sent at

$$
t_{\rm tx}(k) = t_{\rm start} + (k - k_0) T_{\rm code}
$$

on the satellite's clock, and a point partway through a period was sent the code-phase fraction of a period later. The clock correction in subframe 1 then turns satellite time into GPS time. That transmit time is half of the pseudorange measurement on the next page, and the whole chain from code phase through bit and frame to time of week is what makes it unambiguous.

GPS time itself is a continuous count of weeks and seconds since midnight on 6 January 1980, with no leap seconds. It is currently 18 seconds ahead of UTC, and the message carries the offset. The LNAV week number is only 10 bits, so it rolls over every 1024 weeks (about 19.6 years; it last did in April 2019), and the receiver has to know roughly what era it is in to disambiguate. GPS's modern messages use 13 bits, Galileo's 12.

## The rest of the message

Subframes 4 and 5 cycle through 25 pages over 12.5 minutes and carry the almanac, a coarse orbit and clock for every satellite in the constellation, accurate to a kilometer or so and valid for months. It is what lets a warm receiver predict which satellites are up. They also carry the eight Klobuchar parameters of a simple ionosphere model that removes about half of the ionospheric delay for a single-frequency receiver, and the parameters relating GPS time to UTC.

Galileo's message adds a few things of its own, notably the free corrections of the High Accuracy Service (HAS) on E6, which bring the orbit and clock from about a meter to a decimeter or better.

---

**In pyrx:** `bitsync.py` finds the bit edge and the frame in the tracker's output, `lnav.py` holds the word layout, the parity check and the ephemeris assembly, `navdecode.py` runs them on a tracked channel, and `ephemeris.py` holds the resulting records. `pyrx-nav` runs the chain on a recording and prints what it decoded. A test decodes every tracked satellite of two recordings and compares the result with the day's published broadcast ephemeris: every satellite yields the expected subframes with time-of-week tags that agree, the orbit and clock fields it compares match the published values to within half of their least significant bit, and the week number matches exactly. Neither recording contains the ionosphere page, so synthetic tests and a worked example from the interface specification check those parameters instead. Section 4 of the [signal-chain notebook](../../pyrx/notebooks/01-signal-chain.ipynb) decodes the message from a real recording, and section 2 of the [standalone-positioning notebook](../../pyrx/notebooks/02-standalone-pvt.ipynb) turns ephemerides into satellite positions.

Next: [The measurements](07-measurements.md).
