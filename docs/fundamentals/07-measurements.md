# The measurements

> **Who this is for:** You know how the receiver tracks each satellite and reads its message. You want to know exactly what numbers come out of a receiver, what each one means physically, and what is good and bad about each.

> **You will learn:**
> - What a pseudorange is, how it is formed, and what "pseudo" means
> - What the carrier phase measures and why it is precise but ambiguous
> - What Doppler gives and what C/N0 is for
> - How the measurements are shared between receivers, which is where RINEX comes in

This page is conceptual. The series continues with [a standalone fix](08-standalone-fix.md).

---

## Observables

Receiver people call the measurements observables. For each satellite and each signal, at each measurement epoch (typically once a second, sometimes ten times), a receiver produces up to four of them: a pseudorange, a carrier phase, a Doppler, and a signal strength. The first three are all measurements of the same thing, the receiver's distance to the satellite and how it changes, made in different ways with different strengths. The rest of this series is about combining them.

## Pseudorange

The pseudorange is the apparent distance to the satellite: the signal's travel time multiplied by the speed of light. The travel time is the difference between two clock readings, the satellite's at transmission and the receiver's at reception.

The transmit time comes from the previous page. For a chosen receiver epoch, the receiver finds, for each satellite, the point in the code it was receiving at that instant, and knows from the decoded time of week when, by the satellite's clock, it transmitted that point (the next page corrects that clock to GPS time). The receive time is the receiver's own clock, which is the problem. Its clock is wrong by an unknown amount, so every pseudorange is wrong by the same unknown amount, and the travel time of 65 to 85 milliseconds comes out as something like 65 to 85 milliseconds plus or minus the receiver's error. That shared error is why it is called a pseudo-range, and the next page solves for it.

A pseudorange from the L1 C/A code is noisy at the level of a few decimetres to a metre, set by the 293 m chip and the DLL's precision. It is unambiguous: the receiver knows the full distance, with no missing multiples of anything. It is also corrupted by every delay on the path: the ionosphere adds metres, the troposphere adds metres, multipath adds up to tens of metres. The pseudorange is the honest workhorse. Every fix starts from it.

## Carrier phase

The carrier phase is the number of carrier cycles between the satellite and the receiver, as measured by the tracking loop's accumulated replica phase. It is recorded so that it grows as the distance grows. It is measured to about one hundredth of a cycle, two millimetres on L1, a hundred times better than the pseudorange. The ionosphere affects it by the same amount as the pseudorange but with the opposite sign (it advances the phase while delaying the code), which turns out to be useful. And multipath affects it far less, a few centimetres at worst.

The price is that the integer part is unknown. When the tracking loop locks on, it starts counting cycles from zero, and the true count at that moment, some 100 million cycles, is not observable. From then on the receiver counts every cycle gained and lost, so the phase measures changes in distance perfectly while the absolute distance is off by an unknown integer number of wavelengths. That integer is the ambiguity, it is constant for as long as lock holds, and it changes if the loop slips a cycle. RTK is the business of finding those integers. Until they are found, the carrier phase is a very precise measurement of how far the receiver has moved, not of where it is.

```text
                    pseudorange              carrier phase
  precision         0.3 to 1 m               2 mm
  ambiguity         none                     unknown integer cycles
  ionosphere        delays                   advances, same size
  multipath         up to tens of metres     a few centimetres
  survives a fade   yes                      ambiguity has to be re-solved
```

## Doppler

The Doppler observable is the frequency offset the carrier loop is currently applying, in Hz, positive while the satellite approaches. Multiplied by minus the carrier wavelength it becomes the rate of change of distance to the satellite, plus the receiver's and the satellite's clock drift, in metres per second. It gives the receiver's velocity directly: with four satellites the solution is three velocity components and the clock drift, by exactly the same argument as for position. Velocity from Doppler is accurate to a few centimetres per second and needs no ambiguity resolution, so it is available instantly and is what navigation displays show as speed and heading.

## C/N0

The signal strength is reported as the carrier-to-noise-density ratio in dB-Hz, estimated from the correlator outputs. It is not used in the position solution directly, but it is the main input to weighting: a measurement at 30 dB-Hz is noisier and more likely to be multipath than one at 48 dB-Hz, and a solver down-weights it. It is also the first thing a person looks at when a receiver misbehaves.

## Loss-of-lock flags and RINEX

Along with each carrier phase the receiver reports whether lock was continuous since the previous epoch. A slip or a reacquisition is flagged, and any user of the carrier phase has to reset its ambiguity for that satellite when it sees the flag. A Costas loop can also be half a cycle out while the data polarity is unknown, or slip by half a cycle; a separate flag marks those phases, and an RTK engine keeps them out of integer fixing until the flag clears.

Observables are exchanged between receivers and programs in a text format called RINEX (Receiver Independent Exchange Format). An observation file holds, per epoch, one line per satellite with its pseudorange, carrier phase, Doppler and signal strength per signal, with the lock flags. A separate navigation file holds the decoded ephemerides. Public reference stations publish both, every day, for the whole world, which is what makes it possible to develop and test RTK without owning two receivers. Everything in the next two pages is computed from RINEX files, and the `pyrx` receiver produces them from recordings.

---

**In pyrx:** `observables.py` forms pseudoranges and carrier phases from the tracker's state at each epoch and `rinex.py` reads and writes the files. `pyrx-rinex` runs the whole signal chain on a recording and writes observation and navigation files, whose transmit times are checked against Pocket SDR's and whose fix under RTKLIB is checked against RTKLIB's fix from Pocket SDR's observations.

Next: [A standalone fix](08-standalone-fix.md).
