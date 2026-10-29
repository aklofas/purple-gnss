# Purple GNSS documentation

Purple GNSS is an open-source RTK GNSS receiver built from scratch: the antenna and
processing hardware, the firmware on it, and the positioning math that turns
raw satellite measurements into a centimeter-level fix. The project is early.
What exists today is `pyrx`, an offline receiver in Python that takes recorded
radio samples all the way to an RTK fix and is checked against Pocket SDR and
RTKLIB at every stage.

These pages are mostly about GNSS itself. If you are technical but have never
worked with radio or satellite navigation, and you want to understand every
step between a satellite and a position on a map, start with the fundamentals
series. It is written to be read in order.

## Pick your starting point

| What you want to do | Start here | Then read |
|---|---|---|
| Learn how a GNSS fix works, from zero | [What a GNSS is](fundamentals/01-what-gnss-is.md) | The rest of the [fundamentals series](#the-fundamentals-series), in order |
| Run the receiver on a recording | [`pyrx/README.md`](../pyrx/README.md) | The fundamentals page for whichever stage you are looking at |
| See each stage work, with figures | [pyrx/notebooks](../pyrx/notebooks/README.md) | The fundamentals page of the stage you are looking at |
| Look up a term you met in the code or a paper | [Glossary](fundamentals/glossary.md) | The page the glossary entry points to |
| Go deeper than these pages | [Further reading](fundamentals/glossary.md#further-reading) | |

## The fundamentals series

The nine pages each follow the signal one step further on its way from the
satellite to a fix. Each one opens with who it is for and what you will learn,
and closes with a pointer to where that stage lives in `pyrx`. The glossary at
the end collects the terms and further reading.

1. [What a GNSS is](fundamentals/01-what-gnss-is.md): the constellations, the orbits, why four satellites, and why time is the real product.
2. [The signal](fundamentals/02-the-signal.md): carrier, spreading code, and data message, and why the signal arrives below the noise.
3. [From antenna to samples](fundamentals/03-antenna-to-samples.md): what a radio front-end does, for people who have never met one.
4. [Finding a satellite](fundamentals/04-acquisition.md): correlation and the acquisition search.
5. [Staying locked](fundamentals/05-tracking.md): the code and carrier tracking loops.
6. [Reading the message](fundamentals/06-navigation-message.md): frames, ephemeris, and time of week.
7. [The measurements](fundamentals/07-measurements.md): pseudorange, carrier phase, and Doppler.
8. [A standalone fix](fundamentals/08-standalone-fix.md): satellite positions, clocks, the atmosphere, and least squares.
9. [RTK](fundamentals/09-rtk.md): differencing, integer ambiguities, and how centimeters become possible.
10. [Glossary and further reading](fundamentals/glossary.md).

## What kind of pages live here?

- **Understand how it works:** the fundamentals series above. Conceptual, few equations, no code.
- **Run something:** [`pyrx/README.md`](../pyrx/README.md) documents every command, the recordings it was checked on, and the algorithms as implemented.
- **Hardware and firmware:** not yet. The processing board and firmware are being designed; their documentation will appear here when there is something to document.
