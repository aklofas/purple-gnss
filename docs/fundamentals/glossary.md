# Glossary and further reading

> **Who this is for:** Anyone who met a term in the code, a paper or a receiver's status display and wants the one-line meaning and the page of this series that explains it.

---

## Glossary

**Acquisition.** The search that finds which satellites are present and their rough code phase and Doppler. [Finding a satellite](04-acquisition.md).

**Almanac.** A coarse orbit and clock for every satellite in a constellation, valid for months, used to predict which satellites are visible. [Reading the message](06-navigation-message.md).

**Ambiguity.** The unknown integer number of carrier wavelengths in a carrier-phase measurement. Resolving it is the core of RTK. [The measurements](07-measurements.md), [RTK](09-rtk.md).

**Baseline.** The vector from the base receiver to the rover. What an RTK solution actually computes. [RTK](09-rtk.md).

**Base, rover.** In RTK, the receiver at a known point and the receiver whose position is wanted. [RTK](09-rtk.md).

**BOC.** Binary offset carrier, a modulation used by Galileo E1 and GPS L1C that splits the signal's power into two lobes either side of the carrier for sharper timing. [The signal](02-the-signal.md).

**C/A code.** The coarse/acquisition code, the civilian GPS L1 spreading code: 1023 chips, 1.023 Mchip/s, 1 ms period. [The signal](02-the-signal.md).

**Carrier.** The radio sine wave the code and data are modulated onto. [The signal](02-the-signal.md).

**Carrier phase.** The observable formed from the tracking loop's accumulated carrier cycles, precise to millimetres but with an unknown integer part. [The measurements](07-measurements.md).

**Chip.** One element of the spreading code. A C/A chip is 293 m long at the speed of light. [The signal](02-the-signal.md).

**C/N0.** Carrier-to-noise-density ratio, in dB-Hz, the standard measure of GNSS signal strength. 35 to 50 dB-Hz in the open. [The signal](02-the-signal.md).

**Code phase.** Where in its repeating period the received code is at a given sample; the raw timing measurement. [Finding a satellite](04-acquisition.md).

**Correlation.** Multiplying received samples by a replica code and summing. The operation that lifts the signal out of the noise and measures its timing. [The signal](02-the-signal.md).

**Costas loop.** A carrier phase-locked loop insensitive to data-bit sign flips. [Staying locked](05-tracking.md).

**Cycle slip.** A loss of carrier lock that changes the integer ambiguity by one or more cycles. [Staying locked](05-tracking.md).

**Data channel, pilot channel.** The two components of a modern signal: one carries the message; the other carries no data, so once its known secondary code is removed it can be integrated coherently far longer, until the oscillator and the receiver's motion limit it. [The signal](02-the-signal.md).

**DLL.** Delay-locked loop, the feedback loop that keeps the replica code aligned, using early, prompt and late correlators. [Staying locked](05-tracking.md).

**DOP.** Dilution of precision, the factor by which satellite geometry multiplies range error into position error. [A standalone fix](08-standalone-fix.md).

**Doppler.** The frequency shift from the satellite's and receiver's relative motion, up to about 5 kHz; as an observable, the receiver's velocity. [Finding a satellite](04-acquisition.md), [The measurements](07-measurements.md).

**Double difference.** The difference of two receivers' measurements of two satellites, which cancels both receivers' and both satellites' clock errors. [RTK](09-rtk.md).

**ECEF.** Earth-centred, Earth-fixed coordinates: X, Y, Z from the centre of the Earth, rotating with it. The frame positions are solved in. [A standalone fix](08-standalone-fix.md).

**Elevation mask.** The angle above the horizon below which satellites are ignored, typically 10 to 15 degrees. [A standalone fix](08-standalone-fix.md).

**Ephemeris.** The broadcast parameters describing one satellite's orbit, good to a metre or two for a few hours. [Reading the message](06-navigation-message.md).

**Epoch.** One measurement instant, at which the receiver forms observables for every satellite it tracks.

**Fixed, float.** An RTK solution with the ambiguities resolved to integers, and one with them still real-valued. [RTK](09-rtk.md).

**FLL.** Frequency-locked loop, used to pull the carrier frequency in before the PLL can lock. [Staying locked](05-tracking.md).

**Front-end.** The analogue radio chain from the antenna to the analogue-to-digital converter. [From antenna to samples](03-antenna-to-samples.md).

**GPS time.** A continuous count of weeks and seconds since 6 January 1980, with no leap seconds. [Reading the message](06-navigation-message.md).

**HAS.** Galileo's High Accuracy Service, free precise orbit and clock corrections broadcast on E6. [RTK](09-rtk.md).

**IF.** Intermediate frequency, where the signal sits after the mixer, a few MHz instead of 1.5 GHz. [From antenna to samples](03-antenna-to-samples.md).

**Ionosphere, troposphere.** The two layers of atmosphere that delay the signal: the upper, charged one by a frequency-dependent amount, and the lower, neutral one by the same amount at every frequency. [A standalone fix](08-standalone-fix.md).

**I/Q.** In-phase and quadrature, the two components of a complex sample. [From antenna to samples](03-antenna-to-samples.md).

**Klobuchar.** The simple ionosphere model whose eight parameters are broadcast in the GPS message. [A standalone fix](08-standalone-fix.md).

**LAMBDA.** The standard method for finding the integer ambiguities closest to a float solution. [RTK](09-rtk.md).

**LNAV.** The legacy GPS navigation message on L1 C/A: 50 bit/s, 30-bit words, 6-second subframes. [Reading the message](06-navigation-message.md).

**Lock, loss of lock.** The state of a tracking loop that is following the signal, and the moment it stops. [Staying locked](05-tracking.md).

**Multipath.** Signal reaching the antenna by reflection as well as directly. [A standalone fix](08-standalone-fix.md).

**NTRIP.** The internet protocol for streaming RTCM corrections from a base station. [RTK](09-rtk.md).

**Observable.** A measurement a receiver reports: pseudorange, carrier phase, Doppler, C/N0. [The measurements](07-measurements.md).

**Phase centre.** The point an antenna effectively receives at, which moves slightly with direction and frequency. [From antenna to samples](03-antenna-to-samples.md).

**PLL.** Phase-locked loop, the feedback loop that keeps the replica carrier in phase with the received one. [Staying locked](05-tracking.md).

**PPK.** Post-processed kinematic: RTK computed afterwards from logged files. [RTK](09-rtk.md).

**PPP.** Precise point positioning: a single receiver with precise orbits and clocks instead of a base station. [RTK](09-rtk.md).

**PRN.** Pseudo-random noise, the family of spreading codes, and by extension the number identifying a satellite's code. [The signal](02-the-signal.md).

**Pseudorange.** The travel time of the signal times the speed of light, including the receiver's unknown clock error. The basic measurement. [The measurements](07-measurements.md).

**PVT.** Position, velocity and time, the output of a standalone solution. [A standalone fix](08-standalone-fix.md).

**Ratio test.** The check that the best integer ambiguity candidate fits the data much better than the second best, before a fix is accepted. [RTK](09-rtk.md).

**RINEX.** The text file format for exchanging observables and ephemerides between receivers and programs. [The measurements](07-measurements.md).

**RTCM.** The binary format for streaming a base station's observables and corrections to a rover. [RTK](09-rtk.md).

**RTK.** Real-time kinematic: carrier-phase differential positioning against a base station, to a centimetre or two. [RTK](09-rtk.md).

**Saastamoinen.** The standard troposphere delay model. [A standalone fix](08-standalone-fix.md).

**SBAS.** Satellite-based augmentation, metre-level corrections broadcast from geostationary satellites (WAAS, EGNOS). [RTK](09-rtk.md).

**Single difference.** The difference of two receivers' measurements of one satellite, which cancels the satellite's clock. [RTK](09-rtk.md).

**Spreading code.** The pseudo-random chip sequence that identifies a satellite and carries the timing. [The signal](02-the-signal.md).

**Subframe, frame.** The 300-bit, 6-second unit of the LNAV message, and the five-subframe, 30-second group. [Reading the message](06-navigation-message.md).

**TCXO.** Temperature-compensated crystal oscillator, the usual reference oscillator in a receiver. [From antenna to samples](03-antenna-to-samples.md).

**Time of week.** GPS time within the current week, in seconds; the count in each subframe that makes the transmit time unambiguous. [Reading the message](06-navigation-message.md).

**Tracking.** The continuous following of a satellite's code and carrier after acquisition. [Staying locked](05-tracking.md).

**TTFF.** Time to first fix, from power-on to the first position.

**Wide lane.** The difference of two frequencies' carrier phases, which behaves like a signal with a longer wavelength and so an easier integer. [RTK](09-rtk.md).

## Further reading

Books, in the order a newcomer should try them:

- Kai Borre, Dennis Akos, Nicolaj Bertelsen, Peter Rinder and Søren Holdt Jensen, *A Software-Defined GPS and Galileo Receiver*, Birkhäuser, 2007. The shortest route from samples to a fix, with MATLAB code for every stage. The structure of the first seven pages of this series follows it.
- Pratap Misra and Per Enge, *Global Positioning System: Signals, Measurements, and Performance*, 2nd edition, Ganga-Jamuna Press, 2006. The clearest treatment of the measurements, the error sources and the position solution.
- Elliott Kaplan and Christopher Hegarty (editors), *Understanding GPS/GNSS: Principles and Applications*, 3rd edition, Artech House, 2017. The standard reference for receiver design, including the tracking loops in depth.
- Peter Teunissen and Oliver Montenbruck (editors), *Springer Handbook of Global Navigation Satellite Systems*, Springer, 2017. Everything, at research depth. The chapters on carrier-phase ambiguity resolution are the ones to read after the RTK page.

Free material:

- ESA Navipedia, <https://gssc.esa.int/navipedia/>. An encyclopedia of GNSS by the people who run Galileo, at about the level of this series and a step beyond.
- Tomoji Takasu, *Pocket SDR GNSS SDR Algorithm Description*, in the `doc/` directory of <https://github.com/tomojitakasu/PocketSDR>. An implementation-level walk through acquisition, tracking, navigation decoding and PVT in a real open-source receiver. The best next step after this series for anyone who wants to read receiver code.
- RTKLIB manual, <https://www.rtklib.com/>. Appendix E is a compact, precise statement of the positioning models, standalone and RTK, that this project's engine is written against.

Specifications, when you need the exact bit layout:

- IS-GPS-200, the GPS L1/L2 interface specification, and IS-GPS-705 for L5, at <https://www.gps.gov/technical/icwg/>.
- Galileo Open Service Signal-in-Space Interface Control Document, at <https://www.gsc-europa.eu/electronic-library/programme-reference-documents>.
- RINEX 3 and 4 format descriptions, at <https://igs.org/formats-and-standards/>.

Back to the [index](../index.md).
