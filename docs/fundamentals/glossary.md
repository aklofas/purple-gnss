# Glossary and further reading

> **Who this is for:** Anyone who met a term in the code, a paper or a receiver's status display and wants the one-line meaning and the page of this series that explains it.

---

## Glossary

**Acquisition.** The search that finds which satellites are present and their rough code phase and Doppler. [Finding a satellite](04-acquisition.md).

**ADC.** Analog-to-digital converter, the last stage of the front-end, which samples the filtered signal; one or two bits per sample are enough for GNSS. [From antenna to samples](03-antenna-to-samples.md).

**AGC.** Automatic gain control, which keeps the noise amplitude at the right level for the converter's few quantizer thresholds. [From antenna to samples](03-antenna-to-samples.md).

**Aliasing.** Frequencies above half the sample rate folding back onto lower ones, where they cannot be told apart; the Nyquist criterion avoids it, and band-pass sampling uses it on purpose. [From antenna to samples](03-antenna-to-samples.md).

**Almanac.** A coarse orbit and clock for every satellite in a constellation, valid for months, used to predict which satellites are visible and their Doppler for a warm start. [Finding a satellite](04-acquisition.md), [Reading the message](06-navigation-message.md).

**Ambiguity.** The unknown integer number of carrier wavelengths in a carrier-phase measurement, also called the integer ambiguity. Resolving it is the core of RTK. [The measurements](07-measurements.md), [RTK](09-rtk.md).

**Ambiguity resolution.** Finding the integer values of the carrier-phase ambiguities, which turns a float RTK solution into a fixed one. [What a GNSS is](01-what-gnss-is.md), [RTK](09-rtk.md).

**Autocorrelation.** The correlation of a code with shifted copies of itself: the full 1023 at zero shift for a C/A code, and small at every other shift. [The signal](02-the-signal.md).

**Band-pass filter.** A filter that passes one band and rejects the rest; after the LNA it keeps the GNSS band and rejects strong signals such as mobile phones. [From antenna to samples](03-antenna-to-samples.md).

**Band-pass sampling.** Choosing the IF and the sample rate so that aliasing folds the band down to a lower frequency, allowing a lower sample rate. [From antenna to samples](03-antenna-to-samples.md).

**Baseband.** A signal centered at zero frequency, where it lands when the local oscillator sits at the band center. [From antenna to samples](03-antenna-to-samples.md).

**Baseline.** The vector from the base receiver to the rover. What an RTK solution actually computes. [RTK](09-rtk.md).

**Base, rover.** In RTK, the receiver at a known point and the receiver whose position is wanted. [RTK](09-rtk.md).

**Bit synchronization.** Finding which of the 20 code periods of an L1 C/A data bit starts the bit, from where the sign of the prompt correlator output changes. [Reading the message](06-navigation-message.md).

**BOC.** Binary offset carrier, a modulation used by Galileo E1 and GPS L1C that splits the signal's power into two lobes either side of the carrier for sharper timing. [The signal](02-the-signal.md).

**BPSK.** Binary phase-shift keying: multiplying a carrier by +1 or -1, so that a sign flip is a jump of half a cycle in its phase. How the code and data go onto the GPS carrier. [The signal](02-the-signal.md).

**C/A code.** The coarse/acquisition code, the civilian GPS L1 spreading code: 1023 chips, 1.023 Mchip/s, 1 ms period. [The signal](02-the-signal.md).

**Carrier.** The radio sine wave the code and data are modulated onto. [The signal](02-the-signal.md).

**Carrier aiding.** Steering the code loop with the carrier loop's Doppler, since the motion that shifts the carrier compresses the code by the same fraction; the code loop then corrects only what is left. [Staying locked](05-tracking.md).

**Carrier phase.** The observable formed from the tracking loop's accumulated carrier cycles, precise to millimeters but with an unknown integer part. [Staying locked](05-tracking.md), [The measurements](07-measurements.md).

**CDMA.** Code-division multiple access: every satellite transmits on the same frequency and is told apart by its code alone. GLONASS's original signals are the exception. [The signal](02-the-signal.md).

**Chip.** One element of the spreading code. A C/A chip is 293 m long at the speed of light. [The signal](02-the-signal.md).

**Chi-square test.** A check of a fit's residuals: the sum of the squared residuals, each divided by its expected error, is compared with a threshold from the chi-square distribution for the number of measurements beyond the unknowns. A sum too large means a measurement is bad. [A standalone fix](08-standalone-fix.md).

**Choke ring.** A ground plane of concentric rings around a survey antenna that suppresses multipath, especially signals arriving from below the horizon. [From antenna to samples](03-antenna-to-samples.md).

**Circular polarization.** The polarization GNSS satellites transmit, right-hand (RHCP); a reflection off the ground at steep angles largely reverses it to left-hand (LHCP), which the antenna rejects. [From antenna to samples](03-antenna-to-samples.md).

**C/N0.** Carrier-to-noise-density ratio, in dB-Hz, the standard measure of GNSS signal strength. 35 to 50 dB-Hz in the open. [The signal](02-the-signal.md).

**Code phase.** Where in its repeating period the received code is at a given sample; the raw timing measurement. [Finding a satellite](04-acquisition.md).

**Coherent integration.** Summing correlator products with their signs and phases intact, so that the signal adds up in amplitude while noise adds up only in power; limited by data-bit flips, the frequency error and the oscillator. [The signal](02-the-signal.md), [Finding a satellite](04-acquisition.md), [Staying locked](05-tracking.md).

**Cold, warm and hot start.** A receiver start with no prior knowledge, which searches the whole acquisition grid; one with a rough position, the time and an almanac, which searches only where satellites are predicted; and one that also kept recent ephemerides and so skips the wait for them. [Finding a satellite](04-acquisition.md).

**Correlation.** Multiplying received samples by a replica code and summing. The operation that lifts the signal out of the noise and measures its timing. [The signal](02-the-signal.md).

**Correlator.** The part of a tracking channel that multiplies the samples by the replica and sums them over a code period, giving the in-phase and quadrature outputs $I$ and $Q$. [Staying locked](05-tracking.md).

**Costas loop.** A carrier phase-locked loop insensitive to data-bit sign flips. [Staying locked](05-tracking.md).

**Covariance.** For a set of estimates, the matrix whose diagonal holds each one's variance and whose other entries say how their errors move together. A least-squares fix and a Kalman filter both produce one. [A standalone fix](08-standalone-fix.md), [RTK](09-rtk.md).

**Cross-correlation.** Multiplying one signal by another, sample by sample or chip by chip, and adding the products, as a measure of how alike they are at a given alignment. Against a replica it finds a satellite in the noise. [The signal](02-the-signal.md), [Finding a satellite](04-acquisition.md).

**Cycle slip.** A jump in the accumulated carrier phase by a whole number of cycles, or by half a cycle in a Costas loop, when tracking loses count; it changes the ambiguity. [Staying locked](05-tracking.md), [The measurements](07-measurements.md), [RTK](09-rtk.md).

**Data channel, pilot channel.** The two components of a modern signal: one carries the message; the other carries no data, so once its known secondary code is removed it can be integrated coherently far longer, until the oscillator and the receiver's motion limit it. [The signal](02-the-signal.md).

**Decibel, dBm, dB-Hz.** A power ratio on a log scale (10 dB is ten times, 3 dB about double); dBm is a power relative to one milliwatt, and dB-Hz the unit of C/N0. [From antenna to samples](03-antenna-to-samples.md).

**Design matrix.** In a least-squares fit, the matrix with one row per measurement and one column per unknown, holding how each predicted measurement changes with each unknown. For a fix, each row is minus the unit vector to a satellite and a 1 for the clock. It is the Jacobian of the measurement model. [A standalone fix](08-standalone-fix.md), [RTK](09-rtk.md).

**Despreading.** Multiplying by the aligned replica code, which collapses the signal's power from the code's wide band back into a narrow one while the noise stays spread. [The signal](02-the-signal.md).

**Differential GNSS.** Using a receiver at a known location to measure the errors it shares with a nearby receiver, so that the second can remove them. RTK is its carrier-phase form. [What a GNSS is](01-what-gnss-is.md), [RTK](09-rtk.md).

**Direct-sequence spread spectrum.** Spreading a slow message over a wide band by multiplying it by a fast code, as every GNSS signal does. [The signal](02-the-signal.md).

**Discriminator.** In a tracking loop, a function of the correlator outputs whose value estimates the replica's error: how early or late its code is, or how far off its phase or frequency is. [Staying locked](05-tracking.md).

**DLL.** Delay-locked loop, the feedback loop that keeps the replica code aligned, using early, prompt and late correlators. [Staying locked](05-tracking.md).

**DOP.** Dilution of precision, the factor by which satellite geometry multiplies range error into position error, each the square root of a sum of diagonal entries of $(H^\mathsf{T}H)^{-1}$ for the design matrix $H$ (after rotating the position part to local east, north, up for HDOP and VDOP). GDOP covers position and clock, PDOP the position, HDOP and VDOP its horizontal and vertical parts, TDOP the clock. [What a GNSS is](01-what-gnss-is.md), [A standalone fix](08-standalone-fix.md).

**Doppler.** The frequency shift from the satellite's and receiver's relative motion, up to about 5 kHz; as an observable, the range rate, from which the receiver's velocity is solved. [Finding a satellite](04-acquisition.md), [The measurements](07-measurements.md).

**Doppler bin.** One trial frequency of the acquisition search; bins are spaced 500 Hz apart so that no satellite is more than 250 Hz from one. [Finding a satellite](04-acquisition.md).

**Double difference.** The difference of two receivers' measurements of two satellites, which cancels both receivers' and both satellites' clock errors. [RTK](09-rtk.md).

**Downconversion.** Moving a signal to a lower frequency by mixing it with a local oscillator, keeping the difference frequency. [From antenna to samples](03-antenna-to-samples.md).

**Early, prompt and late correlators.** Three correlators of one tracking channel, with the replica code slightly ahead of, at and slightly behind the current estimate; the difference of early and late tells the code loop which way to move. [Staying locked](05-tracking.md).

**ECEF.** Earth-centered, Earth-fixed coordinates: X, Y, Z from the center of the Earth, rotating with it. The frame positions are solved in. [A standalone fix](08-standalone-fix.md).

**Elevation mask.** The angle above the horizon below which satellites are ignored, typically 10 to 15 degrees. [A standalone fix](08-standalone-fix.md).

**Ephemeris.** The broadcast parameters describing one satellite's orbit, good to a meter or two for a few hours. [Reading the message](06-navigation-message.md), [A standalone fix](08-standalone-fix.md).

**Epoch.** One measurement instant, at which the receiver forms observables for every satellite it tracks. [The measurements](07-measurements.md).

**Error budget.** The list of error sources of a fix with the size each leaves after correction, combined into a range error and multiplied by the DOP. [What a GNSS is](01-what-gnss-is.md), [A standalone fix](08-standalone-fix.md).

**False alarm.** An acquisition detection that is only noise; the detection threshold trades false alarms against missed weak satellites. [Finding a satellite](04-acquisition.md).

**Fixed, float.** An RTK solution with the ambiguities resolved to integers, and one with them still real-valued. [RTK](09-rtk.md).

**FLL.** Frequency-locked loop, used to pull the carrier frequency in before the PLL can lock. [Staying locked](05-tracking.md).

**Frame synchronization.** Finding the subframe boundaries in the bit stream, from the preamble, the parity and the 300-bit spacing between preambles; it also settles the sign the Costas loop leaves open. [Reading the message](06-navigation-message.md).

**Friis formula.** The noise figure of a chain of stages, $F_1 + (F_2 - 1)/G_1$ for two, which shows why the first amplifier's noise figure dominates. [From antenna to samples](03-antenna-to-samples.md).

**Front-end.** The analog radio chain from the antenna to the analog-to-digital converter. [From antenna to samples](03-antenna-to-samples.md).

**Geometry-free combination.** The difference of one satellite's carrier phases on two frequencies, in meters. Range, clocks and troposphere cancel, leaving the ionosphere and the ambiguities, so a sudden jump marks a cycle slip. [RTK](09-rtk.md).

**GNSS.** Global Navigation Satellite System, the general term for GPS, Galileo, GLONASS and BeiDou. [What a GNSS is](01-what-gnss-is.md).

**Gold code.** The family of codes the GPS C/A codes belong to, each the chip-by-chip product of two shift-register sequences, chosen so that cross-correlations between members stay small. [The signal](02-the-signal.md).

**GPS time.** A continuous count of weeks and seconds since 6 January 1980, with no leap seconds. [Reading the message](06-navigation-message.md).

**Group delay.** The broadcast term, TGD, that accounts for the different paths of L1 and L2 inside the satellite; a single-frequency receiver applies it to the satellite clock. [Reading the message](06-navigation-message.md), [A standalone fix](08-standalone-fix.md).

**Handover word, telemetry word.** The first two words of every LNAV subframe: the telemetry word starts with the preamble, the handover word with the 17-bit time-of-week count. [Reading the message](06-navigation-message.md).

**HAS.** Galileo's High Accuracy Service, free precise orbit and clock corrections broadcast on E6. [The signal](02-the-signal.md), [Reading the message](06-navigation-message.md).

**IF.** Intermediate frequency, where the signal sits after the mixer, a few MHz instead of 1.5 GHz. [From antenna to samples](03-antenna-to-samples.md).

**Integer least squares.** Finding the integer vector closest to a real-valued estimate, measured in the metric of its covariance; the problem LAMBDA solves for RTK ambiguities. [RTK](09-rtk.md).

**Integrate and dump.** Summing the correlator products over one code period, reporting the sum and starting afresh. [Staying locked](05-tracking.md).

**IODE.** Issue of data, ephemeris: the number that identifies one set of broadcast ephemeris parameters. [Reading the message](06-navigation-message.md).

**Ionosphere, troposphere.** The two layers of atmosphere that delay the signal: the upper, charged one by a frequency-dependent amount, and the lower, neutral one by the same amount at every frequency. [A standalone fix](08-standalone-fix.md).

**I/Q.** In-phase and quadrature, the two components of a complex sample, and of a correlator output. [From antenna to samples](03-antenna-to-samples.md), [Staying locked](05-tracking.md).

**Jacobian.** The matrix of partial derivatives of a set of functions with respect to their unknowns; in a fix, the design matrix. [A standalone fix](08-standalone-fix.md).

**Kalman filter.** A recursive estimator that carries a state and its covariance from epoch to epoch: it predicts them forward, growing the covariance by the process noise, then corrects them with each epoch's measurements through the Kalman gain, which weighs the innovations (measured minus predicted) by the uncertainties. RTK engines use one for the float solution. [A standalone fix](08-standalone-fix.md), [RTK](09-rtk.md).

**Kepler elements.** The six numbers that describe an ideal elliptical orbit: semi-major axis, eccentricity, inclination, longitude of the ascending node, argument of perigee and mean anomaly. The core of the ephemeris. [Reading the message](06-navigation-message.md).

**Kepler's equation.** $M = E - e\sin E$, linking the mean anomaly $M$, which grows steadily with time, to the eccentric anomaly $E$, which locates the satellite on its ellipse; it is solved by iteration. [Reading the message](06-navigation-message.md).

**Klobuchar.** The simple ionosphere model whose eight parameters are broadcast in the GPS message. [A standalone fix](08-standalone-fix.md).

**LAMBDA.** The least-squares ambiguity decorrelation adjustment, the standard method for finding the integer ambiguities closest to a float solution: an integer change of coordinates decorrelates the ambiguities, then a search finds the best and second-best candidates. [RTK](09-rtk.md).

**Least squares.** Fitting unknowns to more measurements than unknowns by minimizing the sum of squared mismatches, in weighted least squares each divided by its expected error. A nonlinear problem such as a fix is linearized around a guess and solved repeatedly. [What a GNSS is](01-what-gnss-is.md), [A standalone fix](08-standalone-fix.md).

**LNA.** Low-noise amplifier, the first amplifier after the antenna element, usually inside an active antenna; it sets the noise performance of the whole chain. [From antenna to samples](03-antenna-to-samples.md).

**LNAV.** The legacy GPS navigation message on L1 C/A: 50 bit/s, 30-bit words, 6-second subframes. [Reading the message](06-navigation-message.md).

**Local oscillator.** The sine wave generated in the receiver, derived from the reference oscillator, that the mixer multiplies the signal by. [From antenna to samples](03-antenna-to-samples.md).

**Lock detector.** A test of whether a tracking loop is still locked, such as the C/N0 estimate or a phase lock indicator comparing the $I$ and $Q$ powers. [Staying locked](05-tracking.md).

**Lock, loss of lock.** The state of a tracking loop that is following the signal, and the moment it stops. [Staying locked](05-tracking.md).

**Loop bandwidth.** How quickly a tracking loop responds: a wide one follows fast changes but lets more noise through, a narrow one averages noise away but loses lock on fast changes. [Staying locked](05-tracking.md).

**Loop filter.** The part of a tracking loop that smooths the discriminator's output and turns it into a frequency command for the NCO. [Staying locked](05-tracking.md).

**Loss-of-lock indicator.** LLI, the flag a receiver reports with each carrier phase when lock was not continuous since the previous epoch, or when the phase may be half a cycle out; RINEX carries it. [The measurements](07-measurements.md), [RTK](09-rtk.md).

**Mapping function.** The factor that stretches a zenith delay to a satellite's elevation, about $1/\sin\varepsilon$ for elevation $\varepsilon$ in its simplest form. [A standalone fix](08-standalone-fix.md).

**Matched filter.** A filter that correlates its input against a known copy of the wanted signal; in white noise no other linear filter gives a higher signal-to-noise ratio. [The signal](02-the-signal.md).

**Medium Earth orbit.** The orbits of the navigation satellites, about 19,100 to 23,200 km up, between low orbits and geostationary ones. [What a GNSS is](01-what-gnss-is.md).

**Mixer.** The front-end stage that multiplies the signal by a local oscillator, moving it down to the intermediate frequency. [From antenna to samples](03-antenna-to-samples.md).

**Multipath.** Signal reaching the antenna by reflection as well as directly. [From antenna to samples](03-antenna-to-samples.md), [The measurements](07-measurements.md), [A standalone fix](08-standalone-fix.md).

**NCO.** Numerically controlled oscillator, the digital oscillator that generates a tracking loop's replica: it adds a step proportional to the commanded frequency to its phase at every sample. [Staying locked](05-tracking.md).

**Noise figure.** How much noise an amplifier adds, in dB: a noise figure of 1 dB adds 26 percent to the noise. [From antenna to samples](03-antenna-to-samples.md).

**Non-coherent integration.** Adding the powers or magnitudes of several correlations, which averages noise down without needing the carrier phase or the data bits to stay constant, at a smaller gain per millisecond than coherent integration. [Finding a satellite](04-acquisition.md), [Staying locked](05-tracking.md).

**NTRIP.** The internet protocol for streaming RTCM corrections from a base station. [RTK](09-rtk.md).

**Nyquist criterion.** The rule that the sample rate must be more than twice the highest frequency present, or aliasing occurs. [From antenna to samples](03-antenna-to-samples.md).

**Observable.** A measurement a receiver reports: pseudorange, carrier phase, Doppler, C/N0. [The measurements](07-measurements.md).

**Parallel code phase search.** The acquisition method that computes the correlation at every code phase at once, by FFTs: a circular correlation becomes a multiplication in the frequency domain. [Finding a satellite](04-acquisition.md).

**Parity.** The 6 check bits at the end of every 30-bit LNAV word, a Hamming code computed from the word's data bits and the end of the previous word, so a single wrong bit fails the check. [Reading the message](06-navigation-message.md).

**Phase center.** The point an antenna effectively receives at, which moves slightly with direction and frequency. [From antenna to samples](03-antenna-to-samples.md).

**Phase noise.** An oscillator's short-term wobble in phase, over tens of milliseconds; it limits coherent integration and the quality of carrier phase. [From antenna to samples](03-antenna-to-samples.md).

**PLL.** Phase-locked loop, the feedback loop that keeps the replica carrier in phase with the received one. [Staying locked](05-tracking.md).

**PPK.** Post-processed kinematic: RTK computed afterwards from logged files. [RTK](09-rtk.md).

**PPP.** Precise point positioning: a single receiver with precise orbits and clocks instead of a base station. [RTK](09-rtk.md).

**Preamble.** The fixed 8-bit pattern 10001011 at the start of every LNAV subframe, used for frame synchronization and to settle the bit sign. [Reading the message](06-navigation-message.md).

**PRN.** Pseudo-random noise, the family of spreading codes, and by extension the number identifying a satellite's code. [The signal](02-the-signal.md).

**Processing gain.** The improvement in signal-to-noise ratio that correlation gives, a factor equal to the number of chips summed: 1023, about 30 dB, for one C/A code period, measured in a band as wide as the chip rate (about 33 dB in the full 2 MHz band). [The signal](02-the-signal.md).

**Pseudorange.** The travel time of the signal times the speed of light, including the receiver's unknown clock error. The basic measurement. [What a GNSS is](01-what-gnss-is.md), [The measurements](07-measurements.md), [A standalone fix](08-standalone-fix.md).

**PVT.** Position, velocity and time, the output of a standalone solution. [A standalone fix](08-standalone-fix.md).

**Quantization.** Rounding each sample to a few levels; the error behaves like a little extra noise, about 2 dB of sensitivity for 1 bit and half a decibel for 2. [From antenna to samples](03-antenna-to-samples.md).

**RAIM.** Receiver autonomous integrity monitoring: a receiver checking its own fix for consistency, from the residuals, and dropping a measurement that does not fit. [A standalone fix](08-standalone-fix.md).

**Ratio test.** The check that the best integer ambiguity candidate fits the data much better than the second best, by the ratio of their squared distances, usually against a threshold of 3, before a fix is accepted. [RTK](09-rtk.md).

**Relativistic correction.** The periodic satellite clock term, tens of nanoseconds, from the orbit's slight eccentricity, computed from the ephemeris; the steady relativistic offset is built into the satellite clocks. [Reading the message](06-navigation-message.md), [A standalone fix](08-standalone-fix.md).

**Replica.** The receiver's own copy of a satellite's code, and of its carrier, which it correlates against the incoming samples. [The signal](02-the-signal.md), [Finding a satellite](04-acquisition.md).

**Residual.** A measurement minus the value the model predicts for it at the current solution. [A standalone fix](08-standalone-fix.md).

**RINEX.** The text file format for exchanging observables and ephemerides between receivers and programs. [The measurements](07-measurements.md).

**RTCM.** The binary format for streaming a base station's observables and corrections to a rover. [RTK](09-rtk.md).

**RTK.** Real-time kinematic: carrier-phase differential positioning against a base station, to a centimeter or two. [RTK](09-rtk.md).

**Saastamoinen.** The standard troposphere delay model. [A standalone fix](08-standalone-fix.md).

**Sagnac correction.** The term that accounts for the Earth's rotation while the signal is in flight, about 70 ms, either as a small extra term in the range or as a rotation of the satellite position. [A standalone fix](08-standalone-fix.md).

**SBAS.** Satellite-based augmentation, meter-level corrections broadcast from geostationary satellites (WAAS, EGNOS). [RTK](09-rtk.md).

**Secondary code.** A slow, known sign pattern laid over the primary code of many pilot signals, which the receiver must align with and remove before integrating longer. [The signal](02-the-signal.md), [Finding a satellite](04-acquisition.md).

**Single difference.** The difference of two receivers' measurements of one satellite, which cancels the satellite's clock. [RTK](09-rtk.md).

**Spreading code.** The pseudo-random chip sequence that identifies a satellite and carries the timing. [The signal](02-the-signal.md).

**Subframe, frame.** The 300-bit, 6-second unit of the LNAV message, and the five-subframe, 30-second group. [Reading the message](06-navigation-message.md).

**TCXO.** Temperature-compensated crystal oscillator, the usual reference oscillator in a receiver. [From antenna to samples](03-antenna-to-samples.md).

**Thermal noise.** The noise a resistor delivers, $kT$ watts per hertz for Boltzmann's constant $k$ and temperature $T$: -174 dBm per hertz at the 290 K reference temperature. An antenna delivers about as much, somewhat less when it looks mostly at the cold sky. [From antenna to samples](03-antenna-to-samples.md).

**Time of week.** GPS time within the current week, in seconds; the count in each subframe that makes the transmit time unambiguous; the handover word carries it as a count of 6-second units. [Reading the message](06-navigation-message.md).

**Tracking.** The continuous following of a satellite's code and carrier after acquisition. [Staying locked](05-tracking.md).

**Trilateration.** Finding a position from distances to points whose positions are known. [What a GNSS is](01-what-gnss-is.md), [A standalone fix](08-standalone-fix.md).

**TTFF.** Time to first fix, from power-on to the first position; it depends on whether the start is cold, warm or hot. [Finding a satellite](04-acquisition.md), [Reading the message](06-navigation-message.md).

**UERE.** User equivalent range error, the combined error of one pseudorange from all sources, which the DOP multiplies into position error. [A standalone fix](08-standalone-fix.md).

**Week number.** The GPS week count; in LNAV it is 10 bits and rolls over every 1024 weeks, about 19.6 years. [Reading the message](06-navigation-message.md).

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

- IS-GPS-200, the GPS L1/L2 interface specification, and IS-GPS-705 for L5, at <https://www.gps.gov/interface-control-documents-icds-interface-specifications-iss>.
- Galileo Open Service Signal-in-Space Interface Control Document, at <https://www.gsc-europa.eu/electronic-library/programme-reference-documents>.
- RINEX 3 and 4 format descriptions, at <https://igs.org/formats-and-standards/>.

Back to the [index](../index.md).
