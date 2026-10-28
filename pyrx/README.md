# pyrx

An offline GNSS receiver in Python. It reads recorded IF samples and will grow stage
by stage into a full receiver: acquisition, tracking, navigation-message decode,
observables, a standalone position fix, and finally an RTK fix. Acquisition,
tracking, navigation-message decode and observables written as RINEX exist so
far. Each stage is checked against an external reference before the next one is
started: Pocket SDR for acquisition, tracking and observables, broadcast ephemeris
files for the navigation decode, RTKLIB for positioning.

## Why Python

This code exists to prove the receiver's algorithms on real signals before they are
ported to the FPGA correlators and the MCU firmware, and then to serve as the
floating-point reference those are checked against. Expressiveness and plotting
matter more than speed here. numpy and scipy supply FFTs, filters and linear
algebra; where profiling shows an inner loop dominating, numba is the escape hatch
that keeps everything in one language. Real-time operation is not a goal.

## Install

```sh
pip install -e '.[dev]'
```

## Tests

```sh
pytest
```

The reference tests compare acquisition, tracking and observables with Pocket
SDR's results on its sample recordings, the navigation decode with the day's
broadcast ephemeris, and the position RTKLIB computes from our RINEX with the one
it computes from Pocket SDR's. They need the environment variable `PYRX_CAPTURES`
pointing at a directory holding these files, and skip when it is unset (the fix
positioning and RTK tests also skip when `rnx2rtkp` is not on the path; the RTK tests take
a few minutes):

| File | Source | SHA-256 |
|---|---|---|
| `L1_20211202_084700_4MHz_IQ.bin` | Pocket SDR sample set, GPS L1, 4 MHz I/Q, 100 ms | `0a8335d2f099e388b474d2afcca1ff91f61cde550dd32bf82fdf199d8a5b8033` |
| `L1_20211201_054600_24MHz_I.bin` | Pocket SDR sample set, GPS L1, 24 MHz real, 6 MHz IF, 100 ms | `05c771f0c152e2bd11ccba0715198afc0aa56e5d420bec0993fa720ac8d508d0` |
| `L1_20211226_082401_24MHz_I.bin` | Pocket SDR sample set, L1+L5 recording of 2021-12-26, GPS L1, 24 MHz real, 6 MHz IF, 30 s | `16c83d58716ae9ba577aac29344766373accb98b34b729afdac8cc2e1352252c` |
| `L1_20220114_004440_8MHz_I.bin` | Pocket SDR sample set, L1+L2 recording of 2022-01-14, GPS L1, 8 MHz real, 2 MHz IF, 30 s | `388c550e2694067566c24fddae61abd1b316eb4bbd0820e0e8fff1818f164de7` |
| `brdc/BRDC00WRD_R_20213600000_01D_GN.rnx` | BKG (`igs.bkg.bund.de/root_ftp/IGS/BRDC/2021/360/`), worldwide GPS broadcast ephemeris, RINEX 3.05, 2021-12-26 | `92fe0754c16b3408691facabad44390ebfa2031ad641334ebe6f8660f8e554c2` |
| `brdc/BRDC00WRD_R_20220140000_01D_GN.rnx` | BKG (`.../BRDC/2022/014/`), worldwide GPS broadcast ephemeris, RINEX 3.05, 2022-01-14 | `ac1488e8b577ddd9b942de2ca8033fa82f8ce56abfe9d6e1781288cfa590d7e2` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.ctrk.log` | Pocket SDR C tracker (`pocket_trk`) log of the 2021-12-26 recording: `$CH` channel state every 0.1 s with the transmit time of week, `$OBS`, `$POS` | `8fcf3d029769683bda4b1fa06c060c6bb61235ea94235d4795d38e07ac298dbf` |
| `pocketsdr-trk/L1_20220114_004440_8MHz_I.ctrk.log` | the same for the 2022-01-14 recording (`$CH` only: no subframe 1 in the window, so Pocket SDR forms no observations) | `3308cc4bc7273ba09ac1a14879b9bce7dca9f696119c0718f48b1a2cb4c09e92` |
| `igs/USUD00JPN_R_20213600000_01D_30S_MO.crx.gz` | BKG (`.../IGS/obs/2021/360/`), IGS station USUD (Usuda, Japan), 30 s observations of 2021-12-26, RINEX 3.04, Hatanaka-compressed; no Doppler logged | `b964f1f0a51ffca461712575c6dbfe915b7eac65cb0f1513c46c75e2dc78e1f2` |
| `igs/USUD00JPN_R_20213600000_01D_GN.rnx.gz` | the station's GPS navigation file of that day from the same directory; the one with the `GPSA`/`GPSB` ionosphere lines, so it is the navigation input for the station and for the 2021-12-26 recording | `911fe136cd28a51ba5e0f81c5494af8c32b3cdf26bac7b68f6d34821a1105649` |
| `igs/igs21900.sp3` | IGS final orbits and clocks of 2021-12-26, 15 min, from BKG (`.../IGS/products/orbits/2190/igs21900.sp3.Z`, decompressed) | `fdb9aae343289d97eec2241af7ee4cac0c774f33fb0d445f1fb1b1c95ed75072` |
| `igs/usud00jpn_20260422.log` | IGS site log of USUD (`files.igs.org/pub/station/log/`): the marker's ITRF coordinates | `80c41b22ee394448803a869da5757eae68ba3bb65aca2d5edd09470236892e41` |
| `igs/MATE00ITA_R_20213600000_01D_30S_MO.crx.gz` | BKG (`.../IGS/obs/2021/360/`), IGS station MATE (Matera, Italy), base of the 11 m pair, 2021-12-26, 30 s, RINEX 3.04, Hatanaka-compressed | `aa785104fa1352ca1f7fbb296f516982ae0fc904d3240a8fededc846efe7ab19` |
| `igs/MAT100ITA_R_20213600000_01D_30S_MO.crx.gz` | MAT1, rover of the Matera pair, same receiver and antenna types as MATE, 11.0 m away | `c776780c6437a7f26ad16043adc486cfcbdefcf38a45529198ed79c9b45ccacb` |
| `igs/TLSE00FRA_R_20213600000_01D_30S_MO.crx.gz` | TLSE (Toulouse, France), base of the 1266 m pair, same day and format | `4c9a77cbd605c7283b374e3e70e409f0e9f5a2959e7d45331a63f7f639445178` |
| `igs/TLSG00FRA_R_20213600000_01D_30S_MO.crx.gz` | TLSG, rover of the Toulouse pair, same antenna type as TLSE (different receivers), 1266 m away | `a41a6ffa9dc37db85a583926c3990f789a914fe88ce41d2bf3fcebf403780bf6` |
| `igs/BRDC00WRD_R_20213600000_01D_MN.rnx.gz` | BKG (`.../IGS/BRDC/2021/360/`), worldwide mixed broadcast ephemeris (GPS and Galileo), RINEX 3.05, 2021-12-26 | `d69a64d9cdda9b2defa6a74405c600baf99c69ae640cc612943bfdd878d806ee` |
| `igs/EUR0E03SNX_20213600000_07D_07D_SOL.SNX.gz` | BKG (`.../EUREF/products/2190/`), EUREF combined weekly SINEX of GPS week 2190: the marker coordinates of the four stations, the truth of the RTK tests | `07f0a8577aec588869fd8db02ebfd56592e480be4b6088da3b8ad98da2335f18` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.obs` | Pocket SDR's observations of the 2021-12-26 recording as RINEX 3.05, converted from its RTCM3 output with `convbin` | `42c66c39f6c62753aa4dae481a72de42f6fd2958018412abae270cee4ee07cb1` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.rnx2rtkp.pos` | RTKLIB `rnx2rtkp` single-point fix from that file with the broadcast ephemeris, the reference for our fix | `304f6d4ad6662085e09668d789e51aa4aa52073e1d0a723da1cfea4494cd841e` |
| `brdc/BRDC00IGS_R_20220140000_01D_MN.rnx` | BKG (`.../BRDC/2022/014/`), IGS combined multi-GNSS broadcast file, RINEX 3.04, 2022-01-14; only its GPSA/GPSB header lines are used | `5e366723dcda6bcfb049b5bc0459284d27581855567c1ff977de7cf76e1f701d` |

The 100 ms 24 MHz recording has sample gaps near 40 ms and 90 ms; the acquisition
tests use its first 10 ms, which is also what Pocket SDR integrates by default.

The Pocket SDR tracking logs in the table were produced with Pocket SDR's
`pocket_trk.py` (its library built with `USE_SOAPY=0 USE_FFTW=1`), run from the
`pocketsdr-trk/` directory with the recordings one level up:

```sh
pocket_trk.py -sig L1CA -prn 4,16,18,26,27,29,31 -f 24 -fi 6.0 -q -log L1_20211226_082401_24MHz_I.trk.log ../L1_20211226_082401_24MHz_I.bin
pocket_trk.py -sig L1CA -prn 10,12,15,23,24,32  -f 8  -fi 2.0 -q -log L1_20220114_004440_8MHz_I.trk.log  ../L1_20220114_004440_8MHz_I.bin
```

The PRN lists are the satellites `pyrx-acq` reports at or above 38 dB-Hz in the
first 10 ms of each file.

The Pocket SDR C tracker files were produced from the same directory (`-fo` being
the LO frequency, L1 minus the IF, in MHz):

```sh
pocket_trk ../L1_20211226_082401_24MHz_I.bin -fmt INT8 -f 24 -fo 1569.42 -sig L1CA -prn 1-32 -ti 0 -log L1_20211226_082401_24MHz_I.ctrk.log -rtcm L1_20211226_082401_24MHz_I.ctrk.rtcm3
pocket_trk ../L1_20220114_004440_8MHz_I.bin  -fmt INT8 -f 8  -fo 1573.42 -sig L1CA -prn 1-32 -ti 0 -log L1_20220114_004440_8MHz_I.ctrk.log
convbin -r rtcm3 -v 3.05 -tr 2021/12/26 08:00:00 -o L1_20211226_082401_24MHz_I.psdr.obs -n L1_20211226_082401_24MHz_I.psdr.nav L1_20211226_082401_24MHz_I.ctrk.rtcm3
rnx2rtkp -p 0 -t -o L1_20211226_082401_24MHz_I.psdr.rnx2rtkp.pos L1_20211226_082401_24MHz_I.psdr.obs ../brdc/BRDC00WRD_R_20213600000_01D_GN.rnx
```

## Sample format

Signed 8-bit samples. Real recordings hold one byte per sample. I/Q recordings
interleave I and Q bytes and the complex sample is `I - jQ`, the Q sign convention
of the Pocket SDR front-end.

## Acquisition from the command line

```sh
pyrx-acq FILE --fs 4e6 --iq [--f-if 0] [--prn 1-32] [--duration 0.01] [--plot DIR]

By default the first 10 ms of the file are integrated (`--duration 0` for the whole file).
```

Prints one line per PRN in the same format as Pocket SDR's `pocket_acq.py`, so the
two outputs can be diffed. With `--plot`, writes `cn0.png` and one search-grid
image per detected PRN.

## Tracking from the command line

```sh
pyrx-trk FILE --fs 8e6 --f-if 2e6 [--iq] [--prn 1-32] [--duration 0] [--log trk.csv] [--plot DIR]
```

Acquires each PRN on the first 10 ms, tracks the detected ones through the file
(`--duration 0`, the default, means all of it), each channel in its own process,
and prints the final C/N0, Doppler and lock state per PRN. `--log` writes a CSV
with columns `time_s,prn,cn0_dbhz,code_offset_ms,doppler_hz,carrier_cycles`, one
row per channel every 0.1 s. `--plot` writes `trk_prnNN.png` per channel with four
panels: prompt I and Q over the first 2 s (the navigation data bits are visible
once the phase lock settles), Doppler, C/N0, and the early, prompt and late
correlator magnitudes.

## Navigation message decode from the command line

```sh
pyrx-nav FILE --fs 8e6 --f-if 2e6 [--iq] [--prn 1-32] [--duration 0] [--json nav.json]
```

Acquires and tracks as `pyrx-trk` does, then decodes each channel's navigation
message and prints one line per PRN: whether bit sync was found, the subframes
decoded as `ID@TOW` (the GPS seconds of week at the subframe's first bit), and the
ephemeris issue of data and `toe` when subframes 1, 2 and 3 with the same issue
were seen. `--json` writes `{"ephemeris": {"<prn>": {field: value, ...}},
"ionosphere": {"alpha": [...], "beta": [...]} | null}` with the complete
ephemerides only.

## RINEX output from the command line

```sh
pyrx-rinex FILE --fs 24e6 --f-if 6e6 [--iq] [--prn 1-32] [--interval 1.0] [--week N] [--marker NAME] --obs out.obs [--nav out.nav]
```

Acquires, tracks and decodes as `pyrx-nav` does, forms observations at every
whole multiple of `--interval` seconds of GPS time, and writes them as a RINEX 3.05
observation file with `C1C L1C D1C S1C` for every channel that has a decoded
subframe. With `--nav`, the decoded ephemerides (and the ionosphere parameters when
seen) are written as a RINEX navigation file. The GPS week comes from subframe 1;
when the recording holds none, give it with `--week`. One line per PRN reports how
many epochs the channel contributed to and its first pseudorange and Doppler, then
the number of epochs and their span.

## Positioning from the command line

```sh
pyrx-pvt OBS NAV [--mask 15] [--ts "YYYY-MM-DD HH:MM:SS"] [--te "YYYY-MM-DD HH:MM:SS"] [-o fix.txt] [-v]
```

Solves every epoch of a RINEX observation file with the ephemerides of a RINEX
navigation file. The inputs can be our own files or public RINEX 3 files, plain,
gzip or Hatanaka compressed; only GPS L1 C/A is used. `--ts` and `--te` bound the
epochs in GPS time, `--mask` is the elevation mask in degrees. The command prints
the number of epochs solved and read, the mean latitude, longitude, height and
receiver clock bias, and the run time; with `-v` one line per solved epoch; with
`-o` those lines go to a file whose header line (starting with `%`) names the
columns: `week tow lat lon height clock_s ns gdop vx vy vz ddtr` (degrees, metres,
seconds, satellites used, geometric dilution of precision, ECEF velocity in m/s and
clock drift in s/s, `nan` without Doppler). It returns 2 when no epoch solves.

## RTK from the command line

```sh
pyrx-rtk ROVER BASE NAV --base-xyz X Y Z [--base-delta E N U] [--rover-delta E N U] [--signals G1C,G2W] [--mode kinematic|static] [--mask 15] [--ts "YYYY-MM-DD HH:MM:SS"] [--te ...] [-o fix.txt] [-v]
```

Solves the baseline from a base station of known position to the rover, epoch by
epoch, from two RINEX observation files (plain, gzip or Hatanaka compressed, ours
or public) and a navigation file. `--base-xyz` is the base marker; the
marker-to-antenna offsets are taken from each file's `ANTENNA: DELTA H/E/N` unless
given. `--signals` lists the signals to use as system letter plus RINEX attribute
(GPS `G1C G2W G5Q`, Galileo `E1C E5Q E7Q`); the seed needs a GPS L1 signal. The
command prints the numbers of fixed and float epochs, the mean fixed baseline in
the base's east-north-up frame and the median ratio, then the run time; `-v`
prints one line per solved epoch; `-o` writes those lines with the header
`% week tow x y z Q ns ratio age` (rover marker in ECEF metres, `Q` 1 fixed 2
float, satellites used, ratio, age of the base epoch). It returns 2 when no
epoch solves.

## Checking the fix

```sh
rnx2rtkp -k single.conf -y 2 -o fix.pos out.obs brdc.rnx
```

RTKLIB's single-point solution from our observation file and the day's broadcast
ephemeris gives a position per epoch; the reference test compares it with the same
solution from Pocket SDR's observations of the 2021-12-26 recording (five
satellites once both have all of them: horizontal within 5 m, vertical within
10 m, pseudorange residual RMS within a factor of two). Without a configuration
file (`rnx2rtkp -p 0`) RTKLIB applies no ionospheric or tropospheric model, which
is how that comparison is made. The positioning tests compare `pyrx-pvt` with
RTKLIB running the same models, from a configuration file with

```
pos1-posmode=single
pos1-elmask=15
pos1-navsys=1
pos1-ionoopt=brdc
pos1-tropopt=saas
pos1-sateph=brdc
out-solformat=xyz
out-outhead=on
out-timesys=gpst
out-timeform=tow
out-outstat=residual
```

(`-y 2` writes the `.stat` file with the residuals and the velocity).

For RTK, `rnx2rtkp -k rtk.conf -y 2 -o fix.pos rover.obs base.obs brdc.rnx` with

```
pos1-posmode=kinematic        # or static
pos1-frequency=l1+l2          # l1, or l1+l2+l5 for GPS L5 and Galileo E5a
pos1-soltype=forward
pos1-elmask=15
pos1-snrmask_r=off
pos1-dynamics=off
pos1-tidecorr=off
pos1-ionoopt=brdc
pos1-tropopt=saas
pos1-sateph=brdc
pos1-navsys=1                 # 9 for GPS and Galileo
pos2-armode=continuous
pos2-arthres=3
pos2-arlockcnt=0
pos2-arelmask=0
pos2-arminfix=10
pos2-elmaskhold=0
pos2-aroutcnt=5
pos2-maxage=30
pos2-slipthres=0.05
pos2-rejionno=30
pos2-niter=1
out-solformat=xyz
out-outhead=on
out-timesys=gpst
out-timeform=tow
out-outstat=residual
ant1-antdelu=<rover ARP height above its marker>
ant2-postype=xyz
ant2-pos1=<base marker X>
ant2-pos2=<base marker Y>
ant2-pos3=<base marker Z>
ant2-antdelu=<base ARP height above its marker>
```

is the configuration `pyrx-rtk` mirrors (RTKLIB applies the antenna offsets as
range corrections, which agrees with moving the stations to their antenna
reference points to well under a millimetre). Epochs where RTKLIB finds no
relative solution produce no output line. USUD is
29 km from the site of both recordings, which is why its day serves as the
long-run check.

## RTK

Observations enter as arrays of every satellite by every requested signal, with
the RINEX loss-of-lock indicators. The base station is its marker plus the antenna
offset; the rover's single-point fix seeds each epoch, solved by the standalone
solver on GPS L1 C/A when it is among the signals, else on the first GPS signal
listed (no GPS signal is an error).

The state is the rover antenna position and one single-differenced (rover minus
base) carrier-phase ambiguity in cycles per satellite and signal. In kinematic mode
the position is reset to the seed with a 30 m sigma every epoch; in static mode it
is carried. Ambiguities are created from phase minus code with a 30-cycle sigma,
removed when a satellite misses more than five epochs that reach the relative
solution, on a loss-of-lock flag at either receiver, or when the single-differenced
geometry-free combination of a satellite's first and another signal jumps by more
than 5 cm.

Residuals use the geometric range with the Earth-rotation term to the antenna
reference point, the satellite clock with its relativistic term, and the
hydrostatic Saastamoinen zenith delay mapped with the Niell dry function. Nothing
else is applied: no ionosphere, no wet troposphere, no group delays, no antenna
phase-centre model (the reference pairs have identical antennas). This is what
RTKLIB's single-baseline mode applies with its default options, so the two solvers
compare number for number on the same files; the deliberate differences (existing
ambiguities aligned to the current clock per signal rather than per frequency, the
geometry-free slip test run past a missing band, the tie order of two reference
candidates at the same elevation) are listed in `rtk.py` and do not change the
default-configuration comparisons. Double differences are formed per system against
the highest satellite, carrier phase and code, with RTKLIB's single-difference
variance `2 (a² + b² / sin² el)` (3 mm for phase, 100 times that for code) plus a
receiver-clock-stability term in the age of the base epoch, and the block
covariance of a shared reference; a double difference beyond 30 m is dropped. One
Kalman update per epoch, linearised at the seed; the update stands but the epoch
reports no solution when fewer than four satellites of the first signal were used.

Integer ambiguities are searched with LAMBDA (decorrelation and a depth-first
search keeping the two best candidates) over the double differences of the
satellites used this epoch whose ambiguity has survived at least one validated
epoch, against the oldest surviving ambiguity of each system and signal (the
choice of reference is a change of basis and does not change the fixed position).
A fix is accepted when at least two ambiguities were searched and the ratio of the
second to the best squared norm is at least 3, or the best norm is zero; the float
states are kept (continuous mode, no hold). Galileo E1/E5a and GPS L5 run through
the same engine with a reference satellite per system.

The reference tests run on the first 6 h (721 epochs) of two IGS station pairs of
2021-12-26, MATE–MAT1 (Matera, 11 m) and TLSE–TLSG (Toulouse, 1266 m), against the
EUREF combined weekly SINEX coordinates of the markers and against RTKLIB with the
configuration above. The six GPS cases (each pair kinematic on L1+L2, kinematic on
L1 and static on L1+L2) assert that the set of fixed epochs equals RTKLIB's; that
the fixed positions are within 1 cm of the marker in the mean (static 5 mm), 2 cm
at the 95th percentile and 3 cm at worst (static 2 cm); and that they agree with
RTKLIB's fixed positions within 5 mm at the 95th percentile and 1 cm at worst.
Time to first fix after restarts every 30 min must stay within twice RTKLIB's mean
and within one epoch of it, with every restart fixing on both sides. GPS L1/L2/L5
with Galileo E1/E5b/E5a on the Matera pair must fix within 1 % of RTKLIB's count
with at least ten satellites on average, within 1 cm mean and 2 cm worst of the
marker and within 1 cm at the 95th percentile and 2 cm worst of RTKLIB's fixed
positions. The hardware's own set, GPS L1/L5 with Galileo E1/E5a, which RTKLIB
cannot run, must fix at least 99 % of the epochs within the same marker bounds.
Measured, the fixed positions agree with RTKLIB's within 2.6 mm in the GPS cases
and 7.8 mm in the multi-GNSS case, and the hardware set fixes 721 of 721.

## Standalone positioning

Satellite position, velocity and clock come from the broadcast ephemeris
(IS-GPS-200 Table 20-IV with the relativistic clock term; velocity and clock drift
by central differences over 1 ms). For each satellite the record whose `toe` is
nearest the epoch is used, within 2 h; a satellite with no such record or with a
health flag set is left out. Times of week are reduced into half a week either way
before any difference with an ephemeris time, so a file crossing the week boundary
is handled.

The ionosphere is corrected with the Klobuchar model from the navigation file's
`GPSA`/`GPSB` coefficients; a file without them (the worldwide GPS-only broadcast
files, or our own navigation file from a recording too short to hold subframe 4
page 18) gets RTKLIB's default coefficients, as `rnx2rtkp` applies. The
troposphere is corrected with Saastamoinen using the
standard atmosphere at the receiver height and 70 % humidity.

Position and receiver clock bias are found by iterated weighted least squares over
the pseudoranges, each corrected for the satellite clock and group delay (`TGD`),
with the Earth-rotation term in the range. The weights, the 15 degree elevation
mask, the chi-square test on the residuals (0.1 % significance) and the GDOP limit
of 30 are those of RTKLIB's single-point mode, so the two solvers compare number
for number on the files tested; the one known difference is that RTKLIB also
leaves out a satellite whose URA index is 10 or more (an accuracy worse than
300 m), which this solver does not. An epoch with fewer than four usable
satellites or failing a check gets no solution and the run continues. Velocity and clock drift then come
from an unweighted least-squares fit of the Dopplers of the satellites used, when
at least four have one.

The reference tests check the broadcast satellite positions and clocks against the
IGS final orbit over a day (within 6 m and 20 ns at every 15 min epoch of every
satellite, at least 1300 of them), and our fix against RTKLIB's with the same
models: on our two recordings within 1 m at every epoch with the same satellites
used (velocity within 0.1 m/s on the 2021-12-26 recording, the only one with
Doppler); on a day of the IGS station USUD, 29 km from the recording site, within
1 m at every common epoch with the same satellites, where at most two epochs may
differ in satellite count (a satellite at the mask) and are then allowed 5 m, and
with 2 mm in the mean and 1 mm at the 99th percentile over the day (measured 0.9 mm
and 0.08 mm: the two solvers apply the same models number for number); the same
bounds over an hour with a navigation file without ionosphere coefficients; and
the day's mean position at USUD within 5 m of the station's published coordinates.

## Observables

Every tracking epoch integrates one code period, so once a subframe has tagged one
epoch with its GPS time, the code start of epoch k was transmitted at
`tow + (k - epoch) * 1 ms` for every epoch of the channel. The receiver clock is
set once per recording: sample 0 is given the GPS time of the latest transmit time
seen at sample 0 plus 68 ms of nominal travel time, which puts pseudoranges near
their true values and leaves the remaining bias to the position solution.
Observation epochs are the receiver times at which that clock reads a multiple of
the interval; a channel is sampled there by interpolating between the two code
starts that bracket the epoch, from its first bit-synced epoch on. Pseudorange is
the light time between receiver and transmit time; carrier phase is the negated
accumulated Doppler phase (it grows with range) less half a cycle while the Costas
loop is locked on inverted data, with every channel's carrier NCO referenced to the
receiver's IF phase at sample 0 so the phases of different channels share one
origin; Doppler is positive for an approaching satellite.

Time: an epoch's `tow` is seconds from the start of its `week`. Within one run it
increases monotonically and may reach or exceed 604800 when a recording crosses
the week boundary; the week field does not change within a run. The RINEX reader
returns `tow` in [0, 604800) with the week of the time stamp. Consumers form a
continuous time as `week * 604800 + tow` when they need one, and reduce every
difference against an ephemeris time of week into half a week either way.

Carrier phase: a channel's phase, and so its RINEX `L1C`, runs from the channel's
first bit-synced epoch to the end of the run or to loss of lock; a lost channel is
not re-acquired within a run. The data polarity, and with it the half cycle added
to the phase, is read from the preamble of every decoded subframe: a subframe of
the opposite polarity means the Costas loop slipped half a cycle since the previous
one. The loss-of-lock indicator written with `L1C` follows Pocket SDR's: bit 1
(slip possible) on the observations within the first 2 s of a channel and on the
first observation after a polarity change; bit 2 (half-cycle ambiguity) where no
decoded subframe establishes the polarity, that is before a channel's first
subframe and between the last subframe before a polarity change and the first
after it. A slip that leaves the polarity unchanged (a whole cycle) is not detected
inside a file. No signal-strength flags are written. The standalone solution does
not use the carrier phase.

## Navigation message decode

GPS L1 C/A carries the 50 bps LNAV message. Once the phase lock has settled, bit
sync finds the data-bit edge among the 20 one-millisecond epochs of a bit by
counting prompt sign changes at each epoch phase; frame sync then looks for the
TLM preamble repeating one subframe (300 bits, 6 s) later with both words of each
candidate passing parity, a valid subframe ID and consecutive TOW counts, trying
both data polarities. From there every 300-bit block whose ten words all pass
parity is a subframe; a parity failure drops that block only. Subframes 1, 2 and 3
decode to the clock and ephemeris parameters, subframe 4 page 18 to the
ionosphere parameters, all scaled per IS-GPS-200 with semi-circles converted to
radians. Each subframe is time-tagged with the tracking epoch of its first bit and
the GPS seconds of week of that bit (the HOW's TOW count refers to the start of
the following subframe). Almanac pages and the UTC parameters are not decoded.

The reference test decodes the two 30 s recordings and requires the clock and
orbit fields of every subframe 1, 2 and 3 (and the week number; not the URA, the
L2 code flags, the fit interval or AODO, which the files do not carry exactly) to
equal the broadcast record with the same issue of data to within half a
transmitted LSB (some records in the files are written to only seven significant
digits), every subframe's time tag to match its position in the file within
0.1 s, every channel to agree on the file start time to within 30 ms, and the
2021-12-26 recording to yield a complete ephemeris on every channel. The 2022-01-14 recording starts inside subframe 1, so its window holds
subframes 2 to 5 and no complete ephemeris. Neither window contains subframe 4
page 18, so the ionosphere parameters are checked only by the synthetic
round-trip tests.

## Tracking loops

Each channel integrates 1 ms (one C/A code period) with early and late taps at
±0.25 chip and a noise tap 10 chips away. A frequency-locked loop pulls in for
the first second (5 Hz bandwidth, then 2 Hz), after which a second-order Costas
phase-locked loop with 5 Hz bandwidth holds the carrier. The code loop is a
0.25 Hz delay-locked loop on 10 ms non-coherent early/late sums, with the code
rate aided by the carrier Doppler. C/N0 is estimated from prompt versus noise-tap
power over 1 s, and a channel is dropped below 25 dB-Hz. Every channel's carrier
NCO starts at the receiver's IF phase at sample 0 (modulo one cycle), so the
accumulated carrier phases of different channels share one origin. These constants match
Pocket SDR's so the two trackers can be compared number for number on the same
recording; they are not the final choices for the hardware receiver.
