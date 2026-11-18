# pyrx

An offline GNSS receiver in Python: acquisition, tracking, navigation-message
decode and observables written as RINEX from recorded GPS L1 C/A IF samples,
acquisition and tracking of the GPS L5, Galileo E5a and Galileo E1 pilot signals with GPS L5
and Galileo E1 and E5a observations beside L1 (the L5 band from a second recording of the same
instants), and a
standalone position fix and an RTK fix from RINEX files, our own or public ones
(GPS L1/L2/L5 and Galileo E1/E5a/E5b). Each stage is checked against an external
reference: Pocket SDR for acquisition, tracking and observables, broadcast
ephemeris files for the navigation decode, IGS orbits, RTKLIB and the EUREF
station coordinates for positioning.

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

The suite runs across all cores (pytest-xdist); `pytest -n 0` runs it serially. Tests
marked `slow` (minutes on a long recording) are left out by default; `pytest -m slow -n 0`
runs them.

The files under `tests/golden/` pin the exact output of the RINEX writers and of every
command on synthetic inputs; `PYRX_UPDATE_GOLDENS=1 pytest` rewrites them after a
deliberate output change. The float-valued ones (observations, positions) were captured
on this project's environment (numpy 2.5, scipy 1.18, x86-64); another numerical stack
may differ in the last printed digit.

The reference tests compare acquisition and tracking with Pocket SDR's results on
its sample recordings (`test_acquisition_reference.py`, `test_tracking_reference.py`;
the L5 and E5a pilots in `test_tracking_pilot_reference.py`, the Galileo E1 pilot in
`test_tracking_e1_reference.py`, one of its two tests `slow`),
the navigation decode with the day's broadcast ephemeris (`test_navdecode_reference.py`),
the transmit times with Pocket SDR's C tracker and the fix RTKLIB computes from our
RINEX with the one it computes from Pocket SDR's (`test_observables_reference.py`), the L5 observations with the L1 ones of the same
satellites (`test_joint_observables_reference.py`), the re-acquisition walk, the fade
flag and the car-roof pilots and joint observations on the project's recordings
(`test_channels_reference.py`, all `slow`),
the broadcast orbits with the IGS final orbit (`test_orbits_reference.py`), the
standalone fix with RTKLIB on the recordings and on a day of an IGS station
(`test_pvt_reference.py`), and the RTK fix with RTKLIB and the EUREF marker
coordinates on two station pairs, the backward pass and the lock count with RTKLIB on
the car-roof drive, that drive's fix by the agreement of both filter directions, and the
car-roof captures with Galileo E5a and with both Galileo bands (the pseudoranges against
the broadcast orbits, E1 against E5a, the agreed fixes) (`test_rtk_reference.py`). They need the
environment variable `PYRX_CAPTURES` pointing at a directory holding these files,
and skip when it is unset; the observables, joint-observables, positioning and RTK
tests also skip when `rnx2rtkp` is not on the path, and the RTK tests take a few
minutes. With `PYRX_REQUIRE_CAPTURES=1` those skips become failures, so a gate run
cannot pass by skipping them (the `fpga` tests on the recording fail the same way):

| File | Source | SHA-256 |
|---|---|---|
| `L1_20211202_084700_4MHz_IQ.bin` | `sample/` directory of the Pocket SDR repository (https://github.com/tomojitakasu/PocketSDR, commit eda8301), GPS L1, 4 MHz I/Q, 100 ms | `0a8335d2f099e388b474d2afcca1ff91f61cde550dd32bf82fdf199d8a5b8033` |
| `L1_20211201_054600_24MHz_I.bin` | the same directory, GPS L1, 24 MHz real, 6 MHz IF, 100 ms | `05c771f0c152e2bd11ccba0715198afc0aa56e5d420bec0993fa720ac8d508d0` |
| `L1_20211226_082401_24MHz_I.bin` | the L1 file of http://gpspp.sakura.ne.jp/pocketsdr/L1L5_20211226_082401_24MHz.zip (Pocket SDR sample data, linked from its `sample/` directory), GPS L1, 24 MHz real, 6 MHz IF, 30 s | `16c83d58716ae9ba577aac29344766373accb98b34b729afdac8cc2e1352252c` |
| `L1_20220114_004440_8MHz_I.bin` | the L1 file of http://gpspp.sakura.ne.jp/pocketsdr/L1L2_20220114_004440_8MHz.zip, GPS L1, 8 MHz real, 2 MHz IF, 30 s | `388c550e2694067566c24fddae61abd1b316eb4bbd0820e0e8fff1818f164de7` |
| `L5_20211226_082401_24MHz_IQ.bin` | the L5 file of L1L5_20211226_082401_24MHz.zip, the archive of the 2021-12-26 L1 file above: the L5/E5a band of the same 30 s, 24 MHz I/Q, LO 1176.45 MHz | `a22e9fb97946e8c4f8347c1f16ece94cab0f127b84b18e28a026482f79fd5e70` |
| `brdc/BRDC00WRD_R_20213600000_01D_GN.rnx` | https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2021/360/ (gzip member), worldwide GPS broadcast ephemeris, RINEX 3.05, 2021-12-26 | `92fe0754c16b3408691facabad44390ebfa2031ad641334ebe6f8660f8e554c2` |
| `brdc/BRDC00WRD_R_20220140000_01D_GN.rnx` | https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2022/014/ (gzip member), worldwide GPS broadcast ephemeris, RINEX 3.05, 2022-01-14 | `ac1488e8b577ddd9b942de2ca8033fa82f8ce56abfe9d6e1781288cfa590d7e2` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.trk.log` | generated: Pocket SDR `pocket_trk.py` (commit eda8301) log of the 2021-12-26 recording, `$CH` channel state every 0.1 s; command below | `b9891175ed8fe7703f7b766e7e91a2bb9595e782d1d0a5d8dbbcf76920eef2f4` |
| `pocketsdr-trk/L1_20220114_004440_8MHz_I.trk.log` | the same for the 2022-01-14 recording | `a9a5dc74a472c91a043465be5986f7b6b865cbb43f0e737ff09d41d358639746` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.ctrk.log` | generated: Pocket SDR C tracker (`pocket_trk`, commit eda8301) log of the 2021-12-26 recording: `$CH` channel state every 0.1 s with the transmit time of week, `$OBS`, `$POS`; command below | `8fcf3d029769683bda4b1fa06c060c6bb61235ea94235d4795d38e07ac298dbf` |
| `pocketsdr-trk/L1_20220114_004440_8MHz_I.ctrk.log` | the same for the 2022-01-14 recording (`$CH` only: no subframe 1 in the window, so Pocket SDR forms no observations) | `3308cc4bc7273ba09ac1a14879b9bce7dca9f696119c0718f48b1a2cb4c09e92` |
| `pocketsdr-trk/L5_20211226_082401_24MHz_IQ.l5fix.ctrk.log` | generated: Pocket SDR C tracker log of the L5 recording, data and pilot channels of GPS L5 and Galileo E5a, from a build with the GPS L5 code generator corrected; command and correction below | `d6057fedb84fc3bfe56967a6efd85e620e2291fecf5858fa311ca3e60be8c3b1` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.e1.ctrk.log` | generated: Pocket SDR C tracker log of the 2021-12-26 L1 recording, the data and pilot channels of Galileo E1 (E1B, E1C); command below | `1fda441198441f290f1e33cc58ab8ebb5cb851f8e651c5d25136dae68b955e22` |
| `igs/USUD00JPN_R_20213600000_01D_30S_MO.crx.gz` | https://igs.bkg.bund.de/root_ftp/IGS/obs/2021/360/, IGS station USUD (Usuda, Japan), 30 s observations of 2021-12-26, RINEX 3.04, Hatanaka-compressed; no Doppler logged | `b964f1f0a51ffca461712575c6dbfe915b7eac65cb0f1513c46c75e2dc78e1f2` |
| `igs/USUD00JPN_R_20213600000_01D_GN.rnx.gz` | the station's GPS navigation file of that day from the same directory; the one with the `GPSA`/`GPSB` ionosphere lines, so it is the navigation input for the station and for the 2021-12-26 recording | `911fe136cd28a51ba5e0f81c5494af8c32b3cdf26bac7b68f6d34821a1105649` |
| `igs/igs21900.sp3` | IGS final orbits and clocks of 2021-12-26, 15 min, https://igs.bkg.bund.de/root_ftp/IGS/products/orbits/2190/igs21900.sp3.Z decompressed | `fdb9aae343289d97eec2241af7ee4cac0c774f33fb0d445f1fb1b1c95ed75072` |
| `igs/usud00jpn_20260422.log` | IGS site log of USUD, https://files.igs.org/pub/station/log/usud00jpn_20260422.log: the marker's ITRF coordinates | `80c41b22ee394448803a869da5757eae68ba3bb65aca2d5edd09470236892e41` |
| `igs/MATE00ITA_R_20213600000_01D_30S_MO.crx.gz` | https://igs.bkg.bund.de/root_ftp/IGS/obs/2021/360/, IGS station MATE (Matera, Italy), base of the 11 m pair, 2021-12-26, 30 s, RINEX 3.04, Hatanaka-compressed | `aa785104fa1352ca1f7fbb296f516982ae0fc904d3240a8fededc846efe7ab19` |
| `igs/MAT100ITA_R_20213600000_01D_30S_MO.crx.gz` | MAT1, rover of the Matera pair, same receiver and antenna types as MATE, 11.0 m away | `c776780c6437a7f26ad16043adc486cfcbdefcf38a45529198ed79c9b45ccacb` |
| `igs/TLSE00FRA_R_20213600000_01D_30S_MO.crx.gz` | TLSE (Toulouse, France), base of the 1266 m pair, same day and format | `4c9a77cbd605c7283b374e3e70e409f0e9f5a2959e7d45331a63f7f639445178` |
| `igs/TLSG00FRA_R_20213600000_01D_30S_MO.crx.gz` | TLSG, rover of the Toulouse pair, same antenna type as TLSE (different receivers), 1266 m away | `a41a6ffa9dc37db85a583926c3990f789a914fe88ce41d2bf3fcebf403780bf6` |
| `igs/BRDC00WRD_R_20213600000_01D_MN.rnx.gz` | https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2021/360/, worldwide mixed broadcast ephemeris (GPS and Galileo), RINEX 3.05, 2021-12-26 | `d69a64d9cdda9b2defa6a74405c600baf99c69ae640cc612943bfdd878d806ee` |
| `igs/EUR0E03SNX_20213600000_07D_07D_SOL.SNX.gz` | https://igs.bkg.bund.de/root_ftp/EUREF/products/2190/, EUREF combined weekly SINEX of GPS week 2190: the marker coordinates of the four stations, the truth of the RTK tests | `07f0a8577aec588869fd8db02ebfd56592e480be4b6088da3b8ad98da2335f18` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.obs` | generated: Pocket SDR's observations of the 2021-12-26 recording as RINEX 3.05, converted from the C tracker's RTCM3 output with `convbin`; command below | `42c66c39f6c62753aa4dae481a72de42f6fd2958018412abae270cee4ee07cb1` |
| `pocketsdr-trk/L1_20211226_082401_24MHz_I.psdr.rnx2rtkp.pos` | generated: RTKLIB 2.4.3 b34 `rnx2rtkp` single-point fix from that file with the broadcast ephemeris, the reference for our fix; command below | `304f6d4ad6662085e09668d789e51aa4aa52073e1d0a723da1cfea4494cd841e` |
| `brdc/BRDC00IGS_R_20220140000_01D_MN.rnx` | https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2022/014/ (gzip member), IGS combined multi-GNSS broadcast file, RINEX 3.04, 2022-01-14; only its GPSA/GPSB header lines are used | `5e366723dcda6bcfb049b5bc0459284d27581855567c1ff977de7cf76e1f701d` |
| `sky/20261103_031515_carroof-driving/ch1.bin` | project recording, not distributed (50 GB per capture; the test reads 20 s of it): Pocket SDR FE 4CH on a car roof, driven through two obstructions, GPS L1, 24 MHz real, 6 MHz IF, 300 s; `test_tracking_reference.py` tracks G12 through the 222 s onset | `3662ffb99bcc61068af4322b766d2be1d2a678feb2fa1b01cef9b2d306ee0113` |
| `sky/20261103_032110_carroof-coldstart/ch1.bin` | project recording, not distributed: the same setup started while moving, with a total loss of lock at 40 to 50 s, 300 s; two `slow` tests in `test_channels_reference.py` walk its first 90 s and G13 from 100 to 160 s | `143a850c69fe5f7dbbba80caf9fc30aca1073dfeee99785575db853dd2678d0e` |
| `sky/20261103_032110_carroof-coldstart/trk_l1ca.log` | generated: Pocket SDR `pocket_trk` L1 C/A log of that recording, the reference of the `slow` test | `4dc83bc579867286d6e938cdc91628f5d9c413723d832c86c9475f1ae85de9d5` |
| `sky/20261103_030551_carroof-static/ch1.bin` | project recording, not distributed: the same front end parked under open sky, GPS L1, 24 MHz real, 6 MHz IF, 300 s; a `slow` test forms L1 and L5 observations from its first 120 s | `f991e49f5f0fd9d3dcaa58a48f528fb59fdf23cfc03d8edb96e9d6469bf44d2b` |
| `sky/20261103_030551_carroof-static/ch2.bin` | the L5/E5a band of the same capture on the same sample clock, 24 MHz I/Q, LO 1176.45 MHz; `slow` tests walk the pilots of its first 60 s and form joint observations from its first 120 s | `2a353935ef7ce19eede08c47d6dd68b62e9c8783f319986ca0e2e7205e2b4115` |
| `sky/20261103_030551_carroof-static/trk_l5_l5fix.log` | generated: the corrected Pocket SDR C tracker's log of that file, as for the L5 recording above | `f711e9169bdd012ac1310f15fd7240632e173c4a1759199e3b4de4d8b58a9993` |
| `sky/20261103_030551_carroof-static/trk_e1b.log` | generated: Pocket SDR C tracker log of that capture's `ch1.bin`, Galileo E1B channels (`-sig E1B -prn 1-36`), the reference of the `slow` E1 test, which walks 60 s of the file from 80 s | `f7807b5c03f5995d16c2c2a49f07139bc3f55fbf853746e5d290ae75d9c2e635` |
| `sky/20261103_031515_carroof-driving/pyrx_l1l5.obs` | generated: `pyrx-rinex ch1.bin --fs 24e6 --f-if 6e6 --l5 ch2.bin` on the drive (291 epochs at 1 s, GPS L1 and L5), the rover of the RTK tests on the car-roof captures | `9f95711a31e419a8b79bc7b21276b0d88d5e34eeafa90c8868f0919fd874df31` |
| `sky/20261103_031515_carroof-driving/pyrx_l1l5.nav` | generated: the navigation file of the same run | `68867ed5a8552c7bc6683f6a84f79040147ee44122662153cdd1862af49a10ab` |
| `sky/20261103_030551_carroof-static/pyrx_l1l5.obs` | generated: `pyrx-rinex ch1.bin --fs 24e6 --f-if 6e6 --l5 ch2.bin` on the parked capture (299 epochs), the rover of the RTK tests on the car-roof captures | `4a34b9a27606570dbe335055071c4c76dda14bfb7ece98affd141f5cf527e778` |
| `sky/20261103_030551_carroof-static/pyrx_l1l5.nav` | generated: the navigation file of the same run | `cb485486a91a6ceec246d3b981bb097d813f14e78bb3fd034d4f0111a96c357c` |
| `cors/base_20261103_0300_0330_gps.rnx` | project file, not distributed: half an hour of 1 s GPS L1 and L5 observations of a public reference station 15 km from the car-roof captures, converted from its RINEX 2.11 file to RINEX 3 (`C1 L1 S1` as `1C`, `C5 L5 S5` as `5Q`), with the station's antenna reference point in `APPROX POSITION XYZ`; the base of those RTK tests | `ebc81edce9e29c37588ab7761b59c4cee74209f9ceb741f0ab0dee50e5fa06d7` |
| `sky/20261103_032110_carroof-coldstart/pyrx_l1l5.obs` | generated: the same command on the capture started while moving (289 epochs) | `292ece824076c902fb7167c19e2a7308f75a35d3cecbf9e671a818399d174674` |
| `sky/20261103_032110_carroof-coldstart/pyrx_l1l5.nav` | generated: the navigation file of the same run | `613c6fb7e289c517ed4811e2163e578a92ba657455830a001c71004f069542e6` |
| `sky/20261103_030551_carroof-static/pyrx_gal.obs` | generated: `pyrx-rinex ch1.bin --fs 24e6 --f-if 6e6 --l5 ch2.bin --galileo` on the parked capture: its GPS rows with the Galileo E5a pilots' beside them | `45c87bc9fe1840cb4ed23b977fcba9c3b0d9d51bfbc959c8c9d5beea5fb3aa41` |
| `sky/20261103_031515_carroof-driving/pyrx_gal.obs` | generated: the same on the drive | `1a183ccd397fa3adf66b8ca561f383b166500af44228c1614850d2bd17e004ec` |
| `sky/20261103_032110_carroof-coldstart/pyrx_gal.obs` | generated: the same on the capture started while moving | `dcbb91a3989023f96918482147270e1c6d3edfe5c12ee3658f1de3b5f6ebbead` |
| `sky/20261103_030551_carroof-static/pyrx_e1e5a.obs` | generated: a whole run of `pyrx-rinex ch1.bin --fs 24e6 --f-if 6e6 --l5 ch2.bin --galileo` on the parked capture: the GPS rows, and the Galileo satellites on E1C and E5Q | `20f69833adeb61c5fcaf57771b4ddc453600fae56feefe5075f8a05d35271a8e` |
| `sky/20261103_031515_carroof-driving/pyrx_e1e5a.obs` | generated: the same on the drive | `4e9993b577bbd48864c3d8fe196b116fa70fcc7a081a8b1f1bf198a7fbfa724b` |
| `sky/20261103_032110_carroof-coldstart/pyrx_e1e5a.obs` | generated: the same on the capture started while moving | `a21fdf49912f10d81a725d40e928dce5d67c6a70aef63acf62e9eac2328cea46` |
| `cors/base_20261103_0300_0330.rnx` | project file, not distributed: the same half hour of the same station with its Galileo satellites beside the GPS ones (`C1 L1 S1` as `E1C`, `C5 L5 S5` as `E5Q`: RINEX 2.11 names no tracked component), the base of the RTK tests with Galileo | `c1690f6daf6acc7803e3c0644a9b232c8207bffffc273aac4aef692ca212f0d9` |
| `brdc/BRDC00IGS_R_20263070000_01D_MN.rnx` | ftp://gssc.esa.int/gnss/data/daily/2026/brdc/ (gzip member), IGS combined multi-GNSS broadcast file, RINEX 3.04, 2026-11-03: the Galileo ephemerides of the car-roof captures | `9b8475997b50d2adb98058298ee892e701c0d2965b5fd744e4089e21e9257879` |

The 100 ms 24 MHz recording has sample gaps near 40 ms and 90 ms; the acquisition
tests use its first 10 ms, which is also what Pocket SDR integrates by default.
The rows marked generated were produced here with the commands below and are kept
in the same directory as the recordings, with every file's checksum in its
`SHA256SUMS`; the downloaded rows and the project recordings carry the checksum of the
file as stored.

The Pocket SDR tracking logs in the table were produced with Pocket SDR's
`pocket_trk.py` (its library built with `USE_SOAPY=0 USE_FFTW=1`), run from the
`pocketsdr-trk/` directory with the recordings one level up:

```sh
pocket_trk.py -sig L1CA -prn 4,16,18,26,27,29,31 -f 24 -fi 6.0 -q -log L1_20211226_082401_24MHz_I.trk.log ../L1_20211226_082401_24MHz_I.bin
pocket_trk.py -sig L1CA -prn 10,12,15,23,24,32  -f 8  -fi 2.0 -q -log L1_20220114_004440_8MHz_I.trk.log  ../L1_20220114_004440_8MHz_I.bin
```

The PRN lists are the satellites `pyrx-acq` reported at or above 38 dB-Hz in the
first 10 ms of each file, the acquisition default when the logs were made.

The Pocket SDR C tracker files were produced from the same directory (`-fo` being
the LO frequency, L1 minus the IF, in MHz):

```sh
pocket_trk ../L1_20211226_082401_24MHz_I.bin -fmt INT8 -f 24 -fo 1569.42 -sig L1CA -prn 1-32 -ti 0 -log L1_20211226_082401_24MHz_I.ctrk.log -rtcm L1_20211226_082401_24MHz_I.ctrk.rtcm3
pocket_trk ../L1_20220114_004440_8MHz_I.bin  -fmt INT8 -f 8  -fo 1573.42 -sig L1CA -prn 1-32 -ti 0 -log L1_20220114_004440_8MHz_I.ctrk.log
convbin -r rtcm3 -v 3.05 -tr 2021/12/26 08:00:00 -o L1_20211226_082401_24MHz_I.psdr.obs -n L1_20211226_082401_24MHz_I.psdr.nav L1_20211226_082401_24MHz_I.ctrk.rtcm3
rnx2rtkp -p 0 -t -o L1_20211226_082401_24MHz_I.psdr.rnx2rtkp.pos L1_20211226_082401_24MHz_I.psdr.obs ../brdc/BRDC00WRD_R_20213600000_01D_GN.rnx
```

The L5 logs come from a copy of Pocket SDR (commit eda8301) with one correction. Its
GPS L5 code generator takes the XB sequence at index `(i + advance) % 10230`;
IS-GPS-705 (3.3.2.2) restarts the XB register at its natural completion after 8191
chips inside the 10230-chip period, so the index is `(i + advance) % 8191`
(`gen_code_L5I` and `gen_code_L5Q` in `src/sdr_code.c`, and the two `np.roll` lines of
`python/sdr_code.py`). Unchanged, the last `advance` chips of every period come from
the wrong part of the sequence, the correlation reads up to 8 dB low depending on the
PRN and component, and weak channels are lost. With the correction the data and pilot
components read equal on every satellite, as the signal is specified. The command:

```sh
pocket_trk ../L5_20211226_082401_24MHz_IQ.bin -fmt INT8X2 -f 24 -fo 1176.45 -sig L5I -prn 1-32 -sig L5Q -prn 1-32 -sig E5AI -prn 1-36 -sig E5AQ -prn 1-36 -ti 0 -log L5_20211226_082401_24MHz_IQ.l5fix.ctrk.log
```

The E1 log of the 2021-12-26 recording comes from the unmodified build, whose hundred
E1 codes equal the tables published with the Galileo ICD:

```sh
pocket_trk ../L1_20211226_082401_24MHz_I.bin -fmt INT8 -f 24 -fo 1569.42 -sig E1B -prn 1-36 -sig E1C -prn 1-36 -ti 0 -log L1_20211226_082401_24MHz_I.e1.ctrk.log
```

It starts its channels one after another and an E1C channel only after the satellite's
E1B channel, so its E1C channels cover the last 6 to 9 s of the file and two of the five
satellites have none.

## Sample format

Signed 8-bit samples. Real recordings hold one byte per sample. I/Q recordings
interleave I and Q bytes and the complex sample is `I - jQ`, the Q sign convention
of the Pocket SDR front-end.

## Acquisition from the command line

```sh
pyrx-acq FILE --fs 4e6 --iq [--f-if 0] [--signal L1CA] [--prn 1-32] [--duration 0.02] [--plot DIR]
```

By default the first 20 ms of the file are integrated (`--duration 0` for the whole
file), and a PRN counts as detected when its peak reads 36 dB-Hz or more.

`--signal` is `L1CA` (GPS L1 C/A, the default), `L5` (the GPS L5 pilot, Q5), `E5A`
(the Galileo E5a pilot, E5a-Q) or `E1` (the Galileo E1 pilot, E1-C); without `--prn` every PRN of the signal is searched
(1 to 32, or 1 to 36 for Galileo). A pilot carries one chip of a secondary code on
every code period, so its sign changes between periods, and a period that straddles a
sign change cancels part of itself in a circular correlation over one period. The
pilots are therefore searched with two periods of samples against the code padded with
zeros to two periods, as Pocket SDR does: one whole period always correlates. On a
recording the plain search read 2 to 3 dB low wherever the code period started in
mid-block, and the code phase hardly moves within a file.

The E1 code lasts 4 ms and is searched with its subcarrier on it (see "Pilot signals"),
two periods, 8 ms, at a time: 20 ms hold four such sums where a 1 ms pilot has nineteen.
The code offset printed is that of a code period's start, anywhere in the 4 ms. The
correlation of this signal has a second peak half a chip either side of the main one, a
quarter of its power; the search takes the largest. The threshold is on the reading, so a
pilot at 36 dB-Hz is found about half the time: of 60 synthetic signals each, 25 at
36 dB-Hz, 59 at 38 and all at 40, the largest peak on the main one in every run and the
offset right to the sample. On the 2021-12-26 recording it finds the five satellites above
41 dB-Hz in the first 20 ms and not the two at 35 dB-Hz.

Prints one line per PRN in the same format as Pocket SDR's `pocket_acq.py`, so the
two outputs can be diffed. With `--plot`, writes `cn0.png` and one search-grid
image per detected PRN.

The search sums whole code periods of a fixed number of samples, so the sample
rate must hold a whole number of samples per millisecond or be high enough that the fraction does not matter; at 16.3676 MHz, a
common front-end rate, a 10 ms search spreads the peak by a quarter
chip and a non-integer rate near 2.5 MHz measured 0.8 chip and 3.7 dB of loss.
Non-integer rates are not supported.

## Tracking from the command line

```sh
pyrx-trk FILE --fs 8e6 --f-if 2e6 [--iq] [--signal L1CA] [--prn 1-32] [--duration 0] [--log trk.csv] [--plot DIR]
```

Searches each PRN on 20 ms windows, at the start and then every 10 s wherever the
PRN is not being tracked, and tracks every detection through the file (`--duration
0`, the default, means all of it) until the data ends or lock is lost; a detection
counts only if the data-bit edges show up in a 6 s trial, which keeps noise peaks
and cross-correlations out. Each PRN runs in its own process. A PRN lost and found
again has several track segments; the command prints the final C/N0, the phase-lock
indicator (`PLI`), the Doppler and the lock state per segment, the later segments of a
PRN as `PRN nn from t s`. The indicator is cos 2φ of the prompt correlator over each
1 s window, 1 when the Costas loop is phase-locked and near 0 when the carrier spins
or there is only noise; a segment ends after two consecutive windows with C/N0 below
25 dB-Hz or the indicator below 0.25 (Pocket SDR's rule at our window length; the
first window's indicator is reported but not counted, since the FLL pull-in spins the
carrier through it). `--log`
writes a CSV with columns
`time_s,prn,cn0_dbhz,phase_lock,code_offset_ms,doppler_hz,carrier_cycles`, one row per
segment every 0.1 s (`phase_lock` empty in an incomplete final window). `--plot` writes `trk_prnNN.png` per channel
(`trk_prnNN_k.png` for a PRN's k-th segment) with four panels: prompt I and Q over the first 2 s (the navigation data bits are visible
once the phase lock settles), Doppler, C/N0, and the early, prompt and late
correlator magnitudes.

With `--signal L5` or `--signal E5A` the channels track the pilot (see "Pilot signals"
below). A pilot has no data bits, so its trial is passed when the secondary code was
found and the trial's last complete window is phase-locked. Each segment's line also
says when the secondary code was found (`SYNC 1.5 s`, seconds into the segment, or
`SYNC none`), and the CSV has one more column, `synced`, 1 from the epoch the
secondary chip is removed. `--signal E1` tracks the Galileo E1 pilot the same way; its
trial also turns away a seed whose loops settled on a side peak of the correlation.

## Navigation message decode from the command line

```sh
pyrx-nav FILE --fs 8e6 --f-if 2e6 [--iq] [--prn 1-32] [--duration 0] [--json nav.json]
```

Acquires and tracks as `pyrx-trk` does, then decodes each segment's navigation
message and prints one line per track segment (a PRN lost and found again has several,
the later ones labeled `PRN nn from t s`): whether bit sync was found, the subframes
decoded as `ID@TOW` (the GPS seconds of week at the subframe's first bit), and the
ephemeris issue of data and `toe` when subframes 1, 2 and 3 with the same issue
were seen. `--json` writes `{"ephemeris": {"<prn>": {field: value, ...}},
"ionosphere": {"alpha": [...], "beta": [...]} | null}` with the complete
ephemerides only.

## RINEX output from the command line

```sh
pyrx-rinex FILE --fs 24e6 --f-if 6e6 [--iq] [--prn 1-32] [--duration 0] [--interval 1.0] [--week N] [--marker NAME] [--marker-type TYPE] --obs out.obs [--nav out.nav] [--l5 L5FILE [--l5-f-if 0] [--galileo]]
```

Acquires, tracks and decodes as `pyrx-nav` does, forms observations at every
whole multiple of `--interval` seconds of GPS time, and writes them as a RINEX 3.05
observation file with `C1C L1C D1C S1C` for every channel that has a decoded
subframe. With `--nav`, the decoded ephemerides (and the ionosphere parameters when
seen) are written as a RINEX navigation file. The GPS week comes from subframe 1;
when the recording holds none, give it with `--week`. `--marker-type` writes the
`MARKER TYPE` header record, which RINEX 3.05 requires for markers other than geodetic
and non-geodetic ones (`GROUND_CRAFT` for a receiver on a vehicle); without it the
record is omitted. One line per track segment reports how
many epochs it contributed to and its first pseudorange and Doppler, then the number
of epochs and their span.

With `--l5 L5FILE` the GPS L5 pilots are tracked in a second recording and their
observations are written beside L1's, eight types per satellite:
`C1C L1C D1C S1C C5Q L5Q D5Q S5Q`. `L5FILE` must be an I/Q recording of the L5 band
made by the same front end on the same sample clock and at the same rate, sample k
of both files being the same instant (the two channels of one capture). Only the
satellites L1 gave a time to are searched on L5, and each L5 segment prints its own
line, `G5Q PRN nn  epochs ...`, or `G5Q PRN nn  no time tag` when it could not be
placed on the L1 time (see "L5 observations" below). Nothing checks that the two files
belong together. If no pilot is tracked at all, which is what a wrong `--l5-f-if` or
sample rate gives, a line on stderr says so and the file is written with empty L5
columns; pilots tracked in a file that does not line up with `FILE` sample for sample
find no place on the L1 time and print `no time tag`.

With `--galileo` as well, the Galileo pilots are tracked too, every satellite number from
1 to 36: E1 in `FILE` and E5a in `L5FILE`, written as `C1C L1C D1C S1C C5Q L5Q D5Q S5Q`
rows of the Galileo satellites in a mixed file; each segment prints `E1C PRN nn  epochs
...` or `E5Q PRN nn  epochs ...` (`no time tag` for a segment that wrote nothing: one that
never found its secondary code, or one whose every epoch disagreed with the other band),
and a band in which no pilot was tracked is named on stderr. A Galileo pilot takes its time from its own secondary code and the receiver's GPS
clock (see "Galileo observations" below), so it needs no GPS channel of its own but the
run needs GPS L1 for that clock. The navigation file stays GPS only: nothing Galileo is
decoded.

For the 2021-12-26 pair, both files of `L1L5_20211226_082401_24MHz.zip`:

```sh
pyrx-rinex L1_20211226_082401_24MHz_I.bin --fs 24e6 --f-if 6e6 --l5 L5_20211226_082401_24MHz_IQ.bin --obs joint.obs --nav joint.nav
```

`--iq` describes `FILE` only; `L5FILE` is always read as I/Q, at `--l5-f-if` (0 for this
pair, recorded at its center frequency). `--t-start` and `--duration` select the same
span of both files. `pyrx-pvt` reads only the L1 columns of the result. Without `--l5`
the file holds `C1C L1C D1C S1C` only, and without `--galileo` no Galileo row.

## Positioning from the command line

```sh
pyrx-pvt OBS NAV [--weights cn0|rtklib] [--mask 15] [--ts "YYYY-MM-DD HH:MM:SS"] [--te "YYYY-MM-DD HH:MM:SS"] [-o fix.txt] [-v]
```

Solves every epoch of a RINEX observation file with the ephemerides of a RINEX
navigation file. The inputs can be our own files or public RINEX 3 files, plain,
gzip or Hatanaka compressed; only GPS L1 C/A is used. `--ts` and `--te` bound the
epochs in GPS time, `--mask` is the elevation mask in degrees, and `--weights rtklib`
turns off the C/N0 term of the pseudorange weights (see Standalone positioning below).
The command prints
the number of epochs solved and read, the mean latitude, longitude, height and
receiver clock bias, and the run time; with `-v` one line per solved epoch; with
`-o` those lines go to a file whose header line (starting with `%`) names the
columns: `week tow lat lon height clock_s ns gdop vx vy vz ddtr` (degrees, meters,
seconds, satellites used, geometric dilution of precision, ECEF velocity in m/s and
clock drift in s/s, `nan` without Doppler). It returns 2 when no epoch solves.

## RTK from the command line

```sh
pyrx-rtk ROVER BASE NAV [--nav-add FILE] --base-xyz X Y Z [--base-delta E N U] [--rover-delta E N U] [--signals G1C,G2W] [--mode kinematic|static] [--slips check|restart] [--min-lock 5] [--fix-sats 5] [--change-limit 0.03] [--backward] [--mask 15] [--ts "YYYY-MM-DD HH:MM:SS"] [--te ...] [-o fix.txt] [-v]
```

Solves the baseline from a base station of known position to the rover, epoch by
epoch, from two RINEX observation files (plain, gzip or Hatanaka compressed, ours
or public) and a navigation file. `--nav-add FILE`, which may be repeated, names a
further navigation file for the systems the files before it lack: every system's
ephemerides come from the first file that has it, and the ionosphere from the first that
has one, so a rover that decodes GPS only can add a public file for Galileo and keep its
own GPS records. `--base-xyz` is the base marker; the
marker-to-antenna offsets are taken from each file's `ANTENNA: DELTA H/E/N` unless
given. `--signals` lists the signals to use as system letter plus RINEX attribute
(GPS `G1C G2W G5Q`, Galileo `E1C E5Q E7Q`); the single-point seed solves on GPS L1
C/A when it is listed and otherwise on the first GPS signal given (it is an L1 solver,
so a GPS signal must be listed). With `--slips check`, the default, a phase the rover
flags as a possible slip is looked at before its ambiguity is given up; `--min-lock` is
the number of updates a new ambiguity waits before the integer search (RTKLIB's
`pos2-arlockcnt`), and `--fix-sats` the number of satellites that search must span for a
fix. `--change-limit` (m, with `--slips check`; 0 turns it off) is how far a flagged phase
or its satellite's other signal may change against the other satellites from one epoch to
the next before the other signal's word that the flagged one is continuous is no longer
taken.
`--slips restart --min-lock 0 --fix-sats 0` is RTKLIB's behavior, which is also what the
library's `RtkOptions` default to (see RTK below). `--backward` runs the filter from the
last epoch to the first; the output stays in time order. The
command prints the numbers of fixed and float epochs, the mean fixed baseline in
the base's east-north-up frame and the median ratio, then the run time; `-v`
prints one line per solved epoch; `-o` writes those lines with the header
`% week tow x y z Q ns ratio age` (rover marker in ECEF meters, `Q` 1 fixed 2
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
is how that comparison is made. The positioning tests compare the solver of
`pyrx-pvt` under RTKLIB's weights (`--weights rtklib`) with RTKLIB running the same
models, from a configuration file with

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
pos1-frequency=l1+2           # l1, or l1+2+3 for GPS L5 and Galileo E5a
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

is the configuration the library's defaults mirror, and `pyrx-rtk` with
`--slips restart --min-lock 0 --fix-sats 0` (RTKLIB applies the antenna offsets as
range corrections, which agrees with moving the stations to their antenna
reference points to well under a millimeter); `pos1-soltype=backward` corresponds to
`--backward` and `pos2-arlockcnt` to `--min-lock`. Epochs where RTKLIB finds no
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
removed when a satellite misses five epochs that reach the relative
solution, on a loss-of-lock flag at either receiver, or when the single-differenced
geometry-free combination of a satellite's first and another signal jumps by more
than 5 cm.

Residuals use the geometric range with the Earth-rotation term to the antenna
reference point, the satellite clock with its relativistic term, and the
hydrostatic Saastamoinen zenith delay mapped with the Niell dry function. Nothing
else is applied: no ionosphere, no wet troposphere, no group delays, no antenna
phase-center model (the reference pairs have identical antennas). This is what
RTKLIB's single-baseline mode applies with its default options, so the two solvers
compare number for number on the same files; the deliberate differences (existing
ambiguities aligned to the current clock per signal rather than per frequency, the
geometry-free slip test run past a missing band, the tie order of two reference
candidates at the same elevation, a slip flagged in a base epoch that pairs with no
rover epoch carried to the next paired one, a slip flagged in an epoch that gets no
solution before the ambiguities are looked at applied at the next one that does) are
listed in `rtk.py` and do not change the default-configuration comparisons. One limit is kept as RTKLIB has it: the loss-of-lock
flags are read only for satellites both receivers contribute at an epoch, so a slip
flagged by one receiver while the other lacks the satellite goes unseen if the satellite
is common again within five epochs (a second signal's geometry-free test can still catch
it). Double differences are formed per system against
the highest satellite, carrier phase and code, with RTKLIB's single-difference
variance `2 (a² + b² / sin² el)` (3 mm for phase, 100 times that for code) plus a
receiver-clock-stability term in the age of the base epoch, and the block
covariance of a shared reference; a double difference beyond 30 m is dropped. One
Kalman update per epoch, linearized at the seed; the update stands but the epoch
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
and 7.0 mm in the multi-GNSS case (1.1 mm at the 95th percentile), and the hardware set
fixes 721 of 721.

### Flagged phases, the lock count and the backward pass

A receiver that flags every phase that may have slipped flags many that did not: on the
car-roof captures pyrx's own L1 fade rule (Observables, below) puts the flag on 11 % and
35 % of the L1 phases, and nearly all of those are continuous. RTKLIB restarts the
ambiguity at every flag, and so does this engine with its default options. Five more
options, all off by default so that the comparisons with RTKLIB stand, change that; the
`pyrx-rtk` command turns on the first three and `change_limit`, which belongs to the first
and is described at the end of this section ("Two signals that move together"; the
measurements before that part were made without it).

`slips="check"` looks at a flagged phase first. A phase is in question when only the
rover flags it (the lost-lock bit, or a change of the half-cycle bit), both receivers
have its phase and code, and its ambiguity was updated at the previous epoch that reached
the filter. Every other flagged phase restarts as before: one the base flags, one after a
gap. Two things can settle a phase in question.

The satellite's other signal. The engine compares the geometry-free difference of a
satellite's first signal and each other one with its earlier value and restarts both on a
step of more than 5 cm. In check mode the earlier value is the one of the last epoch at
which neither phase was in question (or of the last restart of either), so a slip spread
over several flagged epochs is measured whole; and where there is no step, a phase in
question beside an unflagged one is continuous. A step still restarts both, whichever
carries the flag: a fade hits both signals, and on the drive below restarting only the
flagged one was worse than no check at all.

The rest of the epoch. The phases still in question are left out of a trial update on a
copy of the filter, which takes every code and every other phase and so places the rover
relative to the ambiguities that are not in doubt. Each phase in question is then
double-differenced against the highest satellite of its signal that is not in question.
It continues its ambiguity when that innovation is within a quarter cycle
(`CHECK_INNOVATION`: a Costas loop that slips settles half a cycle away, and a quarter is
half the way there) and the innovation's predicted standard deviation, filter state and
measurement noise together, is within an eighth (`CHECK_SIGMA`); otherwise the ambiguity
restarts on this phase. The prediction is that sharp only while enough satellites are
not in question: five of the six in the synthetic tests place the sixth within 0.11
cycle, four only within 0.8 to 1. The epoch's one Kalman update then runs on every phase as usual.
An ambiguity that continues through a flag keeps an established half cycle whatever the
file's half-cycle bit says; a restarted one takes the file's bit.

`min_lock` is RTKLIB's `pos2-arlockcnt`, counted as RTKLIB counts it: an ambiguity
waits that many updates before it enters the integer search when it is begun again after
a slip or an outage, begun on a flagged phase, or begun after the first five solved
epochs; only one that appears unflagged within those first epochs is searched at once.
`min_fix_sats` keeps an epoch float unless the ambiguities eligible for the search span
that many satellites, the reference among them. The number is meant for one system: every
system has its own reference satellite, so with GPS and Galileo five satellites are three
double differences, and the floor counts the satellites of the search less one per system
(five of one system are four; so are three and three of two). On the drive, searches spanning five
satellites or fewer passed the ratio test, some with ratios above 100, at positions the
filter run the other way contradicts by up to a meter.
`backward` runs the filter from the last epoch to the first over the same pairs of
epochs (`solve_rtk` returns the fixes in time order either way). A lost-lock flag sits on
the first phase after a discontinuity, so the backward filter reads that bit from the
indicator it read before and restarts at the earlier phase, as RTKLIB does; like RTKLIB
it takes the half-cycle bit from those indicators too, which follows a change of the bit
one epoch late. RTKLIB pairs a rover epoch with the next base epoch when it runs
backward, this engine with the previous one as it does forward; at equal rates that is
the same epoch.

On the L1 observations of the car-roof drive, where the two engines have the same slip
rules, the backward pass and a lock count of 5 in both directions give RTKLIB's solution
epoch for epoch: the same epochs solved, the same status, the positions within 1 mm
(`test_rtk_reference.py`; `pos1-soltype=backward` and `pos2-arlockcnt=5` in the
configuration above). So does a lock count of 5 on the Matera pair, six hours with
satellites rising and setting. Two details differ from RTKLIB and are not covered by
that: backward, RTKLIB reads the indicators of its very first epoch as it does forward;
and it counts the half-cycle bit of a satellite's first indicator as a change.

There is no truth for the car-roof captures, so the check is judged by agreement: the
filter runs forward and backward, and an epoch is agreed when both fix it within 5 cm
horizontally and 10 cm vertically of each other, apart when both fix it further apart.
The drive, 279 solved epochs at 1 s with fades and one obstruction, GPS L1 and L5 on
five of nine satellites, a base 15 km away, kinematic:

| Options | Fixed forward | Fixed backward | Agreed | Apart |
|---|---|---|---|---|
| default (RTKLIB's rules) | 157 | 122 | 84 | 1 |
| `min_fix_sats=5` | 153 | 119 | 84 | 0 |
| `min_lock=5`, `min_fix_sats=5` | 184 | 143 | 105 | 1 |
| `slips="check"`, `min_fix_sats=5` | 193 | 169 | 134 | 0 |
| `slips="check"`, `min_lock=5`, `min_fix_sats=5` (the command) | 202 | 178 | 144 | 1 |
| `slips="check"`, `min_lock=5`, `min_fix_sats=6` | 198 | 177 | 144 | 0 |
| `slips="check"`, `min_lock=5` | 214 | 188 | 144 | 5 |

RTKLIB 2.4.3 itself, which has no geometry-free test for this pair of signals, fixes 141
epochs forward and 108 backward on the same files. Where the command's forward solution
and the default's both fix (141 epochs) the positions are within 2.2 cm horizontally and
1.8 cm vertically of each other. The first seconds of the drive, before the car pulls
out, fix within 7 mm horizontally and 17 mm vertically of the static solution of a
capture recorded at the same spot ten minutes earlier; that parked capture fixes 132 of
299 epochs in static mode with either set of options, at the same position, and the
command's options do not produce the default's false fix at the very first epoch (ratio
3.1, 2 m off). The test pins the drive at 130 agreed epochs or more, at most one apart,
and 40 more agreed than the default.

The floor is five satellites and not six, which would leave no epoch apart on the drive,
because of what six costs where a truth exists. On the two station pairs above, GPS
only, a quarter and a tenth of the epochs have exactly five satellites; every fix there
is within 2.4 cm of the marker with any of these options; the command's options fix 714
and 721 of the 721 epochs on L1+L2 (default 721 and 721) and 709 and 696 on L1 (716 and
696), a floor of six 521 and 627.

Limits. Agreement of two directions of one engine on one file is not a truth; both can
share a wrong fix, and the one epoch the command fixes apart shows a false fix getting
through with five satellites. The options were chosen on this one drive: lock counts of
3 to 7 give 136 to 147 agreed epochs and 0 to 3 apart, and the two constants of the test
change little between 0.20 and 0.30 cycle and between 0.10 and 0.15 cycle (144 agreed
throughout, one or two apart). The check trusts the phases that carry no flag; an
unflagged slip is caught only by the geometry-free test, as before. And a second
capture, started while driving, has no agreed epoch with any of these options (the
command fixes one epoch forward and 39 backward, none of them the same; the default's 4
and 14 scattered fixes agree nowhere and two are apart): seven satellites, four with an
L5 pilot, the other three flagged on 33 to 56 % of their L1 phases. It needs more
satellites with a pilot.

Galileo supplies them. The same recordings hold nine or ten Galileo satellites, each with
a pilot on E5a, and the base has six to eight of them. With their E5a observations beside
GPS (`pyrx-rinex --l5 --galileo`, "Galileo observations" below; signals `G1C,G5Q,E5Q`; the
Galileo orbits from the day's public broadcast file added to the decoded GPS ones), the
command's options, fixed forward and backward, agreed, apart:

| Capture | GPS L1 + L5 | with Galileo E5a |
|---|---|---|
| started while driving | 1, 39, 0, 0 | 104, 177, 54, 0 |
| the drive | 202, 178, 144, 1 | 242, 255, 227, 4 |
| parked, kinematic mode | 14, 0, 0, 0 | 294, 299, 294, 0 |

The capture started while driving has its first agreed epochs, in stretches of up to 17 s
between the fades, with twelve or more satellites in the solution and no epoch apart. The
drive's four epochs apart are just past the bounds (5.5 to 5.9 cm horizontally at three,
10.1 cm vertically at one), where one direction has just come back from float with half
its ambiguities. On the parked car the fixes with Galileo have their median 3 mm
horizontally and 2 mm vertically from the GPS-only static solution, and every kinematic one
lies within 6 mm and 19 mm of that median. RTKLIB's rules on the same files agree at 14, 97
and 298 epochs: the lock count and the check are needed with Galileo too.

None of this has a truth, and the agreed epochs of the capture started while driving are
evidence at the level of centimeters, not millimeters. They depend on the satellite set:
with the strongest Galileo satellite left out, 61 epochs are agreed, 39 of them the same,
and at those the forward fix lies 14 mm away in the median and more than 3 cm at eight
epochs (5.6 cm at most). On the drive the same comparison keeps 218 of 222 epochs, 6 mm
apart in the median and one epoch over 3 cm. A solution from the Galileo satellites alone
would be a check that shares no satellite with the GPS one, but on one frequency and seven
satellites it is not usable: it fixes 8 and 4 epochs of the capture started while driving,
26 and 8 of the drive, none of the parked car, and no epoch in both directions. That check
waits for Galileo's second signal, E1.

With E1 beside E5a (signals `G1C,G5Q,E1C,E5Q`, the files of a whole run of the command):

| Capture | GPS L1 + L5 | with Galileo E5a | with Galileo E1 and E5a |
|---|---|---|---|
| started while driving | 1, 39, 0, 0 | 104, 177, 54, 0 | 206, 190, 132, 0 |
| the drive | 202, 178, 144, 1 | 242, 255, 227, 4 | 245, 249, 216, 3 |
| parked, kinematic mode | 14, 0, 0, 0 | 294, 299, 294, 0 | 294, 299, 294, 0 |

The capture started while driving goes from 54 agreed epochs to 132, none apart, ten
satellites or more in each. Six Galileo satellites take part on both bands (two more are
tracked but not at the base), four GPS satellites on L1 and L5 and three on L1. It is
firmer than with E5a alone, still without a truth: leaving out one satellite at a time
(seven tried), 77 to 145 epochs stay agreed and none to three stand apart, and the forward
fixes that both runs have lie 3 to 11 mm from each other in the median, 3 to 17 of some 160
more than 3 cm, single ones up to a meter. RTKLIB's rules on the same file agree at 11
epochs.

The drive is not better for E1: 216 agreed and three apart, at 53, 66 and 67 s, 5.4 to
5.6 cm horizontally where the bound is 5. They lie in twenty seconds of a turn, 51 to 70 s,
in which the forward filter is float at five epochs and the backward at one and, where
both fix, they stand 1 to 7 cm apart; before 48 s they agree within millimeters and from
72 s within a centimeter. From 64 to 70 s the difference is a steady 3 to 6 cm one way, so
one direction holds other integers there, and five of those seven epochs still count as
agreed: the bound is a coarse measure. G24, one of the four GPS satellites with a pilot,
is 5 to 13 dB down from 53 s with its L1 phase flagged at every epoch, and G11 follows
from 66 s. The stretch hangs on the satellite set:
without G24 no epoch stands apart (210 agreed), without G11 or without E27 one, without
E13 seven. Each direction by itself stays within 2.5 cm when E1 is taken away, at every
epoch but one. No setting of the lock count or the floor removes the three (floor 6 or 7:
three; lock count 10: four); restarting every flagged phase leaves one but takes the other
capture down to 36 agreed epochs. Which direction is right in that stretch is not known.
The parked car is as with E5a: every kinematic fix within 15 mm of their median, which lies
2 mm from the GPS-only static solution.

**GPS alone against Galileo alone** was to be the check that shares no satellite. With both
bands Galileo alone still does not fix: no epoch of the parked car in kinematic or static
mode (six satellites; the ratio stays below the threshold of 3, at 2.8 at most in static
mode and 2.1 in kinematic), one of the capture started while driving, 2 forward and 14
backward of the drive. On the parked car its float solution in static mode ends 7 cm from
the GPS-only fix in either direction. On the drive
its 16 fixes all lie where the solution with every satellite lies; of the 8 that fall on
epochs the GPS-only run fixes, 6 agree with it and 2 stand 36 cm away, at 78 and 79 s,
where the GPS-only forward fix has seven satellites and the solution with every satellite
sides with Galileo. So the comparison of two fixed solutions is not available on these
recordings; what stands in for it is the whole-cycle check at the GPS-only position in
"Galileo observations" below, on the parked car only.

**Two signals that move together.** The other signal's word is worth less than it looks.
Whether a phase moved can be judged without its satellite's other signal, against the other
satellites (next paragraph). Judged so, from the rover's observations with broadcast
orbits, the phases of the parked car never change by more than 1.2 cm from one second to
the next beyond what the others explain, down to 30 dB-Hz. On the two moving captures the
pilot phases (GPS L5, Galileo E1 and E5a) do so by more than 4 cm, without a flag, 73 and
118 times. Two of those are slips of a whole cycle, each through a fade of 40 to 160 ms
in which the signal's own phase went around once and the loop followed it ("Galileo
observations" below). In many of the others both signals of the satellite move together,
the difference of the two under half the move (19 of 22 such events with both Galileo
pilots unflagged on the drive, 26 of 48 on the capture started while driving), often out
and back at consecutive seconds, and the one GPS case looked at shows the same on the L1
C/A phase and the L5 pilot. Most fall in a second in which the signal faded (61 of the 73
and 109 of the 118).
What they are physically is not established here; a path around the edge of an obstruction
is longer by the same length on both frequencies. The geometry-free difference shows none
of it. So a flagged phase beside an unflagged one was taken for continuous while both
stood centimeters from where the other satellites put them: on the drive, in the turn that
holds the three epochs apart, one GPS satellite's L1 phase is flagged at 21 of 25 epochs
and both its phases move by 6 to 21 cm against the others at three of them.

`change_limit` (m) withdraws that word. Between two epochs the single-differenced residual
of a phase changes by the change of the position error along its line of sight, by a
receiver clock term that is one number per carrier frequency (it also holds the antenna's
turn, one cycle a turn on every phase of a frequency), and by what the phase did on its
own. The engine fits the change of position and the clocks to the phases that neither
receiver flags at either epoch, with the least sum of absolute residuals, once without
each satellite in turn (a phase that moved still pulls on such a solution when few phases
are there), and takes the solution that explains the most of them within 1.5 cm
(`CHANGE_FIT`). A phase that stands more than `change_limit` from it has moved, flagged
(by the lost-lock bit) or not. The consequence is one: where a phase in question or the signal that would vouch for
it has moved, neither vouches, and the phase in question goes to the test against the rest
of the epoch. No phase restarts for having moved, an unflagged phase is not touched, and
the 5 cm geometry-free test is as it was. There is no verdict, and the engine is as
without the option, with fewer than twelve unflagged phases (`CHANGE_MIN_PHASES`), or
when the phases the solution explains, less one per unknown, do not outnumber the
unflagged phases it leaves out: those could be the receiver's motion. The phases of a
frequency are judged only when the solution explains three or more on it
(`CHANGE_MIN_BAND`): a clock fitted to one or two phases explains them whatever they did.
And no verdict is given across more than 5 s (`CHANGE_MAX_STEP`) or where the base epoch
changed by another step than the rover's. The change of a residual also holds the turn of
the line of sight times the error of the position the residual was formed at, in
kinematic mode the epoch's single-point seed: a millimeter over a second, centimeters over
half a minute; and under a base at a lower rate the base's own residuals over its step.
With `slips="restart"` the option does nothing.

How well the verdict holds was tried on synthetic epochs: twelve satellites on two
frequencies at random above 15 degrees, some of the 24 phases flagged, among them one
signal of a satellite whose two phases move together, 300 epochs a case. With sixteen
unflagged phases and 2 mm of noise on a change (a strong pair of phases) a move of 6 cm is
found in 99 % of the epochs and one of 4 cm in 96 %, and no other phase is blamed; with
5 mm (the few phases of the parked car at 30 to 34 dB-Hz show 4) 99 % and 82 %, another
phase blamed in 1 epoch of 300. With twelve unflagged phases, the least for a verdict,
94 % and 89 % at 2 mm with another phase blamed in 2 to 3 % of the epochs, 94 % and 74 % at
5 mm with 3 to 5 %. When nothing moved, a phase is called moved in none of the 300 epochs
at 2 mm and in 3 at most at 5 mm. At 8 mm, more than any phase here that still holds lock
shows, the verdict degrades: a phase is called moved when nothing moved in 4 to 14 % of the
epochs with twelve to twenty unflagged phases. A blamed phase costs little: a flagged phase is tested against the rest of the
epoch where its other signal would have vouched.

Measured with the command's options, without the limit and with 0.03 m (fixed forward and
backward, agreed, apart):

| Capture, signals | without | `change_limit=0.03` |
|---|---|---|
| started while driving, GPS and Galileo E1 and E5a | 206, 190, 132, 0 | 222, 194, 148, 0 |
| the drive, the same | 245, 249, 216, 3 | 247, 249, 221, 0 |
| parked, kinematic, the same | 294, 299, 294, 0 | 294, 299, 294, 0 |
| the drive, GPS and Galileo E5a | 242, 255, 227, 4 | 242, 255, 226, 5 |
| started while driving, GPS and Galileo E5a | 104, 177, 54, 0 | 104, 177, 54, 0 |
| the three captures, GPS L1 and L5 | 1, 39, 0, 0; 202, 178, 144, 1; 14, 0, 0, 0 | the same |

With GPS alone the counts do not move because there is little to judge: on the capture
started while driving no epoch has twelve unflagged phases. On the two station pairs with a
known marker the option cannot act at all: no phase is ever in question there, and at
their 30 s there is no verdict. The command's options fix the same epochs as before (714
and 721 of 721 on L1+L2, 709 and 696 on L1), every fix within 2.4 cm; that says the option
does no harm there, not that it is right. With Galileo E5a alone beside GPS the drive is
not better for it: the L1 frequency then has the GPS phases only, at times too few
unflagged ones for a verdict on them.

What that is worth. The drive has no epoch apart any more, but by a small margin: the three
that stood 5.4 to 5.6 cm apart stand 4.2 cm apart, under a bound of 5. Through the turn
(51 to 72 s) the two directions still differ by 0.8 to 4.2 cm wherever both fix, and over
the whole drive the epochs on which they differ by more than 3 cm go from 12 to 8. Which
direction is right in the turn is as unknown as before. The limit itself matters little
on the drive (221 to 223 agreed and none apart from 0.02 to 0.05 m) and more on the capture
started while driving (148 agreed at 0.02 to 0.03 m, 134 at 0.035 and 0.04, 131 at 0.05,
none apart at any). That capture's gain is thin. Run forward, it follows from one event: at
182 s a flagged L1 phase of a satellite that itself did not move is kept, because a phase of
another satellite that changed by 3 to 3.5 cm is in question beside it and out of the trial
solution; with that ambiguity kept, three epochs from 187 to 189 s fix clearly (ratios of
5.5 to 6.3 against 2.2 to 2.8) and thirteen from 213 to 225 s pass the ratio test at 3.0 to
3.5 where without the limit, or with 0.035 m, they fail it at 2.9 to 3.0. The three
restarts the limit itself adds there bring no fix. Left out one satellite at a
time, the drive keeps 210 to 236 agreed
epochs and has none apart in 15 cases of 19 (three without either of two Galileo
satellites, six without a third, one without a GPS satellite; without the limit 205 to
231 agreed and none apart in 2 cases of 19). The capture started while driving is no
firmer for it: 78 to 151 agreed and none to three apart with one satellite left out (77
to 145 and none to three without the limit), on the same epochs, and single forward fixes
still lie up to a meter from where they lay with every satellite. The rule also leaves the
unflagged signal that moved with the flagged one where it was: its error goes into the
solution as before. A move that is back at the next epoch, the common kind, costs the
flagged phase its ambiguity where it fails the test: it begins again on the displaced
phase and waits out the lock count. A move spread over several flagged epochs in steps
under the limit passes, since the change is taken from one epoch to the next. The run takes
about half again as long with the option (a fit per satellite at every epoch). And the
rule was found on these two captures.

## Standalone positioning

Satellite position, velocity and clock come from the broadcast ephemeris
(IS-GPS-200 Table 20-IV with the relativistic clock term; velocity and clock drift
by central differences over 1 ms). For each satellite the record whose `toe` is
nearest the epoch is used, within 2 h; a satellite with no such record or with a
health flag set is left out. Times of week are reduced into half a week either way
before any difference with an ephemeris time, so a file crossing the week boundary
is handled. Galileo navigation records are read with their calendar epochs taken as
GPS time, as RTKLIB does; the GST to GPST offset is tens of nanoseconds, far below
the orbit's time resolution.

The ionosphere is corrected with the Klobuchar model from the navigation file's
`GPSA`/`GPSB` coefficients; a file without them (the worldwide GPS-only broadcast
files, or our own navigation file from a recording too short to hold subframe 4
page 18) gets RTKLIB's default coefficients, as `rnx2rtkp` applies. The
troposphere is corrected with Saastamoinen using the
standard atmosphere at the receiver height and 70 % humidity.

Position and receiver clock bias are found by iterated weighted least squares over
the pseudoranges, each corrected for the satellite clock and group delay (`TGD`),
with the Earth-rotation term in the range. The 15 degree elevation
mask, the chi-square test on the residuals (0.1 % significance) and the GDOP limit
of 30 are those of RTKLIB's single-point mode, and so are the weights with
`--weights rtklib` (the default of `pvt.solve`): the two solvers then compare number
for number on the files tested; the one known difference is that RTKLIB also
leaves out a satellite whose URA index is 10 or more (an accuracy worse than
300 m), which this solver does not. An epoch with fewer than four usable
satellites or failing a check gets no solution and the run continues.

RTKLIB's pseudorange variance sums the broadcast accuracy (URA), a code-bias term, half
the broadcast ionospheric delay, the Saastamoinen error and a 0.3 m code error growing
toward the horizon. None of it knows how strong the signal was, and a satellite tracked
at 36 dB-Hz has a pseudorange several times as noisy as one at 48 dB-Hz at any
elevation. `pyrx-pvt` therefore adds, by default (`--weights cn0`), a term in the
observation's C/N0 to that variance:

```
(1.5 m)² · 10^(−(C/N0 − 45 dB-Hz) / 10)
```

so 1.5 m at 45 dB-Hz, 4.7 m at 35, 0.8 m at 50; an observation without an `S1C` value
gets none. `S1C` is taken as dB-Hz, which is what a file says with `DBHZ` in its
`SIGNAL STRENGTH UNIT` record and what one without the record is assumed to hold; a file
that declares another unit is solved with RTKLIB's weights, and `pyrx-pvt` says so in
one line on standard error. The slope is the one the residuals show on sky recordings (about ten times
the variance for every 10 dB), and has the form of the signal-strength term of the
demo5 fork of RTKLIB (`stats-errsnr`); RTKLIB 2.4.3 itself has only a signal-strength
mask in this mode. The 1.5 m was taken from a sweep over three five-minute car-roof
recordings as the largest of the values tried (0.5, 1, 1.5, 2, 3 and 5 m) that made
nothing worse on any of them: parked, the
scatter of the fix about its mean went from 2.4 / 1.3 / 8.9 m (north / east / up, RMS)
to 2.0 / 1.3 / 8.4 m with two of eight satellites below 41 dB-Hz; driving, the distance
to RTKLIB's track from Pocket SDR's observations of the same samples went from
2.0 m horizontal and 4.6 m vertical to 1.5 and 3.0 m. The weak satellites are the low
ones there, which the geometry needs, so they are weighted down and not left out. The
chi-square test uses the same variances, so it tolerates a larger residual on a weak
satellite. Velocity and clock drift then come
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
seen at sample 0 plus 68 ms of nominal travel time, rounded to a whole
millisecond, which puts pseudoranges near their true values and leaves the
remaining bias to the position solution.
Observation epochs are the receiver times at which that clock reads a multiple of
the interval; a channel is sampled there by interpolating between the two code
starts that bracket the epoch, from its first bit-synced epoch on. Pseudorange is
the light time between receiver and transmit time, both held as seconds of week in
double precision, so it sits on a lattice of 17 to 35 mm (the resolution of a
double near 600 000 s), well under the code noise; carrier phase is the negated
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
first bit-synced epoch to the end of the run or to loss of lock; a lost satellite
comes back as a new channel with its own carrier-phase ambiguity (the observations
in between are absent), and so does one that was too weak at the previous search.
An epoch whose 1 s tracking window is not phase-locked (the phase-lock indicator
below 0.25, or unknown in an incomplete final window) has no `L1C` while its `C1C`,
`D1C` and `S1C` are written, as the u-blox and NovAtel decoders in RTKLIB withhold
the phase when the receiver's phase-lock flag is off; under the loss rule such a
stretch inside a channel lasts at most 2 s before the channel ends.
The data polarity, and with it the half cycle added
to the phase, is read from the preamble of every decoded subframe: a subframe of
the opposite polarity means the Costas loop slipped half a cycle since the previous
one. The loss-of-lock indicator written with `L1C`: value 1 (bit 0, slip possible) on a
channel's first observation and those within its first 2 s (Pocket SDR's rule), on
the first observation with a phase after any tracking epoch since the previous one
with a phase that was not phase-locked (an unlocked window can lie between sparse
observations), after a fade of the carrier loop (below), and on the first observation
after a polarity change; value 2 (bit 1, half-cycle ambiguity) where no
decoded subframe establishes the polarity, that is before a channel's first
subframe, between the last subframe before a polarity change and the first
after it, and where fades leave it open (below). An RTK engine restarts the satellite's ambiguity at every change of that
flag (RTKLIB's rule), so a polarity change costs the ambiguity twice, when the flag is
set and when it clears; the phase is written flagged rather than blanked or flagged for
the whole interval. No signal-strength flags are written. The standalone solution does
not use the carrier phase.

A moving receiver's phase slips in fades. Under trees and past obstructions the signal
drops 3 to 15 dB for tens of milliseconds and its phase jumps; the Costas loop then
settles on the nearest half cycle while the phase-lock indicator stays high. One such
slip shows as a polarity change up to a subframe (6 s) later; two between subframes
leave the polarity as it was, the phase a whole cycle off after them and half a cycle
off between them. The loop marks these
fades itself: it holds its integrator while the 10 ms prompt power is below half its
running mean. An observation is therefore flagged (value 1) when some half second
that ends no earlier than the epoch of the channel's previous observation with a phase,
and before its own epoch, contains 50 ms or more of hold (the phase of an epoch is the
loop's state before that epoch's update, so a held epoch counts for the phases after
it). The condition thus holds from 50 ms after a
fade begins to up to 0.45 s after it ends, while the loop settles, and every observation
whose stretch since the channel's previous phase contains an epoch where it held carries
the flag (an observation taken in the first 50 ms is written without it, and the next
one carries it); so at one observation a second a fade just before an observation flags
that one and the next.
The flag says a slip was possible, not that one happened. Measured on the car-roof
captures against the L5 pilot of the five satellites that have one (a pilot loop has no
second lock point half a cycle away, so an L1 step shows against it wherever the L5
phase is itself unflagged): of the L1 phases written, 23 on the cold-start drive and 24
on the driving capture are discontinuous from the epoch before (more than 4 cm against
the pilot); without the fade rule 21 of each carried no flag, with it none and 2 (steps
of 5.9 and 4.5 cm at 38 dB-Hz with little or no hold before them, not half cycles). The
price is flags: 34.7 % of the cold-start capture's L1 phases and 11.2 % of the driving
capture's carry one (1.6 % and 1.9 % without the rule), on those five satellites
fifteen and six flagged phases per discontinuity, and an RTK engine that follows RTKLIB
restarts the satellite's ambiguity at each (`pyrx-rtk` looks at the phase first, see
RTK). The parked capture gains none: there the hold fires
only in runs of 30 ms at most on the three weakest satellites (36.6 to 40.5 dB-Hz),
never for 50 ms within a half second. A wider carrier loop is no remedy: on the
cold-start capture the same satellites slip 28, 33 and 36 times at 10, 15 and 20 Hz
against 15 times at 5 Hz. The rule is checked only where a pilot exists, and a slip
outside a fade would not be flagged.

The fades also say where the half cycle is open (value 2). A subframe that decoded held
its polarity throughout, so a fade whose held epochs all lie under decoded subframes is a
slip of whole cycles at most. Where fades lie between the end of one subframe and the
start of the next one of the same polarity, the polarity came back, so they held no slip
or two; with two, the phases between them are half a cycle off, and one long flickering
fade can hold both. The phases from the first held epoch of the first such fade to the
end of the last one's flag therefore carry the half-cycle flag; so do the phases from a
fade after a channel's last subframe to its end, since nothing decoded later says where
the loop settled. That takes the L1 phases with the half-cycle flag from 9.9 % to 13.6 %
on the driving capture and from 10.2 % to 21.5 % on the cold-start drive, and adds three
and five ambiguity restarts to the 270 and 700 an RTK engine makes on them.

### L5 observations

The L5 pilot carries no message, so its channel knows the phase of its 1 ms code but
not which millisecond of the week it is. Nothing is decoded on L5; the time comes from
the same satellite's L1 channel on the same sample clock. Both signals leave the
satellite together and arrive within a fraction of a microsecond of each other, so the
L1 transmit time at a pilot code start, rounded to the code period, is that code
start's transmit time. Every pilot code start that falls inside a time-tagged L1
segment must name the same time for the pilot's first epoch, each within 1 microsecond of the
grid; then the whole L5 segment is on GPS time, the part before L1 was acquired
included. A segment that never overlaps its satellite's L1, or disagrees with it, is
left out. (The pilot's code starts lay 0.1 microsecond from the L1 code grid on the two
front ends measured; a pilot channel locked on something else lands anywhere in the
millisecond and is turned away.) The receiver clock and the epochs are L1's: an instant
at which no L1 channel has an observation gets no `L5Q` either, and a satellite never
tracked on L1 gets none at all.

`C5Q`, `D5Q` and `S5Q` are written from the epoch after the pilot's secondary code was
found; `L5Q` where the pilot's window is phase-locked, blank otherwise, as on L1. The
pilot loop tracks the whole cycle, so there is no half-cycle indicator on `L5Q`; where
the tracker had settled on the inverted pilot before the secondary code was found, the
half cycle is put back. The loss-of-lock indicator is value 1 on a segment's first
observation and within 2 s of the segment's start, on the first phase after an unlocked
window, and on every phase from a coasted sum to the first that holds the loop's return
(below).
RINEX 3.05 (Table A23) makes L5I the reference signal of the band and has the L5Q
phase aligned to it by a quarter cycle; the phase written is aligned (the quarter cycle
is applied), and the header's `SYS / PHASE SHIFT` line for `L5Q` carries the 0.00000
of an already aligned phase. The sign of that quarter cycle follows IS-GPS-705 (Q5
lags I5), the relation the data symbols confirm (see "Pilot signals"); it is common
to all satellites, so it cancels between satellites.

The pilot loop coasts through an outage. With the signal gone a 20 ms sum is noise, its
phase is uniform over the cycle, and a loop that keeps updating on it random-walks its
Doppler: such a loop came back one to three whole cycles away from most synthetic
outages of 0.4 to 1.5 s. The tracker therefore judges each sum by its coherence, the
squared magnitude of the sum over twenty times the summed power of its prompts (1 on a
clean signal, 0.05 on noise, 0.28 at 25 dB-Hz), and takes it into the loop only at 0.20
or above. On any other sum the loop coasts: the discriminator reads zero, which takes
the last error's proportional term back out of the Doppler, and the Doppler rate keeps
steering. One noise sum in seventy still reaches the threshold, so the first sum taken
after a coasted one moves the Doppler by its proportional term only; the integrated
terms take a sum's error when the sum before it was taken too. The integrator hold (see "Tracking loops") stays in
force beside the gate: behind an obstruction the signal is more often attenuated and
flickering than gone, its weak sums pass the gate with a noisy phase, and the hold keeps
them out of the rate estimate the loop then coasts on. In synthetic outages of 0.4 to
1.5 s at 38 and 45 dB-Hz under a steady Doppler ramp of up to 3 Hz/s the loop came back
on the same cycle in 143 runs of 144, where the loop that kept updating did in 52.
Coasting is an extrapolation, and it fails where the rate it extrapolates is wrong. A
maneuver that begins during the outage is one case: with the ramp stopping where the
outage starts, coasting does no better than updating on noise. The seconds after the
secondary code is found are another: the third-order loop's rate estimate is still
settling from the handover (about 1.5 Hz/s off in the half second after the sync, 0.3
Hz/s 1.5 s after it, 0.1 Hz/s from 2 s on), and an 0.8 s outage that began within 0.2 s
of the sync cost a cycle in 10 runs of 23. A second outage 0.1 to 0.2 s after the first
is a third: the return's transient is still in the rate estimate, and 4 runs of 16 came
back a cycle off. And at 35 dB-Hz the estimate is noisy enough that 11 outages of 65
cost a cycle and one more ended the track.

Two five-minute recordings from the roof of a moving car (fifteen and fourteen pilot
channels, an obstruction every few seconds) put the synthetic numbers in proportion. A
building or a tree mostly attenuates: the gate found sums without signal in about 200 of
700 fades and coasted 42 and 34 s of the 70 and 58 channel-minutes tracked. Against the
same loop without the gate the phase-locked time was 1.5 % longer on the driving capture
and the same on the cold start, and the `L5Q` phases numbered 1392 for 1406 and 1225 for 1222. What the recordings do not show is that
a coasted return is on the right cycle. The satellite's L1 phase is a check where it
stayed locked through a fade, and in all 132 such fades the two loops passed it alike
(the geometry-free combination did not move in 124 and stepped by the same amount in
both loops in 8, so not by the pilot's doing), but the gate had coasted in only 7 of
them, for one or two sums. In the fades with coasted stretches less than 0.2 s apart, a
flickering signal, both loops returned from 90 of 132 and stood whole cycles from each
other after 19, and nothing in the recordings says which was right. The window's
indicator shows none of this: it is weighted by the power of the sums, so 0.2 s of
signal and 0.8 s of noise still read 0.95. The tracker therefore marks every coasted
sum, and every `L5Q` from a coasted sum to the first one that holds the loop's return
(the next sum's update, 20 ms after the coasted sum ended) carries the loss-of-lock
indicator.

On the two recordings of 2021-12-26 (`test_joint_observables_reference.py`) the four
satellites on both bands have both signals at every epoch. `C5Q` minus `C1C` is
-29 m for all of them within 2.4 m (mostly the front end's delay between its two
channels; the mean ionospheric difference and the satellites' inter-signal delays, a
few meters, are in it too);
the geometry-free combination of the two phases follows a line within 0.8 to 2.3 mm
RMS over the half minute, which is the two carrier loops agreeing to a hundredth of a
cycle; and RTKLIB's single-point fix is the same from the joint file as from the L1
file. The test requires the common `C5Q` minus `C1C` to lie between -40 and -20 m, each
satellite within 5 m of it, and the geometry-free residual below 5 mm RMS and 10 mm at
worst. On the car-roof capture (a `slow` test on its first 120 s) the four strong
satellites agree in the same way over five minutes, and their `C5Q` agree with
Pocket SDR's 5Q pseudoranges between satellites within 0.2 m in the mean and 0.6 to
0.9 m of scatter, once the 13 ms between the two receivers' clocks is accounted for
(the test requires the geometry-free residual below 5 mm RMS and each satellite's `C5Q`
minus `C1C` within 3 m of the common value).

### Galileo observations

A Galileo satellite has no signal here that carries a decoded time: nothing Galileo is
decoded, and there is no channel of the same satellite to borrow the time from, as the L5
pilot borrows L1's. But the E5a pilot's secondary code is 100 chips long, one chip per
1 ms code period, and the code start that carries its chip 0 leaves the satellite at a
multiple of 100 ms of system time (the tiered code starts with the data component's
page, which starts on the second). So once the secondary code is found, the transmit
time of every code start is known up to a multiple of 100 ms, and the travel time from
a satellite at that height is 60 to 100 ms: the receiver's clock, which the GPS channels
set within 20 ms of true time, says which multiple. pyrx takes the travel time between
22 and 122 ms by that clock (`TRAVEL_TIME_MIN`: the two Galileo satellites in eccentric
orbits are 57 to 106 ms away, and the clock can be 18 ms behind, so a travel time reads 39
to 107 ms). Nothing at run time checks the time a pilot is given beyond that. The run therefore
needs GPS L1 for its clock, and the difference between Galileo's and GPS's system time,
tens of nanoseconds, is in the pseudoranges; a relative solution with a reference
satellite per system does not see it.

The rest is the L5 pilot's: `C5Q`, `D5Q` and `S5Q` from the epoch after the secondary
code was found, `L5Q` where the window is phase-locked, the loss-of-lock indicator within
2 s of a segment's start, after an unlocked window and from a coasted sum to the return,
no half-cycle indicator, and the half cycle of a pilot found inverted put back. The phase
is written aligned to the band's reference signal as RINEX 3.05 Table A23 has it, L5Q a
quarter cycle from L5I, with 0.00000 in the header. That value is the table's. Derived
from the signal's definition, in which the E5a pilot leads the data component where the
L5 pilot lags, it would be the opposite quarter cycle: half a cycle between the two
conventions, the same on every Galileo satellite, so it cancels between satellites and no
relative solution depends on it.

Galileo's orbits come from a navigation file: only its I/NAV records are read, as RTKLIB
does, so a satellite is used when the whole health word of that record is zero, and the
E5a signal's own health bits, which the F/NAV records carry, are not looked at.

Checked on the car-roof captures (`test_rtk_reference.py`): at the single-point fix of
the GPS L1 rows, the E5a pseudorange of every Galileo satellite is the range to the
satellite's broadcast position within the atmosphere, the group delays and the front
end's delay between its bands (-37 to +15 m over every epoch of the three captures; the
test allows 100 m, and a wrong millisecond would be 300 km). Against the Galileo
pseudoranges Pocket SDR forms from the same recordings, differenced between satellites,
they agree within 0.25 m in the mean on every satellite with a minute of common epochs. The GPS rows of a file with Galileo are those of
the file without. What the Galileo observations do for the fix is in "Flagged phases, the lock
count and the backward pass" above.

The E1 pilot is timed by the same rule: its secondary code has 25 chips of one 4 ms code
period each, 100 ms again, and the tracker holds it as one entry per 1 ms epoch. Its
observations are `C1C L1C D1C S1C` under the L5 pilot's rules, the phase with the half
cycle that Table A23 puts between L1C and the band's reference signal L1B, and 0.00000 in
the header. The two pilots of a satellite are tracked in two recordings by channels that
share only the sample clock, and each finds its secondary code on its own, so their
transmit times at one instant check each other: where they stand more than a microsecond
apart (`PILOT_AGREEMENT`; a secondary code found wrongly is a millisecond or more, the
front end and the ionosphere a tenth of a microsecond) neither is written, since nothing
says which is wrong; each is written where it is tracked alone, and a stretch withheld is
not a loss of lock. That rule sees a wrong millisecond, not an E1 channel beside a side
peak of its correlation, which is half a microsecond or a little more off; against that
stands the tracker's guard ("Galileo E1" below).

Checked on the car-roof captures, in files formed by a whole run of the command
(`pyrx_e1e5a.obs`, `test_rtk_reference.py`): the GPS values and the E5a values are those of
the earlier files in every field, so the agreement rule withheld nothing; `C1C` minus `C5Q`
is one value, 31 m, the front end's delay between its bands, on every satellite at every
epoch within 7 m (7, 10 and 8 satellites; the test allows 30 m; a channel beside a side
peak would stand 147 to 183 m off); on the parked car the geometry-free combination of the two phases is a line
over the five minutes, with 1.4 to 2.9 mm of scatter and 11 mm at most (the test allows 5
and 20 mm), on each of the seven satellites. Outside the tests: at the position that the
GPS observations alone give the parked car, the double-differenced E1 and E5a phases
against the base are whole cycles on every pair, within 0.02 to 0.06 cycle RMS over the
299 epochs, one integer each. That is the check of Galileo against GPS at the level of the
observations: the time tags are right, and so is the half cycle put back for a pilot found
inverted (both polarities occur on both bands there). The half cycle that aligns L1C with
L1B is the same on every E1C phase and cancels between satellites, like E5a's quarter
cycle, so no measurement here checks it.

In motion the pilots' phases move without a flag now and then. The geometry-free
combination of a satellite's two phases steps by more than 4 cm from one second to the
next, neither phase flagged, 23 times on the capture started while driving (the largest
29 cm, more than a cycle of either band), 4 times on the drive (7 cm at most) and never on
the parked car (8 mm at most); on GPS L1 and L5 of the two moving captures the same count
is none and two. Which band moved shows against the other satellites, without the other
band (the judgment "Two signals that move together" above describes): in 17 of the 29
steps one band alone, E1 in 8 and E5a in 9, in 5 both, and 7 are not settled (a band
between 1.5 and 2 cm, or no solution at that second). Two are slips of a whole cycle: an
E5a cycle on one satellite, an E1 cycle on another. Neither is the loop
losing a signal that was there. In the first the power of the 20 ms sums fell to an
eighth for 160 ms, their coherence to 0.21 and 0.24, just above the 0.20 at which the loop
coasts and the phase is flagged, and the sum's phase against the loop went to 0.46 cycle
and came back without wrapping; in the second the fade was 40 ms long with a coherence of
0.36. The signal's own phase went around once in the null of the fade, and the loop
followed. The other steps are 4 to 10 cm on one band and gone a second or two later. Far
more often both bands move together, which this combination cannot show; what the RTK
engine does about that is in "Two signals that move together".

## Navigation message decode

GPS L1 C/A carries the 50 bps LNAV message. Once the phase lock has settled, bit
sync finds the data-bit edge among the 20 one-millisecond epochs of a bit by
counting prompt sign changes at each epoch phase; frame sync then looks for the
TLM preamble repeating one subframe (300 bits, 6 s) later with the TLM and HOW
words of both candidates passing parity, a valid subframe ID in the first and
consecutive TOW counts, trying both data polarities. From there every 300-bit
block whose ten words all pass parity and whose TOW count is the one its position
in the sequence requires is a subframe; a block failing either is dropped on its
own, except that a bit error in the last two bits of a block also spoils the
parity of the next block's first word and drops that block too. The HOW alert
flag is not decoded; an alerted satellite's data is used, as RTKLIB and Pocket SDR
do. Subframes 1, 2 and 3 decode to the clock and ephemeris parameters, subframe 4
page 18 to the ionosphere parameters, all scaled per IS-GPS-200 with semi-circles
converted to radians. Each subframe is time-tagged with the tracking epoch of its first bit and
the GPS seconds of week of that bit (the HOW's TOW count refers to the start of
the following subframe). Almanac pages and the UTC parameters are not decoded.
The ephemeris is assembled from a channel's latest subframes 1, 2 and 3 when their
issue of data agree, which assumes a recording of minutes: over hours an issue
number can be reused for a new data set, and no age bound guards against that.

The reference test decodes the two 30 s recordings and requires the clock and
orbit fields of every subframe 1, 2 and 3 (and the week number; not the URA, the
L2 code flags, the fit interval or AODO, which the files do not carry exactly) to
equal the broadcast record with the same issue of data to within half a
transmitted LSB (some records in the files are written to only seven significant
digits), every subframe's time tag to match its position in the file within
0.1 s, every channel to agree on the file start time to within 30 ms, and the
2021-12-26 recording to yield a complete ephemeris on every channel. The 2022-01-14 recording starts inside subframe 1, so its window holds
subframes 2 to 5 and no complete ephemeris. Neither window contains subframe 4
page 18, so the ionosphere parameters are checked by the synthetic round-trip
tests and by a page-18 vector laid out word by word from IS-GPS-200, independent
of the decoder's field table.

## Tracking loops

Each channel integrates 1 ms (one code period of each signal) with early and late taps at
±0.25 chip and a noise tap 10 chips away. A frequency-locked loop pulls in for
the first second, its discriminator on coherent sums of the prompt: 2 ms sums at
5 Hz bandwidth for 0.3 s (pull-in range ±125 Hz, since a weak satellite's acquisition
peak can be 100 Hz off), then 5 ms sums at 2 Hz for 0.7 s (the sums divide the
discriminator's noise, so a 30 dB-Hz channel hands over about a hertz off, 1.1 Hz at
worst measured, where single epochs left 10 Hz). A third-order Costas phase-locked loop with 5 Hz bandwidth (Pocket
SDR's C tracker's constants) then holds the carrier: third order because a moving
antenna's Doppler rate would otherwise lag the phase, 0.11 cycle at the 10 Hz/s a car
produces, and lose the satellite. While the prompt power drops below half its recent
mean (an obstruction's onset) the loop's integrator is held, since on the faded signal
it would wind up and slip the carrier. The code loop is a
0.25 Hz delay-locked loop on 10 ms non-coherent early/late sums, with the code
rate aided by the carrier Doppler. C/N0 is estimated from prompt versus noise-tap
power over 1 s, each new window's estimate averaged in half with the value held
before, so the reported C/N0 (the `S1C` of the observations) is that of completed
windows only and trails a change by a second or two, as a receiver reporting in real
time would; the phase-lock indicator is taken over the same window, and a channel is
dropped after two consecutive windows with C/N0 below 25 dB-Hz or the indicator
below 0.25, the first window judged on C/N0 alone because the pull-in spins the
carrier through it. Every channel's carrier
NCO starts at the receiver's IF phase at sample 0 (modulo one cycle), so the
accumulated carrier phases of different channels share one origin. The constants come
from Pocket SDR's two trackers: the DLL and the C/N0 window match its Python tracker,
the PLL is the third-order loop of its C tracker (a3 = 1.1, b3 = 2.4, W = Bn / 0.7845),
so a locked channel can be compared with them number for number on the same recording;
the FLL on coherent sums (Pocket SDR's works epoch by epoch), the integrator hold and
the two-window loss rule are this receiver's own. None of them are the final choices
for the hardware receiver.

### Pilot signals

GPS L5 and Galileo E5a each send two components a quarter cycle apart at equal power:
one with the navigation data and one without, the pilot. Both have a 10230-chip code
at 10.23 Mchip/s (1 ms), and each code period is multiplied by one chip of a slower
secondary code: 20 chips on the L5 pilot, 100 on the E5a pilot (one code per
satellite), 10 and 20 on the data components, where a data symbol spans one
repetition. The tracker is the same for every signal and is driven by a description
of the signal (`signalspec.py`); for a pilot it runs in stages:

1. Pull-in, 1 s: the FLL on single epochs (5 Hz for 0.5 s, then 2 Hz), because a
   coherent sum over a sign change of the secondary code cancels. This is Pocket SDR's
   pull-in and the one that handed a 30 dB-Hz L1 channel over 10 Hz off; the weakest
   pilot held on the parked car-roof capture, G14, reads 35 dB-Hz.
2. The 1 ms Costas loop, which does not care about the sign changes.
3. From 1.5 s the last prompts are compared with the secondary code at the current
   alignment (Pocket SDR's rule: the correlation of the code with the in-phase prompts
   reaches 0.8 of their mean magnitude), and the match is accepted only while the
   Costas loop is phase-locked (cos 2φ of the last 100 prompts at least 0.5). Without
   that condition the 20-chip code matched noise within a second on every absent PRN
   tried. A window holding an epoch of exact zeros (a gap in a file) is not compared.
4. From then on the secondary chip is removed from every prompt and the carrier loop
   is the same third-order update on the four-quadrant phase of 20 ms coherent sums
   (Pocket SDR's pilot loop): no half-cycle ambiguity, and 13 dB more signal in each
   update. A sum whose prompts do not add up coherently is not used and the loop
   coasts on it (see "L5 observations" above for the rule and what it
   was measured to do); the integrator hold of the 1 ms stage runs on. The phase-lock
   indicator is cos 2φ of the sums; it reads -1 if the loop ever stood half its 50 Hz
   update rate off, where the errors alternate and average to zero.

The code loop, the C/N0 estimate and the loss rule are L1's. At 24 Msps a chip is
2.35 samples and the front end's filter rounds the correlation peak, so the
early-minus-late discriminator's slope is about a fifth below a triangle's and the
code loop's bandwidth correspondingly below 0.25 Hz.

The data component (I5, E5a-I) is correlated every epoch with its own code at the
pilot's carrier and code timing. After the sync its secondary chips are removed and
each symbol's epochs are summed along the axis the interface specifications give the
data relative to the pilot (Q5 lags I5 by a quarter cycle; E5a is I plus jQ), giving a
stream of soft symbols in the track result. Nothing is decoded from it yet and it
steers no loop. On a recording both components read the same power within 0.2 dB,
and on Galileo the uncoded 12-symbol sync pattern of the F/NAV pages appears every
500 symbols with the sign it is sent with, which checks the axis and the sample
convention. On L5 the symbols are convolutionally coded and nothing in the package
decodes them, but a rate-1/2 Viterbi decode of the symbol stream outside it finds the
CNAV preamble upright, followed by the satellite's own PRN and the right time of
week, every 300 bits on four satellites of the 2021 recording, among them one whose
sync settled on the inverted pilot: the L5 symbol sign is as the specification has it.

Checked on the L5 recording of 2021-12-26 against Pocket SDR's C tracker
(`test_tracking_pilot_reference.py`): five GPS and six Galileo satellites from 38 to
50 dB-Hz find the secondary code at 1.5 to 1.6 s and stay locked for the whole file. As
measured: the indicator 0.98 or more in every window after the sync, the Doppler within
0.3 Hz and the mean C/N0 within 1 dB of its pilot channels, the code start within 12 ns of its data
channels (its own code loop is still settling at the end of the file); the test's
bounds are 0.9, 1 Hz, 2 dB and 15 ns.

### Galileo E1

Galileo E1 is tracked on its pilot E1-C through the same stages. Its description
(`signalspec.py`) carries what differs.

**A 4 ms code on 1 ms epochs.** The code is 4092 chips at 1.023 Mchip/s, one memory code
per satellite and component, read from the two tables published with the Galileo Open
Service ICD (`src/pyrx/data/C7_E1B.txt` and `C8_E1C.txt`, unmodified). The codes are the
European Union's: the ICD's Annex H authorizes their use and storage royalty-free to each
user, on terms no license can pass on, and the ICD asks that its terms of use and that
annex go with any part of it. They are beside the tables as published
(`OS-SIS-ICD-2.2-terms-and-annex-H.pdf`), and the three files are not under this
repository's license (`src/pyrx/data/NOTICE.txt`). The tracker keeps its 1 ms epochs and each one
correlates a quarter of the code, so every loop schedule above stays as it is. The
secondary code has 25 chips, one per code period (100 ms), and is held over the four
epochs of a period. Held like that it matches itself at 0.76 one epoch off, against the
sync rule's 0.8, so the match is looked for only at an epoch that ends a code period. The
data component E1-B lies on the pilot's axis with the opposite sign and sends one symbol
per code period.

**A subcarrier.** Every chip is multiplied by one cycle of a square wave (BOC(1,1); the
signal's small BOC(6,1) part is left out of the replica), which the replica carries as two
entries per chip. The correlation then falls three times as steeply as a plain code's: it
crosses zero a third of a chip from the peak and has a second peak, half the height and of
the opposite sign, half a chip either side.

With early and late a quarter chip from prompt, the code discriminator is nine times as
steep as a plain code's and is divided by nine, so that the loop closes a small offset at
the rate of its 0.25 Hz as on the other signals. That holds for a strong signal: early and
late read a quarter of the prompt here, 12 dB down, and the normalized difference loses its
slope in noise sooner than a plain code's. Measured on the synthetic signal, the share of a
0.05 chip offset left after 3 s: 0.16 at 50 dB-Hz, 0.24 at 45, 0.46 at 40, 0.62 at 37, 0.79
at 34, where E5a has 0.14 to 0.22 from 50 down to 37 dB-Hz. The discriminator is also
linear only to a twelfth of a chip, where the late tap reaches the zero of the correlation;
beyond that its reading shrinks as the offset grows. At 45 dB-Hz (no front-end filter): a
start 0.15 chip off is within 0.05 chip after 3.5 s, one 0.2 chip off after 5.6 s, one a
quarter chip off after 8 s, the secondary code found at the usual time and C/N0 reading up
to 7 dB low on the way; a start 0.30 to 0.45 chip off is drawn to the zero of the
correlation at a third of a chip and lost within seconds. An acquisition seed is given to
the sample (0.04 chip at 24 Msps), so the slow range is met after a fade, not at a start.
Nothing flags a channel while it creeps in from a quarter chip or less.

**The side peaks.** From half a chip or more off, the loops stay off the main peak for
good. They do not rest on the second peak itself: with early and late a quarter chip out,
their magnitudes balance 0.625 chip from the main peak, and the loop slides there.
Measured with the guard switched off: on the synthetic signal at 45 dB-Hz a start half a
chip off reads 6 dB low for ten seconds, then moves out to 0.624 chip, 183 m, within 40 s,
8 dB low; on the 2021 recording, of six starts half a chip off, three move out to 0.61
chip in 20 s and hold there 6.3 to 7.6 dB low, and three move inward and are lost after
23 s. The secondary code is found there as on the main peak, and a satellite 7 dB low
looks like a weaker satellite.

Two more taps therefore stand half a chip either side of prompt, and over each 1 s window
each is projected on the prompt: minus the sum of Re(tap times the conjugate of prompt)
over the sum of the prompt's power, with the noise taken out of both sums (the noise of two
taps half a chip apart is correlated by the code's own minus a half, and the noise tap
gives its power). The reading is then the correlation's own ratio at any signal strength:
0.5 on the main peak (0.39 to 0.70 in every window measured, down to windows of
22 dB-Hz), 2 at half a chip and 1.7 at 0.625 chip on the tap toward the main peak (1.55 or
more in every window measured, down to a 31 dB-Hz pilot whose prompt reads 24 there). A
window above 1 (`SIDE_PEAK`) ends the track (`TrackResult.side_peak`), and the track is
cut a window further back, because a window shows a move to a side peak once two thirds of
it are there and the move may lie in the window before. A window whose own C/N0 is below
the loss rule's 25 dB-Hz is not judged, since the difference of two noise sums says
nothing; the loss rule counts it. At a quarter chip off the main peak the tap away from it
reads 1 too, so the guard also ends a channel that stands a quarter chip or more off
through a window. In the walk's trial such a seed is turned away, and the next probe
searches again. The guard reads whole windows: a channel that goes to a side peak writes
nothing of it, but it costs the satellite two seconds and the time to the next probe.

Checked on the synthetic signal (`test_tracking_e1.py`: the two components and their
correlation, the search, the epochs, the code loop, a start half a chip off either way at
38 and 45 dB-Hz ended in its first window, a jump to a side peak in mid-track at 45 and at
33 dB-Hz, the main peak never taken for one from 26 to 50 dB-Hz, an outage) and on
recordings against Pocket
SDR's C tracker (`test_tracking_e1_reference.py`). On the L1 recording of 2021-12-26 the
five Galileo satellites from 41 to 47 dB-Hz find the secondary code at 1.5 to 1.6 s, with
both polarities occurring, and hold for the file. As measured: the indicator 0.97 or more
in every window after the sync; the Doppler within 0.7 Hz and the mean C/N0 within 0.5 dB
of the reference's E1-B channels, the code start within 23 ns of them modulo the 4 ms
(its code loop is still settling; half a chip is 489 ns); E1-B a quarter decibel above
E1-C (+0.23 to +0.27 dB on every satellite of both recordings) with 0.96 or more of
its power on the pilot's axis; the four epochs of a symbol adding up to 3.95 of 4 or
more; and the uncoded ten-symbol sync pattern of the I/NAV page parts at the same symbol
of every 250, with the sign it is sent with whichever polarity the pilot synced with,
which checks the data component's sign and the symbol edges. The test's bounds are 0.9,
1 Hz, 2 dB, 30 ns and 0.5 dB. The `slow` test walks 60 s of the parked car-roof capture:
seven satellites, one segment each, nothing turned away; the same quantities against the
reference's E1-B channels, and every satellite's code start the reference's within
1.1 ns in the mean once the 175 ns that its log times stand from ours on that capture
are removed (its L1 C/A channels stand 176 ns off the same way).

In motion, on the two five-minute car-roof recordings (the one started while moving,
then the drive), the walk's E1 pilots are phase-locked for 1859 s and 2124 s summed over
the satellites, where the reference's E1-C channels are locked 1525 s and 1707 s. The guard ended no track and turned no seed away
in either. Independently of the guard, each E1 channel's code timing was compared with
the same satellite's E5a pilot, tracked in the other band's recording on the same sample
clock: over the 3934 satellite seconds in which both are phase-locked, E1 minus E5a stays
within 22 ns of one value (-104 ns on both recordings, the front end's delay between the
bands), so no channel stood on a side peak or a tenth of a chip off. A weak satellite is
found late or not at all: E19, at 32 and 35 dB-Hz, has 54 s and 71 s where the reference
has 69 s and 217 s, since the search needs 36 dB-Hz in 20 ms.
