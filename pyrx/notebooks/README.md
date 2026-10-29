# From samples to a centimetre fix

These three notebooks follow the Purple GNSS receiver package, `pyrx`, from recorded radio
samples to a position good to about a centimetre. Each step of the receiver gets an
explanation of what it is for, the idea behind it, the equations it uses and a figure of
it working on real data. The notebooks are committed with their outputs, so GitHub shows
every figure and printed number without running anything.

The reader they are written for knows Python and NumPy arrays but has never built a GNSS
receiver. Some familiarity with signals helps in notebook 01; notebook 02 uses least
squares, and notebook 03 introduces a Kalman filter and an integer search. Each notebook
introduces its GNSS terms as they appear. The [GNSS fundamentals](../../docs/fundamentals/01-what-gnss-is.md)
series and its [glossary](../../docs/fundamentals/glossary.md) give more background.

Read them in order: each notebook starts from the measurements the one before it explains.

| Order | Notebook                                             | What it shows                                                                                                                              |
| ----- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | [`01-signal-chain.ipynb`](01-signal-chain.ipynb)     | acquisition, the tracking loops, the navigation message, the observables and a RINEX file, then a comparison with Pocket SDR               |
| 2     | [`02-standalone-pvt.ipynb`](02-standalone-pvt.ipynb) | reading RINEX, broadcast orbits against the precise ephemeris, the atmosphere models, the least-squares fix, then a comparison with RTKLIB |
| 3     | [`03-rtk.ipynb`](03-rtk.ipynb)                       | single and double differences, the float filter, LAMBDA and the ratio test, more signals, then a comparison with RTKLIB                    |

Notebook 01 starts from the raw samples of a 30 s GPS L1 C/A recording. Acquisition
searches for each satellite's repeating spreading code in the noise and gives a first
estimate of its code phase and Doppler shift. The tracking loops then lock onto the code
and the carrier and follow them. The navigation message decoded from the tracked signal
carries the transmit time and the satellite's broadcast orbit. From these the receiver
forms its measurements (pseudorange, carrier phase, Doppler and signal strength) and
writes them to RINEX, the standard exchange format that positioning programs read.

Notebook 02 turns pseudoranges and broadcast ephemerides into satellite positions and
clock corrections, and checks the broadcast orbits against the IGS final orbit. It
corrects each pseudorange for the ionosphere and the troposphere, then solves for the
receiver's position and clock bias by weighted least squares, one epoch at a time. The
real-data example is three hours, 08:00 to 11:00 GPS time, of the IGS station USUD in
Usuda, Japan.

Notebook 03 adds carrier phase and a second receiver. A base and a rover observe the same
satellites, and differencing their measurements cancels the satellite and
receiver clock errors and, over a short baseline, most of the atmosphere. A Kalman
filter estimates the rover position together with real-valued carrier ambiguities (the
float solution). LAMBDA then searches for the integer ambiguities, and the ratio test
decides whether to accept them (the fixed solution). The real-data example is one hour,
00:00 to 01:00 GPS time, of the IGS stations MATE and MAT1 in Matera, Italy, about 11 m
apart. It starts with GPS L1 and L2, then adds GPS L5 and Galileo E1 and E5a.

## Why the data changes

The three notebooks use independent inputs, each chosen for what its step needs to show.
Notebook 01 needs raw samples, so it uses the recording, made in Japan on 2021-12-26. The
RINEX files it writes stay in its own temporary directory. Notebook 02 needs hours of
observations from a station with known coordinates to show how a standalone fix behaves
over time. Notebook 03 needs two receivers a short distance apart with surveyed markers,
so that the RTK error can be measured in centimetres. The station files are from the same
day as the recording. You can run any notebook on its own; none reads files written by
another.

Each notebook ends by comparing `pyrx` with an independent program on the same inputs:
Pocket SDR's tracker logs in 01, RTKLIB's `rnx2rtkp` in 02 and 03. Agreement there shows
that the two programs compute the same thing. How close the result is to the true
position is measured separately, against the station marker coordinates.

## Running them

You need Python 3.12 or later. From the repository root (the directory that contains
`pyrx/`), create and activate a virtual environment, install the package with the
notebook extras and start JupyterLab in this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e 'pyrx[dev,notebooks]'
cd pyrx/notebooks
python -m jupyterlab
```

The activation line is for a POSIX shell; other shells have their own. Open a notebook
and run its cells in order with the kernel of this environment. The first code cell
prints which data path the notebook is on, and why.

To re-execute a notebook in place from `pyrx/notebooks/`, which replaces its saved
outputs, run nbconvert from the same environment:

```sh
python -m nbconvert --to notebook --execute --inplace 01-signal-chain.ipynb
```

In the environment that produced the committed outputs, each notebook runs in under a
minute on either data path.

## Data paths

The committed outputs come from the recording and the IGS station files. To run on them,
set `PYRX_CAPTURES` to the directory that holds the files listed in the
[package README](../README.md), in the layout given there, before starting Jupyter or
nbconvert:

```sh
export PYRX_CAPTURES=/path/to/captures
```

If the variable is unset, or any of the files is missing, the notebooks print the reason
and run on synthetic signals and observations from `pyrx.synth`. Notebook 01 then
synthesises and tracks 22 s of four GPS L1 C/A signals. Notebook 02 uses an hour of
modelled pseudoranges, one epoch every 30 s, with 0.3 m of code noise and a drifting
receiver clock. Notebook 03 uses 40 base and rover epochs 30 s apart on GPS L1 and L2,
with a three-cycle slip in the rover's G03 L1 phase from the halfway epoch on. Position
errors on this path are measured against the position the observations were generated
at.

The receiver steps run and draw their figures on both paths, with different numbers. Some
figures need reference data that only the capture path has, and on the synthetic path
the notebook prints a note in their place:

| Notebook | Needs the captures                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| 01       | the comparison with Pocket SDR's tracker logs                                      |
| 02       | the broadcast orbits against the IGS final orbit; the comparison with RTKLIB       |
| 03       | the run with more signals (GPS L5, Galileo E1 and E5a); the comparison with RTKLIB |

The RTKLIB comparisons also need RTKLIB's `rnx2rtkp` on `PATH`; without it they print a
note and the rest of the notebook runs. The committed comparisons were made with RTKLIB
2.4.3 b34, from the `rtklib_2.4.3` branch of
[tomojitakasu/RTKLIB](https://github.com/tomojitakasu/RTKLIB/tree/rtklib_2.4.3) on GitHub;
build or install its command-line programs to enable them. The Python install above does
not include RTKLIB. Notebook 01 reads Pocket SDR logs that come with the captures and
does not run Pocket SDR itself.

The printed numbers in the committed outputs were produced on the project's own
environment; the millimetre-level comparisons will differ slightly on another.

## Notation

The notebooks share one set of symbols, listed here; each notebook also defines the ones
it uses next to its equations. Subscripts name a receiver: $r$ is the receiver, or the
rover when there are two, and $b$ is the base. Superscripts name a satellite: $s$ is any
satellite and $q$ the reference satellite of a double difference. So $\mathrm{dt}_r$ is
the receiver clock bias and $\mathrm{dt}^s$ the satellite's, and the pseudorange model of
notebook 02 reads

$$P_r^s = \rho_r^s + c(\mathrm{dt}_r - \mathrm{dt}^s) + I_r^s + T_r^s + e_P .$$

Where a signal index is left out, every term refers to the same signal. Carrier phase is
stored in cycles; multiplied by the wavelength it becomes a length in metres,
$\Phi = \lambda\varphi$. In the table, `_` marks a subscript and `^` a superscript, as in
the notebooks' equations. The last column names the `pyrx` field or function that holds
each quantity (modules under `pyrx/src/pyrx/`).

| Symbol                    | Quantity                                                                      | Unit        | Where it lives in `pyrx`                                                           |
| ------------------------- | ----------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------- |
| `P`                       | pseudorange                                                                   | m           | `Observation.pseudorange`, `Observations.pseudorange`                              |
| `φ`                       | carrier phase                                                                 | cycles      | `Observation.carrier_phase`, `Observations.carrier_phase`                          |
| `λ`                       | signal wavelength                                                             | m per cycle | `signals.wavelength(signal)`                                                       |
| `Φ = λφ`                  | carrier phase as a length                                                     | m           | `lam * obs.carrier_phase[row]` in `rtk.Rtk._residuals`                             |
| `ρ`                       | geometric range, including the Earth-rotation term                            | m           | first value returned by `models.geometric_range`                                   |
| `d_rot`                   | Earth-rotation (Sagnac) term, `(Ω_E / c)(x^s y_r − y^s x_r)`                  | m           | `sagnac` in `models.geometric_range`, already inside `ρ`                           |
| `x^s`, `x_r`              | satellite and receiver positions, ECEF                                        | m           | `orbits.SatState.position`; `pvt.Solution.position`                                |
| `dt_r`                    | receiver clock bias relative to GPS time                                      | s           | `pvt.Solution.clock_bias`; the solver's state holds `c * dt_r`                     |
| `dt^s`                    | satellite clock bias, including the relativistic term                         | s           | `orbits.SatState.clock`                                                            |
| `t_g`                     | broadcast group delay                                                         | s           | `ephemeris.Ephemeris.tgd`; the fix uses `P − c * t_g`                              |
| `I`                       | ionospheric delay of the code (the carrier phase advances by the same amount) | m           | `models.klobuchar` in the standalone fix; no ionosphere term in RTK                |
| `T`                       | tropospheric delay                                                            | m           | `models.saastamoinen` in the standalone fix; `zenith_hydrostatic * nmf_dry` in RTK |
| `N`                       | integer carrier ambiguity of one receiver and satellite                       | cycles      | not estimated directly; the synthetic inputs draw it with `synth.ambiguities`      |
| `Δx^s = x_r^s − x_b^s`    | single difference of any quantity `x`, rover minus base                       | unit of `x` | `rtk.Rtk._update_biases`                                                           |
| `∇Δx^{s,q} = Δx^s − Δx^q` | double difference, satellite minus reference satellite                        | unit of `x` | `rtk.Rtk._double_differences`                                                      |
| `ΔN̂`                      | float single-differenced ambiguity                                            | cycles      | `rtk.Rtk.x[3:]`, keyed by `Rtk.keys`                                               |
| `∇ΔN^{s,q}`               | double-differenced ambiguity searched by LAMBDA                               | cycles      | `rtk.Ambiguity`; its `value` is reference minus satellite, `−∇ΔN^{s,q}`            |
| `e_P`, `e_Φ`              | what the model leaves out: noise, multipath, unmodelled biases                | m           | not stored                                                                         |
| `f_D`                     | Doppler shift, positive when the satellite approaches                         | Hz          | `Observation.doppler`, `AcqResult.doppler`                                         |
| `C/N0`                    | carrier-to-noise density ratio                                                | dB-Hz       | `Observation.cn0`, `AcqResult.cn0`                                                 |
| `τ_c`                     | code phase, as an offset into the code period                                 | s           | `AcqResult.code_offset`, `TrackResult.code_offset`                                 |
| `f_s`                     | sample rate                                                                   | Hz          | `samples.SampleFile.fs`                                                            |
| `c`                       | speed of light                                                                | m/s         | `gpstime.C_LIGHT`                                                                  |

A hat marks an estimate. The filter estimates real-valued single-differenced
ambiguities; the integer search works on double differences formed from them. The raw
double differences plotted in notebook 03 and the filter's measurement update both take
satellite minus reference, as the table does. The search results that `pyrx` reports,
`Ambiguity.value`, run the other way, reference minus satellite, so they are
$-\nabla\Delta N^{s,q}$. The search also picks its own reference satellite (the first
eligible one of each system and signal), which need not be the one the measurement
update uses.

The two code-phase offsets have different origins. Acquisition measures from the start
of its sample buffer; tracking records the start of each code period, reduced modulo one
period, from the start of the recording. Neither is a pseudorange.

## Read more

The [GNSS fundamentals](../../docs/fundamentals/01-what-gnss-is.md) explain the receiver
concepts at more length, and the [glossary](../../docs/fundamentals/glossary.md) defines
the terms and lists further reading. The [package README](../README.md) has the
command-line tools, the tests and the provenance of every data file.
