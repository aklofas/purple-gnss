"""RINEX 3 observation and GPS navigation files: writers and readers.

Written layouts follow the RINEX 3.05 tables: fixed-width header lines with the
label in columns 61-80, observation records with one 14.3 field per observable
followed by the two indicator columns, navigation records in seven lines of four
19.12 values. Times are GPS time; the GPS epoch is 1980-01-06. The readers take
our own files and public RINEX 3 files (plain, gzip or Hatanaka compressed):
navigation files and the single-signal observation reader go through georinex;
the multi-signal observation reader parses the text itself, because georinex
drops the loss-of-lock indicators of some columns.
"""

import gzip
import io
import re
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isnan, nan
from pathlib import Path

import numpy as np

from pyrx.ephemeris import Ephemeris, Ionosphere, ura_index, ura_metres
from pyrx.gpstime import GPS_EPOCH, WEEK_S, full_week, gps_datetime, week_tow
from pyrx.observations import Epoch, Observation, Observations, from_epoch

OBS_TYPES = ("C1C", "L1C", "D1C", "S1C")
# georinex navigation variable -> Ephemeris field
_NAV_NAMES = {
    "SVclockBias": "af0",
    "SVclockDrift": "af1",
    "SVclockDriftRate": "af2",
    "IODE": "iode",
    "Crs": "crs",
    "DeltaN": "delta_n",
    "M0": "m0",
    "Cuc": "cuc",
    "Eccentricity": "e",
    "Cus": "cus",
    "sqrtA": "sqrt_a",
    "Toe": "toe",
    "Cic": "cic",
    "Omega0": "omega0",
    "Cis": "cis",
    "Io": "i0",
    "Crc": "crc",
    "omega": "omega",
    "OmegaDot": "omega_dot",
    "IDOT": "idot",
    "CodesL2": "code_l2",
    "L2Pflag": "l2p",
    "health": "health",
    "TGD": "tgd",
    "IODC": "iodc",
}
_GAL_NAMES = {k: v for k, v in _NAV_NAMES.items() if v in (
    "af0", "af1", "af2", "crs", "delta_n", "m0", "cuc", "e", "cus", "sqrt_a", "toe",
    "cic", "omega0", "cis", "i0", "crc", "omega", "omega_dot", "idot",
)}  # fmt: skip


def _line(content: str, label: str) -> str:
    if len(content) > 60:  # the label starts in column 61
        raise ValueError(f"{label} record is longer than 60 characters: {content!r}")
    return f"{content:<60s}{label}\n"


def _field(value: str, width: int, what: str) -> str:
    """`value`, refused when it would run past the `width` characters of its header field."""
    if len(value) > width:
        raise ValueError(f"{what} {value!r} is longer than its {width} characters")
    return value


def _pgm_line(program: str) -> str:
    date = datetime.now(UTC).strftime("%Y%m%d %H%M%S UTC")
    program = _field(program, 20, "program name")
    return _line(f"{program:<20s}{'':<20s}{date:<20s}", "PGM / RUN BY / DATE")


def _e19(value: float) -> str:
    return f"{value:19.12E}"


# RINEX 3.05 Table A23: the phase codes whose phase is shifted against their band's reference
# signal (by a fraction of a cycle, or "must be aligned" to it). The writers take aligned
# phases, so these codes get a 0.00000 correction in SYS / PHASE SHIFT and the reference and
# unshifted codes a blank one, as the standard's example header has it.
PHASE_SHIFTED = {
    "G": {"1S", "1L", "1X", "1P", "1W", "1N", "2C", "2S", "2L", "2X", "5Q", "5X"},
    "E": {"1C", "1X", "5Q", "5X", "7Q", "7X", "8Q", "8X", "6C", "6X"},
}


def _phase_shift_line(signal: str) -> str:
    system, code = signal[0], signal[1:]
    value = f" {0.0:8.5f}" if code in PHASE_SHIFTED.get(system, ()) else ""
    return _line(f"{system} L{code}{value}", "SYS / PHASE SHIFT")


def write_rinex_obs(
    path,
    epochs: list[Epoch],
    marker: str = "PYRX",
    program: str = "pyrx",
    marker_type: str | None = None,
) -> None:
    """Write GPS L1 C/A observations (C1C L1C D1C S1C) as a RINEX 3.05 observation file.

    `marker_type` is written as the MARKER TYPE record when given (RINEX 3.05 requires
    it for markers other than geodetic and non-geodetic ones, e.g. GROUND_CRAFT for a
    receiver on a vehicle).
    """
    epochs = [e for e in epochs if e.obs]
    if any(e.week is None for e in epochs):
        raise ValueError("week unknown: no ephemeris decoded and none given")
    # a record keeps the blanks its last field ends in, and an empty file its G1C header
    rows = [from_epoch(e) for e in epochs]
    _write_observations(path, rows, ("G1C",), marker, program, marker_type, strip=False)


def _write_observations(path, epochs, signals, marker, program, marker_type, strip) -> None:
    """The observation file of both writers: `epochs` as `Observations` with rows, `signals`
    for the header and the columns, `strip` to end each record at its last field."""
    systems = sorted({s[0] for s in signals})
    if not set(systems) <= {"G", "E"}:
        raise ValueError(f"systems {systems}: only GPS (G) and Galileo (E) are written")
    sat_sys = "M: MIXED" if len(systems) != 1 else {"G": "G: GPS", "E": "E: Galileo"}[systems[0]]
    out = [
        _line(
            f"{3.05:9.2f}{'':11s}{'OBSERVATION DATA':<20s}{sat_sys:<20s}", "RINEX VERSION / TYPE"
        ),
        _pgm_line(program),
        _line(marker, "MARKER NAME"),
        *([_line(_field(marker_type, 20, "MARKER TYPE"), "MARKER TYPE")] if marker_type else []),
        _line("", "OBSERVER / AGENCY"),
        _line(f"{'':<20s}{program:<20s}", "REC # / TYPE / VERS"),
        _line("", "ANT # / TYPE"),
        _line(f"{0.0:14.4f}{0.0:14.4f}{0.0:14.4f}", "APPROX POSITION XYZ"),
        _line(f"{0.0:14.4f}{0.0:14.4f}{0.0:14.4f}", "ANTENNA: DELTA H/E/N"),
    ]
    for system in systems:
        codes = [k + sig[1:] for sig in signals if sig[0] == system for k in "CLDS"]
        for i in range(0, len(codes), 13):
            head = f"{system}  {len(codes):3d} " if i == 0 else "       "
            out.append(_line(head + " ".join(codes[i : i + 13]), "SYS / # / OBS TYPES"))
    if epochs:
        t = gps_datetime(epochs[0].week, epochs[0].tow)
        sec = t.second + t.microsecond * 1e-6
        stamp = f"{t.year:6d}{t.month:6d}{t.day:6d}{t.hour:6d}{t.minute:6d}{sec:13.7f}     GPS"
        out.append(_line(stamp, "TIME OF FIRST OBS"))
    out.extend(_phase_shift_line(sig) for sig in signals)
    out.append(_line("", "END OF HEADER"))
    for e in epochs:
        t = gps_datetime(e.week, e.tow)
        sec = t.second + t.microsecond * 1e-6
        out.append(
            f"> {t.year:4d} {t.month:02d} {t.day:02d} {t.hour:02d} {t.minute:02d} "
            f"{sec:10.7f}  0{len(e.sats):3d}\n"
        )
        for i, sat in enumerate(e.sats):
            fields = []
            for j, sig in enumerate(signals):
                if sig[0] != sat[0]:
                    continue
                for a, lli in (
                    (e.pseudorange[i, j], 0),
                    (e.carrier_phase[i, j], e.lli[i, j]),
                    (e.doppler[i, j], 0),
                    (e.cn0[i, j], 0),
                ):
                    fields.append("" if isnan(a) else f"{a:14.3f}{lli or ' '} ")
            record = sat + "".join(f"{f:16s}" for f in fields)
            out.append((record.rstrip() if strip else record) + "\n")
    Path(path).write_text("".join(out))


def _load(path, use="G", **kwargs):
    import georinex

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # xarray, raised inside georinex
        warnings.simplefilter("ignore", RuntimeWarning)  # numpy, on a file with one epoch
        return georinex.load(path, use=use, **kwargs)


def _open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


@dataclass(frozen=True)
class RinexHeader:
    marker: str
    position: np.ndarray  # APPROX POSITION XYZ (m), zeros when absent
    delta_enu: np.ndarray  # ANTENNA: DELTA H/E/N as (E, N, U) (m), zeros when absent
    antenna: str
    receiver: str


def _three_numbers(text: str) -> np.ndarray:
    """The three F14.4 numbers of a header record; zeros for a blank record, which is how a
    file says it has none (the position of a moving receiver). Numbers that fill their
    fields touch, so a record that does not split into three is read by column."""
    if not text.strip():
        return np.zeros(3)
    fields = text.split()
    if len(fields) != 3:
        fields = [text[i : i + 14] for i in (0, 14, 28)]
    return np.array([float(v) for v in fields])


def read_rinex_header(path) -> RinexHeader:
    """Marker, approximate position, antenna offset and equipment of an observation file."""
    marker = antenna = receiver = ""
    position, delta = np.zeros(3), np.zeros(3)
    with _open_text(path) as f:
        for line in f:
            label = line[60:].strip()
            if label == "END OF HEADER":
                break
            if label == "MARKER NAME":
                marker = line[:60].strip()
            elif label in ("APPROX POSITION XYZ", "ANTENNA: DELTA H/E/N"):
                try:
                    values = _three_numbers(line[:42])
                except ValueError:
                    raise ValueError(f"{path}: malformed {label} header record") from None
                if label == "APPROX POSITION XYZ":
                    position = values
                else:
                    delta = values[[1, 2, 0]]  # H, E, N in the file; (E, N, U) here
            elif label == "ANT # / TYPE":
                antenna = line[20:40].rstrip()
            elif label == "REC # / TYPE / VERS":
                receiver = line[20:40].rstrip()
    return RinexHeader(marker, position, delta, antenna, receiver)


def _header_lines(path) -> list[str]:
    """The header of an observation file (plain, gzip or Hatanaka: its header is plain text)."""
    lines = []
    with _open_text(path) as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if line[60:].strip() == "END OF HEADER":
                break
    return lines


def signal_strength_unit(path) -> str:
    """The SIGNAL STRENGTH UNIT of an observation file's header (DBHZ for dB-Hz), empty when
    the file has no such record or a blank one."""
    for line in _header_lines(path):
        if line[60:].strip() == "SIGNAL STRENGTH UNIT":
            return line[:20].strip()
    return ""


def _check_observation_header(lines: list[str], signals: tuple[str, ...]) -> None:
    """Reject what the observation readers do not implement.

    They read RINEX 3 (and 4) observation files whose times are GPS or Galileo
    system time (GST is steered to GPST), store observations unscaled, and return
    GPS and Galileo signals only.
    """
    version = None
    for line in lines:
        label = line[60:].strip()
        if label == "RINEX VERSION / TYPE":
            version = float(line[:9])
            if line[20] != "O":
                raise ValueError("not a RINEX observation file")
        elif label == "TIME OF FIRST OBS":
            tsys = line[48:51].strip()
            if tsys and tsys not in ("GPS", "GAL"):
                raise ValueError(f"time system {tsys} not supported (GPS or GAL)")
        elif (
            label == "SYS / SCALE FACTOR" and line[0] in "GE" and line[2:6].strip() not in ("", "1")
        ):
            raise ValueError("a scale factor other than 1 (SYS / SCALE FACTOR) is not supported")
    if version is None or version < 3:
        raise ValueError("a RINEX 3 observation file is required")
    for sig in signals:
        if sig[0] not in "GE":
            raise ValueError(f"signal {sig}: only GPS (G) and Galileo (E) signals are read")


def _parse_epoch_line(line: str) -> tuple[datetime | None, int, int]:
    """Time (None for an event record, whose time may be blank), epoch flag and record count."""
    flag = int(line[31])
    count = int(line[32:35].strip() or 0)
    if flag > 1:
        return None, flag, count
    f = line[1:31].split()
    t = datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4])) + timedelta(
        seconds=float(f[5])
    )
    return t, flag, count


def read_rinex_observations(
    path, signals: tuple[str, ...] = ("G1C",), tlim: tuple[datetime, datetime] | None = None
) -> list[Observations]:
    """Every satellite by every requested signal, one `Observations` per epoch.

    Parses the RINEX 3 text itself (after Hatanaka or gzip decompression) so the
    loss-of-lock indicators come through. A blank or 0.0 field is a missing value.
    A satellite is kept at an epoch when it has a pseudorange or carrier phase on
    any requested signal; epochs without satellites are dropped. Epochs come out
    in time order; a repeated epoch or satellite keeps its first record and an
    epoch the file ends inside is dropped, as RTKLIB does. Files outside the
    implemented profile raise ValueError (see `_check_observation_header`); so
    does a change of the observation types inside the file.
    """
    import hatanaka

    text = hatanaka.decompress(Path(path))
    lines = (text.decode() if isinstance(text, bytes) else text).splitlines()
    types: dict[str, list[str]] = {}
    system = ""
    i = 0
    for i, line in enumerate(lines):
        label = line[60:].strip()
        if label == "SYS / # / OBS TYPES":
            if line[0] != " ":
                system = line[0]
                types[system] = []
            types[system] += line[7:60].split()
        elif label == "END OF HEADER":
            break
    _check_observation_header(lines[: i + 1], signals)
    # column of each requested signal's C, L, D, S code within its system's record
    columns = []
    for sig in signals:
        codes = types.get(sig[0], [])
        columns.append([codes.index(k + sig[1:]) if k + sig[1:] in codes else None for k in "CLDS"])
    n_sig = len(signals)
    by_time: dict[datetime, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    while i + 1 < len(lines):
        i += 1
        line = lines[i]
        if not line.startswith(">"):
            continue
        t, flag, count = _parse_epoch_line(line)
        records = lines[i + 1 : i + 1 + max(count, 0)]
        i += len(records)
        if flag > 1:  # event records: header lines or comments follow
            if any(r[60:].strip() == "SYS / # / OBS TYPES" for r in records):
                raise ValueError(f"line {i + 1}: observation types change inside the file")
            continue
        if sum(r[:1].isalpha() for r in records) < count or (records and _cut_short(records[-1])):
            if i < len(lines) - 1:
                raise ValueError(f"line {i + 1}: truncated observation record")
            break  # the file ends inside this epoch
        if tlim is not None and not (tlim[0] <= t <= tlim[1]):
            continue
        epoch = by_time.setdefault(t, {})
        for rec in records:
            if rec[:1] not in types or not rec[1:3].strip().isdigit():
                continue
            sat = f"{rec[0]}{int(rec[1:3]):02d}"
            if sat in epoch:
                continue
            row = np.full((4, n_sig), nan)
            lli = np.zeros(n_sig, dtype=int)
            for j, sig in enumerate(signals):
                if sig[0] != sat[0]:
                    continue
                for k, col in enumerate(columns[j]):
                    if col is None:
                        continue
                    field = rec[3 + 16 * col : 3 + 16 * col + 14].strip()
                    if field and float(field) != 0.0:  # 0.0 is a missing value, as a blank
                        row[k, j] = float(field)
                        if k == 1:
                            ind = rec[3 + 16 * col + 14 : 3 + 16 * col + 15].strip()
                            lli[j] = int(ind) if ind else 0
            if np.isfinite(row[0]).any() or np.isfinite(row[1]).any():
                epoch[sat] = (row, lli)
    epochs = []
    for t in sorted(by_time):
        sats = sorted(by_time[t])
        if not sats:
            continue
        rows = np.array([by_time[t][s][0] for s in sats])  # (n_sat, 4, n_sig)
        ns = (t - GPS_EPOCH).total_seconds()
        week = int(ns // WEEK_S)
        epochs.append(
            Observations(
                week,
                round(ns - week * WEEK_S, 6),
                tuple(sats),
                tuple(signals),
                rows[:, 0],
                rows[:, 1],
                rows[:, 2],
                rows[:, 3],
                np.array([by_time[t][s][1] for s in sats], dtype=int),
            )
        )
    return epochs


def _cut_short(rec: str) -> bool:
    """The record line ends inside an observation field: the file was truncated there."""
    n = len(rec) - 3
    return n % 16 < 14 and rec[3 + 16 * (n // 16) :].strip() != ""


def read_rinex_obs(path, tlim: tuple[datetime, datetime] | None = None) -> list[Epoch]:
    """GPS L1 C/A observations (C1C L1C D1C S1C) of a RINEX 3 observation file.

    A blank or 0.0 field is a missing value. A satellite without a pseudorange at
    an epoch is skipped and an epoch without satellites dropped; a missing carrier
    phase, Doppler or C/N0 is `nan`. `tlim` bounds the epochs in GPS time. Files
    outside the implemented profile raise ValueError (see `_check_observation_header`).
    """
    _check_observation_header(_header_lines(path), ())
    ds = _load(path, meas=list(OBS_TYPES), tlim=tlim)
    svs = [int(sv[1:]) for sv in ds.sv.values]
    order = np.argsort(svs)
    empty = np.full((len(ds.time), len(svs)), nan)
    c1, l1, d1, s1 = (
        np.where(ds[k].values == 0.0, nan, ds[k].values) if k in ds else empty for k in OBS_TYPES
    )  # 0.0 is a missing value, as a blank
    epochs = []
    for i, t in enumerate(ds.time.values):
        obs = [
            Observation(svs[j], float(c1[i, j]), float(l1[i, j]), float(d1[i, j]), float(s1[i, j]))
            for j in order
            if not isnan(c1[i, j])
        ]
        if obs:
            week, tow = week_tow(t)
            epochs.append(Epoch(week, tow, obs))
    return epochs


def _read_iono(path) -> Ionosphere | None:
    coeffs = {}
    with _open_text(path) as f:
        for line in f:
            if "END OF HEADER" in line:
                break
            if line[60:].strip() == "IONOSPHERIC CORR" and line[:4] in ("GPSA", "GPSB"):
                values = line[5:53].replace("D", "E").split()
                coeffs[line[:4]] = tuple(float(v) for v in values)
    if "GPSA" in coeffs and "GPSB" in coeffs:
        return Ionosphere(coeffs["GPSA"], coeffs["GPSB"])
    return None


def _gps_record(r, prn, toc) -> Ephemeris:
    f = {ours: float(r[theirs].values) for theirs, ours in _NAV_NAMES.items()}
    for k in ("iode", "iodc", "health", "code_l2", "l2p"):
        f[k] = int(f[k])
    return Ephemeris(
        prn=prn,
        wn=int(r["GPSWeek"].values) % 1024,
        ura=ura_index(float(r["SVacc"].values)),
        toc=toc,
        fit=1 if float(r["FitIntvl"].values) > 4 else 0,
        aodo=0,
        **f,
    )


def _galileo_record(r, prn, toc) -> Ephemeris | None:
    """A Galileo record, or None for an F/NAV record (RTKLIB keeps I/NAV, data-source bit 9)."""
    if not int(r["DataSrc"].values) & 512:
        return None
    f = {ours: float(r[theirs].values) for theirs, ours in _GAL_NAMES.items()}
    iodnav = int(r["IODnav"].values)
    return Ephemeris(
        prn=prn,
        wn=int(r["GALWeek"].values) % 1024,
        code_l2=0,
        ura=ura_index(float(r["SISA"].values)),
        health=int(r["health"].values),
        iodc=iodnav,
        iode=iodnav,
        l2p=0,
        toc=toc,
        fit=0,
        aodo=0,
        system="E",
        tgd=float(r["BGDe5a"].values),
        tgd2=float(r["BGDe5b"].values),
        **f,
    )


_GALILEO_RECORD = re.compile(r"E[ \d]\d \d{4} ")
_FIELD = 19  # columns of a field of a broadcast-orbit line, after four blanks


def _uniform_galileo(path) -> io.StringIO:
    """The navigation file with the spare fields of every Galileo record blanked.

    A Galileo record ends two of its lines in spare fields: the sixth, after IDOT, the data
    sources and the week, and the eighth, after the transmission time. A combined file
    writes them as numbers in some records and leaves them blank in others, and the parser
    behind `_load` takes one layout per satellite: with both kinds in a file it reads the
    fields after a spare one off in one of them (the accuracy as the health word, or a
    group delay as the accuracy), or refuses the file. Blank everywhere, every record reads
    the same.
    """
    with _open_text(path) as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if _GALILEO_RECORD.match(line) and i + 7 < len(lines):
            lines[i + 5] = lines[i + 5][: 4 + 3 * _FIELD].rstrip()
            lines[i + 7] = lines[i + 7][: 4 + _FIELD].rstrip()
    return io.StringIO("\n".join(lines))


def read_rinex_nav(path) -> tuple[list[Ephemeris], Ionosphere | None]:
    """GPS and Galileo ephemerides and Klobuchar coefficients (None without GPSA/GPSB).

    Galileo keeps the I/NAV records only, as RTKLIB's `rnx2rtkp` does.
    """
    ds = _load(_uniform_galileo(path), use={"G", "E"})
    ephs = []
    for sv in ds.sv.values:
        recs = ds.sel(sv=sv).dropna(dim="time", how="all")
        name = str(sv).split("_")[0]  # georinex suffixes a second record at the same time
        prn = int(name[1:])
        for t in recs.time.values:
            r = recs.sel(time=t)
            toc = week_tow(t)[1]
            eph = _galileo_record(r, prn, toc) if name[0] == "E" else _gps_record(r, prn, toc)
            if eph is not None:
                ephs.append(eph)
    return ephs, _read_iono(path)


def write_rinex_observations(
    path,
    epochs: list[Observations],
    marker: str = "PYRX",
    program: str = "pyrx",
    marker_type: str | None = None,
) -> None:
    """Write multi-signal observations as a RINEX 3.05 observation file.

    One `SYS / # / OBS TYPES` line per system with the C, L, D and S codes of its
    signals and a `SYS / PHASE SHIFT` line per phase code: the phases are taken as
    already aligned to their band's reference signal, so a code Table A23 shifts
    carries a 0.00000 correction and a reference code a blank one; a `nan` leaves
    its field blank; a nonzero loss-of-lock indicator is written after the phase.
    `marker_type` as in `write_rinex_obs`.
    """
    epochs = [e for e in epochs if e.sats]
    signals = epochs[0].signals if epochs else ()
    _write_observations(path, epochs, signals, marker, program, marker_type, strip=True)


def write_rinex_nav(
    path,
    ephemerides: list[Ephemeris],
    transmit_tow: dict[int, float],
    iono: Ionosphere | None = None,
    program: str = "pyrx",
) -> None:
    """Write decoded GPS ephemerides (and Klobuchar parameters) as a RINEX 3.05 navigation file.

    `transmit_tow[prn]` is the seconds of week of the subframe 1 each ephemeris came
    from. The record's epoch (toc) and week field are those of the data set's own
    week: an ephemeris broadcast in the last hours of a week has its toe and toc in
    the next one (the subframe's week number is the transmission week), and a reader
    that dates toe by the record's week would otherwise place it a week early. The
    transmission time is written in the record's week too, negative in that case.
    """
    out = [
        _line(
            f"{3.05:9.2f}{'':11s}{'N: GNSS NAV DATA':<20s}{'G: GPS':<20s}", "RINEX VERSION / TYPE"
        ),
        _pgm_line(program),
    ]
    if iono is not None:
        for name, coeffs in (("GPSA", iono.alpha), ("GPSB", iono.beta)):
            values = "".join(f"{c:12.4E}".replace("E", "D") for c in coeffs)
            out.append(_line(f"{name} {values}", "IONOSPHERIC CORR"))
    out.append(_line("", "END OF HEADER"))
    for eph in ephemerides:
        ttr = transmit_tow[eph.prn]
        w = full_week(eph.wn)
        if eph.toe < ttr - WEEK_S / 2:  # the transmission time is then negative in that week
            w, ttr = w + 1, ttr - WEEK_S
        elif eph.toe > ttr + WEEK_S / 2:
            w, ttr = w - 1, ttr + WEEK_S
        t = gps_datetime(w, eph.toc)
        out.append(
            f"G{eph.prn:02d} {t.year:4d} {t.month:02d} {t.day:02d} {t.hour:02d} {t.minute:02d} "
            f"{t.second:02d}{_e19(eph.af0)}{_e19(eph.af1)}{_e19(eph.af2)}\n"
        )
        rows = [
            (eph.iode, eph.crs, eph.delta_n, eph.m0),
            (eph.cuc, eph.e, eph.cus, eph.sqrt_a),
            (eph.toe, eph.cic, eph.omega0, eph.cis),
            (eph.i0, eph.crc, eph.omega, eph.omega_dot),
            (eph.idot, eph.code_l2, w, eph.l2p),
            (ura_metres(eph.ura), eph.health, eph.tgd, eph.iodc),
            (ttr, 4.0 if eph.fit == 0 else 6.0, 0.0, 0.0),
        ]
        for row in rows:
            out.append("    " + "".join(_e19(float(v)) for v in row) + "\n")
    Path(path).write_text("".join(out))
