"""RINEX 3.05 observation and GPS navigation file writers.

Layouts follow the RINEX 3.05 tables: fixed-width header lines with the label in
columns 61-80, observation records with one 14.3 field per observable followed by
the two indicator columns, navigation records in seven lines of four 19.12 values.
Times are GPS time; the GPS epoch is 1980-01-06.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pyrx.lnav import Ephemeris, Ionosphere
from pyrx.observables import Epoch, full_week

GPS_EPOCH = datetime(1980, 1, 6)
OBS_TYPES = ("C1C", "L1C", "D1C", "S1C")


def gps_datetime(week: int, tow: float) -> datetime:
    """Naive datetime in the GPS time scale for a week and seconds of week."""
    return GPS_EPOCH + timedelta(weeks=week, seconds=tow)


def ura_metres(index: int) -> float:
    """SV accuracy in metres from the 4-bit URA index (IS-GPS-200 20.3.3.3.1.3)."""
    if index <= 6:
        return 2.0 ** (1 + index / 2)
    if index < 15:
        return 2.0 ** (index - 2)
    return 6144.0


def _line(content: str, label: str) -> str:
    return f"{content:<60s}{label}\n"


def _pgm_line(program: str) -> str:
    date = datetime.now(UTC).strftime("%Y%m%d %H%M%S UTC")
    return _line(f"{program:<20s}{'':<20s}{date:<20s}", "PGM / RUN BY / DATE")


def _e19(value: float) -> str:
    return f"{value:19.12E}"


def write_rinex_obs(path, epochs: list[Epoch], marker: str = "PYRX", program: str = "pyrx") -> None:
    """Write GPS L1 C/A observations (C1C L1C D1C S1C) as a RINEX 3.05 observation file."""
    epochs = [e for e in epochs if e.obs]
    if any(e.week is None for e in epochs):
        raise ValueError("week unknown: no ephemeris decoded and none given")
    out = [
        _line(
            f"{3.05:9.2f}{'':11s}{'OBSERVATION DATA':<20s}{'G: GPS':<20s}", "RINEX VERSION / TYPE"
        ),
        _pgm_line(program),
        _line(marker, "MARKER NAME"),
        _line("", "OBSERVER / AGENCY"),
        _line(f"{'':<20s}{program:<20s}", "REC # / TYPE / VERS"),
        _line("", "ANT # / TYPE"),
        _line(f"{0.0:14.4f}{0.0:14.4f}{0.0:14.4f}", "APPROX POSITION XYZ"),
        _line(f"{0.0:14.4f}{0.0:14.4f}{0.0:14.4f}", "ANTENNA: DELTA H/E/N"),
        _line(f"G  {len(OBS_TYPES):3d} " + " ".join(OBS_TYPES), "SYS / # / OBS TYPES"),
    ]
    if epochs:
        t = gps_datetime(epochs[0].week, epochs[0].tow)
        sec = t.second + t.microsecond * 1e-6
        stamp = f"{t.year:6d}{t.month:6d}{t.day:6d}{t.hour:6d}{t.minute:6d}{sec:13.7f}     GPS"
        out.append(_line(stamp, "TIME OF FIRST OBS"))
    out.append(_line("G L1C", "SYS / PHASE SHIFT"))
    out.append(_line("", "END OF HEADER"))
    for e in epochs:
        t = gps_datetime(e.week, e.tow)
        sec = t.second + t.microsecond * 1e-6
        out.append(
            f"> {t.year:4d} {t.month:02d} {t.day:02d} {t.hour:02d} {t.minute:02d} "
            f"{sec:10.7f}  0 {len(e.obs):2d}\n"
        )
        for o in e.obs:
            values = (o.pseudorange, o.carrier_phase, o.doppler, o.cn0)
            out.append(f"G{o.prn:02d}" + "".join(f"{v:14.3f}  " for v in values) + "\n")
    Path(path).write_text("".join(out))


def write_rinex_nav(
    path,
    ephemerides: list[Ephemeris],
    transmit_tow: dict[int, float],
    iono: Ionosphere | None = None,
    week: int | None = None,
    program: str = "pyrx",
) -> None:
    """Write decoded GPS ephemerides (and Klobuchar parameters) as a RINEX 3.05 navigation file.

    `transmit_tow[prn]` is the seconds of week of the subframe 1 each ephemeris came
    from. `week` overrides the week derived from each ephemeris' 10-bit week number.
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
        w = week if week is not None else full_week(eph.wn)
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
            (transmit_tow[eph.prn], 4.0 if eph.fit == 0 else 6.0, 0.0, 0.0),
        ]
        for row in rows:
            out.append("    " + "".join(_e19(float(v)) for v in row) + "\n")
    Path(path).write_text("".join(out))
