"""Broadcast ephemeris from a RINEX 3 GPS navigation file, for the decode tests."""

import warnings
from pathlib import Path

import numpy as np

GPS_EPOCH = np.datetime64("1980-01-06T00:00:00")
WEEK_S = 604800

# georinex variable -> pyrx Ephemeris field
_NAMES = {
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
    "GPSWeek": "week",
    "SVacc": "ura",
    "health": "health",
    "TGD": "tgd",
    "IODC": "iodc",
}


def load_gps_nav(path) -> dict[tuple[int, int, int], dict]:
    """Records keyed by (prn, iode, iodc); angles in radians, `toc` in seconds of week."""
    import georinex as gr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ds = gr.load(path, use="G")
    out = {}
    for sv in ds.sv.values:
        recs = ds.sel(sv=sv).dropna(dim="time", how="all")
        for t in recs.time.values:
            r = recs.sel(time=t)
            rec = {ours: float(r[theirs].values) for theirs, ours in _NAMES.items()}
            rec["toc"] = float((t - GPS_EPOCH) / np.timedelta64(1, "s")) % WEEK_S
            for k in ("iode", "iodc", "week", "health"):
                rec[k] = int(rec[k])
            out[(int(sv[1:]), rec["iode"], rec["iodc"])] = rec
    return out


def load_gps_iono(path) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """(alpha, beta) from the header's GPSA/GPSB lines, or None when the file has none."""
    coeffs = {}
    with Path(path).open() as f:
        for line in f:
            if "END OF HEADER" in line:
                break
            if line[60:].strip() == "IONOSPHERIC CORR" and line[:4] in ("GPSA", "GPSB"):
                values = line[5:53].replace("D", "E").split()
                coeffs[line[:4]] = tuple(float(v) for v in values)
    if "GPSA" in coeffs and "GPSB" in coeffs:
        return coeffs["GPSA"], coeffs["GPSB"]
    return None
