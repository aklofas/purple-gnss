"""Synthetic inputs shared by several test modules: a navigation frame, an ephemeris, a
constellation visible from USUD, and the observations the package's own models predict."""

import difflib
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyrx.ephemeris import Ionosphere
from pyrx.gpstime import C_LIGHT, gps_datetime
from pyrx.lnav import assemble_ephemeris, decode_subframe, encode_subframe
from pyrx.models import (
    azimuth_elevation,
    geodetic,
    geometric_range,
    klobuchar,
    nmf_dry,
    saastamoinen,
    zenith_hydrostatic,
)
from pyrx.observations import Epoch, Observation, Observations
from pyrx.orbits import OMEGA_E, sat_id, satellite_state
from pyrx.pvt import IONO_DEFAULT, LAMBDA_L1
from pyrx.signals import wavelength

SF1 = {
    "wn": 142,
    "code_l2": 1,
    "ura": 0,
    "health": 0,
    "iodc": 917,
    "l2p": 0,
    "tgd": -4.19095158577e-09,
    "toc": 28800,
    "af2": 2 * 2**-55,
    "af1": -1.02318153949e-12,
    "af0": -1.97644811124e-04,
}
SF2 = {
    "iode": 149,
    "crs": -37.5625,
    "delta_n": 4.6734e-09,
    "m0": -1.2345678,
    "cuc": -2.0e-06,
    "e": 0.001602733,
    "cus": 5.8e-06,
    "sqrt_a": 5153.668436,
    "toe": 28800,
    "fit": 0,
    "aodo": 2700.0,  # age of data offset is broadcast in units of 900 s
}
SF3 = {
    "cic": 1.2e-07,
    "omega0": 2.1,
    "cis": -9.3e-08,
    "i0": 0.96,
    "crc": 250.0,
    "omega": -0.5,
    "omega_dot": -8.1e-09,
    "iode": 149,
    "idot": 1.5e-10,
}


def frame(tow0=5044):
    """Subframes 1 to 3 of SF1..SF3 as one encoded bit stream starting at `tow0`."""
    out, d29, d30 = [], 0, 0
    for i, f in ((1, SF1), (2, SF2), (3, SF3)):
        sf = encode_subframe(i, (tow0 + i - 1) % 100800, f, d29, d30)
        d29, d30 = int(sf[298]), int(sf[299])
        out.append(sf)
    return np.concatenate(out)


def ephemeris(**changes):
    """The ephemeris of SF1..SF3 decoded for PRN 4, with fields overridden."""
    sfs = [
        decode_subframe(encode_subframe(i, 5044 + i, f), 0, 0, prn=4)
        for i, f in ((1, SF1), (2, SF2), (3, SF3))
    ]
    return replace(assemble_ephemeris(sfs), **changes)


USUD = np.array([-3855263.0361, 3427432.5571, 3741020.2929])
TOW = 28800.0
IONO = Ionosphere(
    (1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07),
    (1.1469e05, -1.6384e05, -1.9661e05, 9.1750e05),
)


def constellation():
    # six satellites on the test orbit, spread so that all are above 15 deg at USUD at toe
    out = []
    # offsets found by search: elevations 76, 24, 28, 27, 70 and 46 deg, GDOP 3.5
    offsets = [
        (-2.31, 0.12),
        (1.83, -3.11),
        (2.54, 2.97),
        (-1.37, -1.37),
        (2.26, 1.88),
        (-1.48, -0.89),
    ]
    for k, (dm, do) in enumerate(offsets):
        out.append(ephemeris(prn=k + 1, m0=-1.2345678 + dm, omega0=2.1 + do, toe=TOW, toc=TOW))
    return out


def truth_epoch(
    ephs, xyz, dtr, vel=None, ddtr=None, week=2190, tow=TOW, iono=IONO_DEFAULT, tropo=True
):
    """Pseudoranges (and Dopplers) of the package's own models; the ionosphere is the one
    the solver assumes when it gets none, so a test may solve without passing it."""
    lat, lon, h = geodetic(xyz)
    obs = []
    for eph in ephs:
        t_tx = tow - dtr - 0.07  # about 70 ms of light time; refined below
        for _ in range(4):
            s = satellite_state(eph, t_tx)
            t_tx = tow - dtr - geometric_range(xyz, s.position)[0] / C_LIGHT
        s = satellite_state(eph, t_tx)
        rho, e = geometric_range(xyz, s.position)
        az, el = azimuth_elevation(xyz, s.position)
        corr = (klobuchar(iono, lat, lon, az, el, tow) if iono else 0.0) + (
            saastamoinen(lat, h, el) if tropo else 0.0
        )
        p = rho + C_LIGHT * (dtr - s.clock) + corr + C_LIGHT * eph.tgd
        d = float("nan")
        if vel is not None:
            rate = e @ (s.velocity - vel) + OMEGA_E / C_LIGHT * (
                s.velocity[1] * xyz[0]
                + s.position[1] * vel[0]
                - s.velocity[0] * xyz[1]
                - s.position[0] * vel[1]
            )
            d = -(rate + C_LIGHT * (ddtr - s.clock_drift)) / LAMBDA_L1
        obs.append(Observation(eph.prn, p, 0.0, d, 45.0))  # carrier phase unused here
    return Epoch(week, tow, obs)


ROVER = USUD + np.array([5.0, -3.0, 2.0])
SIGNALS = ("G1C", "G2W")
ALL_SIGNALS = ("G1C", "G2W", "G5Q", "E1C", "E5Q")


def as_galileo(ephs):
    return [replace(e, prn=e.prn + 10, system="E") for e in ephs]


def ambiguities(rng, ephs, signals):
    return {
        (sat_id(e), s): int(rng.integers(-1000, 1000))
        for e in ephs
        for s in signals
        if s[0] == e.system
    }


_RNG = np.random.default_rng(0)
AMB_R = ambiguities(_RNG, constellation() + as_galileo(constellation()), ALL_SIGNALS)
AMB_B = ambiguities(_RNG, constellation() + as_galileo(constellation()), ALL_SIGNALS)


def synthetic_observations(ephs, xyz, dtr, tow, signals, amb, rng, phase_noise, code_noise, week):
    lat, _, h = geodetic(xyz)
    doy = gps_datetime(week, tow).timetuple().tm_yday
    ephs = sorted(ephs, key=sat_id)
    n, m = len(ephs), len(signals)
    P, L = np.full((n, m), np.nan), np.full((n, m), np.nan)
    for i, eph in enumerate(ephs):
        t_tx = tow - dtr - 0.07
        for _ in range(4):
            s = satellite_state(eph, t_tx)
            t_tx = tow - dtr - geometric_range(xyz, s.position)[0] / C_LIGHT
        s = satellite_state(eph, t_tx)
        rho, _ = geometric_range(xyz, s.position)
        _, el = azimuth_elevation(xyz, s.position)
        model = (
            rho + C_LIGHT * (dtr - s.clock) + zenith_hydrostatic(lat, h) * nmf_dry(lat, h, el, doy)
        )
        for j, sig in enumerate(signals):
            if sig[0] != eph.system:
                continue
            lam = wavelength(sig)
            L[i, j] = model / lam + amb[(sat_id(eph), sig)] + rng.normal(0.0, phase_noise) / lam
            P[i, j] = model + rng.normal(0.0, code_noise)
    return Observations(
        week,
        tow,
        tuple(sat_id(e) for e in ephs),
        tuple(signals),
        P,
        L,
        np.full((n, m), np.nan),
        np.where(np.isfinite(P), 45.0, np.nan),
        np.zeros((n, m), dtype=int),
    )


def synthetic_pair(
    ephs,
    base_xyz,
    rover_xyz,
    tow,
    signals,
    amb_rover,
    amb_base,
    rng,
    phase_noise=0.001,
    code_noise=0.3,
    week=2190,
):
    """Observations of a rover and a base at their antenna reference points from the package's models.

    The default phase noise is a little above the jitter of a 15 Hz PLL at 45 dB-Hz
    (0.7 mm); with it a fixed single-epoch position lands within 1 cm of the truth, while a
    wrong integer moves it by centimetres to a wavelength.
    """
    args = (tow, signals, rng, phase_noise, code_noise, week)
    rover = synthetic_observations(ephs, rover_xyz, 1e-4, args[0], args[1], amb_rover, *args[2:])
    base = synthetic_observations(ephs, base_xyz, 2e-4, args[0], args[1], amb_base, *args[2:])
    return rover, base


def write_iq_int8(x, path):
    """Write complex samples as Pocket SDR interleaved int8 I/Q (Q negated)."""
    raw = np.empty(2 * len(x), dtype=np.int8)
    raw[0::2] = x.real.clip(-127, 127)
    raw[1::2] = (-x.imag).clip(-127, 127)
    raw.tofile(path)


GOLDEN_DIR = Path(__file__).parent / "golden"


def check_golden(name: str, text: str) -> None:
    """Compare `text` with the golden file `tests/golden/<name>` byte for byte.

    Set `PYRX_UPDATE_GOLDENS` to rewrite the file after a deliberate output change.
    """
    path = GOLDEN_DIR / name
    if os.environ.get("PYRX_UPDATE_GOLDENS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    expected = path.read_text()
    if text != expected:
        diff = difflib.unified_diff(
            expected.splitlines(True), text.splitlines(True), f"golden/{name}", "actual"
        )
        raise AssertionError(f"output differs from {name}:\n" + "".join(diff))


def without_date(text: str) -> str:
    """The text with the date of its `PGM / RUN BY / DATE` line replaced."""
    return re.sub(r"\d{8} \d{6} UTC", "<date>", text)


def without_elapsed(text: str) -> str:
    """The text with the value of its `TIME = x.xxx s` lines replaced."""
    return re.sub(r"^TIME = \d+\.\d{3} s$", "TIME = <t> s", text, flags=re.MULTILINE)
