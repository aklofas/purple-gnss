"""Shared pieces of the notebooks: which data to run on, the synthetic inputs when
the recordings are absent, figure styling, and the readers and runners of the
reference results (Pocket SDR logs, RTKLIB's rnx2rtkp).
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from math import log10
from pathlib import Path

import hatanaka
import matplotlib.pyplot as plt
import numpy as np
import pymap3d

from pyrx.models import Station, geodetic
from pyrx.plots import INK, MUTED
from pyrx.pvt import IONO_DEFAULT
from pyrx.synth import (
    TOW,
    USUD,
    ambiguities,
    constellation,
    frame,
    synth_l1ca,
    synthetic_pair,
    truth_epoch,
    write_iq_int8,
)

# marker coordinates (m, ECEF): USUD from its IGS site log, MATE and MAT1 from the EUREF
# combined SINEX of GPS week 2190
USUD_MARKER = USUD
MATE_MARKER = np.array([4641949.24236861, 1393045.74884337, 4133287.70969356])
MAT1_MARKER = np.array([4641951.13677723, 1393053.98279794, 4133281.10948056])

PVT_WINDOW = (datetime(2021, 12, 26, 8, 0), datetime(2021, 12, 26, 11, 0))
RTK_WINDOW = (datetime(2021, 12, 26, 0, 0), datetime(2021, 12, 26, 1, 0))
RTK_BASE_WINDOW = (
    datetime(2021, 12, 25, 23, 59),
    RTK_WINDOW[1],
)  # a base epoch for the first rover epoch
PVT_SPAN = ("2021/12/26 08:00:00", "2021/12/26 11:00:00")
RTK_SPAN = ("2021/12/26 00:00:00", "2021/12/26 01:00:00")


@dataclass(frozen=True)
class Captures:
    """The recordings and reference files of the capture path, all present."""

    root: Path
    recording: Path  # GPS L1, 24 MHz real samples at a 6 MHz IF, 30 s
    recording_nav: Path  # broadcast ephemeris of the recording's day
    trk_log: Path  # Pocket SDR Python tracker log of the recording
    ctrk_log: Path  # Pocket SDR C tracker log of the recording
    usud_obs: Path
    usud_nav: Path
    sp3: Path
    mate_obs: Path
    mat1_obs: Path
    brdc_mixed: Path
    fs: float = 24e6
    f_if: float = 6e6
    iq: bool = False


_FILES = {
    "recording": "L1_20211226_082401_24MHz_I.bin",
    "recording_nav": "igs/USUD00JPN_R_20213600000_01D_GN.rnx.gz",
    "trk_log": "pocketsdr-trk/L1_20211226_082401_24MHz_I.trk.log",
    "ctrk_log": "pocketsdr-trk/L1_20211226_082401_24MHz_I.ctrk.log",
    "usud_obs": "igs/USUD00JPN_R_20213600000_01D_30S_MO.crx.gz",
    "usud_nav": "igs/USUD00JPN_R_20213600000_01D_GN.rnx.gz",
    "sp3": "igs/igs21900.sp3",
    "mate_obs": "igs/MATE00ITA_R_20213600000_01D_30S_MO.crx.gz",
    "mat1_obs": "igs/MAT100ITA_R_20213600000_01D_30S_MO.crx.gz",
    "brdc_mixed": "igs/BRDC00WRD_R_20213600000_01D_MN.rnx.gz",
}


def data_source() -> Captures | None:
    """The capture files named by `PYRX_CAPTURES`, or None (with a printed reason) when
    the variable is unset or a file is missing; the notebooks then run on synthetic data."""
    value = os.environ.get("PYRX_CAPTURES")
    if not value:
        print("PYRX_CAPTURES is not set: running on synthetic data")
        return None
    root = Path(value)
    paths = {name: root / rel for name, rel in _FILES.items()}
    missing = [rel for name, rel in _FILES.items() if not paths[name].is_file()]
    if missing:
        print(f"missing under {root}: {', '.join(missing)}; running on synthetic data")
        return None
    print("running on the recordings named by PYRX_CAPTURES")
    return Captures(root, **paths)


def scratch_dir(name: str) -> Path:
    """A directory for a notebook's intermediate files, under the system temp dir."""
    path = Path(tempfile.gettempdir()) / "pyrx-notebooks" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# synthetic inputs ---------------------------------------------------------------

SYNTH_PRNS = [4, 7, 12, 25]
_SYNTH_DOPPLERS = (800.0, -1200.0, 2500.0, -300.0)  # Hz
_SYNTH_OFFSETS = (500, 1700, 3100, 900)  # samples
SYNTH_CN0 = 45.0  # dB-Hz


def synthetic_recording(
    directory: Path, seconds: float = 22.0, fs: float = 2.5e6
) -> tuple[Path, float, float, bool, list[int]]:
    """An int8 I/Q recording at baseband of four L1 C/A signals carrying three real
    navigation subframes each; returns path, fs, f_if, iq, prns. Written once and reused.

    Each signal is synthesised 10 log10(4) dB above the target so that the sum of the
    four noise realisations lands at `SYNTH_CN0`.
    """
    n = int(fs * seconds)
    path = Path(directory) / f"synthetic_l1ca_{fs / 1e6:g}MHz_{seconds:g}s.bin"
    if path.is_file() and path.stat().st_size == 2 * n:
        return path, fs, 0.0, True, list(SYNTH_PRNS)
    bits = np.concatenate([np.zeros(100, dtype=np.uint8), frame(5044)])
    cn0 = SYNTH_CN0 + 10 * log10(len(SYNTH_PRNS))
    x = np.zeros(n, dtype=np.complex64)
    for i, (prn, dop, off) in enumerate(zip(SYNTH_PRNS, _SYNTH_DOPPLERS, _SYNTH_OFFSETS)):
        x += synth_l1ca(prn, fs, 0.0, dop, off, n, cn0, True, seed=i, bits=bits)
    write_iq_int8(x, path)
    return path, fs, 0.0, True, list(SYNTH_PRNS)


def synthetic_pvt_epochs(n: int = 120, interval: float = 30.0, seed: int = 0):
    """`n` epochs of noisy pseudoranges from the test constellation at USUD with a
    drifting receiver clock; returns epochs, ephemerides, ionosphere, truth position."""
    rng = np.random.default_rng(seed)
    ephs = constellation()
    epochs = []
    for k in range(n):
        dtr = 1e-4 + 2e-9 * k * interval
        epoch = truth_epoch(ephs, USUD, dtr, tow=TOW + k * interval)
        obs = [replace(o, pseudorange=o.pseudorange + rng.normal(0.0, 0.3)) for o in epoch.obs]
        epochs.append(replace(epoch, obs=obs))
    return epochs, ephs, IONO_DEFAULT, USUD


ROVER = USUD + np.array([5.0, -3.0, 2.0])
RTK_SIGNALS = ("G1C", "G2W")


def synthetic_rtk_epochs(n: int = 40, interval: float = 30.0, slip_at: int = 20, seed: int = 0):
    """Rover and base observations of a short baseline at USUD, with the rover's G03 L1
    ambiguity slipping by three cycles from epoch `slip_at` on; returns rover, base,
    ephemerides, base station, rover antenna delta, truth rover marker."""
    rng = np.random.default_rng(seed)
    ephs = constellation()
    amb_r = ambiguities(rng, ephs, RTK_SIGNALS)
    amb_b = ambiguities(rng, ephs, RTK_SIGNALS)
    slipped = {**amb_r, ("G03", "G1C"): amb_r[("G03", "G1C")] + 3}
    rover, base = [], []
    for k in range(n):
        amb = amb_r if k < slip_at else slipped
        r, b = synthetic_pair(ephs, USUD, ROVER, TOW + k * interval, RTK_SIGNALS, amb, amb_b, rng)
        rover.append(r)
        base.append(b)
    return rover, base, ephs, Station(USUD, np.zeros(3)), np.zeros(3), ROVER


# figures ------------------------------------------------------------------------


def figure(nrows: int = 1, ncols: int = 1, figsize=(10, 4), **kw):
    """`plt.subplots` with the package's plot style: muted grid, ink-coloured ticks."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, **kw)
    for ax in np.atleast_1d(axes).flat:
        ax.tick_params(colors=INK)
        ax.grid(color=MUTED, alpha=0.3)
    return fig, axes


def enu(xyz: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """East, north, up (m) of each ECEF row of `xyz` about `origin`."""
    lat, lon, h = geodetic(origin)
    xyz = np.atleast_2d(xyz)
    e, n, u = pymap3d.ecef2enu(xyz[:, 0], xyz[:, 1], xyz[:, 2], np.degrees(lat), np.degrees(lon), h)
    return np.column_stack([e, n, u])


# references ---------------------------------------------------------------------


def read_trk_log(path) -> dict[int, np.ndarray]:
    """Pocket SDR Python tracker `$CH` rows per PRN: time_s, cn0, coff_ms, doppler, adr."""
    rows: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        f = line.split(",")
        if f[0] != "$CH" or f[2] != "L1CA":
            continue
        rows.setdefault(int(f[3]), []).append([float(f[i]) for i in (1, 5, 6, 7, 8)])
    return {prn: np.array(r) for prn, r in rows.items()}


def read_ctrk_log(path) -> dict[int, np.ndarray]:
    """Pocket SDR C tracker `$CH` L1CA rows per PRN: time, cn0, coff_ms, dop, adr, tow_v, tow_ms."""
    rows: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        f = line.split(",")
        if f[0] != "$CH" or f[5] != "L1CA":
            continue
        rows.setdefault(int(f[6]), []).append([float(f[i]) for i in (1, 8, 9, 10, 11, 19, 20)])
    return {prn: np.array(r) for prn, r in rows.items()}


# RTKLIB options mirroring the models pyrx applies, as the reference tests use them
SINGLE_CONF = """\
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
"""

RTK_CONF = """\
pos1-posmode={mode}
pos1-frequency={freq}
pos1-soltype=forward
pos1-elmask=15
pos1-snrmask_r=off
pos1-dynamics=off
pos1-tidecorr=off
pos1-ionoopt=brdc
pos1-tropopt=saas
pos1-sateph=brdc
pos1-navsys={navsys}
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
ant1-antdelu={rover_up}
ant2-postype=xyz
ant2-pos1={base_x}
ant2-pos2={base_y}
ant2-pos3={base_z}
ant2-antdelu={base_up}
"""


def _rnx2rtkp() -> str | None:
    exe = shutil.which("rnx2rtkp")
    if exe is None:
        print("RTKLIB's rnx2rtkp is not on the path: no reference solution")
    return exe


def _plain(path, out_dir) -> Path:
    """A decompressed copy (gzip, Hatanaka) of a RINEX file for rnx2rtkp."""
    path = Path(path)
    if not path.name.endswith((".gz", ".crx", ".Z")):
        return path
    plain = Path(out_dir) / path.name.replace(".crx.gz", ".rnx").replace(".gz", "")
    if plain.exists():
        return plain
    copy = Path(out_dir) / path.name
    shutil.copyfile(path, copy)
    return Path(hatanaka.decompress_on_disk(copy, delete=True))


def _run(exe, conf, out_dir, span, files) -> list[list[str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_path = out_dir / "rnx2rtkp.conf"
    conf_path.write_text(conf)
    pos = out_dir / "rnx2rtkp.pos"
    ts, te = span
    subprocess.run(
        [exe, "-k", str(conf_path), "-y", "2", "-ts", *ts.split(), "-te", *te.split(),
         "-o", str(pos), *(str(_plain(f, out_dir)) for f in files)],
        check=True,
        capture_output=True,
    )  # fmt: skip
    lines = pos.read_text().splitlines()
    return [line.split() for line in lines if line.strip() and not line.startswith("%")]


def rnx2rtkp_single(obs, nav, out_dir, span) -> dict[tuple[int, float], np.ndarray] | None:
    """RTKLIB's single-point solutions keyed by (week, tow to 0.1 s): ECEF position."""
    exe = _rnx2rtkp()
    if exe is None:
        return None
    rows = _run(exe, SINGLE_CONF, out_dir, span, (obs, nav))
    return {(int(f[0]), round(float(f[1]), 1)): np.array([float(v) for v in f[2:5]]) for f in rows}


def rnx2rtkp_rtk(
    rover, base, nav, out_dir, base_xyz, base_up, rover_up, span, freq="l1+l2", navsys=1
) -> dict[tuple[int, float], dict] | None:
    """RTKLIB's kinematic relative solutions keyed by (week, tow to 0.1 s): `xyz`, `q`
    (1 fixed, 2 float), `ns`, `ratio`."""
    exe = _rnx2rtkp()
    if exe is None:
        return None
    conf = RTK_CONF.format(
        mode="kinematic", freq=freq, navsys=navsys, rover_up=rover_up, base_up=base_up,
        base_x=base_xyz[0], base_y=base_xyz[1], base_z=base_xyz[2],
    )  # fmt: skip
    rows = _run(exe, conf, out_dir, span, (rover, base, nav))
    return {
        (int(f[0]), round(float(f[1]), 1)): {
            "xyz": np.array([float(v) for v in f[2:5]]),
            "q": int(f[5]),
            "ns": int(f[6]),
            "ratio": float(f[14]),
        }
        for f in rows
    }
