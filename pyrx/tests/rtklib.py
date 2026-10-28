"""Run RTKLIB's rnx2rtkp and read its solution and status files, for the reference tests."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# RTKLIB single-point mode with the models pyrx applies: GPS only, 15 deg mask, Klobuchar
# from the navigation file, Saastamoinen; ECEF output with the time as week and seconds
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


def parse_pos(path) -> list[dict]:
    """Solutions of a `.pos` written with `-t`: time (hh:mm:ss), lat, lon, height, q, ns."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("%") or not line.strip():
            continue
        f = line.split()
        out.append(
            {
                "time": f[1][:8],
                "lat": float(f[2]),
                "lon": float(f[3]),
                "height": float(f[4]),
                "q": int(f[5]),
                "ns": int(f[6]),
            }
        )
    return out


def parse_stat_residuals(path) -> dict[int, list[float]]:
    """Pseudorange residuals (m) per GPS PRN from the `$SAT` lines of a `.stat` file."""
    out: dict[int, list[float]] = {}
    for line in Path(path).read_text().splitlines():
        f = line.split(",")
        if f[0] == "$SAT" and f[3].startswith("G"):
            out.setdefault(int(f[3][1:]), []).append(float(f[7]))
    return out


def rnx2rtkp_single(obs, nav, out_dir) -> tuple[list[dict], dict[int, list[float]]]:
    """Single-point solutions and pseudorange residuals from rnx2rtkp; skips without the tool."""
    exe = shutil.which("rnx2rtkp")
    if exe is None:
        pytest.skip("rnx2rtkp not on the path")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = out_dir / "fix.pos"
    subprocess.run(
        [exe, "-p", "0", "-t", "-y", "2", "-o", str(pos), str(obs), str(nav)],
        check=True,
        capture_output=True,
    )
    return parse_pos(pos), parse_stat_residuals(pos.with_suffix(".pos.stat"))


def horizontal(a: dict, b: dict) -> float:
    """Horizontal distance (m) between two solutions."""
    dn = (a["lat"] - b["lat"]) * 111.2e3
    de = (a["lon"] - b["lon"]) * 111.2e3 * np.cos(np.radians(b["lat"]))
    return float(np.hypot(dn, de))


def rms(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(v**2)))


def _plain(path, out_dir) -> Path:
    """A copy of a RINEX file decompressed (gzip, Hatanaka) for rnx2rtkp."""
    import hatanaka

    path = Path(path)
    if not path.name.endswith((".gz", ".crx", ".Z")):
        return path
    copy = Path(out_dir) / path.name
    shutil.copyfile(path, copy)
    return Path(hatanaka.decompress_on_disk(copy, delete=True))


def rnx2rtkp_models(
    obs, nav, out_dir, ts: str | None = None, te: str | None = None
) -> tuple[dict[tuple[int, float], dict], dict[tuple[int, float], np.ndarray]]:
    """Single-point solutions with the models of SINGLE_CONF, keyed by (week, tow to 0.1 s).

    Returns the solutions (`xyz` ECEF m, `q`, `ns`) and the ENU velocities of the
    `.stat` file's `$VELACC` lines; skips without the tool.
    """
    exe = shutil.which("rnx2rtkp")
    if exe is None:
        pytest.skip("rnx2rtkp not on the path")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf = out_dir / "single.conf"
    conf.write_text(SINGLE_CONF)
    pos = out_dir / "fix.pos"
    span = [] if ts is None else ["-ts", *ts.split()]
    span += [] if te is None else ["-te", *te.split()]
    obs, nav = _plain(obs, out_dir), _plain(nav, out_dir)
    subprocess.run(
        [exe, "-k", str(conf), "-y", "2", *span, "-o", str(pos), str(obs), str(nav)],
        check=True,
        capture_output=True,
    )
    sols = {}
    for line in pos.read_text().splitlines():
        if line.startswith("%") or not line.strip():
            continue
        f = line.split()
        key = (int(f[0]), round(float(f[1]), 1))
        sols[key] = {"xyz": np.array([float(v) for v in f[2:5]]), "q": int(f[5]), "ns": int(f[6])}
    vel = {}
    for line in pos.with_suffix(".pos.stat").read_text().splitlines():
        f = line.split(",")
        if f[0] == "$VELACC":
            vel[(int(f[1]), round(float(f[2]), 1))] = np.array([float(v) for v in f[4:7]])
    return sols, vel
