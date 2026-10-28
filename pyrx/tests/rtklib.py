"""Run RTKLIB's rnx2rtkp and read its solution and status files, for the observables tests."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


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
