"""Command-line entry points."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from math import degrees, nan
from pathlib import Path

import numpy as np
import pymap3d as pm

from pyrx.acquisition import DETECT_THRESHOLD, acquire_l1ca
from pyrx.models import Station, geodetic
from pyrx.navdecode import decode_lnav
from pyrx.observables import make_observables
from pyrx.plots import plot_acquisition, plot_cn0, plot_tracking, write_track_log
from pyrx.pvt import ELEVATION_MASK, solve
from pyrx.rinex import (
    gps_datetime,
    read_rinex_header,
    read_rinex_nav,
    read_rinex_obs,
    read_rinex_observations,
    write_rinex_nav,
    write_rinex_obs,
)
from pyrx.rtk import RtkOptions, solve_rtk
from pyrx.samples import read_samples
from pyrx.tracking import track_channels

DEFAULT_WINDOW = 0.010  # s; Pocket SDR's default acquisition integration time
NO_DETECTION = (
    f"no PRN detected in the {DEFAULT_WINDOW * 1e3:.0f} ms acquisition window: "
    "check --fs, --f-if and --iq"
)


def parse_prns(s: str) -> list[int]:
    """'1-3,7' -> [1, 2, 3, 7]; repeats are dropped, PRNs outside 1-32 are an argument error."""
    prns: list[int] = []
    try:
        for part in s.split(","):
            lo, hi = part.split("-") if "-" in part else (part, part)
            if not 1 <= int(lo) <= int(hi) <= 32:
                raise ValueError
            prns.extend(p for p in range(int(lo), int(hi) + 1) if p not in prns)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: PRNs 1-32 as e.g. 1-32 or 1,5,7") from None
    return prns


def acq_main(argv: list[str] | None = None) -> int:
    """Acquire GPS L1 C/A PRNs in a recording and print one line per PRN."""
    p = argparse.ArgumentParser(prog="pyrx-acq", description=acq_main.__doc__)
    p.add_argument("file")
    p.add_argument("--fs", type=float, required=True, help="sample rate (Hz)")
    p.add_argument("--f-if", type=float, default=0.0, help="intermediate frequency (Hz)")
    p.add_argument("--iq", action="store_true", help="file holds interleaved I/Q bytes")
    p.add_argument("--prn", type=parse_prns, default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
    p.add_argument("--t-start", type=float, default=0.0, help="start offset in the file (s)")
    p.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_WINDOW,
        help=f"length to integrate (s), 0 for the whole file (default {DEFAULT_WINDOW})",
    )
    p.add_argument("--plot", default=None, help="directory for cn0.png and acq_prnNN.png")
    args = p.parse_args(argv)

    duration = args.duration if args.duration > 0 else None
    samples = read_samples(args.file, args.fs, args.iq, args.t_start, duration)
    t0 = time.perf_counter()
    results = [acquire_l1ca(samples, args.fs, args.f_if, prn) for prn in args.prn]
    elapsed = time.perf_counter() - t0
    for r in results:
        print(
            f"SIG= L1CA, PRN= {r.prn:3d}, COFF= {r.code_offset * 1e3:8.5f} ms, "
            f"DOP= {r.doppler:5.0f} Hz, C/N0= {r.cn0:4.1f} dB-Hz"
        )
    print(f"TIME = {elapsed:.3f} s")
    if args.plot is not None:
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        plot_cn0(results, DETECT_THRESHOLD, out / "cn0.png")
        for r in results:
            if r.detected:
                plot_acquisition(r, args.fs, out / f"acq_prn{r.prn:02d}.png")
    return 0


def trk_main(argv: list[str] | None = None) -> int:
    """Acquire and track GPS L1 C/A PRNs in a recording; print one line per PRN."""
    p = argparse.ArgumentParser(prog="pyrx-trk", description=trk_main.__doc__)
    p.add_argument("file")
    p.add_argument("--fs", type=float, required=True, help="sample rate (Hz)")
    p.add_argument("--f-if", type=float, default=0.0, help="intermediate frequency (Hz)")
    p.add_argument("--iq", action="store_true", help="file holds interleaved I/Q bytes")
    p.add_argument("--prn", type=parse_prns, default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
    p.add_argument("--t-start", type=float, default=0.0, help="start offset in the file (s)")
    p.add_argument("--duration", type=float, default=0.0, help="length to track (s), 0 = all")
    p.add_argument("--log", default=None, help="CSV of channel state every 0.1 s")
    p.add_argument("--plot", default=None, help="directory for trk_prnNN.png")
    args = p.parse_args(argv)

    head = read_samples(args.file, args.fs, args.iq, args.t_start, DEFAULT_WINDOW)
    seeds = []
    for prn in args.prn:
        acq = acquire_l1ca(head, args.fs, args.f_if, prn)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
    if not seeds:
        print(NO_DETECTION, file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    duration = args.duration if args.duration > 0 else None
    results = track_channels(
        args.file, args.fs, args.iq, args.f_if, seeds, t_start=args.t_start, duration=duration
    )
    elapsed = time.perf_counter() - t0
    for r in results:
        state = "LOCK" if r.locked else f"LOST at {r.lost_at:.1f} s"
        print(f"PRN {r.prn:2d}  C/N0 {r.cn0[-1]:4.1f} dB-Hz  DOP {r.doppler[-1]:7.1f} Hz  {state}")
    print(f"TIME = {elapsed:.3f} s")
    if args.log is not None:
        write_track_log(results, args.log)
    if args.plot is not None:
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            plot_tracking(r, out / f"trk_prn{r.prn:02d}.png")
    return 0


def nav_main(argv: list[str] | None = None) -> int:
    """Acquire, track and decode the navigation message of GPS L1 C/A PRNs in a recording."""
    p = argparse.ArgumentParser(prog="pyrx-nav", description=nav_main.__doc__)
    p.add_argument("file")
    p.add_argument("--fs", type=float, required=True, help="sample rate (Hz)")
    p.add_argument("--f-if", type=float, default=0.0, help="intermediate frequency (Hz)")
    p.add_argument("--iq", action="store_true", help="file holds interleaved I/Q bytes")
    p.add_argument("--prn", type=parse_prns, default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
    p.add_argument("--t-start", type=float, default=0.0, help="start offset in the file (s)")
    p.add_argument("--duration", type=float, default=0.0, help="length to decode (s), 0 = all")
    p.add_argument("--json", default=None, help="write decoded ephemerides and ionosphere here")
    args = p.parse_args(argv)

    head = read_samples(args.file, args.fs, args.iq, args.t_start, DEFAULT_WINDOW)
    seeds = []
    for prn in args.prn:
        acq = acquire_l1ca(head, args.fs, args.f_if, prn)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
    if not seeds:
        print(NO_DETECTION, file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    duration = args.duration if args.duration > 0 else None
    tracks = track_channels(
        args.file, args.fs, args.iq, args.f_if, seeds, t_start=args.t_start, duration=duration
    )
    results = [decode_lnav(r) for r in tracks]
    elapsed = time.perf_counter() - t0
    for res in results:
        bits = "sync" if res.stream is not None else "none"
        sfs = ",".join(f"{sf.sf_id}@{sf.tow_s}" for sf in res.subframes) or "none"
        e = res.ephemeris
        eph = f"IODE {e.iode} toe {e.toe:.0f}" if e is not None else "incomplete"
        print(f"PRN {res.prn:2d}  bits {bits}  subframes {sfs}  ephemeris {eph}")
    print(f"TIME = {elapsed:.3f} s")
    if args.json is not None:
        iono = next((r.ionosphere for r in results if r.ionosphere is not None), None)
        data = {
            "ephemeris": {str(r.prn): asdict(r.ephemeris) for r in results if r.ephemeris},
            "ionosphere": None if iono is None else {"alpha": iono.alpha, "beta": iono.beta},
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=1))
    return 0


def rinex_main(argv: list[str] | None = None) -> int:
    """Acquire, track and decode GPS L1 C/A PRNs; write RINEX observation and navigation files."""
    p = argparse.ArgumentParser(prog="pyrx-rinex", description=rinex_main.__doc__)
    p.add_argument("file")
    p.add_argument("--fs", type=float, required=True, help="sample rate (Hz)")
    p.add_argument("--f-if", type=float, default=0.0, help="intermediate frequency (Hz)")
    p.add_argument("--iq", action="store_true", help="file holds interleaved I/Q bytes")
    p.add_argument("--prn", type=parse_prns, default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
    p.add_argument("--t-start", type=float, default=0.0, help="start offset in the file (s)")
    p.add_argument("--duration", type=float, default=0.0, help="length to process (s), 0 = all")
    p.add_argument("--interval", type=float, default=1.0, help="observation epoch interval (s)")
    p.add_argument("--week", type=int, default=None, help="GPS week (needed without subframe 1)")
    p.add_argument("--marker", default="PYRX", help="RINEX marker name")
    p.add_argument("--obs", required=True, help="RINEX observation file to write")
    p.add_argument("--nav", default=None, help="RINEX navigation file to write")
    args = p.parse_args(argv)

    head = read_samples(args.file, args.fs, args.iq, args.t_start, DEFAULT_WINDOW)
    seeds = []
    for prn in args.prn:
        acq = acquire_l1ca(head, args.fs, args.f_if, prn)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
    if not seeds:
        print(NO_DETECTION, file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    duration = args.duration if args.duration > 0 else None
    tracks = track_channels(
        args.file, args.fs, args.iq, args.f_if, seeds, t_start=args.t_start, duration=duration
    )
    lnavs = [decode_lnav(r) for r in tracks]
    epochs = make_observables(tracks, lnavs, args.interval, args.week)
    elapsed = time.perf_counter() - t0
    for r, ln in zip(tracks, lnavs):
        mine = [o for e in epochs for o in e.obs if o.prn == r.prn]
        if not mine:
            print(f"PRN {r.prn:2d}  no time tag")
            continue
        print(
            f"PRN {r.prn:2d}  epochs {len(mine)}  P {mine[0].pseudorange / 1e3:.1f} km  "
            f"D {mine[0].doppler:.1f} Hz"
        )
    if not epochs or epochs[0].week is None:
        what = "no observations" if not epochs else "week unknown (no subframe 1); give --week"
        print(f"{what}, nothing written", file=sys.stderr)
        return 2
    week = epochs[0].week
    write_rinex_obs(args.obs, epochs, args.marker)
    print(
        f"EPOCHS {len(epochs)}  from {gps_datetime(week, epochs[0].tow)} "
        f"to {gps_datetime(week, epochs[-1].tow)}"
    )
    if args.nav is not None:
        ephs = [ln.ephemeris for ln in lnavs if ln.ephemeris is not None]
        if ephs:
            tow1 = {ln.prn: sf.tow_s for ln in lnavs for sf in ln.subframes if sf.sf_id == 1}
            iono = next((ln.ionosphere for ln in lnavs if ln.ionosphere is not None), None)
            write_rinex_nav(args.nav, ephs, tow1, iono)
        else:
            print("no ephemeris decoded, navigation file not written")
    print(f"TIME = {elapsed:.3f} s")
    return 0


def _solution_line(sol) -> str:
    lat, lon, h = geodetic(sol.position)
    v = sol.velocity if sol.velocity is not None else (nan, nan, nan)
    ddtr = sol.clock_drift if sol.clock_drift is not None else nan
    return (
        f"{sol.week} {sol.tow:.3f} {degrees(lat):.9f} {degrees(lon):.9f} {h:.4f} "
        f"{sol.clock_bias:.12f} {len(sol.prns)} {sol.gdop:.2f} "
        f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {ddtr:.12f}"
    )


def pvt_main(argv: list[str] | None = None) -> int:
    """Standalone position, clock and velocity per epoch from RINEX observation and navigation files."""
    p = argparse.ArgumentParser(prog="pyrx-pvt", description=pvt_main.__doc__)
    p.add_argument("obs", help="RINEX observation file (plain, .gz or Hatanaka)")
    p.add_argument("nav", help="RINEX navigation file")
    p.add_argument("--mask", type=float, default=ELEVATION_MASK, help="elevation mask (deg)")
    p.add_argument("--ts", default=None, help="first epoch, GPS time 'YYYY-MM-DD HH:MM:SS'")
    p.add_argument("--te", default=None, help="last epoch, GPS time 'YYYY-MM-DD HH:MM:SS'")
    p.add_argument("-o", "--out", default=None, help="write one line per solved epoch here")
    p.add_argument("-v", "--verbose", action="store_true", help="print one line per epoch")
    args = p.parse_args(argv)

    tlim = None
    if args.ts is not None or args.te is not None:
        ts = datetime.fromisoformat(args.ts) if args.ts else datetime(1980, 1, 6)
        te = datetime.fromisoformat(args.te) if args.te else datetime(2100, 1, 1)
        tlim = (ts, te)
    t0 = time.perf_counter()
    epochs = read_rinex_obs(args.obs, tlim)
    ephs, iono = read_rinex_nav(args.nav)
    sols = solve(epochs, ephs, iono, args.mask)
    elapsed = time.perf_counter() - t0
    if args.verbose:
        for sol in sols:
            print(_solution_line(sol))
    if not sols:
        print(f"no epoch solved of {len(epochs)} read", file=sys.stderr)
        return 2
    llh = np.array([geodetic(s.position) for s in sols])
    lat, lon, h = llh.mean(axis=0)
    clock_ms = 1e3 * float(np.mean([s.clock_bias for s in sols]))
    print(
        f"EPOCHS {len(sols)} of {len(epochs)}  mean lat {degrees(lat):.7f} "
        f"lon {degrees(lon):.7f} height {h:.2f}  clock {clock_ms:.3f} ms"
    )
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["% week tow lat lon height clock_s ns gdop vx vy vz ddtr"]
        lines += [_solution_line(s) for s in sols]
        out.write_text("\n".join(lines) + "\n")
    print(f"TIME = {elapsed:.3f} s")
    return 0


def _time_limits(ts: str | None, te: str | None):
    if ts is None and te is None:
        return None
    start = datetime.fromisoformat(ts) if ts else datetime(1980, 1, 6)
    end = datetime.fromisoformat(te) if te else datetime(2100, 1, 1)
    return (start, end)


def rtk_main(argv: list[str] | None = None) -> int:
    """RTK baseline per epoch from rover and base RINEX observation files and a navigation file."""
    p = argparse.ArgumentParser(prog="pyrx-rtk", description=rtk_main.__doc__)
    p.add_argument("rover", help="rover RINEX observation file (plain, .gz or Hatanaka)")
    p.add_argument("base", help="base RINEX observation file")
    p.add_argument("nav", help="RINEX navigation file")
    p.add_argument("--base-xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument(
        "--base-delta", type=float, nargs=3, default=None, metavar=("E", "N", "U"),
        help="base marker to antenna reference point (m); default from the file header",
    )  # fmt: skip
    p.add_argument(
        "--rover-delta", type=float, nargs=3, default=None, metavar=("E", "N", "U"),
        help="rover marker to antenna reference point (m); default from the file header",
    )  # fmt: skip
    p.add_argument("--signals", default="G1C,G2W", help="signals to use, e.g. G1C,G2W,E1C,E5Q")
    p.add_argument("--mode", choices=("kinematic", "static"), default="kinematic")
    p.add_argument("--mask", type=float, default=ELEVATION_MASK, help="elevation mask (deg)")
    p.add_argument("--ts", default=None, help="first epoch, GPS time 'YYYY-MM-DD HH:MM:SS'")
    p.add_argument("--te", default=None, help="last epoch, GPS time 'YYYY-MM-DD HH:MM:SS'")
    p.add_argument("-o", "--out", default=None, help="write one line per solved epoch here")
    p.add_argument("-v", "--verbose", action="store_true", help="print one line per epoch")
    args = p.parse_args(argv)

    signals = tuple(args.signals.split(","))
    tlim = _time_limits(args.ts, args.te)
    t0 = time.perf_counter()
    rover_delta = args.rover_delta or read_rinex_header(args.rover).delta_enu
    base_delta = args.base_delta or read_rinex_header(args.base).delta_enu
    rover = read_rinex_observations(args.rover, signals, tlim)
    base = read_rinex_observations(args.base, signals, tlim)
    ephs, iono = read_rinex_nav(args.nav)
    station = Station(np.array(args.base_xyz), np.asarray(base_delta, dtype=float))
    fixes = solve_rtk(
        rover, base, ephs, station, np.asarray(rover_delta, dtype=float),
        RtkOptions(mode=args.mode, mask_deg=args.mask), iono,
    )  # fmt: skip
    elapsed = time.perf_counter() - t0

    def line(f):
        q = 1 if f.status == "fix" else 2
        x, y, z = f.position
        return f"{f.week} {f.tow:.3f} {x:.4f} {y:.4f} {z:.4f} {q} {f.ns} {f.ratio:.1f} {f.age:.2f}"

    if args.verbose:
        for f in fixes:
            print(line(f))
    if not fixes:
        print(f"no epoch solved of {len(rover)} rover epochs read", file=sys.stderr)
        return 2
    fixed = [f for f in fixes if f.status == "fix"]
    lat, lon, h = geodetic(station.marker)
    if fixed:
        enu = np.array([pm.ecef2enu(*f.position, lat, lon, h, deg=False) for f in fixed])
        e, n, u = enu.mean(axis=0)
        ratio = float(np.median([f.ratio for f in fixed]))
    else:
        e = n = u = ratio = nan
    print(
        f"EPOCHS {len(fixed)} fixed {len(fixes) - len(fixed)} float of {len(rover)}  "
        f"mean fixed baseline E {e:.4f} N {n:.4f} U {u:.4f} m  ratio median {ratio:.1f}"
    )
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(["% week tow x y z Q ns ratio age", *map(line, fixes)]) + "\n")
    print(f"TIME = {elapsed:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(acq_main())
