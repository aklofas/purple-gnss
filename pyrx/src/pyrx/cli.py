"""Command-line entry points."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from pyrx.acquisition import DETECT_THRESHOLD, acquire_l1ca
from pyrx.navdecode import decode_lnav
from pyrx.observables import make_observables
from pyrx.plots import plot_acquisition, plot_cn0, plot_tracking, write_track_log
from pyrx.rinex import gps_datetime, write_rinex_nav, write_rinex_obs
from pyrx.samples import read_samples
from pyrx.tracking import track_channels

DEFAULT_WINDOW = 0.010  # s; Pocket SDR's default acquisition integration time


def parse_prns(s: str) -> list[int]:
    """'1-3,7' -> [1, 2, 3, 7]."""
    prns: list[int] = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            prns.extend(range(int(lo), int(hi) + 1))
        else:
            prns.append(int(part))
    return prns


def acq_main(argv: list[str] | None = None) -> int:
    """Acquire GPS L1 C/A PRNs in a recording and print one line per PRN."""
    p = argparse.ArgumentParser(prog="pyrx-acq", description=acq_main.__doc__)
    p.add_argument("file")
    p.add_argument("--fs", type=float, required=True, help="sample rate (Hz)")
    p.add_argument("--f-if", type=float, default=0.0, help="intermediate frequency (Hz)")
    p.add_argument("--iq", action="store_true", help="file holds interleaved I/Q bytes")
    p.add_argument("--prn", default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
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
    results = [acquire_l1ca(samples, args.fs, args.f_if, prn) for prn in parse_prns(args.prn)]
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
    p.add_argument("--prn", default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
    p.add_argument("--t-start", type=float, default=0.0, help="start offset in the file (s)")
    p.add_argument("--duration", type=float, default=0.0, help="length to track (s), 0 = all")
    p.add_argument("--log", default=None, help="CSV of channel state every 0.1 s")
    p.add_argument("--plot", default=None, help="directory for trk_prnNN.png")
    args = p.parse_args(argv)

    head = read_samples(args.file, args.fs, args.iq, args.t_start, DEFAULT_WINDOW)
    seeds = []
    for prn in parse_prns(args.prn):
        acq = acquire_l1ca(head, args.fs, args.f_if, prn)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
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
    p.add_argument("--prn", default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
    p.add_argument("--t-start", type=float, default=0.0, help="start offset in the file (s)")
    p.add_argument("--duration", type=float, default=0.0, help="length to decode (s), 0 = all")
    p.add_argument("--json", default=None, help="write decoded ephemerides and ionosphere here")
    args = p.parse_args(argv)

    head = read_samples(args.file, args.fs, args.iq, args.t_start, DEFAULT_WINDOW)
    seeds = []
    for prn in parse_prns(args.prn):
        acq = acquire_l1ca(head, args.fs, args.f_if, prn)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
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
    p.add_argument("--prn", default="1-32", help="PRNs, e.g. 1-32 or 1,5,7")
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
    for prn in parse_prns(args.prn):
        acq = acquire_l1ca(head, args.fs, args.f_if, prn)
        if acq.detected:
            seeds.append((prn, acq.code_offset, acq.doppler, acq.cn0))
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
            write_rinex_nav(args.nav, ephs, tow1, iono, week)
        else:
            print("no ephemeris decoded, navigation file not written")
    print(f"TIME = {elapsed:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(acq_main())
