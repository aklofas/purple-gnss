"""Command-line entry points."""

import argparse
import time
from pathlib import Path

from pyrx.acquisition import DETECT_THRESHOLD, acquire_l1ca
from pyrx.plots import plot_acquisition, plot_cn0, plot_tracking, write_track_log
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


if __name__ == "__main__":
    raise SystemExit(acq_main())
