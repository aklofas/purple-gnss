"""Command-line entry points."""

import argparse
import time
from pathlib import Path

from pyrx.acquisition import acquire_l1ca
from pyrx.plots import plot_acquisition, plot_cn0
from pyrx.samples import read_samples


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
    p.add_argument("--duration", type=float, default=None, help="length to read (s)")
    p.add_argument("--plot", default=None, help="directory for cn0.png and acq_prnNN.png")
    args = p.parse_args(argv)

    samples = read_samples(args.file, args.fs, args.iq, args.t_start, args.duration)
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
        plot_cn0(results, 38.0, out / "cn0.png")
        for r in results:
            if r.detected:
                plot_acquisition(r, args.fs, out / f"acq_prn{r.prn:02d}.png")
    return 0
