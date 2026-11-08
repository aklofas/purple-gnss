"""Track segments of each PRN over a whole recording, re-acquiring where it is not tracked.

A segment is one continuous track (`TrackResult`) of a PRN. For each PRN the
recording is walked in time order: the PRN is searched at the probe instant, a
detection is tracked until the data ends or lock is lost, and the next probe is the
first instant on the grid `t_start + REACQ_INTERVAL * k` at or after the segment's
end (or one grid step after a probe that found nothing). Nothing is probed twice,
so the walk ends after at most one probe per grid instant.

A detection is accepted only after a trial: the seed is tracked for `TRIAL`
seconds and kept when `bit_sync` finds the data-bit edges in that stretch. The
tracker's own loss rule cannot filter a false alarm, because its C/N0 estimate on
an absent PRN settles near 30 dB-Hz and never crosses `CN0_LOST`, so a phantom
seed would otherwise run to the end of the file. An accepted seed is tracked again
from the same seed, so the segment does not depend on the trial.

PRNs run in parallel processes; within a PRN everything is sequential, so the
segments do not depend on the worker count.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from pyrx.acquisition import DEFAULT_WINDOW, acquire_l1ca
from pyrx.navdecode import settled_bit_stream
from pyrx.samples import SampleFile, read_samples
from pyrx.tracking import TrackResult, track_l1ca

REACQ_INTERVAL = 10.0  # s between probes of a PRN that is not being tracked
TRIAL = 6.0  # s a detection is tracked before bit sync decides whether it is real


def _bit_syncs(track: TrackResult) -> bool:
    """True when the data-bit edges stand out once the PLL has settled, as `decode_lnav` judges."""
    return settled_bit_stream(track) is not None


def track_prn(
    path,
    fs: float,
    iq: bool,
    f_if: float,
    prn: int,
    t_start: float = 0.0,
    duration: float | None = None,
    interval: float = REACQ_INTERVAL,
    window: float = DEFAULT_WINDOW,
    rejected: list[tuple[int, float]] | None = None,
) -> list[TrackResult]:
    """Segments of PRN `prn` in the recording at `path`, in time order.

    Probes start at `t_start` and fall on the grid `t_start + interval * k`; each
    integrates `window` seconds. `duration` limits the walk to that many seconds
    from `t_start` (the whole file when None). When `rejected` is given, every
    detection the trial turned down is appended to it as `(prn, probe instant)`.
    """
    source = SampleFile(path, fs, iq)
    end = source.n_samples / fs
    if duration is not None:
        end = min(end, t_start + duration)
    segments: list[TrackResult] = []
    k = 0  # probe index on the grid
    while (t := t_start + interval * k) + window <= end:
        acq = acquire_l1ca(read_samples(path, fs, iq, t, window), fs, f_if, prn)
        if acq.detected:
            seed = (acq.code_offset, acq.doppler, acq.cn0)
            trial = track_l1ca(source, f_if, prn, *seed, t_start=t, duration=min(TRIAL, end - t))
            if _bit_syncs(trial):
                seg = track_l1ca(source, f_if, prn, *seed, t_start=t, duration=end - t)
                segments.append(seg)
                if seg.lost_at is None:
                    break
                # the first grid instant at or after the loss; a ceil of the float quotient
                # overshoots when the loss lands on a grid instant
                k = int((seg.lost_at - t_start) // interval)
                if t_start + interval * k < seg.lost_at:
                    k += 1
                continue
            if rejected is not None:
                rejected.append((prn, t))
        k += 1
    return segments


def _prn_job(args):
    path, fs, iq, f_if, prn, kwargs = args
    rejected: list[tuple[int, float]] = []
    return track_prn(path, fs, iq, f_if, prn, rejected=rejected, **kwargs), rejected


def track_prns(
    path,
    fs: float,
    iq: bool,
    f_if: float,
    prns: list[int],
    workers: int | None = None,
    rejected: list[tuple[int, float]] | None = None,
    **kwargs,
) -> list[TrackResult]:
    """All segments of the given PRNs, PRNs in the given order and segments in time order.

    `kwargs` and `rejected` go to `track_prn`. Each worker process reopens the file,
    so nothing large is pickled in.
    """
    jobs = [(os.fspath(path), fs, iq, f_if, prn, kwargs) for prn in prns]
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers <= 1 or len(jobs) == 1:
        per_prn = [_prn_job(j) for j in jobs]
    else:
        # spawn, not fork: the parent holds BLAS threads and forking them is unsafe
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            per_prn = list(pool.map(_prn_job, jobs))
    if rejected is not None:
        rejected.extend(r for _, rej in per_prn for r in rej)
    return [seg for segs, _ in per_prn for seg in segs]
