"""Reading recorded IF samples.

Files hold signed 8-bit samples. Real recordings have one byte per sample; I/Q
recordings interleave I and Q bytes, and the complex sample is I - jQ (the Q sign
convention of the Pocket SDR front-end that produced the recordings).
"""

import os

import numpy as np


def read_samples(
    path: str | os.PathLike,
    fs: float,
    iq: bool,
    t_start: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """Return complex64 samples from `path`, starting `t_start` seconds in.

    `duration` seconds are read (all remaining samples when None). Raises
    ValueError if the file holds fewer samples than requested.
    """
    bytes_per_sample = 2 if iq else 1
    start = round(fs * t_start)
    count = -1 if duration is None else round(fs * duration)
    raw = np.fromfile(
        path,
        dtype=np.int8,
        count=-1 if count < 0 else count * bytes_per_sample,
        offset=start * bytes_per_sample,
    )
    if count >= 0 and len(raw) < count * bytes_per_sample:
        raise ValueError(
            f"{path}: requested {count} samples from sample {start}, "
            f"file holds {len(raw) // bytes_per_sample}"
        )
    if iq:
        return (raw[0::2].astype(np.float32) - 1j * raw[1::2].astype(np.float32)).astype(
            np.complex64
        )
    return raw.astype(np.complex64)
