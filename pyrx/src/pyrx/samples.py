"""Reading recorded IF samples.

Files hold signed 8-bit samples. Real recordings have one byte per sample; I/Q
recordings interleave I and Q bytes, and the complex sample is I - jQ (the Q sign
convention of the Pocket SDR front-end that produced the recordings).
"""

import os

import numpy as np


def _to_complex(raw: np.ndarray, iq: bool) -> np.ndarray:
    """int8 bytes (interleaved I,Q when `iq`) to complex64 with Pocket SDR's I - jQ."""
    if iq:
        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        return (i - 1j * q).astype(np.complex64)
    return raw.astype(np.complex64)


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
    return _to_complex(raw, iq)


class SampleFile:
    """A recording read block by block through a memory map, never loaded whole."""

    def __init__(self, path: str | os.PathLike, fs: float, iq: bool):
        self.path = path
        self.fs = fs
        self.iq = iq
        self._bytes_per_sample = 2 if iq else 1
        self._raw = np.memmap(path, dtype=np.int8, mode="r")
        self.n_samples = len(self._raw) // self._bytes_per_sample

    def read(self, start: int, count: int) -> np.ndarray:
        """`count` complex64 samples beginning at sample index `start`."""
        if start < 0 or start + count > self.n_samples:
            raise ValueError(
                f"{self.path}: samples {start}..{start + count} outside 0..{self.n_samples}"
            )
        b = self._bytes_per_sample
        return _to_complex(np.asarray(self._raw[start * b : (start + count) * b]), self.iq)


class ArraySamples:
    """In-memory complex64 samples with the same block interface as SampleFile."""

    def __init__(self, samples: np.ndarray, fs: float):
        self._samples = np.asarray(samples, dtype=np.complex64)
        self.fs = fs
        self.n_samples = len(self._samples)

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or start + count > self.n_samples:
            raise ValueError(f"samples {start}..{start + count} outside 0..{self.n_samples}")
        return self._samples[start : start + count]
