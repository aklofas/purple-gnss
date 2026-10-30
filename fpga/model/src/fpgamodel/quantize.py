"""The 2-bit sample encoding and the word layouts the register bus uses.

A sample is the 2-bit code {sign, mag}: bit 1 set means negative, bit 0 set means
magnitude 3, clear means magnitude 1 (the MAX2771's 2-bit output). Playback words hold
16 samples, code words hold 32 chips.
"""

import numpy as np


def to_sign_mag(x: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """2-bit codes for real samples; the magnitude bit is set where |x| >= `threshold`.

    The threshold defaults to the standard deviation of `x`, close to the optimal 2-bit
    quantizer for Gaussian noise, except for input that already holds only the values +/-1
    and +/-3 (a 2-bit recording), which comes back unchanged in meaning: there the
    threshold is 2, so a stretch of nothing but +/-1 stays at magnitude 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if threshold is None:
        threshold = 2.0 if _holds_only_two_bit_values(x) else float(x.std())
    sign = (x < 0).astype(np.uint8)
    mag = (np.abs(x) >= threshold).astype(np.uint8)
    return (sign << 1) | mag


def _holds_only_two_bit_values(x: np.ndarray, block: int = 1 << 20) -> bool:
    """True when every sample is one of -3, -1, 1, 3; checked a block at a time so a long
    recording costs no input-sized temporary."""
    for start in range(0, len(x), block):
        a = np.abs(x[start : start + block])
        if not ((a == 1.0) | (a == 3.0)).all():
            return False
    return len(x) > 0


def values(codes: np.ndarray) -> np.ndarray:
    """Sample values +/-1, +/-3 (int8) of 2-bit codes."""
    codes = np.asarray(codes)
    mag = np.where(codes & 1, 3, 1).astype(np.int8)
    return np.where(codes & 2, -mag, mag).astype(np.int8)


def pack_words(codes: np.ndarray) -> np.ndarray:
    """Playback words: 16 codes per uint32, code i of a word in bits 2i+1:2i, zero padded."""
    codes = np.asarray(codes, dtype=np.uint64)
    n_words = -(-len(codes) // 16)
    padded = np.zeros(n_words * 16, dtype=np.uint64)
    padded[: len(codes)] = codes
    shifts = (2 * np.arange(16, dtype=np.uint64))[np.newaxis, :]
    return (padded.reshape(n_words, 16) << shifts).sum(axis=1).astype(np.uint32)


def code_bits(chips: np.ndarray) -> np.ndarray:
    """Code words: 32 chips per uint32, chip i in bit i & 31 of word i >> 5, zero padded.

    A set bit is a chip of value -1 (`pyrx.codes.gps_ca` returns +1/-1).
    """
    bits = (np.asarray(chips) < 0).astype(np.uint64)
    n_words = -(-len(bits) // 32)
    padded = np.zeros(n_words * 32, dtype=np.uint64)
    padded[: len(bits)] = bits
    shifts = np.arange(32, dtype=np.uint64)[np.newaxis, :]
    return (padded.reshape(n_words, 32) << shifts).sum(axis=1).astype(np.uint32)
