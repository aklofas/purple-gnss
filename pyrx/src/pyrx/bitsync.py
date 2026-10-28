"""Bit and frame synchronisation of a tracked GPS L1 C/A channel.

Bit sync finds which of the 20 one-millisecond epochs in a 20 ms data bit
carries the bit edge, by counting prompt sign changes at each epoch phase:
noise flips the prompt sign at any phase, data bits only at the edge phase,
so the edge phase is accepted when its count stands out against the others.
Frame sync then looks for the TLM preamble repeating one subframe later with
consistent parity, subframe ID and TOW, and decodes every parity-clean
subframe from there on. Received polarity is unknown until the preamble is
matched, so both polarities are tried.
"""

from dataclasses import dataclass, replace

import numpy as np

from pyrx.lnav import (
    PREAMBLE,
    SUBFRAME_BITS,
    SUBFRAME_S,
    WEEK_S,
    WORD_BITS,
    Subframe,
    decode_how,
    decode_subframe,
)

EPOCHS_PER_BIT = 20
TOW_COUNTS = WEEK_S // SUBFRAME_S  # the HOW count wraps to 0 at the week boundary
MIN_EDGE_CONTRAST = 3.0  # winning phase count over the mean count of the other phases
MIN_EDGES = 5


@dataclass
class BitStream:
    """50 bps bits recovered from prompt correlations.

    bits: 0/1 as received (1 where the 20 ms prompt I sum is negative); the
        polarity is resolved later by frame sync.
    epoch: tracking epoch index of each bit's first millisecond.
    phase: epoch index modulo 20 of the bit edges.
    score: fraction of all sign changes that fell on that phase.
    inverted: set by frame sync: True when the received bits had to be
        complemented to match the preamble, None until a frame start is found.
    """

    bits: np.ndarray
    epoch: np.ndarray
    phase: int
    score: float
    inverted: bool | None = None


def bit_sync(prompt_i: np.ndarray, start_epoch: int) -> BitStream | None:
    """Bits from per-epoch prompt I starting at epoch `start_epoch`; None without a clear edge phase."""
    p = np.asarray(prompt_i, dtype=np.float64)
    if len(p) < 2 * EPOCHS_PER_BIT:
        return None
    s = np.sign(p)
    edges = np.nonzero(s[1:] != s[:-1])[0] + 1  # epoch index where the sign differs from the last
    counts = np.bincount(edges % EPOCHS_PER_BIT, minlength=EPOCHS_PER_BIT)
    if counts.sum() == 0:
        return None
    phase = int(counts.argmax())
    peak = int(counts[phase])
    others = (counts.sum() - peak) / (EPOCHS_PER_BIT - 1)
    if peak < MIN_EDGES or peak < MIN_EDGE_CONTRAST * others:
        return None
    score = peak / counts.sum()
    n_bits = (len(p) - phase) // EPOCHS_PER_BIT
    sums = p[phase : phase + n_bits * EPOCHS_PER_BIT].reshape(n_bits, EPOCHS_PER_BIT).sum(axis=1)
    epoch = start_epoch + phase + EPOCHS_PER_BIT * np.arange(n_bits)
    return BitStream((sums < 0).astype(np.uint8), epoch, phase, float(score))


def _preamble_at(b: np.ndarray, p: int) -> bool:
    return bool(np.array_equal(b[p : p + len(PREAMBLE)], PREAMBLE))


def frame_sync(stream: BitStream, prn: int = 0) -> list[Subframe]:
    """Subframes in order from the first verified subframe start; empty when none is found.

    A start is verified by the preamble appearing there and one subframe later,
    the TLM and HOW words passing parity, a valid subframe ID, and the TOW
    counts of the two subframes differing by one. From there every 300-bit
    block whose ten words all pass parity and whose TOW count is the one its
    position in the sequence requires is returned; a block with a parity
    failure or another count is dropped on its own.
    """
    n = len(stream.bits)
    # the candidate's second subframe must hold its TLM and HOW words in full
    for invert in (0, 1):
        b = stream.bits ^ invert
        for p in range(n - SUBFRAME_BITS - 2 * WORD_BITS + 1):
            if not (_preamble_at(b, p) and _preamble_at(b, p + SUBFRAME_BITS)):
                continue
            d29, d30 = (int(b[p - 2]), int(b[p - 1])) if p >= 2 else (0, 0)
            how = decode_how(b[p : p + 30], b[p + 30 : p + 60], d29, d30)
            q = p + SUBFRAME_BITS
            how_next = decode_how(b[q : q + 30], b[q + 30 : q + 60], int(b[q - 2]), int(b[q - 1]))
            if how is None or how_next is None:
                continue
            if not 1 <= how[1] <= 5 or how_next[0] != (how[0] + 1) % TOW_COUNTS:
                continue
            stream.inverted = bool(invert)
            return _decode_from(b, p, d29, d30, stream.epoch, prn, how[0])
    return []


def _decode_from(b, p, d29, d30, epoch, prn, count) -> list[Subframe]:
    out = []
    while p + SUBFRAME_BITS <= len(b):
        sf = decode_subframe(b[p : p + SUBFRAME_BITS], d29, d30, prn)
        # parity is a weak code: a block can pass it with a changed count
        if sf is not None and sf.tow_s == (count * SUBFRAME_S - SUBFRAME_S) % WEEK_S:
            out.append(replace(sf, epoch=int(epoch[p])))
        d29, d30 = int(b[p + SUBFRAME_BITS - 2]), int(b[p + SUBFRAME_BITS - 1])
        p += SUBFRAME_BITS
        count = (count + 1) % TOW_COUNTS
    return out
