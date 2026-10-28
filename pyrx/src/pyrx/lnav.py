"""GPS LNAV navigation message: parity, subframe layout, encoding and decoding.

Bit positions follow IS-GPS-200 Figure 20-1: 1-based within the 300-bit
subframe, MSB first, counting the six parity bits that end every 30-bit word.
Scale factors and signedness come from Tables 20-I, 20-III and 20-X, the parity
equations from Table 20-XIV. Angles broadcast in semi-circles are returned in
radians so they compare directly with RINEX navigation files.
"""

from dataclasses import dataclass
from math import pi

import numpy as np

PREAMBLE = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
WORD_BITS = 30
SUBFRAME_BITS = 300
SUBFRAME_S = 6
WEEK_S = 604800
IONO_PAGE_SV_ID = 56  # subframe 4 page 18 carries the ionosphere parameters

# Table 20-XIV: for each of D25..D30, the previous word's bit it starts from
# (D29* or D30*) and the data bits d1..d24 XORed into it.
_PARITY = (
    (29, (1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23)),
    (30, (2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24)),
    (29, (1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22)),
    (30, (2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23)),
    (30, (1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24)),
    (29, (3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24)),
)


def _parity(data: np.ndarray, d29: int, d30: int) -> np.ndarray:
    prev = {29: d29, 30: d30}
    return np.array(
        [(prev[p] + int(data[[i - 1 for i in taps]].sum())) & 1 for p, taps in _PARITY],
        dtype=np.uint8,
    )


def data_bits(word: np.ndarray, d30: int) -> np.ndarray:
    """The 24 data bits of a received word, with the D30* inversion undone."""
    return (np.asarray(word[:24], dtype=np.uint8) ^ d30).astype(np.uint8)


def parity_ok(word: np.ndarray, d29: int, d30: int) -> bool:
    """True when the word's six parity bits match its data and the previous word's D29*, D30*."""
    return bool(np.array_equal(_parity(data_bits(word, d30), d29, d30), word[24:30]))


def encode_word(data: np.ndarray, d29: int, d30: int) -> np.ndarray:
    """24 data bits to the 30 transmitted bits, given the previous word's D29*, D30*."""
    data = np.asarray(data, dtype=np.uint8)
    return np.concatenate([data ^ d30, _parity(data, d29, d30)]).astype(np.uint8)


# A field is (name, chunks, signed, scale, semicircles): chunks are (start, nbits)
# in subframe bit numbering, most significant chunk first.
def _f(name, *chunks, signed=False, scale=1, sc=False):
    return (name, chunks, signed, scale, sc)


_TOW_COUNT = (31, 17)
_SF_ID = (50, 3)
_DATA_ID = (61, 2)
_PAGE = _f("page_sv_id", (63, 6))

_SF1 = [
    _f("wn", (61, 10)),
    _f("code_l2", (71, 2)),
    _f("ura", (73, 4)),
    _f("health", (77, 6)),
    _f("iodc", (83, 2), (211, 8)),
    _f("l2p", (91, 1)),
    _f("tgd", (197, 8), signed=True, scale=2**-31),
    _f("toc", (219, 16), scale=2**4),
    _f("af2", (241, 8), signed=True, scale=2**-55),
    _f("af1", (249, 16), signed=True, scale=2**-43),
    _f("af0", (271, 22), signed=True, scale=2**-31),
]
_SF2 = [
    _f("iode", (61, 8)),
    _f("crs", (69, 16), signed=True, scale=2**-5),
    _f("delta_n", (91, 16), signed=True, scale=2**-43, sc=True),
    _f("m0", (107, 8), (121, 24), signed=True, scale=2**-31, sc=True),
    _f("cuc", (151, 16), signed=True, scale=2**-29),
    _f("e", (167, 8), (181, 24), scale=2**-33),
    _f("cus", (211, 16), signed=True, scale=2**-29),
    _f("sqrt_a", (227, 8), (241, 24), scale=2**-19),
    _f("toe", (271, 16), scale=2**4),
    _f("fit", (287, 1)),
    _f("aodo", (288, 5)),
]
_SF3 = [
    _f("cic", (61, 16), signed=True, scale=2**-29),
    _f("omega0", (77, 8), (91, 24), signed=True, scale=2**-31, sc=True),
    _f("cis", (121, 16), signed=True, scale=2**-29),
    _f("i0", (137, 8), (151, 24), signed=True, scale=2**-31, sc=True),
    _f("crc", (181, 16), signed=True, scale=2**-5),
    _f("omega", (197, 8), (211, 24), signed=True, scale=2**-31, sc=True),
    _f("omega_dot", (241, 24), signed=True, scale=2**-43, sc=True),
    _f("iode", (271, 8)),
    _f("idot", (279, 14), signed=True, scale=2**-43, sc=True),
]
_SF4_PAGE18 = [
    _PAGE,
    _f("alpha0", (69, 8), signed=True, scale=2**-30),
    _f("alpha1", (77, 8), signed=True, scale=2**-27),
    _f("alpha2", (91, 8), signed=True, scale=2**-24),
    _f("alpha3", (99, 8), signed=True, scale=2**-24),
    _f("beta0", (107, 8), signed=True, scale=2**11),
    _f("beta1", (121, 8), signed=True, scale=2**14),
    _f("beta2", (129, 8), signed=True, scale=2**16),
    _f("beta3", (137, 8), signed=True, scale=2**16),
]


def _spec(sf_id: int, page_sv_id: int | None) -> list:
    if sf_id == 1:
        return _SF1
    if sf_id == 2:
        return _SF2
    if sf_id == 3:
        return _SF3
    if sf_id == 4 and page_sv_id == IONO_PAGE_SV_ID:
        return _SF4_PAGE18
    if sf_id in (4, 5):
        return [_PAGE]
    return []


def _get(d: np.ndarray, start: int, nbits: int) -> int:
    v = 0
    for b in d[start - 1 : start - 1 + nbits]:
        v = (v << 1) | int(b)
    return v


def _set(d: np.ndarray, start: int, nbits: int, v: int) -> None:
    for i in range(nbits - 1, -1, -1):
        d[start - 1 + i] = v & 1
        v >>= 1


def _unpack(d: np.ndarray, spec: list) -> dict:
    out = {}
    for name, chunks, signed, scale, sc in spec:
        v, n = 0, 0
        for start, nbits in chunks:
            v = (v << nbits) | _get(d, start, nbits)
            n += nbits
        if signed and v >= 1 << (n - 1):
            v -= 1 << n
        out[name] = v if scale == 1 and not sc else v * scale * (pi if sc else 1.0)
    return out


def _pack(d: np.ndarray, spec: list, fields: dict) -> None:
    for name, chunks, signed, scale, sc in spec:
        n = sum(nbits for _, nbits in chunks)
        value = fields[name]
        v = int(value) if scale == 1 and not sc else round(value / (pi if sc else 1.0) / scale)
        v &= (1 << n) - 1
        for start, nbits in reversed(chunks):
            _set(d, start, nbits, v & ((1 << nbits) - 1))
            v >>= nbits


@dataclass(frozen=True)
class Subframe:
    """One parity-checked subframe.

    tow_s: GPS seconds of week at the subframe's first bit (the HOW's TOW count
        refers to the start of the following subframe).
    fields: decoded parameters, named by subframe; subframe 4 page 18 holds the
        ionosphere parameters, other subframe 4/5 pages only `page_sv_id`.
    epoch: tracking epoch index of the first bit, -1 when not known.
    bits: the 300 bits as transmitted.
    """

    prn: int
    sf_id: int
    tow_s: int
    fields: dict
    epoch: int
    bits: np.ndarray


def decode_subframe(bits300: np.ndarray, d29: int, d30: int, prn: int = 0) -> Subframe | None:
    """Decode 300 received bits.

    None when any word fails parity, the ID is not 1..5, or a subframe 4 or 5
    carries a data ID other than 01 (the only one defined, 20.3.3.5.1.1).
    """
    bits300 = np.asarray(bits300, dtype=np.uint8)
    d = np.zeros(SUBFRAME_BITS, dtype=np.uint8)
    for w in range(10):
        word = bits300[WORD_BITS * w : WORD_BITS * (w + 1)]
        if not parity_ok(word, d29, d30):
            return None
        d[WORD_BITS * w : WORD_BITS * w + 24] = data_bits(word, d30)
        d29, d30 = int(word[28]), int(word[29])
    sf_id = _get(d, *_SF_ID)
    if not 1 <= sf_id <= 5:
        return None
    if sf_id in (4, 5) and _get(d, *_DATA_ID) != 1:
        return None
    tow_s = (_get(d, *_TOW_COUNT) * SUBFRAME_S - SUBFRAME_S) % WEEK_S
    fields = _unpack(d, _spec(sf_id, _get(d, *_PAGE[1][0])))
    return Subframe(prn, sf_id, tow_s, fields, -1, bits300)


def decode_how(word1: np.ndarray, word2: np.ndarray, d29: int, d30: int) -> tuple[int, int] | None:
    """(TOW count, subframe ID) from a subframe's TLM and HOW; None unless both pass parity."""
    if not parity_ok(word1, d29, d30) or not parity_ok(word2, int(word1[28]), int(word1[29])):
        return None
    d = np.zeros(2 * WORD_BITS, dtype=np.uint8)
    d[WORD_BITS : WORD_BITS + 24] = data_bits(word2, int(word1[29]))
    return _get(d, *_TOW_COUNT), _get(d, *_SF_ID)


def encode_subframe(
    sf_id: int, tow_count: int, fields: dict, d29: int = 0, d30: int = 0
) -> np.ndarray:
    """300 transmitted bits for a subframe, the inverse of `decode_subframe`.

    TLM and HOW flags are zero. The two non-information bits of the HOW and of
    word 10 are chosen so those words end with D29 = D30 = 0 (Figure 20-2,
    20.3.3.3.1), so the next subframe can be decoded with D29* = D30* = 0.
    """
    d = np.zeros(SUBFRAME_BITS, dtype=np.uint8)
    d[:8] = PREAMBLE
    _set(d, *_TOW_COUNT, tow_count)
    _set(d, *_SF_ID, sf_id)
    if sf_id in (4, 5):
        _set(d, *_DATA_ID, 1)
    _pack(d, _spec(sf_id, fields.get("page_sv_id")), fields)
    out = np.empty(SUBFRAME_BITS, dtype=np.uint8)
    for w in range(10):
        data = d[WORD_BITS * w : WORD_BITS * w + 24]
        if w in (1, 9):
            for free in range(4):
                data[22], data[23] = free >> 1, free & 1
                word = encode_word(data, d29, d30)
                if word[28] == 0 and word[29] == 0:
                    break
        else:
            word = encode_word(data, d29, d30)
        out[WORD_BITS * w : WORD_BITS * (w + 1)] = word
        d29, d30 = int(word[28]), int(word[29])
    return out


@dataclass(frozen=True)
class Ephemeris:
    """Clock and orbit parameters of one satellite from subframes 1, 2 and 3.

    Units: seconds, metres, radians and their rates; `toc` and `toe` are seconds
    of week and `wn` the 10-bit week number. A Galileo record read from RINEX
    uses the same fields (`iode` and `iodc` hold IODnav, `ura` the SISA index).
    """

    prn: int
    wn: int
    code_l2: int
    ura: int
    health: int
    iodc: int
    l2p: int
    tgd: float
    toc: float
    af2: float
    af1: float
    af0: float
    iode: int
    crs: float
    delta_n: float
    m0: float
    cuc: float
    e: float
    cus: float
    sqrt_a: float
    toe: float
    fit: int
    aodo: int
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    system: str = "G"  # "G" GPS, "E" Galileo (same Kepler set; `tgd` is then BGD E5a/E1)
    tgd2: float = 0.0  # s, Galileo BGD E5b/E1


@dataclass(frozen=True)
class Ionosphere:
    """Klobuchar model coefficients from subframe 4 page 18."""

    alpha: tuple[float, float, float, float]
    beta: tuple[float, float, float, float]


def assemble_ephemeris(subframes: list[Subframe]) -> Ephemeris | None:
    """An Ephemeris from the latest subframes 1, 2 and 3 when their issue of data agree."""
    latest = {}
    for sf in subframes:
        if sf.sf_id in (1, 2, 3):
            latest[sf.sf_id] = sf
    if len(latest) < 3:
        return None
    sf1, sf2, sf3 = latest[1], latest[2], latest[3]
    iode = sf2.fields["iode"]
    if iode != sf3.fields["iode"] or iode != sf1.fields["iodc"] & 0xFF:
        return None
    return Ephemeris(
        prn=sf1.prn,
        **sf1.fields,
        **sf2.fields,
        **{k: v for k, v in sf3.fields.items() if k != "iode"},
    )
