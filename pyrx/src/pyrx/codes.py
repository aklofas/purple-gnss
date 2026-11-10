"""PRN code generation and resampling.

GPS L1 C/A codes follow IS-GPS-200: two 10-stage LFSRs (G1, G2), the output chip
being G1 XOR a delayed G2. GPS L5 codes follow IS-GPS-705: the product of XA, a
13-stage register short-cycled at 8190 chips, and XB, a 13-stage register that runs
its natural 8191-chip period from a start state per PRN and component, both restarted
with the 10230-chip (1 ms) code. Galileo E5a codes follow the Open Service ICD: two
14-stage registers, the second started from a value per PRN and component, truncated
to 10230 chips. The secondary codes multiply one chip onto each period of a primary
code: the Neuman-Hofman codes on L5 (10 chips on I5, 20 on Q5), CS20 on E5a-I and one
100-chip code per satellite on E5a-Q. Chips are returned as +1 for chip value 0 and -1
for chip value 1, so a code multiplies directly onto a signal.
"""

import numpy as np

CA_CHIP_RATE = 1.023e6
CA_PERIOD = 1e-3
CA_LENGTH = 1023

# G2 delay in chips per PRN 1..32, IS-GPS-200 Table 3-Ia.
_CA_G2_DELAY = (
    5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252, 254, 255, 256, 257, 258,
    469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860, 861, 862,
)  # fmt: skip

_ca_cache: dict[int, np.ndarray] = {}


def _lfsr(
    taps: tuple[int, ...], length: int, stages: int = 10, state: list[int] | None = None
) -> np.ndarray:
    """Output (chips 0/1) of a shift register seeded with all ones, or with `state` (stage 1
    first); `taps` are 1-based stage numbers and the output is the last stage."""
    state = [1] * stages if state is None else list(state)
    out = np.empty(length, dtype=np.int8)
    for i in range(length):
        out[i] = state[-1]
        feedback = 0
        for t in taps:
            feedback ^= state[t - 1]
        state = [feedback] + state[:-1]
    return out


def gps_ca(prn: int) -> np.ndarray:
    """One period (1023 chips) of the GPS L1 C/A code for `prn` in 1..32, as int8 +1/-1."""
    if not 1 <= prn <= len(_CA_G2_DELAY):
        raise ValueError(f"PRN {prn} outside 1..{len(_CA_G2_DELAY)}")
    if prn not in _ca_cache:
        g1 = _lfsr((3, 10), CA_LENGTH)
        g2 = _lfsr((2, 3, 6, 8, 9, 10), CA_LENGTH)
        chips = g1 ^ np.roll(g2, _CA_G2_DELAY[prn - 1])
        _ca_cache[prn] = (1 - 2 * chips).astype(np.int8)
    return _ca_cache[prn]


L5_CHIP_RATE = 10.23e6
L5_LENGTH = 10230
_L5_XA_LENGTH = 8190
_L5_XB_LENGTH = 8191

# XB code advance in chips per PRN 1..32, IS-GPS-705 Table 3-Ia
_L5I_XB_ADVANCE = (
    266, 365, 804, 1138, 1509, 1559, 1756, 2084, 2170, 2303, 2527, 2687, 2930, 3471, 3940, 4132,
    4332, 4924, 5343, 5443, 5641, 5816, 5898, 5918, 5955, 6243, 6345, 6477, 6518, 6875, 7168, 7187,
)  # fmt: skip
_L5Q_XB_ADVANCE = (
    1701, 323, 5292, 2020, 5429, 7136, 1041, 5947, 4315, 148, 535, 1939, 5206, 5910, 3595, 5135,
    6082, 6990, 3546, 1523, 4548, 4484, 1893, 3961, 7106, 5299, 4660, 276, 4389, 3783, 1591, 1601,
)  # fmt: skip


def _signs(bits: str) -> np.ndarray:
    return np.array([1 - 2 * int(b) for b in bits], dtype=np.int8)


NH10 = _signs("0000110101")  # on every I5 code period, IS-GPS-705 3.3.3.1.2
NH20 = _signs("00000100110101001110")  # on every Q5 code period, 3.3.2.3

_register_cache: dict[str, np.ndarray] = {}
_code_cache: dict[tuple[str, int], np.ndarray] = {}


def _gps_l5(component: str, advances: tuple[int, ...], prn: int) -> np.ndarray:
    if not 1 <= prn <= len(advances):
        raise ValueError(f"PRN {prn} outside 1..{len(advances)}")
    key = (component, prn)
    if key not in _code_cache:
        if "XA" not in _register_cache:
            _register_cache["XA"] = _lfsr((9, 10, 12, 13), _L5_XA_LENGTH, 13)
            _register_cache["XB"] = _lfsr((1, 3, 4, 6, 7, 8, 12, 13), _L5_XB_LENGTH, 13)
        xa, xb = _register_cache["XA"], _register_cache["XB"]
        i = np.arange(L5_LENGTH)
        # XA restarts after 8190 chips; XB, advanced to the PRN's start state, restarts at
        # its natural completion after 8191; both are reset with the 10230-chip period
        chips = xa[i % _L5_XA_LENGTH] ^ xb[(i + advances[prn - 1]) % _L5_XB_LENGTH]
        _code_cache[key] = (1 - 2 * chips).astype(np.int8)
    return _code_cache[key]


def gps_l5i(prn: int) -> np.ndarray:
    """One period (10230 chips) of the GPS L5 I5 (data) code for `prn` in 1..32."""
    return _gps_l5("L5I", _L5I_XB_ADVANCE, prn)


def gps_l5q(prn: int) -> np.ndarray:
    """One period (10230 chips) of the GPS L5 Q5 (pilot) code for `prn` in 1..32."""
    return _gps_l5("L5Q", _L5Q_XB_ADVANCE, prn)


# Base register 2 start values (octal) per code number 1..36, Galileo OS SIS ICD Tables 16, 17
_E5AI_START = (
    0o30305, 0o14234, 0o27213, 0o20577, 0o23312, 0o33463, 0o15614, 0o12537, 0o01527, 0o30236,
    0o27344, 0o07272, 0o36377, 0o17046, 0o06434, 0o15405, 0o24252, 0o11631, 0o24776, 0o00630,
    0o11560, 0o17272, 0o27445, 0o31702, 0o13012, 0o14401, 0o34727, 0o22627, 0o30623, 0o27256,
    0o01520, 0o14211, 0o31465, 0o22164, 0o33516, 0o02737,
)  # fmt: skip
_E5AQ_START = (
    0o25652, 0o05142, 0o24723, 0o31751, 0o27366, 0o24660, 0o33655, 0o27450, 0o07626, 0o01705,
    0o12717, 0o32122, 0o16075, 0o16644, 0o37556, 0o02477, 0o02265, 0o06430, 0o25046, 0o12735,
    0o04262, 0o11230, 0o00037, 0o06137, 0o04312, 0o20606, 0o11162, 0o22252, 0o30533, 0o24614,
    0o07767, 0o32705, 0o05052, 0o27553, 0o03711, 0o02041,
)  # fmt: skip
_E5A_TAPS_1 = 0o40503  # feedback taps of base register 1, ICD Table 15
_E5A_TAPS_2 = 0o50661
_E5_STAGES = 14


def _octal_taps(octal: int) -> tuple[int, ...]:
    """1-based stage numbers of the feedback taps: bit j of the octal value is tap a_j."""
    return tuple(j for j in range(1, _E5_STAGES + 1) if octal >> j & 1)


def _gal_e5a(component: str, starts: tuple[int, ...], prn: int) -> np.ndarray:
    if not 1 <= prn <= len(starts):
        raise ValueError(f"PRN {prn} outside 1..{len(starts)}")
    key = (component, prn)
    if key not in _code_cache:
        if "E5a1" not in _register_cache:
            _register_cache["E5a1"] = _lfsr(_octal_taps(_E5A_TAPS_1), L5_LENGTH, _E5_STAGES)
        # bit j - 1 of the start value is the start of stage j
        state = [starts[prn - 1] >> j & 1 for j in range(_E5_STAGES)]
        register_2 = _lfsr(_octal_taps(_E5A_TAPS_2), L5_LENGTH, _E5_STAGES, state)
        _code_cache[key] = (1 - 2 * (_register_cache["E5a1"] ^ register_2)).astype(np.int8)
    return _code_cache[key]


def gal_e5ai(prn: int) -> np.ndarray:
    """One period (10230 chips) of the Galileo E5a-I (data) code number `prn` in 1..36."""
    return _gal_e5a("E5aI", _E5AI_START, prn)


def gal_e5aq(prn: int) -> np.ndarray:
    """One period (10230 chips) of the Galileo E5a-Q (pilot) code number `prn` in 1..36."""
    return _gal_e5a("E5aQ", _E5AQ_START, prn)


def _hex_signs(digits: str, length: int) -> np.ndarray:
    return _signs(f"{int(digits, 16):0{4 * len(digits)}b}"[:length])


GAL_CS20 = _hex_signs("842E9", 20)  # the secondary code of every E5a-I, ICD Table 20

# CS100_1..36, the secondary codes of E5a-Q by code number, ICD Table 20
_CS100 = (
    "83F6F69D8F6E15411FB8C9B1C", "66558BD3CE0C7792E83350525", "59A025A9C1AF0651B779A8381",
    "D3A32640782F7B18E4DF754B7", "B91FCAD7760C218FA59348A93", "BAC77E933A779140F094FBF98",
    "537785DE280927C6B58BA6776", "EFCAB4B65F38531ECA22257E2", "79F8CAE838475EA5584BEFC9B",
    "CA5170FEA3A810EC606B66494", "1FC32410652A2C49BD845E567", "FE0A9A7AFDAC44E42CB95D261",
    "B03062DC2B71995D5AD8B7DBE", "F6C398993F598E2DF4235D3D5", "1BB2FB8B5BF24395C2EF3C5A1",
    "2F920687D238CC7046EF6AFC9", "34163886FC4ED7F2A92EFDBB8", "66A872CE47833FB2DFD5625AD",
    "99D5A70162C920A4BB9DE1CA8", "81D71BD6E069A7ACCBEDC66CA", "A654524074A9E6780DB9D3EC6",
    "C3396A101BEDAF623CFC5BB37", "C3D4AB211DF36F2111F2141CD", "3DFF25EAE761739265AF145C1",
    "994909E0757D70CDE389102B5", "B938535522D119F40C25FDAEC", "C71AB549C0491537026B390B7",
    "0CDB8C9E7B53F55F5B0A0597B", "61C5FA252F1AF81144766494F", "626027778FD3C6BB4BAA7A59D",
    "E745412FF53DEBD03F1C9A633", "3592AC083F3175FA724639098", "52284D941C3DCAF2721DDB1FD",
    "73B3D8F0AD55DF4FE814ED890", "94BF16C83BD7462F6498E0282", "A8C3DE1AC668089B0B45B3579",
)  # fmt: skip


def gal_cs100(prn: int) -> np.ndarray:
    """The 100-chip secondary code of the Galileo E5a-Q code number `prn` in 1..36."""
    if not 1 <= prn <= len(_CS100):
        raise ValueError(f"PRN {prn} outside 1..{len(_CS100)}")
    return _hex_signs(_CS100[prn - 1], 100)


def sample_code(
    code: np.ndarray, chip_rate: float, fs: float, n: int, phase_chips: float = 0.0, start: int = 0
) -> np.ndarray:
    """`n` samples of `code` at sample rate `fs` from sample `start`, sample k holding chip
    floor(phase + (start + k)*rate/fs)."""
    chip_index = np.floor(
        phase_chips + np.arange(start, start + n, dtype=np.float64) * (chip_rate / fs)
    )
    return code[chip_index.astype(np.int64) % len(code)].astype(np.float32)
