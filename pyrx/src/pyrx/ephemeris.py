"""The broadcast ephemeris and ionosphere records and the URA accuracy scale, shared by the
navigation decode, the RINEX files and the positioning. Imports nothing from either."""

from dataclasses import dataclass

# upper bounds of the URA index intervals (IS-GPS-200 20.3.3.3.1.3); index 15 is "no accuracy"
URA_BOUNDS = (
    2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0, 3072.0, 6144.0,
)  # fmt: skip


def ura_metres(index: int) -> float:
    """SV accuracy in meters from the 4-bit URA index (IS-GPS-200 20.3.3.3.1.3)."""
    if index <= 6:
        return 2.0 ** (1 + index / 2)
    if index < 15:
        return 2.0 ** (index - 2)
    return 6144.0


def ura_index(metres: float) -> int:
    """4-bit URA index whose interval holds an SV accuracy in meters; 15 above the table."""
    for index, bound in enumerate(URA_BOUNDS):
        if metres <= bound:
            return index
    return 15


@dataclass(frozen=True)
class Ephemeris:
    """Clock and orbit parameters of one satellite from subframes 1, 2 and 3.

    Units: seconds, meters, radians and their rates; `toc` and `toe` are seconds
    of week and `wn` the 10-bit week number. A Galileo record read from RINEX
    uses the same fields (`iode` and `iodc` hold IODnav; `ura` is the GPS URA
    index whose interval holds the SISA in meters, not the SISA index).
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
    aodo: float  # s, age of data offset (broadcast in units of 900 s)
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
