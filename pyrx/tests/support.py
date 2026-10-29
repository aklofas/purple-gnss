"""Shared test helpers: fixed ambiguity sets, the golden-file comparison and the
normalisers of dated or timed output."""

import difflib
import os
import re
from pathlib import Path

import numpy as np

from pyrx.ephemeris import Ionosphere
from pyrx.synth import USUD, ambiguities, as_galileo, constellation

IONO = Ionosphere(
    (1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07),
    (1.1469e05, -1.6384e05, -1.9661e05, 9.1750e05),
)


ROVER = USUD + np.array([5.0, -3.0, 2.0])
SIGNALS = ("G1C", "G2W")
ALL_SIGNALS = ("G1C", "G2W", "G5Q", "E1C", "E5Q")


_RNG = np.random.default_rng(0)
AMB_R = ambiguities(_RNG, constellation() + as_galileo(constellation()), ALL_SIGNALS)
AMB_B = ambiguities(_RNG, constellation() + as_galileo(constellation()), ALL_SIGNALS)


GOLDEN_DIR = Path(__file__).parent / "golden"


def check_golden(name: str, text: str) -> None:
    """Compare `text` with the golden file `tests/golden/<name>` byte for byte.

    Set `PYRX_UPDATE_GOLDENS` to rewrite the file after a deliberate output change.
    """
    path = GOLDEN_DIR / name
    if os.environ.get("PYRX_UPDATE_GOLDENS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    expected = path.read_text()
    if text != expected:
        diff = difflib.unified_diff(
            expected.splitlines(True), text.splitlines(True), f"golden/{name}", "actual"
        )
        raise AssertionError(f"output differs from {name}:\n" + "".join(diff))


def without_date(text: str) -> str:
    """The text with the date of its `PGM / RUN BY / DATE` line replaced."""
    return re.sub(r"\d{8} \d{6} UTC", "<date>", text)


def without_elapsed(text: str) -> str:
    """The text with the value of its `TIME = x.xxx s` lines replaced."""
    return re.sub(r"^TIME = \d+\.\d{3} s$", "TIME = <t> s", text, flags=re.MULTILINE)
