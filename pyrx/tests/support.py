"""Shared test helpers: fixed ambiguity sets, the golden-file comparison and the
normalizers of dated or timed output."""

import dataclasses
import difflib
import os
import re
from pathlib import Path

import numpy as np

from pyrx.codes import CA_CHIP_RATE, CA_PERIOD, gps_ca
from pyrx.ephemeris import Ionosphere
from pyrx.signals import L1_FREQ
from pyrx.synth import USUD, ambiguities, as_galileo, constellation, synth_l1ca

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


def synth_l1ca_ramp(
    prn: int,
    fs: float,
    doppler: float,
    doppler_rate: float,
    code_offset: int,
    n: int,
    cn0_dbhz: float,
    seed: int = 0,
) -> np.ndarray:
    """`n` real float32 samples of a GPS L1 C/A signal at IF 0 whose Doppler ramps.

    The carrier sits at `doppler + doppler_rate * t` Hz and the code rate is scaled to
    match at every sample; random data bits every 20 code periods from `code_offset`;
    noise as `synth_l1ca`'s real case.
    """
    rng = np.random.default_rng(seed)
    code = gps_ca(prn)
    period = round(fs * CA_PERIOD)
    sym = rng.choice([-1.0, 1.0], size=(n - code_offset) // (20 * period) + 2)
    noise_std = np.sqrt(fs / 10 ** (cn0_dbhz / 10.0) / 4.0)
    out = np.empty(n, dtype=np.float32)
    cycles = 0.0  # carrier phase (cycles) and code phase (chips) carried across blocks
    chips = -code_offset * CA_CHIP_RATE * (1.0 + doppler / L1_FREQ) / fs
    block = 1 << 20
    for start in range(0, n, block):
        k = np.arange(start, min(start + block, n))
        fd = doppler + doppler_rate * k / fs
        c = cycles + np.cumsum(fd) / fs
        ch = chips + np.cumsum(CA_CHIP_RATE * (1.0 + fd / L1_FREQ)) / fs
        cycles, chips = c[-1], ch[-1]
        signal = code[np.floor(ch).astype(np.int64) % len(code)] * np.cos(2.0 * np.pi * (c % 1.0))
        signal *= sym[(k - code_offset) // (20 * period) + 1]
        out[k] = signal + noise_std * rng.standard_normal(len(k))
    return out


def synth_l1ca_faded(
    prn: int,
    fs: float,
    doppler: float,
    code_offset: int,
    n: int,
    cn0_dbhz: float,
    fade_from: float,
    fade_len: float,
    fade_db: float,
    seed: int = 0,
) -> np.ndarray:
    """`synth_l1ca`'s I/Q signal with data bits whose signal, not its noise, is `fade_db`
    down for `fade_len` seconds from `fade_from`.

    The noiseless signal comes from the same call at a very high C/N0: the bits are drawn
    after the noise with the same count, so they agree. I/Q at IF 0 because a real signal
    at IF 0 folds its noise onto the in-phase axis: it tracks 6 dB above the stated C/N0
    with almost no Costas phase noise, which is not the signal the hold is for.
    """
    noisy = synth_l1ca(
        prn, fs, 0.0, doppler, code_offset, n, cn0_dbhz, True, seed=seed, data_bits=True
    )
    clean = synth_l1ca(
        prn, fs, 0.0, doppler, code_offset, n, 200.0, True, seed=seed, data_bits=True
    )
    t = np.arange(n) / fs
    fade = (t >= fade_from) & (t < fade_from + fade_len)
    out = noisy.copy()
    out[fade] -= clean[fade] * (1.0 - 10 ** (-fade_db / 20.0))
    return out


def assert_tracks_equal(a, b) -> None:
    """Every field of two TrackResults equal: arrays element for element, scalars by value."""
    assert type(a) is type(b)
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=f.name)
        else:
            assert x == y, f.name
