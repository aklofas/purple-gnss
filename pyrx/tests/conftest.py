import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    # the reference-recording tests take up to a minute each: scheduled first so that a
    # parallel run is not left waiting on one of them
    items.sort(key=lambda item: not item.path.stem.endswith("_reference"))


def capture_file(captures: Path, name: str) -> Path:
    """Path of a recording in the captures directory; skip the test if it is missing."""
    path = captures / name
    if not path.is_file():
        pytest.skip(f"recording {name} not found in {captures}")
    return path


@pytest.fixture
def captures() -> Path:
    """Directory of recordings named by PYRX_CAPTURES; skip when unset."""
    value = os.environ.get("PYRX_CAPTURES")
    if not value or not Path(value).is_dir():
        pytest.skip("PYRX_CAPTURES not set")
    return Path(value)


@pytest.fixture(scope="session")
def gap_file(tmp_path_factory) -> Path:
    """38.6 s of PRN 4 at 2.5 MHz with the signal replaced by noise from 14.5 s to 24 s.

    Two frames of subframes 1 to 3 (TOW counts 5044.. and 5047..) after a 2 s lead-in,
    then the TLM and HOW words of a third frame (frame sync verifies a subframe start by
    the next subframe's HOW): subframes at 2-8, 8-14, 14-20, 20-26, 26-32 and 32-38 s. The
    gap covers the end of the first frame's subframe 2 and all of subframe 3 and the second
    frame's subframe 1, so a track from the start decodes subframes 1 and 2 only and a
    track from 25 s on decodes subframe 3 only.
    """
    import numpy as np

    from pyrx.synth import frame, synth_l1ca, write_iq_int8

    fs = 2.5e6
    seq = np.concatenate(
        [np.zeros(100, dtype=np.uint8), frame(5044), frame(5047), frame(5050)[:60]]
    )
    n = int(fs * (len(seq) * 0.020 + 0.6))
    x = synth_l1ca(4, fs, 0.0, 800.0, 500, n, 45.0, True, bits=seq)
    a, b = int(fs * 14.5), int(fs * 24.0)
    noise_var = fs / 10 ** (45.0 / 10)
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=np.sqrt(noise_var / 2), size=(b - a, 2))
    x[a:b] = (noise[:, 0] + 1j * noise[:, 1]).astype(np.complex64)
    path = tmp_path_factory.mktemp("gap") / "gap.bin"
    write_iq_int8(x, path)
    return path
