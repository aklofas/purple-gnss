import os
from pathlib import Path

import pytest


def skip_or_fail(reason: str) -> None:
    """Skip a test that lacks the recording, or fail it when PYRX_REQUIRE_CAPTURES is set: a
    gate run must not pass by skipping the tests that need the recording."""
    if os.environ.get("PYRX_REQUIRE_CAPTURES"):
        pytest.fail(f"{reason} (PYRX_REQUIRE_CAPTURES is set)")
    pytest.skip(reason)


def capture_file(captures: Path, name: str) -> Path:
    """Path of a recording in the captures directory; skip the test if it is missing."""
    path = captures / name
    if not path.is_file():
        skip_or_fail(f"recording {name} not found in {captures}")
    return path


@pytest.fixture
def captures() -> Path:
    """Directory of recordings named by PYRX_CAPTURES; skip when unset."""
    value = os.environ.get("PYRX_CAPTURES")
    if not value or not Path(value).is_dir():
        skip_or_fail("PYRX_CAPTURES not set")
    return Path(value)
