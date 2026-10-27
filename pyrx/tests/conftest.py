import os
from pathlib import Path

import pytest


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
