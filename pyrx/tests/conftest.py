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
