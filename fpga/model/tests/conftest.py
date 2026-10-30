import os
from pathlib import Path

import pytest


@pytest.fixture
def captures() -> Path:
    """Directory of recordings named by PYRX_CAPTURES; skip when unset."""
    value = os.environ.get("PYRX_CAPTURES")
    if not value or not Path(value).is_dir():
        pytest.skip("PYRX_CAPTURES not set")
    return Path(value)
