"""Shared pytest fixtures. The src modules use flat imports (`import config`), so src/ goes on sys.path."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import config  # noqa: E402

# A trimmed copy of one 2026 SNOWPACK station (forecast run, 2026-10-14 to 2026-10-31).
SNOWPACK_FIXTURE_ROOT = Path(__file__).parent / "data" / "snowpack"
FIXTURE_PRO = SNOWPACK_FIXTURE_ROOT / "2026" / "zone003" / "100977" / "100977_res.pro"


@pytest.fixture
def snowpack_root(monkeypatch):
    """Points the pipeline's snowpack data root at the test fixture directory."""
    monkeypatch.setitem(config.PATHS["EXTERNAL_DATA"], "snowpack_output", SNOWPACK_FIXTURE_ROOT)
    return SNOWPACK_FIXTURE_ROOT


@pytest.fixture
def fixture_station():
    """Snowpack locations table containing only the fixture station."""
    return pd.DataFrame({"id": [100977], "zone": [3], "lat": [40.74507], "lon": [-107.05675], "elev": [2912]})


@pytest.fixture
def fixture_polygon():
    """A polygon whose centroid sits next to the fixture station."""
    return {"polygon": 1, "centroid_lat": 40.745, "centroid_lon": -107.056}
