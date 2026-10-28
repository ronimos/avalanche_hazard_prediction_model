"""Checks that cached training inputs are refreshed when a new water year is added to target_years."""
import json

import pandas as pd
import pytest

import config
import dataset_utils as dsu


@pytest.fixture
def target_years(monkeypatch):
    monkeypatch.setitem(config.PROCESSING_CONFIG, "target_years", [2023, 2024, 2025])


@pytest.mark.parametrize("newest_observation, covered", [
    ("2025-04-01T16:20:00Z", False),  # Last season's cache: ends in water year 2024
    ("2025-11-15T12:00:00Z", True),   # Reaches water year 2025
])
def test_avalanche_records_cover_target_years(target_years, newest_observation, covered):
    records = [{"observed_at": "2023-12-01T00:00:00Z"}, {"observed_at": newest_observation}]

    assert dsu.avalanche_records_cover_target_years(records) is covered


def test_avalanche_records_without_dates_are_not_covered(target_years):
    assert not dsu.avalanche_records_cover_target_years([{"observed_at": None}])


class FakeSession:
    """Stands in for requests.Session: serves one page of records, then an empty page."""
    pages = [[{"observed_at": "2026-01-10T12:00:00Z"}], []]

    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        page = self.pages[min(self.calls, len(self.pages) - 1)]
        self.calls += 1
        response = type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: page})
        return response()


def test_stale_avalanche_cache_is_refetched(target_years, tmp_path, monkeypatch):
    cache = tmp_path / "avalanche_records.json"
    cache.write_text(json.dumps([{"observed_at": "2025-04-01T16:20:00Z"}]))
    monkeypatch.setattr(dsu.requests, "Session", FakeSession)
    monkeypatch.setattr(dsu.time, "sleep", lambda _: None)

    records = dsu.load_or_fetch_avalanche_data(cache)

    assert records == [{"observed_at": "2026-01-10T12:00:00Z"}]
    assert json.loads(cache.read_text()) == records


def test_current_avalanche_cache_is_not_refetched(target_years, tmp_path, monkeypatch):
    cached = [{"observed_at": "2026-01-10T12:00:00Z"}]
    cache = tmp_path / "avalanche_records.json"
    cache.write_text(json.dumps(cached))
    monkeypatch.setattr(dsu.requests, "Session", lambda: pytest.fail("should not hit the API"))

    assert dsu.load_or_fetch_avalanche_data(cache) == cached


def test_download_manifest_rebuilt_when_target_years_change(target_years, tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"water_year": [2023, 2024], "stale": [1, 1]}).to_csv(manifest_path, index=False)
    monkeypatch.setitem(config.PATHS["PROCESSED_DATA"], "download_manifest", manifest_path)
    # Danger ratings live in the git-ignored data/external, so supply a minimal table
    danger_path = tmp_path / "danger_ratings.csv"
    pd.DataFrame({"date": ["2024-01-15", "2025-12-15"], "polygon": [2, 2], "atl": [3, 2], "ntl": [2, 2], "btl": [1, 1]}).to_csv(danger_path, index=False)
    monkeypatch.setitem(config.PATHS["RAW_DATA"], "danger_ratings", danger_path)

    manifest = dsu.create_download_manifest()

    assert "stale" not in manifest.columns
    assert set(manifest["water_year"]) == {2023, 2025}


def test_download_manifest_reused_when_target_years_match(target_years, tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"water_year": [2023, 2024, 2025], "cached": [1, 1, 1]}).to_csv(manifest_path, index=False)
    monkeypatch.setitem(config.PATHS["PROCESSED_DATA"], "download_manifest", manifest_path)

    assert "cached" in dsu.create_download_manifest().columns
