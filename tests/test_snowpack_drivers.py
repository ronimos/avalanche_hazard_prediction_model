"""Checks the xsnow-based driver table against the raw SMET text."""
import numpy as np
import pandas as pd
import pytest

import snowpack_drivers as sd
from conftest import SNOWPACK_FIXTURE_ROOT

STATION = SNOWPACK_FIXTURE_ROOT / "2026" / "zone003" / "100977"


def read_raw_smet(path):
    """Parses a SMET file directly, without xsnow, applying nodata only."""
    lines = path.read_text().splitlines()
    fields = next(line for line in lines if line.startswith("fields")).split("=")[1].split()
    start = lines.index("[DATA]") + 1
    df = pd.DataFrame([line.split() for line in lines[start:]], columns=fields)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").apply(pd.to_numeric).replace(-999, np.nan)


@pytest.fixture(scope="module")
def smet():
    return sd.read_station_smet(STATION)


@pytest.fixture(scope="module")
def daily(smet):
    return sd.daily_drivers(smet).set_index("date")


def test_station_found_by_id_in_any_zone(snowpack_root):
    sd.season_station_dirs.cache_clear()
    assert sd.station_dir(2026, 100977) == snowpack_root / "2026" / "zone003" / "100977"
    assert sd.station_dir(2026, 999999) is None


def test_all_five_streams_map_to_one_station(smet):
    slopes = smet.drop_duplicates("slope").set_index("slope")["inclination"]
    assert slopes.to_dict() == {0: 0, 1: 38, 2: 38, 3: 38, 4: 38}


def test_missing_directory_returns_none(tmp_path):
    assert sd.read_station_smet(tmp_path) is None


def test_days_are_mountain_time(daily):
    # 2026-10-14T12:00Z is 06:00 MDT on Oct 14; 2026-10-31T18:00Z is 12:00 MDT on Oct 31
    assert daily.index.min() == pd.Timestamp("2026-10-14")
    assert daily.index.max() == pd.Timestamp("2026-10-31")


def test_new_snow_matches_raw_file(daily):
    raw = read_raw_smet(STATION / "100977_res.smet")
    raw.index = raw.index.tz_localize("UTC").tz_convert(sd.LOCAL_TZ).tz_localize(None)
    last_hn24 = raw["HN24"].groupby(raw.index.normalize()).last() * 0.01  # cm -> m, as xsnow reports it

    pd.testing.assert_series_equal(daily["HN24"], last_hn24, check_names=False, check_freq=False, atol=1e-5, check_dtype=False)
    assert daily["HN72"].iloc[5] == pytest.approx(daily["HN24"].iloc[3:6].sum())


def test_stability_is_least_stable_slope(daily):
    raws = [read_raw_smet(STATION / f"100977{k}_res.smet") for k in range(1, 5)]
    sn = pd.concat([r["Sn"] for r in raws], axis=1).min(axis=1)
    sn.index = sn.index.tz_localize("UTC").tz_convert(sd.LOCAL_TZ).tz_localize(None)
    expected = sn.groupby(sn.index.normalize()).min()

    pd.testing.assert_series_equal(daily["Sn_min"], expected, check_names=False, check_freq=False, atol=1e-5, check_dtype=False)
