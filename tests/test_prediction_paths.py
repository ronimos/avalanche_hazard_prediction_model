"""Checks that forecast dates resolve to the right water-year folder (e.g. /ssd/snowpack/fcst/2026/...)."""
from datetime import datetime

import dataset_utils as dsu
import make_prediction_dataset as mpd


def test_october_2026_forecast_reads_2026_folder():
    manifest = mpd.create_prediction_manifest(datetime(2026, 10, 29), save_manifest=False)

    assert set(manifest["water_year"]) == {2026}
    assert manifest["URL"].str.contains("/2026/zone").all()
    assert manifest["polygon"].nunique() == 147


def test_early_october_window_reaches_back_into_previous_water_year():
    manifest = mpd.create_prediction_manifest(datetime(2026, 10, 5), save_manifest=False)

    september = manifest[manifest["date"].dt.month == 9]
    assert not september.empty
    assert set(september["water_year"]) == {2025}
    assert set(manifest.loc[manifest["date"].dt.month == 10, "water_year"]) == {2026}


def test_snowpack_features_from_2026_pro_file(snowpack_root, fixture_polygon, fixture_station):
    df = dsu.process_snowpack_data_for_polygon_and_year(
        fixture_polygon, 2026, fixture_station, start_date="2026-10-19", end_date="2026-10-29"
    )

    assert df is not None
    assert df["date"].min().date().isoformat() == "2026-10-19"
    assert df["date"].max().date().isoformat() == "2026-10-29"
    assert "weak_layer_depth" in df.columns


def test_weather_features_from_2026_smet_file(snowpack_root, fixture_polygon, fixture_station):
    df = dsu.process_weather_data_for_polygon_and_year(fixture_polygon, 2026, fixture_station)

    assert df is not None
    assert df["date"].min().isoformat() == "2026-10-14"
    assert {"wind_speed", "wind_speed_mean_3d"} <= set(df.columns)


def test_missing_year_folder_returns_none(snowpack_root, fixture_polygon, fixture_station, caplog):
    df = dsu.process_snowpack_data_for_polygon_and_year(fixture_polygon, 2025, fixture_station)

    assert df is None
    assert "2025/zone003/100977/100977_res.pro" in caplog.text
