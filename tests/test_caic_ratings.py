import json

import numpy as np
import pandas as pd
import pytest

import caic_ratings as cr
import config


def _forecast(polygons, issued, days):
    """An AvID forecast; `days` maps position -> (date, (alp, tln, btl) words)."""
    return {"issueDateTime": issued, "polygons": polygons, "title": "Zone",
            "dangerRatings": {"days": [{"position": pos, "date": f"{date}T23:30:00Z", "alp": r[0], "tln": r[1], "btl": r[2]}
                                       for pos, (date, r) in days.items()]}}


NUMBERS = {"aaa-1": 101, "bbb-2": 102}


def test_ratings_map_polygons_levels_and_positions():
    forecasts = [_forecast(["AAA-1", "zzz-old"], "2025-12-14T23:30:00Z",
                           {1: ("2025-12-15", ("considerable", "moderate", "noRating")),
                            2: ("2025-12-16", ("high", "considerable", "low"))})]

    ratings = cr.ratings_from_forecasts(forecasts, NUMBERS)

    assert ratings["polygon"].unique().tolist() == [101]  # case-insensitive; unknown IDs skipped
    day1 = ratings[ratings["position"] == 1].iloc[0]
    assert (day1["date"], day1["atl"], day1["ntl"]) == (pd.Timestamp("2025-12-15"), 3, 2)
    assert np.isnan(day1["btl"])
    assert ratings.loc[ratings["position"] == 2, "atl"].item() == 4


def test_latest_issued_forecast_wins():
    forecasts = [_forecast(["aaa-1"], "2025-12-15T13:00:00Z", {1: ("2025-12-15", ("high", "high", "high"))}),
                 _forecast(["aaa-1"], "2025-12-14T23:30:00Z", {1: ("2025-12-15", ("low", "low", "low"))})]

    ratings = cr.ratings_from_forecasts(forecasts, NUMBERS)

    assert len(ratings) == 1 and ratings["atl"].item() == 4


def test_day_ratings_keep_day_one_only():
    forecasts = [_forecast(["aaa-1"], "2025-12-14T23:30:00Z",
                           {1: ("2025-12-15", ("low",) * 3), 2: ("2025-12-16", ("high",) * 3)})]

    days = cr.day_ratings(cr.ratings_from_forecasts(forecasts, NUMBERS))

    assert days["date"].tolist() == [pd.Timestamp("2025-12-15")]
    assert list(days.columns) == ["polygon", "date", *cr.BANDS]


def test_fetch_current_adds_to_history(tmp_path, monkeypatch):
    history_path = tmp_path / "history.csv"
    monkeypatch.setattr(cr, "polygon_numbers", lambda: NUMBERS)
    pulls = iter([
        [_forecast(["aaa-1"], "2025-12-13T23:30:00Z", {1: ("2025-12-14", ("low",) * 3)})],
        [_forecast(["aaa-1", "bbb-2"], "2025-12-14T23:30:00Z", {1: ("2025-12-15", ("moderate",) * 3)})],
    ])
    monkeypatch.setattr(cr, "fetch_forecasts", lambda: next(pulls))

    cr.fetch_current(history_path)
    history = cr.fetch_current(history_path)

    assert len(history) == 3
    assert cr.load_history(history_path).equals(history)


def test_fetch_needs_url_and_key(monkeypatch):
    monkeypatch.setitem(config.AVID_CONFIG, "api_key", None)

    with pytest.raises(RuntimeError, match="AVID_API_BASE"):
        cr.fetch_forecasts()


def test_import_writes_all_positions(tmp_path, monkeypatch):
    path, out = tmp_path / "forecasts.json", tmp_path / "avid.csv"
    path.write_text(json.dumps([_forecast(["aaa-1"], "2025-12-14T23:30:00Z",
                                          {1: ("2025-12-15", ("low",) * 3), 2: ("2025-12-16", ("low",) * 3)})]))
    monkeypatch.setattr(cr, "polygon_numbers", lambda: NUMBERS)

    cr.import_forecasts(path, out)

    assert pd.read_csv(out)["position"].tolist() == [1, 2]


def test_dotenv_loader_keeps_existing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nAVID_TEST_A="from file"\nexport AVID_TEST_B=b\nAVID_TEST_C=c\n')
    monkeypatch.delenv("AVID_TEST_A", raising=False)
    monkeypatch.delenv("AVID_TEST_B", raising=False)
    monkeypatch.setenv("AVID_TEST_C", "from environment")

    config._load_dotenv(env)

    import os
    assert os.environ["AVID_TEST_A"] == "from file"
    assert os.environ["AVID_TEST_B"] == "b"
    assert os.environ["AVID_TEST_C"] == "from environment"
    for key in ("AVID_TEST_A", "AVID_TEST_B"):
        monkeypatch.delenv(key)
