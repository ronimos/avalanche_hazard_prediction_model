import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import hazard_change as hc
import likelihood_data as ld


def _ratings(polygon, start, values):
    """Daily ratings from `start`, with the same value in every band unless a tuple is given."""
    rows = []
    for i, value in enumerate(values):
        atl, ntl, btl = value if isinstance(value, tuple) else (value,) * 3
        rows.append({"date": pd.Timestamp(start) + pd.Timedelta(days=i), "polygon": polygon,
                     "atl": atl, "ntl": ntl, "btl": btl})
    return pd.DataFrame(rows)


def _table(ratings, region="North", season=2023, **drivers):
    """Modeling-table rows for every rated polygon-day, with all drivers at 0 unless given."""
    table = ratings[["polygon", "date"]].assign(season=season, region=region, day_of_season=0,
                                                Sn_capped=0, Ss_capped=0)
    for column in ld.DRIVERS + ld.ACTIVITY_FEATURES:
        table[column] = drivers.get(column, 0.0)
    return table


def test_change_table_uses_consecutive_days_only():
    ratings = pd.concat([_ratings(1, "2024-01-01", [2, 3, 3]), _ratings(1, "2024-01-05", [1])])
    ratings = ratings.astype({"atl": float}).reset_index(drop=True)
    ratings.loc[1, "atl"] = np.nan  # not rated on Jan 2 above treeline

    day = hc.build_change_table(_table(ratings), ratings).set_index("date")

    assert day.loc["2024-01-02", "ntl_prev"] == 2
    assert day.loc["2024-01-02", "ntl_change"] == 1
    assert day.loc["2024-01-03", "ntl_trend"] == 1
    assert day.loc["2024-01-03", "ntl_change"] == 0
    # Jan 4 is missing, so Jan 5 has no rating for t-1
    assert np.isnan(day.loc["2024-01-05", "ntl_prev"])
    # An unrated day breaks the band's chain
    assert np.isnan(day.loc["2024-01-03", "atl_prev"])


def test_load_ratings_treats_zero_as_not_rated(tmp_path, monkeypatch):
    path = tmp_path / "danger_rating.csv"
    _ratings(1, "2024-01-01", [(0, 2, 1), (3, 2, 1)]).to_csv(path, index=False)
    monkeypatch.setitem(hc.config.PATHS["RAW_DATA"], "danger_ratings", path)
    monkeypatch.setitem(hc.config.PATHS["RAW_DATA"], "avid_ratings", tmp_path / "missing.csv")

    ratings = hc.load_ratings()

    assert ratings["atl"].isna().sum() == 1
    assert ratings["ntl"].tolist() == [2, 2]


def test_load_ratings_adds_avid_days_the_file_lacks(tmp_path, monkeypatch):
    path, avid_path = tmp_path / "danger_rating.csv", tmp_path / "avid.csv"
    _ratings(1, "2024-01-01", [2]).to_csv(path, index=False)
    pd.DataFrame({"polygon": [1, 1, 1], "date": ["2024-01-01", "2025-12-01", "2025-12-01"], "position": [1, 1, 2],
                  "issued": "x", "atl": [3, 4, 1], "ntl": [3, 4, 1], "btl": [3, 4, 1]}).to_csv(avid_path, index=False)
    monkeypatch.setitem(hc.config.PATHS["RAW_DATA"], "danger_ratings", path)
    monkeypatch.setitem(hc.config.PATHS["RAW_DATA"], "avid_ratings", avid_path)

    ratings = hc.load_ratings().set_index("date")

    # The ratings file wins where both have a day; outlook positions aren't ratings
    assert ratings.loc["2024-01-01", "atl"] == 2
    assert ratings.loc["2025-12-01", "atl"] == 4
    assert len(ratings) == 2


def test_change_is_clipped_to_one_level():
    table = hc.build_change_table(_table(_ratings(1, "2024-01-01", [1, 3, 1])), _ratings(1, "2024-01-01", [1, 3, 1]))

    assert table["atl_change"].tolist()[1:] == [1, -1]


def test_regional_drivers_are_means_over_the_region():
    ratings = pd.concat([_ratings(1, "2024-01-01", [2]), _ratings(2, "2024-01-01", [2]), _ratings(3, "2024-01-01", [2])])
    table = _table(ratings)
    table["HN24"] = [0.1, 0.3, 0.9]
    table["region"] = ["North", "North", "South"]

    out = hc.build_change_table(table, ratings).set_index("polygon")

    assert out.loc[1, "HN24_region"] == pytest.approx(0.2)
    assert out.loc[3, "HN24_region"] == pytest.approx(0.9)


def test_impossible_changes_are_masked():
    proba = np.full((3, 3), 1 / 3)

    masked = hc._mask_impossible(proba, np.array([1, 2, 5]))

    assert masked[0, 0] == 0 and masked[2, 2] == 0
    np.testing.assert_allclose(masked.sum(axis=1), 1)
    np.testing.assert_allclose(masked[1], 1 / 3)


def test_transition_model_frequencies():
    train = pd.DataFrame({"atl_prev": [2] * 4 + [3], "atl_trend": [0] * 5, "atl_change": [1, 0, 0, 0, -1]})

    model = hc.TransitionModel("atl", smoothing=0).fit(train)
    proba = model.predict_proba(pd.DataFrame({"atl_prev": [2, 1], "atl_trend": [0, 1]}))

    np.testing.assert_allclose(proba[0], [0, 0.75, 0.25])
    # An unseen rating and trend falls back to the overall frequencies, then masks the decrease from Low
    np.testing.assert_allclose(proba[1], np.array([0, 3, 1]) / 4)


def _synthetic(seasons=(2023, 2024), polygons=range(1, 9), days=60, seed=0):
    """Ratings that go up the day after heavy new snow, on otherwise random days."""
    rng = np.random.default_rng(seed)
    ratings, tables = [], []
    for season in seasons:
        for polygon in polygons:
            snow = rng.random(days) < 0.2
            values = [2] + [3 if s else 2 for s in snow[1:]]
            r = _ratings(polygon, f"{season + 1}-01-01", values)
            t = _table(r, season=season)
            t["HN24"] = np.where(snow, 0.4, 0.0) + rng.normal(0, 0.05, days)
            ratings.append(r)
            tables.append(t)
    return hc.build_change_table(pd.concat(tables), pd.concat(ratings))


def test_change_model_learns_increase_from_new_snow():
    df = hc.band_rows(_synthetic(), "atl")
    train, test = df[df["season"] == 2023], df[df["season"] == 2024]

    proba = hc.ChangeModel("atl", C=1.0).fit(train).predict_proba(test)

    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert roc_auc_score(test["atl_change"] == 1, proba[:, 2]) > 0.9


def test_change_model_handles_a_change_missing_from_training():
    df = hc.band_rows(_synthetic(), "atl")
    train = df[df["atl_change"] >= 0]  # no decreases

    proba = hc.ChangeModel("atl").fit(train).predict_proba(df)

    assert proba.shape == (len(df), 3)
    assert (proba[:, 0] == 0).all()


def test_held_out_season_never_reaches_fit(monkeypatch):
    fitted_on = []

    class Recording:
        def __init__(self, band):
            pass

        def fit(self, df):
            fitted_on.append(set(df["season"]))
            return self

        def predict_proba(self, df):
            return np.tile([0.1, 0.8, 0.1], (len(df), 1))

    monkeypatch.setattr(hc, "MODELS", {"recording": Recording})
    df = _synthetic(seasons=(2023, 2024, 2025))

    predictions = hc.leave_one_season_out(df, "atl")

    assert fitted_on == [{2024, 2025}, {2023, 2025}, {2023, 2024}]
    assert len(predictions) == len(hc.band_rows(df, "atl"))


def test_baseline_skill_is_zero_and_better_model_is_positive():
    predictions = pd.concat([hc.leave_one_season_out(_synthetic(), band) for band in ["atl"]], ignore_index=True)

    scores = hc.scores(predictions).set_index(["season", "model"])

    assert (scores.xs("transition", level="model")["log_loss_skill"] == 0).all()
    assert (scores.xs("change_logistic", level="model")["log_loss_skill"] > 0).all()


def _history(polygon, ratings_by_day, outlook=None):
    """AvID history rows: day-1 ratings per day, plus an optional outlook (position 2) row."""
    rows = [{"polygon": polygon, "date": pd.Timestamp(day), "position": 1, "issued": "x", "atl": v, "ntl": v, "btl": v}
            for day, v in ratings_by_day.items()]
    if outlook:
        day, v = outlook
        rows.append({"polygon": polygon, "date": pd.Timestamp(day), "position": 2, "issued": "x", "atl": v, "ntl": v, "btl": v})
    return pd.DataFrame(rows)


def test_forecast_inputs_use_ratings_before_the_forecast_day_only():
    ratings = pd.DataFrame({"polygon": 1, "date": pd.to_datetime(["2025-12-13", "2025-12-14", "2025-12-15"]),
                            "atl": [2, 3, 4], "ntl": [2, 3, 4], "btl": [2, 3, 4]})
    drivers = _table(_ratings(1, "2025-12-15", [0]))

    rows = hc.forecast_inputs(pd.Timestamp("2025-12-15"), ratings, drivers)

    assert len(rows) == 1
    assert rows["atl_prev"].item() == 3 and rows["atl_trend"].item() == 1
    assert np.isnan(rows["atl"].item())


def _with_outlook(df, seed=1):
    """Adds an outlook that calls 80% of the changes and is otherwise the rating for t-1."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for band in hc.BANDS:
        right = rng.random(len(df)) < 0.8
        df[f"{band}_outlook"] = np.where(right, df[band], df[f"{band}_prev"])
    return df


def test_change_table_joins_outlooks():
    ratings = _ratings(1, "2024-01-01", [2, 3])
    outlooks = pd.DataFrame({"polygon": [1], "date": [pd.Timestamp("2024-01-02")], "atl": [4], "ntl": [3], "btl": [2]})

    day = hc.build_change_table(_table(ratings), ratings, outlooks).set_index("date")

    assert day.loc["2024-01-02", "atl_outlook"] == 4
    assert np.isnan(day.loc["2024-01-01", "atl_outlook"])
    # Without outlooks the columns are there but empty
    assert hc.build_change_table(_table(ratings), ratings)["ntl_outlook"].isna().all()


def test_outlook_model_uses_the_outlook():
    df = hc.band_rows(_with_outlook(_synthetic()), "atl")
    flat = np.tile([0.1, 0.8, 0.1], (len(df), 1))
    train, test = df["season"] == 2023, df["season"] == 2024

    proba = hc.OutlookModel("atl").fit(flat[train], df[train]).predict_proba(flat[test], df[test])

    np.testing.assert_allclose(proba.sum(axis=1), 1)
    up = df.loc[test, "atl_change"] == 1
    assert roc_auc_score(up, proba[:, 2]) > 0.85
    # From Low there is no decrease
    assert (proba[df.loc[test, "atl_prev"].to_numpy() == 1, 0] == 0).all()


def test_outlook_evaluation_holds_out_seasons_and_blocks(monkeypatch):
    df = _with_outlook(_synthetic(seasons=(2023, 2024)))
    df.loc[df["season"] == 2023, [f"{b}_outlook" for b in hc.BANDS]] = np.nan  # outlooks only in 2024
    change_fits, outlook_fits = [], []
    real_change_fit, real_outlook_fit = hc.ChangeModel.fit, hc.OutlookModel.fit

    def change_fit(self, d):
        change_fits.append(set(d["season"]))
        return real_change_fit(self, d)

    def outlook_fit(self, proba, d):
        outlook_fits.append(set(d["date"]))
        return real_outlook_fit(self, proba, d)

    monkeypatch.setattr(hc.ChangeModel, "fit", change_fit)
    monkeypatch.setattr(hc.OutlookModel, "fit", outlook_fit)

    out = hc.evaluate_outlook(df, "atl")

    # The change model for 2024 never sees 2024
    assert change_fits == [{2023}]
    assert set(out["model"]) == {"change_logistic", "outlook_only", "change_with_outlook"}
    assert set(out["season"]) == {2024}
    # Each outlook fit leaves out one 14-day block: 59 days with a rating for t-1 -> 5 blocks, 2 fits per block
    assert len(outlook_fits) == 10
    rows = hc.band_rows(df, "atl")
    days = set(rows.loc[rows["season"] == 2024, "date"])
    assert sorted(len(days - fit) for fit in outlook_fits) == [3, 3] + [14] * 8


def test_predict_gives_probabilities_and_caic_ratings(tmp_path, monkeypatch):
    df = _with_outlook(_synthetic())
    models = {band: hc.ChangeModel(band).fit(hc.band_rows(df, band)) for band in hc.BANDS}
    outlook_models = {}
    for band in hc.BANDS:
        rows = hc.band_rows(df, band)
        # Flat probabilities stand in for held-out ones; in-sample ones are too good to leave room for the outlook
        outlook_models[band] = hc.OutlookModel(band).fit(np.tile([0.1, 0.8, 0.1], (len(rows), 1)), rows)
    path = tmp_path / "hazard_change_model.joblib"
    import joblib
    joblib.dump({"models": models, "outlook_models": outlook_models, "seasons": [2023, 2024]}, path)
    monkeypatch.setitem(hc.config.PATHS["ARTIFACTS"], "hazard_change_model", path)
    date = pd.Timestamp("2025-12-15")
    monkeypatch.setattr(hc.likelihood_forecast, "forecast_table",
                        lambda d: _table(pd.concat([_ratings(1, d, [0]), _ratings(2, d, [0])]), season=2025))
    history = pd.concat([_history(1, {"2025-12-13": 2, "2025-12-14": 2}, outlook=("2025-12-15", 3)),
                         _history(2, {"2025-12-10": 2})])

    out = hc.predict(date, history=history)

    # Polygon 2 wasn't rated yesterday, so it gets no forecast
    assert out["polygon"].tolist() == [1]
    assert list(out.columns) == ["polygon", *hc.output_columns()]
    assert out["atl_caic_today"].item() == 2 and out["atl_caic_outlook"].item() == 3
    for prefix in ["p", "drivers_p"]:
        total = out[[f"atl_{prefix}_{c}" for c in hc.CHANGE_NAMES]].sum(axis=1).item()
        assert total == pytest.approx(1)
    # The outlook calls an increase, so the probability with it is higher
    assert out["atl_p_up"].item() > out["atl_drivers_p_up"].item()


def test_predict_with_an_old_artifact_gives_drivers_only(tmp_path, monkeypatch):
    df = _synthetic()
    models = {band: hc.ChangeModel(band).fit(hc.band_rows(df, band)) for band in hc.BANDS}
    path = tmp_path / "hazard_change_model.joblib"
    import joblib
    joblib.dump({"models": models, "seasons": [2023, 2024]}, path)
    monkeypatch.setitem(hc.config.PATHS["ARTIFACTS"], "hazard_change_model", path)
    monkeypatch.setattr(hc.likelihood_forecast, "forecast_table", lambda d: _table(_ratings(1, d, [0]), season=2025))

    out = hc.predict(pd.Timestamp("2025-12-15"), history=_history(1, {"2025-12-14": 3}))

    assert np.isnan(out["atl_p_up"].item()) and out["atl_drivers_p_up"].notna().item()


def test_predict_without_ratings_returns_empty_columns(tmp_path, monkeypatch):
    import joblib
    path = tmp_path / "model.joblib"
    joblib.dump({"models": {}, "seasons": []}, path)
    monkeypatch.setitem(hc.config.PATHS["ARTIFACTS"], "hazard_change_model", path)
    monkeypatch.setattr(hc.likelihood_forecast, "forecast_table", lambda d: _table(_ratings(1, d, [0])))

    out = hc.predict(pd.Timestamp("2025-10-30"), history=_history(1, {}).reindex(columns=hc.caic_ratings.COLUMNS))

    assert out.empty and list(out.columns) == ["polygon", *hc.output_columns()]
