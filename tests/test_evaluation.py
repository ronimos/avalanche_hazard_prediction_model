import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import evaluation as ev
import likelihood_data as ld


def _drivers(polygons=(1, 2), seasons=(2023, 2024, 2025), days=("11-15", "01-10", "06-01")):
    rows = []
    for season in seasons:
        for polygon in polygons:
            for md in days:
                year = season if md.startswith(("10", "11", "12")) else season + 1
                rows.append({"polygon": polygon, "date": f"{year}-{md}", "season": season,
                             **{d: 1.0 for d in ld.DRIVERS}})
    return pd.DataFrame(rows)


def test_modeling_table_labels_and_season_window():
    drivers = _drivers()
    drivers.loc[drivers.index[0], "Ss_min"] = 6.0
    activity = pd.DataFrame({"polygon": [1, 2], "date": ["2023-11-15", "2024-01-10"],
                             "d2plus_avalanches": [2, 0], "natural_avalanches": [0, 1], "n_avalanches": [2, 1]})

    table = ld.build_modeling_table(drivers, activity, label="d2plus")

    assert not (table["date"].dt.month == 6).any()
    assert table["y"].sum() == 1
    assert table.loc[(table["polygon"] == 1) & (table["date"] == "2023-11-15"), "y"].item() == 1
    assert table.loc[(table["polygon"] == 1) & (table["date"] == "2023-11-15"), "day_of_season"].item() == 45
    assert table["Ss_capped"].sum() == 1
    assert ld.build_modeling_table(drivers, activity, label="natural")["y"].sum() == 1


class RecordingModel:
    """Stub model that remembers which seasons it was fit on."""
    fitted_on = []

    def fit(self, df):
        RecordingModel.fitted_on.append(set(df["season"]))
        return self

    def predict(self, df):
        return np.full(len(df), 0.5)


def test_held_out_season_never_reaches_fit():
    RecordingModel.fitted_on = []
    table = ld.build_modeling_table(_drivers(), pd.DataFrame(columns=["polygon", "date", "d2plus_avalanches"]))

    predictions = ev.leave_one_season_out(table, RecordingModel)

    assert RecordingModel.fitted_on == [{2024, 2025}, {2023, 2025}, {2023, 2024}]
    assert len(predictions) == len(table)
    assert set(predictions["season"]) == {2023, 2024, 2025}


def test_polygon_groups_come_from_training_seasons_only():
    table = ld.build_modeling_table(_drivers(), pd.DataFrame(columns=["polygon", "date", "d2plus_avalanches"]))
    table.loc[(table["polygon"] == 1) & (table["season"] == 2025), "y"] = 1  # events only in 2025

    predictions = ev.leave_one_season_out(table, RecordingModel)
    held_out_2025 = predictions[predictions["season"] == 2025].drop_duplicates("polygon").set_index("polygon")

    assert held_out_2025.loc[1, "polygon_group"] == "none"


def test_within_polygon_auc_ignores_differences_between_polygons():
    # Polygon 1 has high scores and polygon 2 low ones, but each ranks its own days perfectly
    df = pd.DataFrame({"polygon": [1, 1, 1, 2, 2, 2], "y": [0, 0, 1, 0, 1, 1],
                       "p": [0.80, 0.85, 0.90, 0.01, 0.02, 0.03]})

    assert ev.within_polygon_auc(df) == pytest.approx(1.0)


def test_baselines_fit_and_predict_probabilities():
    activity = pd.DataFrame({"polygon": [1], "date": ["2024-01-09"], "d2plus_avalanches": [1], "n_avalanches": [2]})
    regions = pd.Series({1: "A", 2: "A", 3: "B"})
    table = ld.build_modeling_table(_drivers(polygons=(1, 2, 3)), activity, regions=regions)
    table["y"] = (np.arange(len(table)) % 3 == 0).astype(int)

    for make_model in ev.BASELINES.values():
        p = make_model().fit(table).predict(table)
        assert p.shape == (len(table),)
        assert ((p > 0) & (p < 1)).all()


def _random_predictions(seed=0, polygons=5, days=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-12-01", periods=days)
    df = pd.DataFrame([(p, d) for p in range(polygons) for d in dates], columns=["polygon", "date"])
    df["season"] = 2024
    df["p"] = rng.uniform(size=len(df))
    df["y"] = (rng.uniform(size=len(df)) < df["p"] * 0.3).astype(int)
    return df


def test_fast_aucs_match_sklearn():
    df = _random_predictions()

    assert ev._rank_auc(df["y"].to_numpy(), df["p"].to_numpy()) == pytest.approx(roc_auc_score(df["y"], df["p"]))
    assert ev._fast_within_polygon_auc(df["polygon"].to_numpy(), df["y"].to_numpy(), df["p"].to_numpy()) == pytest.approx(ev.within_polygon_auc(df))


def test_bootstrap_is_paired_and_brackets_the_estimate():
    a = _random_predictions()
    b = a.assign(p=a["p"] ** 2)  # same ranking, different calibration

    out = ev.block_bootstrap({"a": a, "b": b}, reference="a", n_boot=50).set_index(["model", "metric"])

    assert out.loc[("b - a", "auc"), ["estimate", "ci_low", "ci_high"]].abs().max() < 1e-12
    row = out.loc[("a", "brier")]
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"]


def test_recent_activity_uses_only_days_before_the_forecast():
    regions = pd.Series({1: "A", 2: "A", 3: "B"})
    activity = pd.DataFrame({"polygon": [1, 2, 1], "date": ["2025-01-10", "2025-01-10", "2025-01-13"],
                             "d2plus_avalanches": [1, 0, 3], "n_avalanches": [1, 4, 3]})

    by_polygon, by_region = ld.recent_activity(activity, regions, lag=1, window=2)
    poly = by_polygon.set_index(["polygon", "date"])["polygon_recent_any"]
    region = by_region.set_index(["region", "date"])

    assert poly.get((1, pd.Timestamp("2025-01-10")), 0) == 0  # same-day record is not used
    assert poly[(2, pd.Timestamp("2025-01-11"))] == 1  # day after a polygon's last record
    assert poly[(1, pd.Timestamp("2025-01-11"))] == 1  # window Jan 9-10
    assert poly[(1, pd.Timestamp("2025-01-12"))] == 1  # window Jan 10-11
    assert poly[(1, pd.Timestamp("2025-01-13"))] == 0  # window Jan 11-12
    assert region.loc[("A", pd.Timestamp("2025-01-11")), "region_recent_any"] == pytest.approx(np.log1p(5))
    assert region.loc[("A", pd.Timestamp("2025-01-11")), "region_recent_d2plus"] == pytest.approx(np.log1p(1))
    assert ("B", pd.Timestamp("2025-01-11")) not in region.index


def test_activity_lag_shifts_the_window():
    regions = pd.Series({1: "A"})
    activity = pd.DataFrame({"polygon": [1], "date": ["2025-01-10"], "d2plus_avalanches": [1], "n_avalanches": [1]})

    by_polygon, _ = ld.recent_activity(activity, regions, lag=2, window=2)
    poly = by_polygon.set_index("date")["polygon_recent_any"]

    assert poly.get(pd.Timestamp("2025-01-11"), 0) == 0  # window Jan 8-9
    assert poly[pd.Timestamp("2025-01-12")] == 1         # window Jan 9-10
