"""Checks the occupancy-detection model on simulated data with a known driver effect."""
import numpy as np
import pandas as pd
import pytest

import bayes_model as bm
import likelihood_data as ld


def test_weekend_and_holidays():
    dates = pd.Series(pd.to_datetime(["2025-12-26", "2026-01-05", "2026-01-03", "2026-01-19", "2026-01-20"]))
    # Fri in winter break, an ordinary Monday, Saturday, MLK Day, ordinary Tuesday
    assert bm.weekend_or_holiday(dates).tolist() == [1, 0, 1, 1, 0]


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(1)
    polygons = ld.polygon_regions().index[:20]
    dates = pd.date_range("2024-12-01", "2025-03-31")
    df = pd.DataFrame([(p, d) for p in polygons for d in dates], columns=["polygon", "date"])
    df["season"] = 2024
    df["day_of_season"] = (df["date"] - pd.Timestamp("2024-10-01")).dt.days
    for name in ld.DRIVERS:
        df[name] = rng.normal(size=len(df))
    df["Sn_capped"] = df["Ss_capped"] = 0
    # Only new snow drives occurrence; a fixed 40% of avalanches get reported
    psi = 1 / (1 + np.exp(-(-2.0 + 1.5 * df["HN24"])))
    df["y"] = (rng.uniform(size=len(df)) < psi * 0.4).astype(int)
    return df


@pytest.mark.slow
def test_recovers_driver_effect_and_predicts_probabilities(simulated):
    model = bm.OccupancyDetectionModel(draws=200, tune=300, chains=2, n_predict_draws=100).fit(simulated)

    effects = model.driver_effects()["mean"]
    assert effects["drivers__HN24"] > 0.8
    assert effects["drivers__HN24"] == effects.abs().max()

    p = model.predict(simulated)
    assert p.shape == (len(simulated),)
    assert ((p > 0) & (p < 1)).all()
    assert (model.last_occurrence >= p).all()
