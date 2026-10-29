"""Checks the retraining backtest never trains on days it predicts, or on the most recent days."""
import numpy as np
import pandas as pd

import evaluation as ev
import retraining as rt


def test_retrain_dates():
    weekly = rt.retrain_dates(2024, "weekly")
    assert weekly[0] == pd.Timestamp("2024-12-15")
    assert weekly[1] == pd.Timestamp("2024-12-22")
    assert weekly[-1] <= pd.Timestamp("2025-05-31")
    assert rt.retrain_dates(2024, "monthly")[:3] == list(pd.to_datetime(["2024-12-15", "2025-01-15", "2025-02-15"]))


def test_recalibration_matches_season_level():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=20000) < 0.02).astype(int)
    p = np.full(len(y), 0.10)  # five times too high
    assert abs(rt._recalibrated(p, y, np.array([0.10]))[0] - y.mean()) < 0.003
    assert rt._recalibrated(p, np.zeros_like(y), p[:3]).tolist() == p[:3].tolist()


class StubModel:
    """Remembers what it was trained on and predicts the share of events it saw."""
    fits = []

    def fit(self, df, sample_weight=None):
        StubModel.fits.append((df, sample_weight))
        self.rate = df["y"].mean()
        return self

    def predict(self, df):
        return np.full(len(df), self.rate)


def _table():
    dates = pd.date_range("2023-11-01", "2024-05-31").append(pd.date_range("2024-11-01", "2025-05-31"))
    df = pd.DataFrame({"polygon": 1, "date": dates})
    df["season"] = np.where(df["date"] < "2024-07-01", 2023, 2024)
    df["y"] = (df["date"].dt.day == 1).astype(int)
    return df


def test_refit_uses_only_earlier_days_of_the_current_season(monkeypatch):
    monkeypatch.setattr(ev, "activity_logistic_model", StubModel)
    StubModel.fits = []
    table = _table()

    out = rt.replay_season(table, 2024, "weighted", "monthly")

    assert out["date"].min() == pd.Timestamp("2024-12-15")
    assert len(out) == ((table["season"] == 2024) & (table["date"] >= "2024-12-15")).sum()
    for (train, weight), start in zip(StubModel.fits, rt.retrain_dates(2024, "monthly")):
        current = train[train["season"] == 2024]
        assert current["date"].max() < start - pd.Timedelta(days=rt.GAP_DAYS)
        assert (weight[train["season"].to_numpy() == 2024] == rt.CURRENT_WEIGHT).all()
        assert (train["season"] == 2023).sum() == (table["season"] == 2023).sum()
