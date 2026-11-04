import joblib
import numpy as np
import pandas as pd

import config
import likelihood_data as ld
import likelihood_forecast as lf


def _season_drivers(polygons=(1, 2), last_day=("2026-12-16", "2026-12-15")):
    """Driver table as `build_driver_table` returns it; each polygon's run ends on its own last day."""
    rows = []
    for polygon, last in zip(polygons, last_day):
        for date in pd.date_range("2026-12-13", last):
            rows.append({"polygon": polygon, "date": date.strftime("%Y-%m-%d"), "season": 2026, "station": "000001",
                         **{d: 1.0 for d in ld.DRIVERS}})
    return pd.DataFrame(rows)


class FixedModel:
    """Stub with the parts of OccupancyDetectionModel that `predict` uses: 5 draws, occurrence 0.1 to 0.5."""
    _polygon_index = pd.Index([1, 2, 3])

    def predict_draws(self, df):
        occurrence = np.repeat(np.linspace(0.1, 0.5, 5)[:, None], len(df), axis=1)
        return occurrence * 0.2, occurrence


class FixedShare:
    """Natural share stub: every release natural with probability `share`."""

    def __init__(self, share):
        self.share = share

    def predict_draws(self, df, weekday=True):
        return np.full((5, len(df)), self.share)


def test_season_of_uses_water_year():
    assert lf.season_of(pd.Timestamp("2026-10-31")) == 2026
    assert lf.season_of(pd.Timestamp("2027-03-01")) == 2026


def test_forecast_table_skips_the_runs_unfinished_last_day(monkeypatch):
    monkeypatch.setattr(lf.sd, "build_driver_table", lambda seasons: _season_drivers())

    table = lf.forecast_table(pd.Timestamp("2026-12-15"))

    # Polygon 2's run ends on Dec 15, so that day is incomplete
    assert table["polygon"].tolist() == [1]
    assert table["day_of_season"].item() == 75
    assert table["y"].item() == 0


def test_predict_returns_occurrence_probability_interval_and_record_probability(monkeypatch, tmp_path):
    path = tmp_path / "likelihood_model.joblib"
    joblib.dump(FixedModel(), path)
    monkeypatch.setitem(config.PATHS["ARTIFACTS"], "likelihood_model", path)
    monkeypatch.setitem(config.PATHS["ARTIFACTS"], "natural_share_model", tmp_path / "missing.joblib")
    monkeypatch.setattr(lf.sd, "build_driver_table", lambda seasons: _season_drivers(last_day=("2026-12-16", "2026-12-16")))

    out = lf.predict(pd.Timestamp("2026-12-15"))

    assert list(out.columns) == ["polygon", *lf.OUTPUT_COLUMNS.values(), *lf.SNOW_COLUMNS.values()]
    assert out["polygon"].tolist() == [1, 2]
    assert (out["dry_slab_probability_q10"] < out["dry_slab_probability"]).all()
    assert (out["dry_slab_probability"] < out["dry_slab_probability_q90"]).all()
    # Occurrence, not the lower probability that an avalanche is also recorded
    assert (out["dry_slab_record_probability"] < out["dry_slab_probability"]).all()


def _with_share(monkeypatch, tmp_path, share):
    for key, model in [("likelihood_model", FixedModel()), ("natural_share_model", FixedShare(share))]:
        joblib.dump(model, tmp_path / f"{key}.joblib")
        monkeypatch.setitem(config.PATHS["ARTIFACTS"], key, tmp_path / f"{key}.joblib")
    monkeypatch.setattr(lf.sd, "build_driver_table", lambda seasons: _season_drivers(last_day=("2026-12-16", "2026-12-16")))
    return lf.predict(pd.Timestamp("2026-12-15"))


def test_predict_adds_natural_columns_when_the_share_model_exists(monkeypatch, tmp_path):
    out = _with_share(monkeypatch, tmp_path, 0.6)

    assert list(out.columns) == ["polygon", *lf.OUTPUT_COLUMNS.values(), *lf.SNOW_COLUMNS.values(), *lf.NATURAL_COLUMNS.values()]
    # Natural only is natural or triggered times the share, draw by draw
    for key, column in lf.NATURAL_COLUMNS.items():
        np.testing.assert_allclose(out[column], 0.6 * out[lf.OUTPUT_COLUMNS[key]])


def test_natural_never_exceeds_natural_or_triggered(monkeypatch, tmp_path):
    out = _with_share(monkeypatch, tmp_path, 1.0)

    for key, column in lf.NATURAL_COLUMNS.items():
        np.testing.assert_allclose(out[column], out[lf.OUTPUT_COLUMNS[key]])
