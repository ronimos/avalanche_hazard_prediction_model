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
    """Stub with the parts of OccupancyDetectionModel that `predict` uses."""
    _polygon_index = pd.Index([1, 2, 3])

    def predict_interval(self, df, quantiles):
        n = len(df)
        return pd.DataFrame({"p_record": np.full(n, 0.05), "p_record_q10": np.full(n, 0.02),
                             "p_record_q90": np.full(n, 0.09), "p_occurrence": np.full(n, 0.3)}, index=df.index)


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


def test_predict_returns_record_probability_and_interval(monkeypatch, tmp_path):
    path = tmp_path / "likelihood_model.joblib"
    joblib.dump(FixedModel(), path)
    monkeypatch.setitem(config.PATHS["ARTIFACTS"], "likelihood_model", path)
    monkeypatch.setattr(lf.sd, "build_driver_table", lambda seasons: _season_drivers(last_day=("2026-12-16", "2026-12-16")))

    out = lf.predict(pd.Timestamp("2026-12-15"))

    assert list(out.columns) == ["polygon", *lf.OUTPUT_COLUMNS.values()]
    assert out["polygon"].tolist() == [1, 2]
    assert (out["dry_slab_probability_q10"] < out["dry_slab_probability"]).all()
    assert (out["dry_slab_probability"] < out["dry_slab_probability_q90"]).all()
