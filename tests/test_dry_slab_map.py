import json
import re

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import config
import dry_slab_map as dsm


def _polygons():
    return gpd.GeoDataFrame({"polygon_number": [1, 2], "mountain_range": ["Front Range", None],
                             "referenc_region": ["Front Range", "Aspen"]},
                            geometry=[box(-106, 39, -105.9, 39.1), box(-107, 39, -106.9, 39.1)], crs=4326)


def _predictions(date="2026-12-15"):
    return pd.DataFrame({"date": [date], "polygon": [1], "predicted_hazard": [2],
                         "dry_slab_probability": [0.12], "dry_slab_probability_q10": [0.08],
                         "dry_slab_probability_q90": [0.17], "dry_slab_record_probability": [0.01],
                         "snow_depth_m": [0.85], "new_snow_72h_m": [0.3]})


def test_map_features_carry_values_and_leave_missing_polygons_empty():
    features = dsm.map_features(_polygons(), _predictions())["features"]

    first, second = (f["properties"] for f in features)
    assert first == {"id": 1, "range": "Front Range", "region": "Front Range", "p": 0.12, "lo": 0.08, "hi": 0.17,
                     "rec": 0.01, "hs": 0.85, "hn72": 0.3, "danger": "Moderate"}
    assert second["p"] is None and second["range"] == "" and second["danger"] == "No data"


def test_render_fills_every_placeholder():
    features = dsm.map_features(_polygons(), _predictions())
    html = dsm.render(features, pd.Timestamp("2026-10-31"), pd.Timestamp("2026-10-30 08:35"))

    assert not re.search(r"__[A-Z_]+__", html)
    assert "Saturday, Oct 31, 2026" in html and "1 polygons" in html
    assert "extrapolation" in html and config.METHODS_DOC_URL in html
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert len(data["features"]) == 2


def test_no_map_without_dry_slab_predictions(monkeypatch, tmp_path):
    csv = tmp_path / "predictions.csv"
    _predictions().drop(columns=["dry_slab_probability"]).to_csv(csv, index=False)
    monkeypatch.setitem(config.PATHS["RESULTS"], "hazard_predictions_csv", csv)

    assert dsm.create_dry_slab_map("2026-12-15") is None
