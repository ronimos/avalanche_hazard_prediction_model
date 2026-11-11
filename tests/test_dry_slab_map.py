import json
import re

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon, box

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
    no_change = {f"{band}_{short}": None for band in dsm.BANDS for short in [*dsm.CHANGE_FIELDS.values(), "why"]}
    assert first.pop("label") == [39.05, -105.95]
    assert first == {"id": 1, "range": "Front Range", "region": "Front Range", "p": 0.12, "lo": 0.08, "hi": 0.17,
                     "rec": 0.01, "np": None, "nlo": None, "nhi": None, "nrec": None, "hs": 0.85, "hn72": 0.3, "danger": "Moderate", **no_change}
    assert second["p"] is None and second["range"] == "" and second["danger"] == "No data"


def test_map_features_carry_natural_only_probability():
    predictions = _predictions().assign(natural_dry_slab_probability=0.07, natural_dry_slab_probability_q10=0.04,
                                        natural_dry_slab_probability_q90=0.11, natural_dry_slab_record_probability=0.005)

    first = dsm.map_features(_polygons(), predictions)["features"][0]["properties"]

    assert (first["np"], first["nlo"], first["nhi"], first["nrec"]) == (0.07, 0.04, 0.11, 0.005)


def test_map_features_carry_rating_change():
    predictions = _predictions().assign(atl_caic_today=3.0, atl_caic_outlook=4.0, atl_p_down=0.05,
                                        atl_p_same=0.6, atl_p_up=0.35, atl_drivers_p_up=0.5,
                                        atl_drivers_why="Up: new snow raises it")

    first = dsm.map_features(_polygons(), predictions)["features"][0]["properties"]

    assert (first["atl_t"], first["atl_o"], first["atl_d"], first["atl_s"], first["atl_u"]) == (3, 4, 0.05, 0.6, 0.35)
    assert first["atl_du"] == 0.5 and first["atl_why"] == "Up: new snow raises it"
    assert first["ntl_u"] is None and first["atl_dd"] is None


def test_label_point_is_inside_a_concave_polygon_and_its_largest_part():
    u_shape = Polygon([(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)])
    lat, lon = dsm.label_point(MultiPolygon([box(10, 10, 10.1, 10.1), u_shape]))

    assert u_shape.contains(Point(lon, lat))


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


def _write_map(path, day):
    path.write_text(f"<html><body><p>map {day}</p></body></html>", encoding="utf-8")
    return path


def test_publish_archives_each_day_and_keeps_the_newest_live(tmp_path):
    web = tmp_path / "web"
    for day in ["2026-12-14", "2026-12-16"]:
        dsm.publish(day, web, _write_map(tmp_path / f"{day}.html", day))
    # Re-publishing an older day must not replace the live page
    dsm.publish("2026-12-15", web, _write_map(tmp_path / "2026-12-15.html", "2026-12-15"))

    assert json.loads((web / "maps" / "dates.json").read_text()) == ["2026-12-14", "2026-12-15", "2026-12-16"]
    live = (web / "next_day_prediction.html").read_text()
    assert "map 2026-12-16" in live and 'const DAY = "2026-12-16"' in live
    archived = (web / "maps" / "2026-12-15.html").read_text()
    assert "map 2026-12-15" in archived and archived.index("day-nav") < archived.index("</body>")


def test_add_day_nav_is_added_once():
    once = dsm.add_day_nav("<body></body>", pd.Timestamp("2026-12-15"))

    assert dsm.add_day_nav(once, pd.Timestamp("2026-12-15")) == once
    assert "__DAY__" not in once


def test_publish_without_a_map_does_nothing(tmp_path):
    assert dsm.publish("2026-12-15", tmp_path / "web", tmp_path / "missing.html") is None
    assert not (tmp_path / "web").exists()


def test_add_day_nav_hooks_the_polygon_panel_inside_the_page_script():
    page = "<body><script>\n(() => {\nconst layers = new Map();\n})();\n</script>\n</body>"

    html = dsm.add_day_nav(page, pd.Timestamp("2026-12-15"))

    assert html.index("window.dsmPolygon") < html.index("})();") < html.index("day-nav")
