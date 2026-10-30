import json

import pandas as pd
import pytest

import avalanche_observations as ao

HEADER = list(ao.COLUMN_MAP) + ["Comments"]


def _row(obs_id, date, lon, lat, n="1", trigger="N", dsize="D2", elevation="&#62;TL", known="Estimated", atype="SS"):
    return {"Observation ID": obs_id, "Date": date, "Date Known": known, "Longitude": lon, "latitude": lat,
            "#": n, "Elevation": elevation, "Aspect": "NE", "Type": atype, "Trigger": trigger,
            "Relative Size": "R2", "Destructive Size": dsize, "Area": "Gunnison", "Comments": "-"}


@pytest.fixture
def records_csv(tmp_path):
    rows = [
        _row("a", "2025-01-10T18:00:00Z", -106.5, 39.5),                           # inside polygon 1
        _row("a", "2025-01-10T18:00:00Z", -106.5, 39.5),                           # exact duplicate -> dropped
        _row("a", "2025-01-10T20:00:00Z", -106.6, 39.6, n="3", trigger="AS"),      # same report, another avalanche
        _row("b", "2025-01-11T05:30:00Z", -105.5, 39.5, dsize="D1.5", known="Known"),  # polygon 2, 22:30 local on Jan 10
        _row("c", "2025-01-12T18:00:00Z", -104.0, 39.5, n="-"),                    # outside both polygons
        _row("d", "2025-01-12T18:00:00Z", -123.1, 49.5, dsize="U"),                # out of state
        _row("e", "2025-01-10T21:00:00Z", -106.4, 39.4, n="2", atype="WS"),       # wet slab in polygon 1
    ]
    path = tmp_path / "records.csv"
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)
    return path


@pytest.fixture
def polygons_geojson(tmp_path):
    def square(x0, y0):
        return [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]]
    features = [{"type": "Feature", "properties": {"polygon_number": n},
                 "geometry": {"type": "Polygon", "coordinates": square(x0, 39.0)}}
                for n, x0 in [(1, -107.0), (2, -106.0)]]
    path = tmp_path / "polygons.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_load_cleans_records(records_csv):
    df = ao.load_avalanche_records(records_csv)

    assert len(df) == 6
    assert df["destructive_size"].tolist()[:3] == [2.0, 2.0, 1.5]
    assert df["natural"].tolist()[:2] == [True, False]
    assert set(df["elevation_band"]) == {">TL"}
    assert df.loc[df["obs_id"] == "b", "date_known"].item()


def test_occurrence_date_uses_mountain_time(records_csv):
    df = ao.load_avalanche_records(records_csv)

    assert df.loc[df["obs_id"] == "b", "date"].item() == pd.Timestamp("2025-01-10")


def test_missing_avalanche_count_defaults_to_one(records_csv):
    df = ao.load_avalanche_records(records_csv)
    c = df[df["obs_id"] == "c"].iloc[0]

    assert c["n_avalanches"] == 1
    assert not c["n_avalanches_reported"]


def test_records_assigned_by_point_in_polygon(records_csv, polygons_geojson):
    assigned = ao.assign_to_polygons(ao.load_avalanche_records(records_csv), polygons_geojson)

    assert assigned.groupby("obs_id")["polygon"].first().to_dict() == {"a": 1, "b": 2, "e": 1}
    assert len(assigned) == 4


def test_daily_activity_counts(records_csv, polygons_geojson):
    assigned = ao.assign_to_polygons(ao.load_avalanche_records(records_csv), polygons_geojson)
    daily = ao.summarize_daily_activity(assigned).set_index("polygon")

    assert daily.loc[1, "n_records"] == 3
    assert daily.loc[1, "n_avalanches"] == 6
    assert daily.loc[1, "natural_avalanches"] == 3
    assert daily.loc[1, "d2plus_avalanches"] == 6
    assert daily.loc[1, "dry_slab_d2plus_avalanches"] == 4
    assert daily.loc[2, "dry_slab_d2plus_avalanches"] == 0
    assert daily.loc[2, "d2plus_avalanches"] == 0
