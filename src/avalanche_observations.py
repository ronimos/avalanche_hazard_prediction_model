# -*- coding: utf-8 -*-
"""
================================================================================
Avalanche Observations: Cleaning and Polygon Assignment
================================================================================

Loads a CAIC avalanche-record export (one row per avalanche or group of
similar avalanches), cleans it, and assigns each record to a backcountry
forecast polygon by testing whether its latitude/longitude falls inside the
polygon.

The output is the observation half of the avalanche likelihood model: a record
confirms that avalanches occurred in a polygon on a day, but the absence of a
record does not confirm that none did.

Usage:
    python avalanche_observations.py

Author: Ron Simenhois
"""
import html
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import config

LOCAL_TZ = "America/Denver"

# CAIC export column -> pipeline column
COLUMN_MAP = {
    "Observation ID": "obs_id",
    "Date": "occurred_at",
    "Date Known": "date_known",
    "Longitude": "longitude",
    "latitude": "latitude",
    "#": "n_avalanches",
    "Elevation": "elevation_band",
    "Aspect": "aspect",
    "Type": "avalanche_type",
    "Trigger": "trigger",
    "Relative Size": "relative_size",
    "Destructive Size": "destructive_size",
    "Area": "forecast_zone",
}


def _parse_size(series: pd.Series, prefix: str) -> pd.Series:
    """Converts size codes such as 'D2.5' or 'R3' to floats; '-' and 'U' become NaN."""
    return pd.to_numeric(series.astype(str).str.removeprefix(prefix), errors="coerce")


def load_avalanche_records(csv_path: Path) -> pd.DataFrame:
    """
    Loads and cleans a CAIC avalanche-record export.

    Rows that repeat an observation ID are kept: one observation report can
    describe several avalanches. Only exact duplicate rows are dropped.

    Args:
        csv_path (Path): Path to the CAIC CSV export.

    Returns:
        pd.DataFrame: One row per record with the occurrence date in local
                      (Mountain) time, numeric sizes, and the avalanche count.
    """
    raw = pd.read_csv(csv_path, low_memory=False, dtype=str)
    n_raw = len(raw)
    raw = raw.drop_duplicates()
    logging.info(f"Loaded {n_raw} avalanche records from '{csv_path}'; dropped {n_raw - len(raw)} exact duplicates.")

    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
    df["date"] = df["occurred_at"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    df["date_known"] = df["date_known"].str.lower().eq("known")

    # '#' is the number of avalanches in the record; '-' and 0 mean it wasn't given
    count = pd.to_numeric(df["n_avalanches"], errors="coerce")
    df["n_avalanches_reported"] = count.notna() & (count > 0)
    df["n_avalanches"] = count.where(df["n_avalanches_reported"], 1).astype(int)

    df["destructive_size"] = _parse_size(df["destructive_size"], "D")
    df["relative_size"] = _parse_size(df["relative_size"], "R")
    df["natural"] = df["trigger"].eq("N")
    df["elevation_band"] = df["elevation_band"].map(html.unescape)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["water_year"] = np.where(df["date"].dt.month >= 10, df["date"].dt.year, df["date"].dt.year - 1)

    invalid = df["occurred_at"].isna() | df["latitude"].isna() | df["longitude"].isna()
    if invalid.any():
        logging.warning(f"Dropping {invalid.sum()} records without a valid date or coordinates.")
    return df[~invalid].reset_index(drop=True)


def assign_to_polygons(records: pd.DataFrame, polygons_path: Path) -> pd.DataFrame:
    """
    Assigns each record to the forecast polygon that contains its coordinates.

    Records outside every polygon (out of state, or between polygons) are
    dropped and counted in the log.

    Args:
        records (pd.DataFrame): Output of `load_avalanche_records`.
        polygons_path (Path): GeoJSON of forecast polygons with a
                              'polygon_number' property.

    Returns:
        pd.DataFrame: The records with an integer 'polygon' column.
    """
    polygons = gpd.read_file(polygons_path)[["polygon_number", "geometry"]].to_crs("EPSG:4326")
    points = gpd.GeoDataFrame(
        records, geometry=gpd.points_from_xy(records["longitude"], records["latitude"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin(points, polygons, how="left", predicate="within")

    # A point on a shared border can fall in two polygons; keep the first match
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched = joined["polygon_number"].isna()
    logging.info(
        f"Assigned {(~unmatched).sum()} of {len(joined)} records to "
        f"{joined.loc[~unmatched, 'polygon_number'].nunique()} polygons; {unmatched.sum()} fell outside every polygon."
    )
    assigned = pd.DataFrame(joined[~unmatched].drop(columns=["geometry", "index_right"]))
    assigned["polygon"] = assigned.pop("polygon_number").astype(int)
    return assigned.reset_index(drop=True)


def summarize_daily_activity(assigned: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates assigned records to one row per polygon and occurrence date.

    Args:
        assigned (pd.DataFrame): Output of `assign_to_polygons`.

    Returns:
        pd.DataFrame: Per polygon-day counts of records and avalanches, natural
                      avalanches, D2+ avalanches, and the largest size.
    """
    df = assigned.assign(
        natural_avalanches=assigned["n_avalanches"].where(assigned["natural"], 0),
        d2plus_avalanches=assigned["n_avalanches"].where(assigned["destructive_size"] >= 2, 0),
    )
    return df.groupby(["polygon", "date"], as_index=False).agg(
        n_records=("obs_id", "size"),
        n_avalanches=("n_avalanches", "sum"),
        natural_avalanches=("natural_avalanches", "sum"),
        d2plus_avalanches=("d2plus_avalanches", "sum"),
        max_destructive_size=("destructive_size", "max"),
        any_date_known=("date_known", "any"),
    )


def main() -> None:
    records = load_avalanche_records(config.PATHS["RAW_DATA"]["avalanche_records"])
    assigned = assign_to_polygons(records, config.PATHS["RAW_DATA"]["polygons"])
    daily = summarize_daily_activity(assigned)

    assigned.to_csv(config.PATHS["PROCESSED_DATA"]["avalanche_observations"], index=False)
    daily.to_csv(config.PATHS["PROCESSED_DATA"]["daily_polygon_activity"], index=False)
    logging.info(f"Saved {len(assigned)} assigned records and {len(daily)} polygon-days with activity.")


if __name__ == "__main__":
    main()
