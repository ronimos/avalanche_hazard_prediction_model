# -*- coding: utf-8 -*-
"""
================================================================================
Likelihood Model Dataset: Polygon-Days with Drivers and Labels
================================================================================

Joins the daily snowpack drivers (`snowpack_drivers.py`) with the recorded
avalanche activity (`avalanche_observations.py`) into one modeling table: a
row per forecast polygon and local date, November to May.

A label of 1 means at least one qualifying avalanche was recorded that day. A
label of 0 means none was recorded, not that none occurred. The main label is
D2+ dry slab avalanches; wet avalanches are outside the model's scope.

Recent recorded activity is also added as a forecastable feature. Avalanche
cycles last several days and cover whole regions, so what was recorded in the
region over the previous days carries information the snowpack drivers miss.
It only uses days before the forecast day, offset by a lag that allows for
reports arriving late (the export has no report timestamps to check this).

Author: Ron Simenhois
"""
from typing import Iterable, Optional

import geopandas as gpd

import numpy as np
import pandas as pd

import config

# Daily drivers used by every model, as named in the driver table
DRIVERS = [
    "HN24", "HN72", "SWE_change_24h", "SWE_change_3d", "VW_mean", "VW_max",
    "TA_max", "TA_max_change_24h", "MS_Rain", "MS_Water",
    "Sn_min", "zSn_at_min", "Ss_min", "zSs_at_min", "HS_mod",
]

# SNOWPACK writes 6 when it finds no weak layer to evaluate
STABILITY_CAP = 6.0

# Label name -> activity column whose count must be positive
LABELS = {
    "dry_slab": "dry_slab_d2plus_avalanches",
    "d2plus": "d2plus_avalanches",
    "natural": "natural_avalanches",
    "any": "n_avalanches",
}
DEFAULT_LABEL = "dry_slab"

SEASON_MONTHS = (11, 12, 1, 2, 3, 4, 5)

# Recorded activity before the forecast day: own polygon, and the polygon's forecast region
ACTIVITY_FEATURES = ["polygon_recent_any", "region_recent_dry_slab", "region_recent_any"]

# Activity window: `ACTIVITY_WINDOW` days ending `lag` days before the forecast day
ACTIVITY_WINDOW = 2
# The forecast for day t is issued on the morning of t-1, when only reports through t-2 can be in
DEFAULT_ACTIVITY_LAG = 2


def polygon_regions() -> pd.Series:
    """CAIC forecast region of each polygon, indexed by polygon number."""
    polygons = gpd.read_file(config.PATHS["RAW_DATA"]["polygons"])
    return polygons.set_index(polygons["polygon_number"].astype(int))["referenc_region"].sort_index()


def _trailing_sum(daily: pd.DataFrame, key: str, columns: list, lag: int, window: int) -> pd.DataFrame:
    """For each key and date t, the sum of `columns` over days t-lag-window+1 .. t-lag."""
    out = []
    for k, g in daily.groupby(key):
        g = g.set_index("date")[columns].groupby(level=0).sum()
        # Run past the last record so the days that still see it in their window are kept
        days = pd.date_range(g.index.min(), g.index.max() + pd.Timedelta(days=lag + window - 1))
        g = g.reindex(days, fill_value=0).rename_axis("date")
        out.append(g.rolling(window, min_periods=1).sum().shift(lag).dropna().assign(**{key: k}))
    return pd.concat(out).reset_index()


def recent_activity(activity: pd.DataFrame, regions: pd.Series, lag: int = DEFAULT_ACTIVITY_LAG,
                    window: int = ACTIVITY_WINDOW) -> tuple:
    """
    Recorded activity in each polygon and region before each date.

    Args:
        activity (pd.DataFrame): Daily polygon activity (rows only for days with records).
        regions (pd.Series): Forecast region of each polygon.
        lag (int): Days between the end of the window and the forecast day (1 = through yesterday).
        window (int): Days in the window.

    Returns:
        tuple: (per polygon-date frame with 'polygon_recent_any',
                per region-date frame with 'region_recent_dry_slab' and 'region_recent_any').
    """
    activity = activity.assign(date=pd.to_datetime(activity["date"]), region=activity["polygon"].map(regions),
                               any_record=1)
    by_polygon = _trailing_sum(activity, "polygon", ["any_record"], lag, window)
    by_polygon["polygon_recent_any"] = (by_polygon.pop("any_record") > 0).astype(int)
    by_region = _trailing_sum(activity, "region", ["dry_slab_d2plus_avalanches", "n_avalanches"], lag, window)
    by_region["region_recent_dry_slab"] = np.log1p(by_region.pop("dry_slab_d2plus_avalanches"))
    by_region["region_recent_any"] = np.log1p(by_region.pop("n_avalanches"))
    return by_polygon, by_region


def build_modeling_table(drivers: pd.DataFrame, activity: pd.DataFrame, label: str = DEFAULT_LABEL,
                         months: Iterable[int] = SEASON_MONTHS, regions: Optional[pd.Series] = None,
                         activity_lag: int = DEFAULT_ACTIVITY_LAG) -> pd.DataFrame:
    """
    Joins drivers and recorded activity into one polygon-day table.

    Args:
        drivers (pd.DataFrame): Daily driver table, one row per polygon and date.
        activity (pd.DataFrame): Daily polygon activity, rows only for days
                                 with at least one record.
        label (str): Which avalanches count as an event; a key of LABELS.
        months (Iterable[int]): Calendar months to keep.
        regions (Optional[pd.Series]): Forecast region of each polygon. When
                                       given, recent-activity features are added.
        activity_lag (int): Lag for the recent-activity window, in days.

    Returns:
        pd.DataFrame: Drivers plus 'y' (0/1), 'day_of_season', indicators for
                      capped stability indices and, with regions, ACTIVITY_FEATURES.
    """
    drivers = drivers.assign(date=pd.to_datetime(drivers["date"]))
    activity = activity.assign(date=pd.to_datetime(activity["date"]))
    table = drivers.merge(activity[["polygon", "date", LABELS[label]]], on=["polygon", "date"], how="left")
    table = table[table["date"].dt.month.isin(list(months))].copy()

    table["y"] = (table.pop(LABELS[label]).fillna(0) > 0).astype(int)
    season_start = pd.to_datetime(table["season"].astype(str) + "-10-01")
    table["day_of_season"] = (table["date"] - season_start).dt.days
    table["Sn_capped"] = (table["Sn_min"] >= STABILITY_CAP).astype(int)
    table["Ss_capped"] = (table["Ss_min"] >= STABILITY_CAP).astype(int)

    if regions is not None:
        by_polygon, by_region = recent_activity(activity, regions, lag=activity_lag)
        table["region"] = table["polygon"].map(regions)
        table = table.merge(by_polygon, on=["polygon", "date"], how="left").merge(by_region, on=["region", "date"], how="left")
        table[ACTIVITY_FEATURES] = table[ACTIVITY_FEATURES].fillna(0)
    return table.sort_values(["season", "polygon", "date"]).reset_index(drop=True)


def load_modeling_table(label: str = DEFAULT_LABEL, activity_lag: int = DEFAULT_ACTIVITY_LAG) -> pd.DataFrame:
    """Builds the modeling table, with recent-activity features, from the processed driver and activity files."""
    drivers = pd.read_csv(config.PATHS["PROCESSED_DATA"]["snowpack_drivers"], dtype={"station": str})
    activity = pd.read_csv(config.PATHS["PROCESSED_DATA"]["daily_polygon_activity"])
    return build_modeling_table(drivers, activity, label, regions=polygon_regions(), activity_lag=activity_lag)
