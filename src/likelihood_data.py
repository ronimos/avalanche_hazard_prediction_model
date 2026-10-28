# -*- coding: utf-8 -*-
"""
================================================================================
Likelihood Model Dataset: Polygon-Days with Drivers and Labels
================================================================================

Joins the daily snowpack drivers (`snowpack_drivers.py`) with the recorded
avalanche activity (`avalanche_observations.py`) into one modeling table: a
row per forecast polygon and local date, November to May.

A label of 1 means at least one qualifying avalanche was recorded that day. A
label of 0 means none was recorded, not that none occurred.

Author: Ron Simenhois
"""
from typing import Iterable

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
    "d2plus": "d2plus_avalanches",
    "natural": "natural_avalanches",
    "any": "n_avalanches",
}

SEASON_MONTHS = (11, 12, 1, 2, 3, 4, 5)


def build_modeling_table(drivers: pd.DataFrame, activity: pd.DataFrame, label: str = "d2plus",
                         months: Iterable[int] = SEASON_MONTHS) -> pd.DataFrame:
    """
    Joins drivers and recorded activity into one polygon-day table.

    Args:
        drivers (pd.DataFrame): Daily driver table, one row per polygon and date.
        activity (pd.DataFrame): Daily polygon activity, rows only for days
                                 with at least one record.
        label (str): Which avalanches count as an event; a key of LABELS.
        months (Iterable[int]): Calendar months to keep.

    Returns:
        pd.DataFrame: Drivers plus 'y' (0/1), 'day_of_season' and indicators
                      for capped stability indices.
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
    return table.sort_values(["season", "polygon", "date"]).reset_index(drop=True)


def load_modeling_table(label: str = "d2plus") -> pd.DataFrame:
    """Builds the modeling table from the processed driver and activity files."""
    drivers = pd.read_csv(config.PATHS["PROCESSED_DATA"]["snowpack_drivers"], dtype={"station": str})
    activity = pd.read_csv(config.PATHS["PROCESSED_DATA"]["daily_polygon_activity"])
    return build_modeling_table(drivers, activity, label)
