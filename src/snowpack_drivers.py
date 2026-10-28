# -*- coding: utf-8 -*-
"""
================================================================================
Snowpack Drivers: Daily Features from SNOWPACK SMET Output
================================================================================

Builds the daily driver table for the avalanche likelihood model: one row per
forecast polygon and local (Mountain) date, from the SNOWPACK station nearest
each polygon's centroid.

Each station directory holds five SMET streams, read with xsnow:
    <station>_res.smet            flat field
    <station>1_res.smet .. 4_res  38-degree slopes facing N, E, S, W

xsnow names a SMET stream by its header `station_id` (e.g. '1009771') and a
PRO stream by its file stem (e.g. '1009771_res'), so the two can't be merged
or grouped automatically. SMET and PRO files are read separately, and every
stream is mapped to its station explicitly.

Loading and weather drivers come from the flat field; stability drivers take
the least stable of the four slopes.

In these runs `wind_trans24` is always 0, `VW_drift` equals `VW`, and
`HN72_24` almost always equals `HN24`, so 72 h new snow is summed from daily
`HN24` and wind is summarized from `VW`. Load comes from SWE change, because
the 24 h precipitation field changed between seasons (`PSUM24` before 2025,
`HNW24` from 2025). xsnow returns heights in meters.

Zone folders changed in 2025 (90 zones before, 134 after), so stations are
looked up by ID within each season rather than by zone.

Usage:
    python snowpack_drivers.py

Author: Ron Simenhois
"""
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import xsnow

import config
import dataset_utils as dsu

LOCAL_TZ = "America/Denver"

# 24 h accumulations SNOWPACK reports at every timestep; the day's value is its last one
LAST_OF_DAY = ["HN24", "SWE", "HS_mod", "MS_Water"]
STABILITY = {"Sn": "zSn", "Ss": "zSs"}


@lru_cache(maxsize=None)
def season_station_dirs(season: int) -> Dict[str, Path]:
    """Maps each zero-padded station ID to its output directory for one season."""
    season_root = config.PATHS["EXTERNAL_DATA"]["snowpack_output"] / str(season)
    return {d.name: d for d in season_root.glob("zone*/*") if d.is_dir()}


def station_dir(season: int, station_id: int) -> Optional[Path]:
    """Directory of a station's SNOWPACK output for one season, or None if it wasn't run."""
    return season_station_dirs(season).get(f"{station_id:06d}")


def read_station_smet(directory: Path) -> Optional[pd.DataFrame]:
    """
    Reads a station's five SMET streams with xsnow.

    Args:
        directory (Path): Station directory, e.g. .../2025/zone003/100977.

    Returns:
        Optional[pd.DataFrame]: Long table indexed by (time, slope) with
                                'inclination' and the SMET variables, or None
                                if the directory has no SMET files.
    """
    files = sorted(directory.glob("*_res.smet"))
    if not files:
        return None
    station = directory.name
    streams = {f.name.removesuffix("_res.smet"): station for f in files}
    xs = xsnow.read([str(f) for f in files], locations=streams, n_cpus_use=1)
    try:
        ds = xs.to_xarray().sel(location=station, realization=0)
        variables = [v for v in LAST_OF_DAY + ["TA", "VW", "MS_Rain"] + list(STABILITY) + list(STABILITY.values())
                     if v in ds.data_vars]
        df = ds[variables].to_dataframe().reset_index()
        df["inclination"] = ds["inclination"].values[df["slope"].to_numpy()]
    finally:
        xs.close()
    return df[["time", "slope", "inclination"] + variables]


def daily_drivers(smet: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a station's SMET timesteps to daily drivers by local date.

    Args:
        smet (pd.DataFrame): Output of `read_station_smet`.

    Returns:
        pd.DataFrame: One row per local date with the model drivers.
    """
    smet = smet.copy()
    smet["date"] = (pd.to_datetime(smet["time"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
                    .dt.tz_localize(None).dt.normalize())
    flat = smet[smet["inclination"] == 0].sort_values("time").groupby("date")

    daily = flat[LAST_OF_DAY].last()
    daily["TA_max"] = flat["TA"].max()
    daily["VW_mean"] = flat["VW"].mean()
    daily["VW_max"] = flat["VW"].max()
    daily["MS_Rain"] = flat["MS_Rain"].sum(min_count=1)
    daily = daily.asfreq("D")
    daily["HN72"] = daily["HN24"].rolling(3, min_periods=3).sum()
    daily["SWE_change_24h"] = daily["SWE"].diff()
    daily["SWE_change_3d"] = daily["SWE"] - daily["SWE"].shift(3)
    daily["TA_max_change_24h"] = daily["TA_max"].diff()

    # Least stable slope and timestep of the day, with the depth of that minimum
    slopes = smet[smet["inclination"] > 0]
    for index, depth in STABILITY.items():
        valid = slopes.dropna(subset=[index])
        weakest = valid.loc[valid.groupby("date")[index].idxmin(), ["date", index, depth]].set_index("date")
        daily[f"{index}_min"] = weakest[index]
        daily[f"{depth}_at_min"] = weakest[depth]

    daily.index.name = "date"
    return daily.reset_index()


def polygon_stations() -> pd.DataFrame:
    """Nearest SNOWPACK station to each forecast polygon's centroid."""
    centroids = dsu.get_polygon_centroids(config.PATHS["RAW_DATA"]["polygons"], config.PROCESSING_CONFIG["geoprojection_epsg"])
    locations = pd.read_csv(config.PATHS["RAW_DATA"]["snowpack_locations"])
    stations = dsu.find_closest_stations(centroids, locations)
    return centroids[["polygon_number"]].join(stations).rename(columns={"polygon_number": "polygon"}).reset_index(drop=True)


def build_driver_table(seasons: Iterable[int]) -> pd.DataFrame:
    """
    Builds the daily driver table for every polygon and season.

    Polygons that share a station reuse one read.

    Args:
        seasons (Iterable[int]): Water years (Oct-Sep, labeled by starting year).

    Returns:
        pd.DataFrame: One row per polygon and local date.
    """
    mapping = polygon_stations()
    tables = []
    for season in seasons:
        for station_id, group in mapping.groupby("st_id"):
            directory = station_dir(season, int(station_id))
            smet = read_station_smet(directory) if directory else None
            if smet is None:
                logging.warning(f"No SMET output for station {int(station_id):06d} in {season}; skipping polygons {group['polygon'].tolist()}.")
                continue
            drivers = daily_drivers(smet)
            drivers["season"] = season
            drivers["station"] = f"{int(station_id):06d}"
            tables.extend(drivers.assign(polygon=int(p)) for p in group["polygon"])
        logging.info(f"Season {season}: drivers built for {sum(t['season'].iat[0] == season for t in tables)} polygons.")
    table = pd.concat(tables, ignore_index=True)
    first = ["polygon", "date", "season", "station"]
    return table[first + [c for c in table.columns if c not in first]]


def main() -> None:
    table = build_driver_table(config.PROCESSING_CONFIG["target_years"])
    output = config.PATHS["PROCESSED_DATA"]["snowpack_drivers"]
    table.to_csv(output, index=False)
    logging.info(f"Saved {len(table)} polygon-days of drivers to '{output}'.")


if __name__ == "__main__":
    main()
