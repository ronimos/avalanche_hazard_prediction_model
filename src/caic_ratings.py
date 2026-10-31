# -*- coding: utf-8 -*-
"""
================================================================================
CAIC Ratings: Danger Ratings per Polygon from AvID Forecasts
================================================================================

Turns CAIC avalanche forecasts from the AvID API into danger ratings per
forecast polygon, day and elevation band.

Each forecast lists the polygons it covers (by UUID, the `id` of the polygon
GeoJSON) and rates three days. Forecasts go out around 4 PM Mountain time:
day 1 (position 1) is the next day's rating, and days 2 and 3 are CAIC's
outlook. Occasional morning updates rate the same day. So at noon on t-1, the
current forecast gives the rating for t-1 and CAIC's outlook for t.

    - `import_forecasts` writes the ratings in a saved set of forecasts (for
      example AvyScribe's all_forecasts.json) for training.
    - `fetch_current` pulls today's forecasts from AvID and appends them to a
      history file, which gives the daily run yesterday's rating too.

The API URL and key come from AVID_API_BASE and AVID_API_KEY (or API_KEY) in
the project's .env.

Usage:
    python caic_ratings.py --import /home/ron/AvyScribe/data/all_forecasts.json
    python caic_ratings.py --fetch

Author: Ron Simenhois
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config

BANDS = ["atl", "ntl", "btl"]
# AvID band key -> our band name
AVID_BANDS = {"alp": "atl", "tln": "ntl", "btl": "btl"}
# AvID rating word -> level; 'noRating' and anything else is missing
LEVELS = {"low": 1, "moderate": 2, "considerable": 3, "high": 4, "extreme": 5}
COLUMNS = ["polygon", "date", "position", "issued", *BANDS]


def polygon_numbers(polygons_path: Optional[Path] = None) -> Dict[str, int]:
    """Polygon UUID (lower case) -> polygon number, from the polygon GeoJSON."""
    with open(polygons_path or config.PATHS["RAW_DATA"]["polygons"], encoding="utf-8") as f:
        features = json.load(f)["features"]
    return {f["properties"]["id"].lower(): int(f["properties"]["polygon_number"]) for f in features}


def ratings_from_forecasts(forecasts: List[dict], numbers: Dict[str, int]) -> pd.DataFrame:
    """
    One row per polygon, rated day and forecast position.

    Args:
        forecasts (List[dict]): AvID forecast objects.
        numbers (Dict[str, int]): Output of `polygon_numbers`. Polygons not in
                                  it (an older polygon set) are skipped.

    Returns:
        pd.DataFrame: COLUMNS, with 'date' the local day rated and the latest
                      issued forecast kept when several rate the same polygon,
                      day and position.
    """
    rows, unknown = [], set()
    for forecast in forecasts:
        days = (forecast.get("dangerRatings") or {}).get("days") or []
        for polygon_id in forecast.get("polygons") or []:
            number = numbers.get(polygon_id.lower())
            if number is None:
                unknown.add(polygon_id)
                continue
            for day in days:
                rows.append({"polygon": number, "date": day["date"][:10], "position": day.get("position"),
                             "issued": forecast.get("issueDateTime"),
                             **{band: LEVELS.get(day.get(key), np.nan) for key, band in AVID_BANDS.items()}})
    if unknown:
        logging.info(f"Skipped {len(unknown)} polygon IDs that aren't in the polygon file.")
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    ratings = pd.DataFrame(rows)
    ratings["date"] = pd.to_datetime(ratings["date"])
    ratings = ratings.sort_values("issued").drop_duplicates(["polygon", "date", "position"], keep="last")
    return ratings[COLUMNS].sort_values(["date", "polygon", "position"]).reset_index(drop=True)


def day_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """CAIC's rating for each polygon and day: the day-1 position of the latest forecast for it."""
    days = ratings[ratings["position"] == 1][["polygon", "date", *BANDS]].reset_index(drop=True)
    return days.assign(date=pd.to_datetime(days["date"]))


def outlook_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """CAIC's outlook for each polygon and day t: the day-2 position, from the forecast issued on t-2."""
    days = ratings[ratings["position"] == 2][["polygon", "date", *BANDS]].reset_index(drop=True)
    return days.assign(date=pd.to_datetime(days["date"]))


def import_forecasts(path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """Writes the ratings in a saved JSON list of AvID forecasts to the AvID ratings file."""
    with open(path, encoding="utf-8") as f:
        forecasts = json.load(f)
    ratings = ratings_from_forecasts(forecasts, polygon_numbers())
    output_path = output_path or config.PATHS["RAW_DATA"]["avid_ratings"]
    ratings.to_csv(output_path, index=False)
    logging.info(f"{len(day_ratings(ratings))} polygon-day ratings from {len(forecasts)} forecasts saved to '{output_path}'")
    return ratings


def fetch_forecasts() -> List[dict]:
    """Current forecasts from the AvID API."""
    settings = config.AVID_CONFIG
    if not settings["base_url"] or not settings["api_key"]:
        raise RuntimeError("AVID_API_BASE and AVID_API_KEY (or API_KEY) must be set, for example in the project's .env.")
    response = requests.get(settings["base_url"], params={"productType": settings["product_type"]},
                            headers={"x-api-key": settings["api_key"], "Accept": "application/json"},
                            timeout=settings["timeout_sec"])
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        data = data.get("results", data.get("data", [data]))
    return data


def fetch_current(history_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Pulls the current forecasts and adds them to the ratings history.

    Returns:
        pd.DataFrame: The whole history (COLUMNS), including this pull.
    """
    history_path = history_path or config.PATHS["PROCESSED_DATA"]["caic_ratings_history"]
    current = ratings_from_forecasts(fetch_forecasts(), polygon_numbers())
    logging.info(f"Fetched CAIC ratings for {current['polygon'].nunique()} polygons, days "
                 f"{', '.join(str(d.date()) for d in sorted(current['date'].unique()))}.")
    history = load_history(history_path)
    combined = pd.concat([history, current], ignore_index=True) if not history.empty else current
    combined = combined.sort_values("issued").drop_duplicates(["polygon", "date", "position"], keep="last")
    combined = combined.sort_values(["date", "polygon", "position"]).reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)
    return combined


def load_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Ratings saved by earlier pulls, or an empty table."""
    path = path or config.PATHS["PROCESSED_DATA"]["caic_ratings_history"]
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["date"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Danger ratings per polygon from CAIC's AvID forecasts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--import", dest="import_path", type=Path, help="Saved JSON list of AvID forecasts to import for training.")
    group.add_argument("--fetch", action="store_true", help="Pull the current forecasts and add them to the history.")
    args = parser.parse_args()
    if args.import_path:
        import_forecasts(args.import_path)
    else:
        fetch_current()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
