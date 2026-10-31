# -*- coding: utf-8 -*-
"""
================================================================================
Dry Slab Map: Daily Web Map of the Avalanche Likelihood Forecast
================================================================================

Builds the page published as next_day_prediction.html: the forecast polygons on
a street map or satellite imagery (Leaflet), colored by the probability of a
D2+ dry slab avalanche or by the chance that tomorrow's danger rating goes up
or down (`hazard_change.py`), with the 80% range, snowpack context, CAIC's
ratings and outlook, and the forecast's limitations.

It reads the day's predictions CSV written by predict.py, so it runs after it.

Usage:
    python dry_slab_map.py --date 2026-12-15

Author: Ron Simenhois
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

import config
import likelihood_data as ld

TEMPLATE = Path(__file__).parent / "templates" / "dry_slab_map.html"
# Degrees; keeps the page small without visibly changing polygon outlines
SIMPLIFY_TOLERANCE = 0.002
DANGER_LEVELS = {1: "Low", 2: "Moderate", 3: "Considerable", 4: "High"}
BANDS = ["atl", "ntl", "btl"]
# Rating change output column suffix -> property suffix on the page
CHANGE_FIELDS = {"caic_today": "t", "caic_outlook": "o", "p_down": "d", "p_same": "s", "p_up": "u",
                 "drivers_p_down": "dd", "drivers_p_same": "ds", "drivers_p_up": "du"}


def output_path(date: pd.Timestamp) -> Path:
    """Where the map for a forecast day is written."""
    return config.RESULTS_DIR / f"dry_slab_map_{date:%Y-%m-%d}.html"


def _value(row: pd.Series, column: str, digits: int = 5) -> Optional[float]:
    value = row.get(column)
    return round(float(value), digits) if pd.notna(value) else None


def map_features(polygons: gpd.GeoDataFrame, predictions: pd.DataFrame) -> dict:
    """GeoJSON FeatureCollection of the polygons with the values the page shows."""
    merged = polygons.merge(predictions, left_on="polygon_number", right_on="polygon", how="left")
    features = []
    for _, row in merged.iterrows():
        danger = row.get("predicted_hazard")
        features.append({
            "type": "Feature",
            "geometry": row.geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True).__geo_interface__,
            "properties": {
                "id": int(row["polygon_number"]),
                "range": row.get("mountain_range") if pd.notna(row.get("mountain_range")) else "",
                "region": row.get("referenc_region") if pd.notna(row.get("referenc_region")) else "",
                "p": _value(row, "dry_slab_probability"),
                "lo": _value(row, "dry_slab_probability_q10"),
                "hi": _value(row, "dry_slab_probability_q90"),
                "rec": _value(row, "dry_slab_record_probability"),
                "hs": _value(row, "snow_depth_m", 3),
                "hn72": _value(row, "new_snow_72h_m", 3),
                "danger": DANGER_LEVELS.get(int(danger), "No data") if pd.notna(danger) else "No data",
                **{f"{band}_{short}": _value(row, f"{band}_{field}", 3)
                   for band in BANDS for field, short in CHANGE_FIELDS.items()},
            },
        })
    return {"type": "FeatureCollection", "features": features}


def render(features: dict, date: pd.Timestamp, generated: datetime) -> str:
    """Fills the page template for one forecast day."""
    season_note = ("" if date.month in ld.SEASON_MONTHS else
                   " This day is outside the model's November to May training months, so its values are an extrapolation.")
    replacements = {
        "__DATA__": json.dumps(features, separators=(",", ":")),
        "__DATE_LONG__": f"{date:%A, %b} {date.day}, {date:%Y}",
        "__GENERATED__": f"{generated:%b} {generated.day}, {generated:%H:%M}",
        "__N__": str(sum(f["properties"]["p"] is not None for f in features["features"])),
        "__SEASON_NOTE__": season_note,
        "__METHODS_URL__": config.METHODS_DOC_URL,
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


def create_dry_slab_map(prediction_date: Union[str, datetime]) -> Optional[Path]:
    """
    Writes the dry slab map for one forecast day from the predictions CSV.

    Args:
        prediction_date (Union[str, datetime]): Forecast day, a datetime or 'YYYY-MM-DD'.

    Returns:
        Optional[Path]: The page written, or None when the day has no dry slab predictions.
    """
    date = pd.Timestamp(prediction_date).normalize()
    predictions = pd.read_csv(config.PATHS["RESULTS"]["hazard_predictions_csv"])
    predictions = predictions[pd.to_datetime(predictions["date"]).dt.normalize() == date]
    if predictions.empty or "dry_slab_probability" not in predictions or predictions["dry_slab_probability"].isna().all():
        logging.warning(f"No dry slab predictions for {date.date()}; the dry slab map was not written.")
        return None

    polygons = gpd.read_file(config.PATHS["RAW_DATA"]["polygons"]).to_crs(4326)
    polygons["polygon_number"] = polygons["polygon_number"].astype(int)
    path = output_path(date)
    path.write_text(render(map_features(polygons, predictions), date, datetime.now()), encoding="utf-8")
    logging.info(f"Dry slab map saved to '{path}'")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the dry slab map for one forecast day.")
    parser.add_argument("--date", required=True, help="Forecast day, YYYY-MM-DD.")
    args = parser.parse_args()
    create_dry_slab_map(args.date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
