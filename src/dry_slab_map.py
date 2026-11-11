# -*- coding: utf-8 -*-
"""
================================================================================
Dry Slab Map: Daily Web Map of the Avalanche Likelihood Forecast
================================================================================

Builds the page published as next_day_prediction.html: the forecast polygons on
a street map or satellite imagery (Leaflet), colored by the probability of a
D2+ dry slab avalanche (natural or triggered, or natural only) or by the forecast change in tomorrow's danger rating
(`hazard_change.py`; hue = direction, depth = chance), with the 80% range, snowpack context, CAIC's
ratings and outlook, and the forecast's limitations.

It reads the day's predictions CSV written by predict.py, so it runs after it.

`--publish DIR` copies an already written map to the web folder instead: DIR/maps/<date>.html plus
DIR/next_day_prediction.html when it's the newest day, with a forecast day picker added, and
DIR/maps/dates.json listing the archived days so earlier forecasts can be looked up.

Usage:
    python dry_slab_map.py --date 2026-12-15
    python dry_slab_map.py --date 2026-12-15 --publish /home/www/html/ron/prediction_model

Author: Ron Simenhois
"""
import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.ops import polylabel

import config
import likelihood_data as ld

TEMPLATE = Path(__file__).parent / "templates" / "dry_slab_map.html"
DAY_NAV = Path(__file__).parent / "templates" / "day_nav.html"
LATEST_PAGE = "next_day_prediction.html"
ARCHIVE_DIR = "maps"
# Lets the day picker read and set the panel's polygon, which lives inside the page script. Added at the end
# of that script so it also works on maps written before the picker existed (those have select but no lock).
POLYGON_HOOK = ("window.dsmPolygon = { get: () => typeof locked !== \"undefined\" && locked != null ? locked : null, "
                "set: id => { if (layers.has(id)) (typeof lock === \"function\" ? lock : select)(id); } };\n")
SCRIPT_END = re.compile(r"\n\}\)\(\);\r?\n</script>")
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


def label_point(geometry) -> list:
    """[lat, lon] for the polygon's number: the point farthest inside its largest part."""
    part = max(getattr(geometry, "geoms", [geometry]), key=lambda g: g.area)
    point = polylabel(part, tolerance=0.002)
    return [round(point.y, 4), round(point.x, 4)]


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
                "label": label_point(row.geometry),
                "range": row.get("mountain_range") if pd.notna(row.get("mountain_range")) else "",
                "region": row.get("referenc_region") if pd.notna(row.get("referenc_region")) else "",
                "p": _value(row, "dry_slab_probability"),
                "lo": _value(row, "dry_slab_probability_q10"),
                "hi": _value(row, "dry_slab_probability_q90"),
                "rec": _value(row, "dry_slab_record_probability"),
                # Natural releases only (None until that model is trained)
                "np": _value(row, "natural_dry_slab_probability"),
                "nlo": _value(row, "natural_dry_slab_probability_q10"),
                "nhi": _value(row, "natural_dry_slab_probability_q90"),
                "nrec": _value(row, "natural_dry_slab_record_probability"),
                "hs": _value(row, "snow_depth_m", 3),
                "hn72": _value(row, "new_snow_72h_m", 3),
                "danger": DANGER_LEVELS.get(int(danger), "No data") if pd.notna(danger) else "No data",
                **{f"{band}_{short}": _value(row, f"{band}_{field}", 3)
                   for band in BANDS for field, short in CHANGE_FIELDS.items()},
                **{f"{band}_why": row.get(f"{band}_drivers_why") if isinstance(row.get(f"{band}_drivers_why"), str) else None
                   for band in BANDS},
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


def add_day_nav(html: str, date: pd.Timestamp) -> str:
    """Adds the forecast day picker to a map page (any template version); no-op if it's already there."""
    if "<!-- day-nav:" in html:
        return html
    ends = list(SCRIPT_END.finditer(html))
    if ends:
        at = ends[-1].start() + 1
        html = html[:at] + POLYGON_HOOK + html[at:]
    nav = DAY_NAV.read_text(encoding="utf-8").replace("__DAY__", f"{date:%Y-%m-%d}")
    head, sep, tail = html.rpartition("</body>")
    return head + nav + sep + tail if sep else html + nav


def _write_shared(path: Path, text: str) -> None:
    """Writes a file the scheduled run (caic) and manual runs (ron) both overwrite: atomic, group-writable."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.chmod(tmp, 0o664)
    except PermissionError:
        pass
    os.replace(tmp, path)


def archived_days(web_dir: Union[str, Path]) -> list:
    """Forecast days with a map in the web folder's archive, oldest first."""
    archive = Path(web_dir) / ARCHIVE_DIR
    return sorted(p.stem for p in archive.glob("*.html") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem))


def publish(prediction_date: Union[str, datetime], web_dir: Union[str, Path], source: Optional[Path] = None) -> Optional[Path]:
    """
    Publishes a written map: web_dir/maps/<date>.html, web_dir/maps/dates.json, and
    web_dir/next_day_prediction.html when no later day is archived (re-publishing an old day
    doesn't replace the live page).

    Args:
        prediction_date (Union[str, datetime]): Forecast day, a datetime or 'YYYY-MM-DD'.
        web_dir (Union[str, Path]): The folder that holds next_day_prediction.html.
        source (Optional[Path]): Map to publish; defaults to the day's map in results/.

    Returns:
        Optional[Path]: The archived page, or None when the day has no map.
    """
    date = pd.Timestamp(prediction_date).normalize()
    source = Path(source) if source else output_path(date)
    if not source.exists():
        logging.warning(f"No map at '{source}'; nothing published for {date.date()}.")
        return None
    web_dir = Path(web_dir)
    archive = web_dir / ARCHIVE_DIR
    if not archive.exists():
        archive.mkdir(parents=True)
        try:
            os.chmod(archive, 0o2775)
        except PermissionError:
            pass
    html = add_day_nav(source.read_text(encoding="utf-8"), date)
    day = f"{date:%Y-%m-%d}"
    page = archive / f"{day}.html"
    _write_shared(page, html)
    days = archived_days(web_dir)
    _write_shared(archive / "dates.json", json.dumps(days))
    if day == days[-1]:
        latest = web_dir / LATEST_PAGE
        # Copy in place so the live page keeps its owner and group
        if latest.exists():
            with open(latest, "w", encoding="utf-8") as f:
                f.write(html)
        else:
            _write_shared(latest, html)
    logging.info(f"Published the {day} map to '{page}'" + (f" and '{web_dir / LATEST_PAGE}'" if day == days[-1] else ""))
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the dry slab map for one forecast day.")
    parser.add_argument("--date", required=True, help="Forecast day, YYYY-MM-DD.")
    parser.add_argument("--publish", metavar="WEB_DIR",
                        help="Publish the day's already written map to this folder instead of writing it.")
    parser.add_argument("--source", type=Path, help="With --publish: the map file to publish (default: the day's map in results/).")
    args = parser.parse_args()
    if args.publish:
        if publish(args.date, args.publish, args.source) is None:
            raise SystemExit(1)
    else:
        create_dry_slab_map(args.date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
