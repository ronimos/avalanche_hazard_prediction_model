# -*- coding: utf-8 -*-
"""
================================================================================
Avalanche Likelihood Forecast: Operational Training and Prediction
================================================================================

Trains the operational avalanche likelihood model and predicts one day with it.

The operational model is the occupancy-detection model (`bayes_model.py`) with
the snowpack drivers only. Recent recorded activity is left out: there is no
confirmed live source of avalanche records, and at the operational lag of 2
days it added little on held-out seasons. The model is refit once before each
season on all past seasons; mid-season retraining didn't help.

For each polygon it gives the probability that a D2+ dry slab avalanche
occurs, natural or triggered, whether or not anyone reports it (posterior mean,
with the 10% and 90% posterior quantiles). The model assumes every polygon has
the same avalanche rate given the snowpack, so the busiest polygons, treated as
near-fully reported, set the level for all of them; the interval doesn't cover
that assumption (see docs/methods.md). The probability that one is recorded is
also given, since it is the part that can be checked against the records.

Drivers for the forecast day come from the current season's SNOWPACK run
(SNOWPACK_DATA_ROOT). The run ends partway through its last day, so a day is
used only when the run continues past it.

Usage:
    python likelihood_forecast.py --train
    python likelihood_forecast.py --date 2026-12-15

Author: Ron Simenhois
"""
import argparse
import logging
from datetime import datetime

import joblib
import pandas as pd

import config
import likelihood_data as ld
import snowpack_drivers as sd

# Longer chains than the evaluation folds: one fit a season, on every season, has to converge
TRAIN_DRAWS = 1000
TRAIN_TUNE = 2000

# Posterior summary -> output column
OUTPUT_COLUMNS = {
    "p_occurrence": "dry_slab_probability",
    "p_occurrence_q10": "dry_slab_probability_q10",
    "p_occurrence_q90": "dry_slab_probability_q90",
    "p_record": "dry_slab_record_probability",
}

# Snowpack context for the map (CSV only), driver -> output column
SNOW_COLUMNS = {"HS_mod": "snow_depth_m", "HN72": "new_snow_72h_m"}


def season_of(date: pd.Timestamp) -> int:
    """Water year (Oct-Sep) of a date, labeled by its starting year."""
    return date.year if date.month >= 10 else date.year - 1


def train() -> None:
    """Fits the operational model on every season in the modeling table and saves it."""
    import bayes_model

    table = ld.load_modeling_table()
    model = bayes_model.OccupancyDetectionModel(drivers=ld.DRIVERS, draws=TRAIN_DRAWS, tune=TRAIN_TUNE).fit(table)
    # The posterior draws used for prediction are kept; the full trace isn't needed
    model.idata = None
    path = config.PATHS["ARTIFACTS"]["likelihood_model"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Likelihood model trained on seasons {list(model._seasons)} and saved to '{path}'.")


def forecast_table(date: pd.Timestamp) -> pd.DataFrame:
    """
    Model inputs for every polygon on one day, from the current season's SNOWPACK run.

    Args:
        date (pd.Timestamp): Forecast day (local date).

    Returns:
        pd.DataFrame: One row per polygon with a complete day of drivers.
    """
    drivers = sd.build_driver_table([season_of(date)])
    drivers["date"] = pd.to_datetime(drivers["date"])
    complete = drivers.groupby("polygon")["date"].transform("max") > date
    day = drivers[(drivers["date"] == date) & complete]
    no_activity = pd.DataFrame(columns=["polygon", "date", ld.LABELS[ld.DEFAULT_LABEL]])
    return ld.build_modeling_table(day, no_activity, months=range(1, 13))


def predict(date: pd.Timestamp) -> pd.DataFrame:
    """
    Dry slab occurrence probability, its 80% interval and the record probability for every polygon on one day.

    Args:
        date (pd.Timestamp): Forecast day (local date).

    Returns:
        pd.DataFrame: 'polygon' plus the OUTPUT_COLUMNS and SNOW_COLUMNS.
    """
    model = joblib.load(config.PATHS["ARTIFACTS"]["likelihood_model"])
    if date.month not in ld.SEASON_MONTHS:
        logging.warning(f"{date.date()} is outside the November-May training months; the likelihood is an extrapolation.")
    table = forecast_table(date)
    missing = sorted(set(model._polygon_index) - set(table["polygon"]))
    if missing:
        logging.warning(f"No complete drivers for {date.date()} in {len(missing)} polygons: {missing}")
    if table.empty:
        return pd.DataFrame(columns=["polygon", *OUTPUT_COLUMNS.values(), *SNOW_COLUMNS.values()])
    summary = model.predict_interval(table, quantiles=(0.1, 0.9))
    return pd.concat([table[["polygon"]], summary[list(OUTPUT_COLUMNS)].rename(columns=OUTPUT_COLUMNS),
                      table[list(SNOW_COLUMNS)].rename(columns=SNOW_COLUMNS)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the operational avalanche likelihood model or predict one day.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Fit on all seasons in the modeling table and save the model.")
    group.add_argument("--date", type=str, help="Forecast day, YYYY-MM-DD.")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        print(predict(pd.Timestamp(datetime.strptime(args.date, "%Y-%m-%d"))).round(4).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
