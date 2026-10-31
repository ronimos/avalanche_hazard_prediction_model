# -*- coding: utf-8 -*-
"""
================================================================================
Hazard Change: Probability That Tomorrow's Danger Rating Changes
================================================================================

A first guess for forecasters. When the run finishes at noon on day t-1, CAIC's
rating for t-1 is already out, and carrying it forward to t is right on about
80% of days. Picking a single rating for t doesn't beat that, so this model
gives, for each elevation band, the probability that the rating for t goes
down, stays the same or goes up from the rating for t-1.

Features: the polygon's ratings for t-1 (all bands) and the trend from t-2,
the snowpack drivers for t (SNOWPACK forced by NWP) and their regional means.
Ratings move by zone, so the regional means carry the storm signal. Recent
recorded activity added at most 0.01 of log-loss skill and has no live feed,
so it is left out.

Ratings come from data/raw/danger_rating.csv (2023-24 and 2024-25) and the
AvID forecasts (`caic_ratings.py`, 2025-26 on).

The baseline knows only the rating for t-1 and its trend: the training
seasons' transition frequencies.

Evaluation is leave-one-season-out, as for the likelihood model:
    - log loss and Brier score of the three-way probability, and their skill
      against the baseline;
    - AUC of the probability of an increase and of a decrease;
    - reliability: predicted against observed rate, binned.

CAIC's outlook for t (day 2 of the forecast issued at 4 PM on t-2) is out
by noon on t-1 and calls changes far better than the drivers. It exists only
from 2025-26, so it isn't a feature of the change model. Instead a small
second model (`OutlookModel`) combines the change model's probabilities with
the outlook's direction. It is trained on the change model's held-out-season
probabilities for the seasons with outlooks, and evaluated within those
seasons by leaving out one 14-day block at a time.

Daily run: `predict` pulls the current CAIC forecasts from AvID (today's
rating and CAIC's outlook for tomorrow), adds yesterday's rating from the
ratings history, and gives, per band, the three probabilities with the
outlook and the drivers-only ones, for every polygon CAIC rated today.
Drivers come from the same SNOWPACK build as the likelihood forecast.

Usage:
    python hazard_change.py                    # held-out-season evaluation
    python hazard_change.py --train            # fit on all seasons and save
    python hazard_change.py --date 2026-12-15  # predict one day

Author: Ron Simenhois
"""
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional

import joblib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import caic_ratings
import config
import likelihood_data as ld
import likelihood_forecast

BANDS = ["atl", "ntl", "btl"]
LEVELS = (1, 5)  # Low to Extreme; CAIC's 0 means no rating
# Change from the rating for t-1, in model column order
CHANGES = [-1, 0, 1]
CHANGE_NAMES = ["down", "same", "up"]

# Drivers averaged over the forecast region, since ratings are issued by zone
REGIONAL_DRIVERS = ["HN24", "HN72", "SWE_change_3d", "VW_max", "TA_max_change_24h"]
DRIVER_FEATURES = ld.DRIVERS + ["day_of_season", "Sn_capped", "Ss_capped"] + [f"{c}_region" for c in REGIONAL_DRIVERS]
OUTPUT_DIR = config.RESULTS_DIR / "hazard_change"
# Days per block when evaluating the outlook model within a season
OUTLOOK_BLOCK_DAYS = 14


def rating_features(band: str) -> List[str]:
    """Rating features for one band: its rating for t-1, its trend, and the other bands' ratings."""
    return [f"{band}_prev", f"{band}_trend"] + [f"{other}_prev" for other in BANDS if other != band]


def load_ratings() -> pd.DataFrame:
    """CAIC danger ratings per polygon and day, with 0 (not rated) as missing: the ratings file plus, for days it lacks, the AvID ratings."""
    ratings = pd.read_csv(config.PATHS["RAW_DATA"]["danger_ratings"], parse_dates=["date"])
    avid_path = config.PATHS["RAW_DATA"]["avid_ratings"]
    if avid_path.exists():
        avid = caic_ratings.day_ratings(pd.read_csv(avid_path, parse_dates=["date"]))
        ratings = pd.concat([ratings, avid], ignore_index=True).drop_duplicates(["polygon", "date"], keep="first")
    ratings[BANDS] = ratings[BANDS].where(ratings[BANDS] > 0)
    return ratings.sort_values(["polygon", "date"]).reset_index(drop=True)


def load_outlooks() -> pd.DataFrame:
    """CAIC's outlook per polygon and day t: day 2 of the AvID forecast issued at 4 PM on t-2."""
    path = config.PATHS["RAW_DATA"]["avid_ratings"]
    if not path.exists():
        return pd.DataFrame(columns=["polygon", "date", *BANDS])
    return caic_ratings.outlook_ratings(pd.read_csv(path, parse_dates=["date"]))


def build_change_table(table: pd.DataFrame, ratings: pd.DataFrame, outlooks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Joins ratings, with what is known at noon on t-1, to the modeling table.

    Args:
        table (pd.DataFrame): Modeling table from `likelihood_data` (drivers for day t).
        ratings (pd.DataFrame): Output of `load_ratings`.
        outlooks (Optional[pd.DataFrame]): CAIC's outlook per polygon and day t (`load_outlooks`).

    Returns:
        pd.DataFrame: One row per polygon and day t with, per band, the rating
                      ('atl'), the rating for t-1 ('atl_prev'), the change from
                      t-2 to t-1 ('atl_trend'), the change to t ('atl_change',
                      clipped to -1/0/1) and CAIC's outlook for t ('atl_outlook',
                      missing without one), plus regional driver means.
    """
    ratings = ratings.copy()
    by_polygon = ratings.groupby("polygon")
    for band in BANDS:
        # Only consecutive days count; a gap in the ratings leaves the lag missing
        prev1 = by_polygon[band].shift(1).where(by_polygon["date"].diff(1) == pd.Timedelta(days=1))
        prev2 = by_polygon[band].shift(2).where(by_polygon["date"].diff(2) == pd.Timedelta(days=2))
        ratings[f"{band}_prev"] = prev1
        ratings[f"{band}_trend"] = prev1 - prev2
        ratings[f"{band}_change"] = np.sign(ratings[band] - prev1)

    df = ratings.merge(table, on=["polygon", "date"], how="inner")
    if outlooks is not None and len(outlooks):
        renamed = outlooks[["polygon", "date", *BANDS]].rename(columns={b: f"{b}_outlook" for b in BANDS})
        df = df.merge(renamed.astype({"polygon": df["polygon"].dtype}), on=["polygon", "date"], how="left")
    for band in BANDS:
        if f"{band}_outlook" not in df:
            df[f"{band}_outlook"] = np.nan
    for column in REGIONAL_DRIVERS:
        df[f"{column}_region"] = df.groupby(["region", "date"])[column].transform("mean")
    return df.reset_index(drop=True)


def _mask_impossible(proba: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """No decrease from Low and no increase from Extreme; renormalizes."""
    proba = proba.copy()
    proba[prev <= LEVELS[0], 0] = 0.0
    proba[prev >= LEVELS[1], 2] = 0.0
    return proba / proba.sum(axis=1, keepdims=True)


class TransitionModel:
    """Baseline: training-season frequency of each change given the rating for t-1 and its trend."""

    def __init__(self, band: str, smoothing: float = 1.0):
        self.band = band
        self.smoothing = smoothing

    def _key(self, df: pd.DataFrame) -> pd.Series:
        return df[f"{self.band}_prev"].astype(int).astype(str) + "_" + np.sign(df[f"{self.band}_trend"].fillna(0)).astype(int).astype(str)

    def fit(self, df: pd.DataFrame) -> "TransitionModel":
        counts = pd.crosstab(self._key(df), df[f"{self.band}_change"]).reindex(columns=CHANGES, fill_value=0)
        self.table_ = (counts + self.smoothing).div((counts + self.smoothing).sum(axis=1), axis=0)
        self.overall_ = df[f"{self.band}_change"].value_counts(normalize=True).reindex(CHANGES, fill_value=0).values
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.table_.reindex(self._key(df)).to_numpy()
        proba[np.isnan(proba).any(axis=1)] = self.overall_
        return _mask_impossible(proba, df[f"{self.band}_prev"].to_numpy())


class ChangeModel:
    """Multinomial logistic regression of the change on ratings and drivers."""

    def __init__(self, band: str, C: float = 0.003):
        self.band = band
        self.columns = rating_features(band) + DRIVER_FEATURES
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=C, max_iter=5000))

    def fit(self, df: pd.DataFrame) -> "ChangeModel":
        self.pipeline.fit(df[self.columns], df[f"{self.band}_change"].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        classes = list(self.pipeline.classes_)
        raw = self.pipeline.predict_proba(df[self.columns])
        proba = np.column_stack([raw[:, classes.index(c)] if c in classes else np.zeros(len(df)) for c in CHANGES])
        return _mask_impossible(proba, df[f"{self.band}_prev"].to_numpy())


class OutlookModel:
    """
    Multinomial logistic regression of the change on the change model's log
    probabilities, the direction of CAIC's outlook from the rating for t-1
    (down, same, up or missing) and the rating for t-1.
    """

    def __init__(self, band: str, C: float = 1.0, use_model: bool = True):
        self.band = band
        self.use_model = use_model
        self.model = LogisticRegression(C=C, max_iter=3000)

    def features(self, proba: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        prev = df[f"{self.band}_prev"].to_numpy()
        direction = np.sign(df[f"{self.band}_outlook"].to_numpy() - prev)
        parts = [np.column_stack([direction == c for c in CHANGES] + [np.isnan(direction)]),
                 np.column_stack([prev == level for level in range(LEVELS[0], LEVELS[1] + 1)])]
        if self.use_model:
            parts.insert(0, np.log(np.clip(proba, 1e-4, 1)))
        return np.hstack(parts).astype(float)

    def fit(self, proba: np.ndarray, df: pd.DataFrame) -> "OutlookModel":
        self.model.fit(self.features(proba, df), df[f"{self.band}_change"].astype(int))
        return self

    def predict_proba(self, proba: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        classes = list(self.model.classes_)
        raw = self.model.predict_proba(self.features(proba, df))
        out = np.column_stack([raw[:, classes.index(c)] if c in classes else np.zeros(len(df)) for c in CHANGES])
        return _mask_impossible(out, df[f"{self.band}_prev"].to_numpy())


MODELS = {"transition": TransitionModel, "change_logistic": ChangeModel}


def band_rows(df: pd.DataFrame, band: str) -> pd.DataFrame:
    """Rows where the band is rated on both t-1 and t."""
    return df.dropna(subset=[band, f"{band}_prev"])


def leave_one_season_out(df: pd.DataFrame, band: str) -> pd.DataFrame:
    """
    Held-out-season probabilities of each change, for every model.

    Returns:
        pd.DataFrame: Held-out rows with 'model', 'season', 'polygon', 'date',
                      'change' and one probability column per CHANGE_NAMES.
    """
    rows = band_rows(df, band)
    folds = []
    for season in sorted(rows["season"].unique()):
        train, test = rows[rows["season"] != season], rows[rows["season"] == season]
        for name, make_model in MODELS.items():
            proba = make_model(band).fit(train).predict_proba(test)
            out = test[["season", "polygon", "date", f"{band}_prev"]].rename(columns={f"{band}_prev": "prev"})
            out = out.assign(model=name, band=band, change=test[f"{band}_change"].astype(int).to_numpy())
            folds.append(out.assign(**{n: proba[:, i] for i, n in enumerate(CHANGE_NAMES)}))
    return pd.concat(folds, ignore_index=True)


def outlook_seasons(rows: pd.DataFrame, band: str) -> List[int]:
    """Seasons with CAIC outlooks for the band."""
    return sorted(int(s) for s in rows.loc[rows[f"{band}_outlook"].notna(), "season"].unique())


def held_out_change_proba(rows: pd.DataFrame, band: str, seasons: List[int]) -> pd.DataFrame:
    """Rows of `seasons` with the change model's probabilities from a fit on the other seasons."""
    parts = []
    for season in seasons:
        train, test = rows[rows["season"] != season], rows[rows["season"] == season]
        proba = ChangeModel(band).fit(train).predict_proba(test)
        parts.append(test.assign(**{f"model_{n}": proba[:, i] for i, n in enumerate(CHANGE_NAMES)}))
    return pd.concat(parts) if parts else rows.iloc[:0]


def evaluate_outlook(df: pd.DataFrame, band: str) -> pd.DataFrame:
    """
    Within the seasons with outlooks: the change model's held-out-season
    probabilities, and the outlook model with and without them, scored by
    leaving out one OUTLOOK_BLOCK_DAYS block at a time.

    Returns:
        pd.DataFrame: Held-out rows with 'model' ('change_logistic',
                      'outlook_only', 'change_with_outlook'), 'season',
                      'polygon', 'date', 'prev', 'outlook', 'change' and one
                      probability column per CHANGE_NAMES.
    """
    rows = band_rows(df, band)
    rows = held_out_change_proba(rows, band, outlook_seasons(rows, band)).reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame()
    base = rows[[f"model_{n}" for n in CHANGE_NAMES]].to_numpy()
    block = rows["season"].astype(str) + "_" + ((rows["date"] - rows.groupby("season")["date"].transform("min")).dt.days // OUTLOOK_BLOCK_DAYS).astype(str)
    versions = {"change_logistic": base}
    for name, use_model in [("outlook_only", False), ("change_with_outlook", True)]:
        proba = np.zeros_like(base)
        for b in block.unique():
            test = (block == b).to_numpy()
            model = OutlookModel(band, use_model=use_model).fit(base[~test], rows[~test])
            proba[test] = model.predict_proba(base[test], rows[test])
        versions[name] = proba
    out = rows[["season", "polygon", "date", f"{band}_prev", f"{band}_outlook"]].rename(
        columns={f"{band}_prev": "prev", f"{band}_outlook": "outlook"})
    out = out.assign(band=band, change=rows[f"{band}_change"].astype(int).to_numpy())
    return pd.concat([out.assign(model=name, **{n: p[:, i] for i, n in enumerate(CHANGE_NAMES)})
                      for name, p in versions.items()], ignore_index=True)


def outlook_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Log loss and AUCs per band and model, overall and on days the outlook shows no change."""
    records = []
    for (band, model), g in predictions.groupby(["band", "model"]):
        y, proba = g["change"].to_numpy(), g[CHANGE_NAMES].to_numpy()
        quiet = (np.sign(g["outlook"] - g["prev"]).fillna(0) == 0).to_numpy()
        records.append({
            "band": band, "model": model, "n": len(g), "log_loss": log_loss(y, proba, labels=CHANGES),
            "auc_up": roc_auc_score(y == 1, g["up"]), "auc_down": roc_auc_score(y == -1, g["down"]),
            "quiet_rate_up": np.mean(y[quiet] == 1), "quiet_mean_p_up": g["up"][quiet].mean(),
            "quiet_auc_up": roc_auc_score(y[quiet] == 1, g["up"][quiet]),
            "quiet_rate_down": np.mean(y[quiet] == -1), "quiet_mean_p_down": g["down"][quiet].mean(),
            "quiet_auc_down": roc_auc_score(y[quiet] == -1, g["down"][quiet]),
        })
    return pd.DataFrame(records)


def _brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = (y[:, None] == np.array(CHANGES)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Log loss, Brier score, AUC of an increase and of a decrease, per band, season and model, with skill against the baseline."""
    records = []
    for (band, season, model), g in predictions.groupby(["band", "season", "model"]):
        y, proba = g["change"].to_numpy(), g[CHANGE_NAMES].to_numpy()
        records.append({
            "band": band, "season": season, "model": model, "n": len(g),
            "up_rate": np.mean(y == 1), "down_rate": np.mean(y == -1),
            "log_loss": log_loss(y, proba, labels=CHANGES), "brier": _brier(y, proba),
            "auc_up": roc_auc_score(y == 1, g["up"]), "auc_down": roc_auc_score(y == -1, g["down"]),
            "mean_p_up": g["up"].mean(), "mean_p_down": g["down"].mean(),
        })
    out = pd.DataFrame(records)
    base = out[out["model"] == "transition"].set_index(["band", "season"])
    for metric in ["log_loss", "brier"]:
        ref = out.set_index(["band", "season"]).index.map(base[metric])
        out[f"{metric}_skill"] = 1 - out[metric] / ref.to_numpy()
    return out


def reliability(predictions: pd.DataFrame, edges: List[float] = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)) -> pd.DataFrame:
    """Predicted against observed rate of an increase and of a decrease, in probability bins, all seasons pooled."""
    records = []
    for direction, value in [("up", 1), ("down", -1)]:
        binned = predictions.assign(bin=pd.cut(predictions[direction], list(edges), include_lowest=True),
                                    hit=predictions["change"] == value)
        summary = binned.groupby(["band", "model", "bin"], observed=True).agg(
            n=("hit", "size"), predicted=(direction, "mean"), observed=("hit", "mean")).reset_index()
        records.append(summary.assign(direction=direction))
    return pd.concat(records, ignore_index=True)


def output_columns() -> List[str]:
    """
    Daily output per band: CAIC's rating for t-1, CAIC's outlook for t, the
    three probabilities with the outlook ('p_up') and from the drivers only
    ('drivers_p_up').
    """
    return [f"{band}_{name}" for band in BANDS
            for name in ["caic_today", "caic_outlook", *[f"p_{change}" for change in CHANGE_NAMES],
                         *[f"drivers_p_{change}" for change in CHANGE_NAMES]]]


def train() -> None:
    """
    Fits the change model per band on every season with ratings, and the
    outlook model on its held-out-season probabilities for the seasons with
    outlooks, and saves them.
    """
    table = build_change_table(ld.load_modeling_table(), load_ratings(), load_outlooks())
    models, outlook_models = {}, {}
    for band in BANDS:
        rows = band_rows(table, band)
        models[band] = ChangeModel(band).fit(rows)
        with_outlook = held_out_change_proba(rows, band, outlook_seasons(rows, band))
        base = with_outlook[[f"model_{n}" for n in CHANGE_NAMES]].to_numpy()
        outlook_models[band] = OutlookModel(band).fit(base, with_outlook)
    path = config.PATHS["ARTIFACTS"]["hazard_change_model"]
    path.parent.mkdir(parents=True, exist_ok=True)
    seasons = sorted(int(s) for s in table["season"].unique())
    o_seasons = sorted(int(s) for s in table.loc[table["atl_outlook"].notna(), "season"].unique())
    joblib.dump({"models": models, "outlook_models": outlook_models, "seasons": seasons, "outlook_seasons": o_seasons}, path)
    logging.info(f"Hazard change models trained on seasons {seasons}, outlook models on {o_seasons}; saved to '{path}'.")


def forecast_inputs(date: pd.Timestamp, ratings: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Model inputs for day t, one row per polygon rated on t-1 with drivers for t.

    Args:
        date (pd.Timestamp): Forecast day t.
        ratings (pd.DataFrame): AvID ratings (`caic_ratings.COLUMNS`), or ratings per polygon
                                and day (`caic_ratings.day_ratings`). Only ratings for days
                                before t and, from AvID, the outlook for t are used.
        drivers (pd.DataFrame): Modeling-table rows for day t (`likelihood_forecast.forecast_table`).
    """
    outlooks = None
    if "position" in ratings:
        outlooks = caic_ratings.outlook_ratings(ratings)
        outlooks = outlooks[outlooks["date"] == date]
        ratings = caic_ratings.day_ratings(ratings)
    known = ratings[ratings["date"] < date]
    rated_yesterday = known.loc[known["date"] == date - pd.Timedelta(days=1), ["polygon"]]
    # A row for day t without ratings, so the lags for t line up
    ratings = pd.concat([known, rated_yesterday.assign(date=date)], ignore_index=True)
    drivers = drivers.assign(region=drivers["polygon"].map(ld.polygon_regions()))
    return build_change_table(drivers, ratings.sort_values(["polygon", "date"]), outlooks)


def predict(date: pd.Timestamp, history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Probability that each band's rating goes down, stays or goes up on day t, for every polygon CAIC rated on t-1,
    with CAIC's outlook and from the drivers only.

    Args:
        date (pd.Timestamp): Forecast day t.
        history (Optional[pd.DataFrame]): AvID ratings (`caic_ratings.COLUMNS`). By default the
                                          current forecasts are pulled and added to the history.

    Returns:
        pd.DataFrame: 'polygon' plus `output_columns()`.
    """
    fitted = joblib.load(config.PATHS["ARTIFACTS"]["hazard_change_model"])
    history = caic_ratings.fetch_current() if history is None else history
    rows = forecast_inputs(date, history, likelihood_forecast.forecast_table(date))
    if rows.empty:
        logging.warning(f"No CAIC ratings for {(date - pd.Timedelta(days=1)).date()}; no rating change forecast for {date.date()}.")
        return pd.DataFrame(columns=["polygon", *output_columns()]).astype({"polygon": int})

    outlook_models = fitted.get("outlook_models")
    if outlook_models is None:
        logging.warning("The hazard change artifact has no outlook models; retrain it. Only drivers-only probabilities are given.")
    out = rows[["polygon"]].copy()
    for band in BANDS:
        rated = rows[f"{band}_prev"].notna().to_numpy()
        drivers_only = np.full((len(rows), len(CHANGES)), np.nan)
        with_outlook = np.full((len(rows), len(CHANGES)), np.nan)
        if rated.any():
            drivers_only[rated] = fitted["models"][band].predict_proba(rows[rated])
            if outlook_models is not None:
                with_outlook[rated] = outlook_models[band].predict_proba(drivers_only[rated], rows[rated])
        out[f"{band}_caic_today"] = rows[f"{band}_prev"].to_numpy()
        out[f"{band}_caic_outlook"] = rows[f"{band}_outlook"].to_numpy()
        for i, change in enumerate(CHANGE_NAMES):
            out[f"{band}_p_{change}"] = with_outlook[:, i]
        for i, change in enumerate(CHANGE_NAMES):
            out[f"{band}_drivers_p_{change}"] = drivers_only[:, i]
    return out[["polygon", *output_columns()]].reset_index(drop=True)


def evaluate() -> None:
    """Held-out-season evaluation, and the outlook model within the outlook seasons; writes results to OUTPUT_DIR."""
    table = build_change_table(ld.load_modeling_table(), load_ratings(), load_outlooks())
    predictions = pd.concat([leave_one_season_out(table, band) for band in BANDS], ignore_index=True)
    metrics, calibration = scores(predictions), reliability(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    calibration.to_csv(OUTPUT_DIR / "reliability.csv", index=False)
    outlook = pd.concat([evaluate_outlook(table, band) for band in BANDS], ignore_index=True)
    outlook_metrics = outlook_scores(outlook)
    outlook.to_csv(OUTPUT_DIR / "outlook_predictions.csv", index=False)
    outlook_metrics.to_csv(OUTPUT_DIR / "outlook_metrics.csv", index=False)

    pd.set_option("display.width", 200)
    logging.info(f"Held-out-season scores:\n{metrics.round(3).to_string(index=False)}")
    logging.info(f"Reliability:\n{calibration.round(3).to_string(index=False)}")
    logging.info(f"With CAIC's outlook, {OUTLOOK_BLOCK_DAYS}-day blocks held out:\n{outlook_metrics.round(3).to_string(index=False)}")
    logging.info(f"Results saved to '{OUTPUT_DIR}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate, train or run the danger rating change model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Fit on all seasons with ratings and save the models.")
    group.add_argument("--date", type=str, help="Forecast day, YYYY-MM-DD: pull CAIC's forecasts and predict.")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.date:
        print(predict(pd.Timestamp(datetime.strptime(args.date, "%Y-%m-%d"))).round(3).to_string(index=False))
    else:
        evaluate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # Run through the imported module, so saved models refer to hazard_change.ChangeModel, not __main__
    import hazard_change
    hazard_change.main()
