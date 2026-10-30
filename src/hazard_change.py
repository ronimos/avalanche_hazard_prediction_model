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
the snowpack drivers for t (SNOWPACK forced by NWP), their regional means, and
recorded activity through t-2 (`likelihood_data`). Ratings move by zone, so
the regional means carry the storm signal.

The baseline knows only the rating for t-1 and its trend: the training
seasons' transition frequencies.

Evaluation is leave-one-season-out, as for the likelihood model:
    - log loss and Brier score of the three-way probability, and their skill
      against the baseline;
    - AUC of the probability of an increase and of a decrease;
    - reliability: predicted against observed rate, binned.

Usage:
    python hazard_change.py

Author: Ron Simenhois
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import likelihood_data as ld

BANDS = ["atl", "ntl", "btl"]
LEVELS = (1, 4)  # Low to High; CAIC's 0 means no rating
# Change from the rating for t-1, in model column order
CHANGES = [-1, 0, 1]
CHANGE_NAMES = ["down", "same", "up"]

# Drivers averaged over the forecast region, since ratings are issued by zone
REGIONAL_DRIVERS = ["HN24", "HN72", "SWE_change_3d", "VW_max", "TA_max_change_24h"]
DRIVER_FEATURES = (ld.DRIVERS + ["day_of_season", "Sn_capped", "Ss_capped"] + ld.ACTIVITY_FEATURES
                   + [f"{c}_region" for c in REGIONAL_DRIVERS])
OUTPUT_DIR = config.RESULTS_DIR / "hazard_change"


def rating_features(band: str) -> List[str]:
    """Rating features for one band: its rating for t-1, its trend, and the other bands' ratings."""
    return [f"{band}_prev", f"{band}_trend"] + [f"{other}_prev" for other in BANDS if other != band]


def load_ratings() -> pd.DataFrame:
    """CAIC danger ratings per polygon and day, with 0 (not rated) as missing."""
    ratings = pd.read_csv(config.PATHS["RAW_DATA"]["danger_ratings"], parse_dates=["date"])
    ratings[BANDS] = ratings[BANDS].where(ratings[BANDS] > 0)
    return ratings.sort_values(["polygon", "date"]).reset_index(drop=True)


def build_change_table(table: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Joins ratings, with what is known at noon on t-1, to the modeling table.

    Args:
        table (pd.DataFrame): Modeling table from `likelihood_data` (drivers for day t).
        ratings (pd.DataFrame): Output of `load_ratings`.

    Returns:
        pd.DataFrame: One row per polygon and day t with, per band, the rating
                      ('atl'), the rating for t-1 ('atl_prev'), the change from
                      t-2 to t-1 ('atl_trend') and the change to t ('atl_change',
                      clipped to -1/0/1), plus regional driver means.
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
    for column in REGIONAL_DRIVERS:
        df[f"{column}_region"] = df.groupby(["region", "date"])[column].transform("mean")
    return df.reset_index(drop=True)


def _mask_impossible(proba: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """No decrease from Low and no increase from High (the data has no Extreme); renormalizes."""
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

    def __init__(self, band: str, C: float = 0.01):
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


def main() -> None:
    table = build_change_table(ld.load_modeling_table(), load_ratings())
    predictions = pd.concat([leave_one_season_out(table, band) for band in BANDS], ignore_index=True)
    metrics, calibration = scores(predictions), reliability(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    calibration.to_csv(OUTPUT_DIR / "reliability.csv", index=False)

    pd.set_option("display.width", 200)
    logging.info(f"Held-out-season scores:\n{metrics.round(3).to_string(index=False)}")
    logging.info(f"Reliability:\n{calibration.round(3).to_string(index=False)}")
    logging.info(f"Results saved to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
