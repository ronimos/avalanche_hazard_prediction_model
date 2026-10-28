# -*- coding: utf-8 -*-
"""
================================================================================
Evaluation Harness: Leave-One-Season-Out Validation
================================================================================

Scores avalanche likelihood models by fitting on all seasons but one and
predicting the held-out season, for every season in turn. Polygon-days are
never split at random: neighboring days share storms and rolling drivers, and
a random split inflated the previous model's AUC from about 0.5 to 0.90.

Any model works if `make_model()` returns an object with `fit(df)` and
`predict(df) -> probabilities`, where df is a slice of the modeling table.
Everything the model learns, including scaling and imputation, happens in
`fit`, so nothing from the held-out season reaches it.

Metrics per held-out season:
    - ROC AUC and average precision: ranking of polygon-days. Valid under
      positive-unlabeled labels as long as reporting doesn't depend on the
      drivers themselves.
    - Brier score: calibration of the probability of a *record*, since true
      occurrence is never observed.
Skill is also reported by polygon group (how many events each polygon had in
the training seasons), to show whether quiet polygons get useful predictions.

Usage:
    python evaluation.py --label d2plus

Author: Ron Simenhois
"""
import argparse
import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import config
import likelihood_data as ld

# Polygon groups by number of event days in the training seasons
GROUP_EDGES = [-1, 0, 10, np.inf]
GROUP_NAMES = ["none", "1-10", "over 10"]


class SklearnModel:
    """Wraps a scikit-learn pipeline in the harness's fit(df) / predict(df) interface."""

    def __init__(self, pipeline: Pipeline, columns: List[str]):
        self.pipeline = pipeline
        self.columns = columns

    def fit(self, df: pd.DataFrame) -> "SklearnModel":
        self.pipeline.fit(df[self.columns], df["y"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[self.columns])[:, 1]


def climatology_model() -> SklearnModel:
    """Baseline 1: smooth day-of-season curve plus a fixed effect per polygon."""
    features = ColumnTransformer([
        ("season_curve", SplineTransformer(n_knots=6, degree=3), ["day_of_season"]),
        ("polygon", OneHotEncoder(handle_unknown="ignore"), ["polygon"]),
    ])
    return SklearnModel(make_pipeline(features, LogisticRegression(C=1.0, max_iter=2000)), ["day_of_season", "polygon"])


def driver_logistic_model() -> SklearnModel:
    """Baseline 2: climatology plus the snowpack drivers, without pooling or a detection model."""
    features = ColumnTransformer([
        ("season_curve", SplineTransformer(n_knots=6, degree=3), ["day_of_season"]),
        ("polygon", OneHotEncoder(handle_unknown="ignore"), ["polygon"]),
        ("drivers", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ld.DRIVERS),
        ("capped", "passthrough", ["Sn_capped", "Ss_capped"]),
    ])
    columns = ["day_of_season", "polygon", *ld.DRIVERS, "Sn_capped", "Ss_capped"]
    return SklearnModel(make_pipeline(features, LogisticRegression(C=1.0, max_iter=5000)), columns)


BASELINES: Dict[str, Callable[[], SklearnModel]] = {
    "climatology": climatology_model,
    "driver_logistic": driver_logistic_model,
}


def leave_one_season_out(table: pd.DataFrame, make_model: Callable) -> pd.DataFrame:
    """
    Fits on all seasons but one and predicts the held-out season, for each season.

    Args:
        table (pd.DataFrame): Modeling table from `likelihood_data`.
        make_model (Callable): Returns a fresh, unfitted model.

    Returns:
        pd.DataFrame: The held-out rows of every fold with 'p' (predicted
                      probability) and 'polygon_group' (by training events).
    """
    folds = []
    for season in sorted(table["season"].unique()):
        train, test = table[table["season"] != season], table[table["season"] == season]
        model = make_model().fit(train)

        train_events = train.groupby("polygon")["y"].sum().reindex(test["polygon"].unique(), fill_value=0)
        groups = pd.cut(train_events, GROUP_EDGES, labels=GROUP_NAMES)
        folds.append(test.assign(p=model.predict(test), polygon_group=test["polygon"].map(groups).astype(str)))
    return pd.concat(folds, ignore_index=True)


def _scores(df: pd.DataFrame) -> pd.Series:
    """AUC, average precision and Brier score for one slice of predictions."""
    both_classes = df["y"].nunique() == 2
    return pd.Series({
        "rows": len(df),
        "events": int(df["y"].sum()),
        "base_rate": df["y"].mean(),
        "auc": roc_auc_score(df["y"], df["p"]) if both_classes else np.nan,
        "average_precision": average_precision_score(df["y"], df["p"]) if both_classes else np.nan,
        "brier": brier_score_loss(df["y"], df["p"]),
    })


def within_polygon_auc(df: pd.DataFrame) -> float:
    """
    Event-weighted mean of each polygon's own AUC over its days.

    Pooled AUC mostly rewards knowing which polygons get reports at all.
    This one only rewards ranking a polygon's days against each other, so it
    doesn't depend on detection differences between polygons.
    """
    per_polygon = [
        (roc_auc_score(g["y"], g["p"]), g["y"].sum())
        for _, g in df.groupby("polygon") if g["y"].nunique() == 2
    ]
    if not per_polygon:
        return np.nan
    aucs, weights = zip(*per_polygon)
    return float(np.average(aucs, weights=weights))


def season_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores per held-out season."""
    scores = predictions.groupby("season")[["y", "p"]].apply(_scores)
    scores["within_polygon_auc"] = predictions.groupby("season")[["polygon", "y", "p"]].apply(within_polygon_auc)
    return scores


def group_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Scores per held-out season and polygon group."""
    return predictions.groupby(["season", "polygon_group"])[["y", "p"]].apply(_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-season-out evaluation of the likelihood baselines.")
    parser.add_argument("--label", choices=sorted(ld.LABELS), default="d2plus")
    args = parser.parse_args()

    table = ld.load_modeling_table(args.label)
    logging.info(f"Modeling table: {len(table)} polygon-days, {table['y'].sum()} with a recorded {args.label} event.")

    output_dir = config.RESULTS_DIR / "evaluation"
    output_dir.mkdir(exist_ok=True)
    by_season, by_group = [], []
    for name, make_model in BASELINES.items():
        predictions = leave_one_season_out(table, make_model)
        predictions[["polygon", "date", "season", "y", "p", "polygon_group"]].to_csv(
            output_dir / f"predictions_{name}_{args.label}.csv", index=False)
        by_season.append(season_metrics(predictions).assign(model=name))
        by_group.append(group_metrics(predictions).assign(model=name))

    by_season = pd.concat(by_season).reset_index().set_index(["model", "season"])
    by_group = pd.concat(by_group).reset_index().set_index(["model", "season", "polygon_group"])
    by_season.to_csv(output_dir / f"season_metrics_{args.label}.csv")
    by_group.to_csv(output_dir / f"group_metrics_{args.label}.csv")
    print(by_season.round(3).to_string())
    print(by_group.round(3).to_string())


if __name__ == "__main__":
    main()
