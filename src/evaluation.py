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
    python evaluation.py --label dry_slab

Author: Ron Simenhois
"""
import argparse
import logging
from typing import Callable, Dict, List, Optional

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

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> "SklearnModel":
        fit_params = {} if sample_weight is None else {f"{self.pipeline.steps[-1][0]}__sample_weight": sample_weight}
        self.pipeline.fit(df[self.columns], df["y"], **fit_params)
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


def driver_logistic_model(drivers: List[str] = ld.DRIVERS) -> SklearnModel:
    """Baseline 2: climatology plus the snowpack drivers, without pooling or a detection model."""
    features = ColumnTransformer([
        ("season_curve", SplineTransformer(n_knots=6, degree=3), ["day_of_season"]),
        ("polygon", OneHotEncoder(handle_unknown="ignore"), ["polygon"]),
        ("drivers", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), drivers),
        ("capped", "passthrough", ["Sn_capped", "Ss_capped"]),
    ])
    columns = ["day_of_season", "polygon", *drivers, "Sn_capped", "Ss_capped"]
    return SklearnModel(make_pipeline(features, LogisticRegression(C=1.0, max_iter=5000)), columns)


def activity_logistic_model() -> SklearnModel:
    """Driver logistic regression plus recent recorded activity in the polygon and its region."""
    return driver_logistic_model(ld.DRIVERS + ld.ACTIVITY_FEATURES)


BASELINES: Dict[str, Callable[[], SklearnModel]] = {
    "climatology": climatology_model,
    "driver_logistic": driver_logistic_model,
    "activity_logistic": activity_logistic_model,
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
    folds, models = [], {}
    for season in sorted(table["season"].unique()):
        train, test = table[table["season"] != season], table[table["season"] == season]
        model = make_model().fit(train)
        models[season] = model

        train_events = train.groupby("polygon")["y"].sum().reindex(test["polygon"].unique(), fill_value=0)
        groups = pd.cut(train_events, GROUP_EDGES, labels=GROUP_NAMES)
        folds.append(test.assign(p=model.predict(test), polygon_group=test["polygon"].map(groups).astype(str)))
    predictions = pd.concat(folds, ignore_index=True)
    predictions.attrs["models"] = models  # fitted model per held-out season, for diagnostics
    return predictions


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


def _rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC from ranks (Mann-Whitney); fast enough to repeat inside the bootstrap."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(p).rank().to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _fast_within_polygon_auc(polygon: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Vectorized `within_polygon_auc` for the bootstrap: per-polygon ranks, event-weighted."""
    df = pd.DataFrame({"polygon": polygon, "y": y, "p": p})
    df["rank"] = df.groupby("polygon")["p"].rank()
    g = df.assign(pos_rank=df["rank"] * df["y"]).groupby("polygon").agg(n=("y", "size"), n1=("y", "sum"), r1=("pos_rank", "sum"))
    g = g[(g["n1"] > 0) & (g["n1"] < g["n"])]
    if g.empty:
        return np.nan
    auc = (g["r1"] - g["n1"] * (g["n1"] + 1) / 2) / (g["n1"] * (g["n"] - g["n1"]))
    return float(np.average(auc, weights=g["n1"]))


BOOTSTRAP_METRICS = {
    "auc": lambda d, p: _rank_auc(d["y"].to_numpy(), p),
    "within_polygon_auc": lambda d, p: _fast_within_polygon_auc(d["polygon"].to_numpy(), d["y"].to_numpy(), p),
    "average_precision": lambda d, p: average_precision_score(d["y"], p) if d["y"].nunique() == 2 else np.nan,
    "brier": lambda d, p: brier_score_loss(d["y"], p),
}


def block_bootstrap(predictions: Dict[str, pd.DataFrame], reference: str, block_days: int = 7,
                    n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    Confidence intervals for each model's metrics and for its difference from a reference model.

    Days in each held-out season are cut into consecutive blocks of `block_days`,
    covering all polygons, and blocks are resampled with replacement. A block keeps
    a storm cycle and its region-wide effects together, so the intervals account
    for days and polygons not being independent. All models are scored on the same
    resamples, so differences are paired.

    Args:
        predictions (Dict[str, pd.DataFrame]): Model name -> output of `leave_one_season_out`.
        reference (str): Model that differences are taken against.
        block_days (int): Days per block.
        n_boot (int): Number of resamples.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: One row per season, model (or 'model - reference') and metric,
                      with the point estimate and a 95% percentile interval.
    """
    keys = ["polygon", "date", "season", "y"]
    wide = predictions[reference][keys + ["p"]].rename(columns={"p": reference})
    for name, df in predictions.items():
        if name != reference:
            wide = wide.merge(df[keys + ["p"]].rename(columns={"p": name}), on=keys, how="inner")
    models = list(predictions)
    rng = np.random.default_rng(seed)

    rows = []
    for season, d in wide.groupby("season"):
        d = d.reset_index(drop=True)
        block = ((d["date"] - d["date"].min()).dt.days // block_days).to_numpy()
        block_rows = [np.flatnonzero(block == b) for b in np.unique(block)]

        def score(idx):
            sub = d.iloc[idx]
            return {(m, k): f(sub, sub[m].to_numpy()) for m in models for k, f in BOOTSTRAP_METRICS.items()}

        point = score(np.arange(len(d)))
        draws = []
        for _ in range(n_boot):
            picks = rng.integers(len(block_rows), size=len(block_rows))
            draws.append(score(np.concatenate([block_rows[b] for b in picks])))
        draws = pd.DataFrame(draws)

        for metric in BOOTSTRAP_METRICS:
            for m in models:
                series = {m: (point[(m, metric)], draws[(m, metric)])}
                if m != reference:
                    series[f"{m} - {reference}"] = (point[(m, metric)] - point[(reference, metric)],
                                                    draws[(m, metric)] - draws[(reference, metric)])
                for name, (estimate, samples) in series.items():
                    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
                    rows.append({"season": season, "model": name, "metric": metric,
                                 "estimate": estimate, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-season-out evaluation of avalanche likelihood models.")
    parser.add_argument("--label", choices=sorted(ld.LABELS), default=ld.DEFAULT_LABEL)
    parser.add_argument("--models", nargs="+", default=list(BASELINES), help="Models to score; 'bayes' adds the occupancy-detection model.")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--reference", default="climatology", help="Model that bootstrap differences are taken against.")
    parser.add_argument("--activity-lag", type=int, default=ld.DEFAULT_ACTIVITY_LAG,
                        help="Days between the end of the recent-activity window and the forecast day.")
    args = parser.parse_args()

    table = ld.load_modeling_table(args.label, activity_lag=args.activity_lag)
    logging.info(f"Modeling table: {len(table)} polygon-days, {table['y'].sum()} with a recorded {args.label} event.")

    output_dir = config.RESULTS_DIR / "evaluation" / f"lag{args.activity_lag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, by_season, by_group = {}, [], []
    for name in args.models:
        make_model = BASELINES.get(name) or _extra_model(name)
        predictions[name] = leave_one_season_out(table, make_model)
        predictions[name].drop(columns=[c for c in table.columns if c not in ("polygon", "date", "season", "y")]).to_csv(
            output_dir / f"predictions_{name}_{args.label}.csv", index=False)
        by_season.append(season_metrics(predictions[name]).assign(model=name))
        _save_fit_details(name, predictions[name].attrs["models"], output_dir, args.label)
        by_group.append(group_metrics(predictions[name]).assign(model=name))

    by_season = pd.concat(by_season).reset_index().set_index(["model", "season"])
    by_group = pd.concat(by_group).reset_index().set_index(["model", "season", "polygon_group"])
    by_season.to_csv(output_dir / f"season_metrics_{args.label}.csv")
    by_group.to_csv(output_dir / f"group_metrics_{args.label}.csv")
    print(by_season.round(3).to_string())
    print(by_group.round(3).to_string())

    intervals = block_bootstrap(predictions, reference=args.reference, n_boot=args.n_boot)
    intervals.to_csv(output_dir / f"bootstrap_{args.label}.csv", index=False)
    print(intervals.round(3).to_string(index=False))


def _save_fit_details(name: str, models: Dict, output_dir, label: str) -> None:
    """Writes convergence diagnostics and coefficient summaries for models that report them."""
    for season, model in models.items():
        if hasattr(model, "diagnostics"):
            logging.info(f"{name}, held-out {season}: {model.diagnostics}")
            pd.Series(model.diagnostics).to_csv(output_dir / f"diagnostics_{name}_{label}_heldout{season}.csv")
        if hasattr(model, "summary"):
            model.summary.to_csv(output_dir / f"summary_{name}_{label}_heldout{season}.csv")
        if hasattr(model, "driver_effects"):
            model.driver_effects().to_csv(output_dir / f"driver_effects_{name}_{label}_heldout{season}.csv")


def _extra_model(name: str) -> Callable:
    """Models outside this module, imported only when asked for (they may be slow or heavy)."""
    if name == "bayes":
        import bayes_model
        return bayes_model.OccupancyDetectionModel
    if name == "bayes_activity":
        import bayes_model
        return lambda: bayes_model.OccupancyDetectionModel(drivers=ld.DRIVERS + ld.ACTIVITY_FEATURES)
    raise ValueError(f"Unknown model '{name}'.")


if __name__ == "__main__":
    main()
