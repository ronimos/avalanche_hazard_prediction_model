# -*- coding: utf-8 -*-
"""
================================================================================
Natural Share: Which Part of the Dry Slab Likelihood Is Natural
================================================================================

Natural D2+ dry slab avalanches are a subset of all D2+ dry slab avalanches, so
the natural-only likelihood is the all-trigger likelihood times the chance
that a release is natural:

    P(natural release) = P(release) * s[p,t]

s is fit on the polygon-days with a D2+ dry slab record only, with outcome 1
when at least one of that day's records is natural (trigger 'N'; an unknown
trigger counts as not natural, as in the label):

    natural[p,t] ~ Bernoulli(s[p,t])
    logit s = a + f(day of season) + beta . x[p,t] + region effect + polygon effect
              + lambda * weekend_or_holiday[t]

The polygon and region effects are partially pooled, so a polygon with a few
records, all natural, gets a share pulled toward the average rather than 100%.
The weekend term absorbs the extra reports of triggered avalanches on busy
days; the forecast share for occurrence is taken at a weekday, the share for
a record on the actual day.

The share is the share of recorded releases. Reading it as the share of all
releases assumes naturals and triggered avalanches are reported at the same
rate, which they aren't exactly (naturals are mostly found after the fact).

This replaces a separately fit natural-only occupancy model, which could come
out above the all-trigger one in polygons whose few records were all natural.

Held out (seasons 2023/24/25), the all-trigger model times this share ranks
natural days as well as the separate model did (within-polygon AUC
.827/.827/.911 for both) and the predicted share matches the observed one
within 0.03.

Usage:
    python natural_share.py --evaluate     # held-out seasons, against the separate natural model
    python likelihood_forecast.py --train --label natural_dry_slab

Author: Ron Simenhois
"""
import argparse
import logging
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import config
import likelihood_data as ld
from bayes_model import summarize_draws, weekend_or_holiday


def event_table() -> pd.DataFrame:
    """Modeling table rows with a D2+ dry slab record, with 'natural' (0/1) for a natural one among them."""
    table = ld.load_modeling_table("dry_slab")
    natural = ld.load_modeling_table("natural_dry_slab")[["polygon", "date", "y"]].rename(columns={"y": "natural"})
    events = table[table["y"] == 1].merge(natural, on=["polygon", "date"], how="left")
    return events.reset_index(drop=True)


class NaturalShareModel:
    """Hierarchical logistic model of the natural share of recorded D2+ dry slab releases."""

    def __init__(self, drivers=ld.DRIVERS, draws: int = 1000, tune: int = 1000, chains: int = 4,
                 target_accept: float = 0.9, n_predict_draws: int = 400, seed: int = 0):
        self.drivers = drivers
        self.draws, self.tune, self.chains, self.target_accept = draws, tune, chains, target_accept
        self.n_predict_draws = n_predict_draws
        self.seed = seed
        self.diagnostics = {}

    def _design(self, df: pd.DataFrame, fit: bool = False):
        if fit:
            self._features = ColumnTransformer([
                ("drivers", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), self.drivers),
                ("capped", "passthrough", ["Sn_capped", "Ss_capped"]),
            ]).fit(df)
            self._season_curve = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(df[["day_of_season"]])
        X = self._features.transform(df).astype(float)
        B = self._season_curve.transform(df[["day_of_season"]])
        polygon = self._polygon_index.get_indexer(df["polygon"])
        return X, B, polygon

    def fit(self, events: pd.DataFrame) -> "NaturalShareModel":
        regions = ld.polygon_regions()
        self._polygon_index = pd.Index(regions.index)
        region_names = pd.Index(sorted(regions.unique()))
        region_of_polygon = region_names.get_indexer(regions.to_numpy())
        X, B, polygon = self._design(events, fit=True)
        weekend = weekend_or_holiday(events["date"])
        y = events["natural"].to_numpy()
        logging.info(f"Fitting natural share on {len(events)} dry slab record-days ({y.sum()} with a natural record).")

        with pm.Model():
            a = pm.Normal("a", 0.0, 1.5)
            w_season = pm.Normal("w_season", 0.0, 1.0, shape=B.shape[1])
            beta = pm.Normal("beta", 0.0, 0.5, shape=X.shape[1])
            sigma_region = pm.HalfNormal("sigma_region", 1.0)
            region_raw = pm.Normal("region_raw", 0.0, 1.0, shape=len(region_names))
            sigma_polygon = pm.HalfNormal("sigma_polygon", 1.0)
            polygon_raw = pm.Normal("polygon_raw", 0.0, 1.0, shape=len(self._polygon_index))
            u = pm.Deterministic("u", sigma_region * region_raw[region_of_polygon] + sigma_polygon * polygon_raw)
            lam = pm.Normal("lambda_weekend", 0.0, 1.0)
            logit_s = a + pm.math.dot(B, w_season) + pm.math.dot(X, beta) + u[polygon] + lam * weekend
            pm.Bernoulli("natural", logit_p=logit_s, observed=y)
            idata = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains, nuts_sampler="numpyro",
                              target_accept=self.target_accept, random_seed=self.seed, progressbar=False)

        names = ["a", "w_season", "beta", "sigma_region", "sigma_polygon", "u", "lambda_weekend"]
        summary = az.summary(idata, var_names=names, kind="diagnostics")
        self.diagnostics = {"max_rhat": float(summary["r_hat"].max()), "min_ess_bulk": float(summary["ess_bulk"].min()),
                            "divergences": int(idata.sample_stats["diverging"].sum())}
        ok = self.diagnostics["max_rhat"] < 1.01 and self.diagnostics["divergences"] == 0 and self.diagnostics["min_ess_bulk"] > 400
        (logging.info if ok else logging.warning)(f"Convergence {'passed' if ok else 'FAILED'}: {self.diagnostics}")
        self.summary = az.summary(idata, var_names=[n for n in names if n != "u"])

        post = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
        keep = np.random.default_rng(self.seed).choice(post.sizes["sample"], size=min(self.n_predict_draws, post.sizes["sample"]), replace=False)
        self._post = {n: np.asarray(post[n].isel(sample=keep).transpose("sample", ...))
                      for n in ["a", "w_season", "beta", "u", "lambda_weekend"]}
        return self

    def predict_draws(self, df: pd.DataFrame, weekday: bool = True) -> np.ndarray:
        """
        Natural share for each posterior draw, (draws, rows).

        Args:
            weekday (bool): True gives the share of releases (weekend term off);
                            False the share of records on the actual day.
        """
        X, B, polygon = self._design(df)
        weekend = np.zeros(len(df)) if weekday else weekend_or_holiday(df["date"])
        p = self._post
        u = np.where(polygon >= 0, p["u"][:, np.clip(polygon, 0, None)], 0.0)
        return expit(p["a"][:, None] + p["w_season"] @ B.T + p["beta"] @ X.T + u + p["lambda_weekend"][:, None] * weekend)


def natural_summary(record: np.ndarray, occurrence: np.ndarray, share: NaturalShareModel, df: pd.DataFrame,
                    quantiles=(0.1, 0.9)) -> pd.DataFrame:
    """
    Natural-only posterior summary from the all-trigger draws (draws, rows) and the share model.

    Draw i of the all-trigger model is paired with draw i of the share model
    (the two posteriors are independent), so every draw, and so the mean and
    each quantile, is at most its all-trigger value.
    """
    s_occ = share.predict_draws(df, weekday=True)
    s_rec = share.predict_draws(df, weekday=False)
    idx = np.arange(len(record)) % len(s_occ)
    return summarize_draws(record * s_rec[idx], occurrence * s_occ[idx], df.index, quantiles)


def evaluate() -> pd.DataFrame:
    """
    Held-out seasons: the all-trigger model's held-out record probability times
    a share fit without that season, scored on natural labels, next to the
    separately fit natural model.
    """
    import evaluation

    out_dir = config.RESULTS_DIR / "evaluation" / "lag2"
    events = event_table()
    labels = ld.load_modeling_table("natural_dry_slab")[["polygon", "date", "y"]]
    held = pd.read_csv(out_dir / "predictions_bayes_dry_slab.csv", parse_dates=["date"]).drop(columns="y")
    held = held.merge(labels, on=["polygon", "date"])
    table = ld.load_modeling_table("dry_slab").merge(held[["polygon", "date"]], on=["polygon", "date"])
    folds = []
    for season in sorted(held["season"].unique()):
        model = NaturalShareModel(seed=int(season)).fit(events[events["season"] != season])
        rows = table[table["season"] == season]
        share = model.predict_draws(rows, weekday=False).mean(axis=0)
        test = held[held["season"] == season].merge(rows[["polygon", "date"]].assign(share=share), on=["polygon", "date"])
        test["p"] = test["p"] * test["share"]
        folds.append(test)
        # Share calibration on the held-out season's record-days
        ev = events[events["season"] == season]
        logging.info(f"Season {season}: held-out natural share observed {ev['natural'].mean():.3f}, "
                     f"predicted {model.predict_draws(ev, weekday=False).mean():.3f}; {model.diagnostics}")
    predictions = pd.concat(folds, ignore_index=True)
    predictions.to_csv(out_dir / "predictions_share_natural_dry_slab.csv", index=False)
    separate = pd.read_csv(out_dir / "predictions_bayes_natural_dry_slab.csv", parse_dates=["date"])
    scores = pd.concat({"share x all-trigger": evaluation.season_metrics(predictions),
                        "separate natural model": evaluation.season_metrics(separate)}, names=["model"])
    scores.to_csv(out_dir / "metrics_share_natural_dry_slab.csv")
    compared = predictions.merge(separate[["polygon", "date", "p"]], on=["polygon", "date"], suffixes=("", "_separate"))
    logging.info("Records predicted / observed per season:\n"
                 + compared.groupby("season")[["p", "p_separate", "y"]].sum().round(1).to_string())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural share of D2+ dry slab releases.")
    parser.add_argument("--evaluate", action="store_true", help="Score on held-out seasons.")
    args = parser.parse_args()
    if args.evaluate:
        print(evaluate().round(3).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
