# -*- coding: utf-8 -*-
"""
================================================================================
Bayesian Occupancy-Detection Model for Avalanche Likelihood
================================================================================

A record on polygon p and day t needs an avalanche to occur (probability psi)
and to be reported (probability d):

    y[p,t] ~ Bernoulli(psi[p,t] * d[p,t])
    logit psi = a0 + f(day of season) + beta . x[p,t] + gamma[region, t]
    logit d   = delta[p] + eta[season] + lambda * weekend_or_holiday[t]
    eta[s]    ~ Normal(0, 0.5), the same prior used for an unseen season; a season
                that is partly in the training data (mid-season retraining)
                keeps its fitted effect
    delta[p]  = mu_d + region effect + polygon effect

Identifiability. With one report stream per polygon-day and rare records,
logit(psi * d) is close to logit psi + logit d, so a polygon effect on
occurrence and one on detection can't be told apart. The polygon baseline
therefore sits in detection, where access differences dominate; occurrence
varies between polygons only through the snowpack drivers. The overall
detection level mu_d gets a weakly informative prior (median 0.2) so the
occurrence intercept is anchored. Rankings don't depend on that prior.

gamma is a shared effect for a forecast region on a day (storm cycles). Its
value on an unseen day is unknown, so predictions integrate over it, as they
do over the season effect eta for an unseen season.

The model follows the evaluation harness interface: fit(df), predict(df).

Author: Ron Simenhois
"""
import logging
import os
from typing import Dict, List

# Let NumPyro run chains in parallel on CPU; must be set before jax is imported
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import likelihood_data as ld

# Prior median 0.2 for the share of D2+ avalanches that get reported in an average polygon
DETECTION_PRIOR_MU = np.log(0.2 / 0.8)
DETECTION_PRIOR_SD = 0.75

# Season-to-season variation in reporting effort, on the logit scale
SEASON_EFFECT_SD = 0.5

# Quadrature nodes for averaging over the unknown storm effect of a region-day
STORM_QUADRATURE_NODES = 15


def weekend_or_holiday(dates: pd.Series) -> np.ndarray:
    """1 on Saturdays, Sundays, US federal holidays and Dec 24 to Jan 1, else 0."""
    dates = pd.to_datetime(dates)
    holidays = USFederalHolidayCalendar().holidays(dates.min(), dates.max())
    winter_break = ((dates.dt.month == 12) & (dates.dt.day >= 24)) | ((dates.dt.month == 1) & (dates.dt.day == 1))
    return (dates.dt.day_of_week.ge(5) | dates.isin(holidays) | winter_break).astype(int).to_numpy()


class OccupancyDetectionModel:
    """Hierarchical occupancy-detection model fit with NUTS (PyMC + NumPyro)."""

    def __init__(self, drivers: List[str] = ld.DRIVERS, draws: int = 500, tune: int = 1000, chains: int = 4,
                 target_accept: float = 0.95, n_predict_draws: int = 400, seed: int = 0):
        self.drivers = drivers
        self.draws, self.tune, self.chains, self.target_accept = draws, tune, chains, target_accept
        self.n_predict_draws = n_predict_draws
        self.seed = seed
        self.diagnostics: Dict[str, float] = {}

    def _design(self, df: pd.DataFrame, fit: bool = False):
        """Driver matrix, season-curve basis and index arrays for a slice of the modeling table."""
        if fit:
            self._drivers = ColumnTransformer([
                ("drivers", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), self.drivers),
                ("capped", "passthrough", ["Sn_capped", "Ss_capped"]),
            ]).fit(df)
            self._season_curve = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit(df[["day_of_season"]])
        X = self._drivers.transform(df).astype(float)
        B = self._season_curve.transform(df[["day_of_season"]])
        polygon = self._polygon_index.get_indexer(df["polygon"])
        region = self._region_of_polygon[polygon]
        return X, B, polygon, region, weekend_or_holiday(df["date"])

    def fit(self, df: pd.DataFrame) -> "OccupancyDetectionModel":
        regions = ld.polygon_regions()
        self._polygon_index = pd.Index(regions.index)
        self._region_names = pd.Index(sorted(regions.unique()))
        self._region_of_polygon = self._region_names.get_indexer(regions.to_numpy())

        X, B, polygon, region, weekend = self._design(df, fit=True)
        region_day, region_days = pd.factorize(pd.MultiIndex.from_arrays([region, df["date"].to_numpy()]))
        season, seasons = pd.factorize(df["season"])
        self._seasons = pd.Index(seasons)
        y = df["y"].to_numpy()
        logging.info(f"Fitting occupancy-detection model on {len(df)} polygon-days ({y.sum()} records), "
                     f"{X.shape[1]} drivers, {len(region_days)} region-days, seasons {list(seasons)}.")

        with pm.Model():
            a0 = pm.Normal("a0", -2.0, 1.5)
            w_season = pm.Normal("w_season", 0.0, 1.0, shape=B.shape[1])
            beta = pm.Normal("beta", 0.0, 1.0, shape=X.shape[1])
            sigma_gamma = pm.HalfNormal("sigma_gamma", 1.0)
            gamma_raw = pm.Normal("gamma_raw", 0.0, 1.0, shape=len(region_days))

            mu_d = pm.Normal("mu_d", DETECTION_PRIOR_MU, DETECTION_PRIOR_SD)
            sigma_region = pm.HalfNormal("sigma_region", 1.0)
            region_raw = pm.Normal("region_raw", 0.0, 1.0, shape=len(self._region_names))
            sigma_polygon = pm.HalfNormal("sigma_polygon", 1.0)
            polygon_raw = pm.Normal("polygon_raw", 0.0, 1.0, shape=len(self._polygon_index))
            delta = pm.Deterministic(
                "delta", mu_d + sigma_region * region_raw[self._region_of_polygon] + sigma_polygon * polygon_raw)
            # Two or three training seasons can't estimate a spread for season effects (it
            # produced a funnel and divergences), so the spread is fixed by the prior
            eta = pm.Normal("eta", 0.0, SEASON_EFFECT_SD, shape=len(seasons))
            lam = pm.Normal("lambda_weekend", 0.0, 1.0)

            logit_psi = a0 + pt.dot(B, w_season) + pt.dot(X, beta) + sigma_gamma * gamma_raw[region_day]
            logit_d = delta[polygon] + eta[season] + lam * weekend
            p_record = pm.math.sigmoid(logit_psi) * pm.math.sigmoid(logit_d)
            pm.Bernoulli("y", p=pt.clip(p_record, 1e-9, 1 - 1e-9), observed=y)

            idata = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains, nuts_sampler="numpyro",
                              target_accept=self.target_accept, random_seed=self.seed, progressbar=False)

        self.idata = idata
        self._check_convergence(idata)
        self._store_posterior(idata)
        return self

    def _check_convergence(self, idata) -> None:
        names = ["a0", "w_season", "beta", "sigma_gamma", "mu_d", "sigma_region", "sigma_polygon", "eta", "lambda_weekend", "delta"]
        summary = az.summary(idata, var_names=names, kind="diagnostics")
        self.diagnostics = {
            "max_rhat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "min_ess_tail": float(summary["ess_tail"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum()),
        }
        ok = self.diagnostics["max_rhat"] < 1.01 and self.diagnostics["divergences"] == 0 and self.diagnostics["min_ess_bulk"] > 400
        (logging.info if ok else logging.warning)(f"Convergence {'passed' if ok else 'FAILED'}: {self.diagnostics}")
        self.summary = az.summary(idata, var_names=[n for n in names if n != "delta"])

    def _store_posterior(self, idata) -> None:
        post = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
        keep = np.random.default_rng(self.seed).choice(post.sizes["sample"], size=min(self.n_predict_draws, post.sizes["sample"]), replace=False)
        self._post = {name: np.asarray(post[name].isel(sample=keep).transpose("sample", ...))
                      for name in ["a0", "w_season", "beta", "sigma_gamma", "delta", "lambda_weekend", "eta"]}

    def _predict_parts(self, df: pd.DataFrame):
        """
        Posterior-mean occurrence and record probabilities, integrating over unseen region-day effects
        and the effect of a season the model wasn't fit on.
        """
        X, B, polygon, region, weekend = self._design(df)
        season = self._seasons.get_indexer(df["season"])
        seen = season >= 0
        region_day, region_days = pd.factorize(pd.MultiIndex.from_arrays([region, df["date"].to_numpy()]))
        rng = np.random.default_rng(self.seed + 1)
        post = self._post
        psi_sum = np.zeros(len(df))
        record_sum = np.zeros(len(df))
        for s in range(len(post["a0"])):
            gamma = rng.normal(0.0, post["sigma_gamma"][s], size=len(region_days))[region_day]
            eta = np.where(seen, post["eta"][s][season], rng.normal(0.0, SEASON_EFFECT_SD))
            psi = expit(post["a0"][s] + B @ post["w_season"][s] + X @ post["beta"][s] + gamma)
            d = expit(post["delta"][s][polygon] + eta + post["lambda_weekend"][s] * weekend)
            psi_sum += psi
            record_sum += psi * d
        n = len(post["a0"])
        return psi_sum / n, record_sum / n

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Probability that an avalanche is recorded (psi * d), comparable with the baselines."""
        psi, record = self._predict_parts(df)
        self.last_occurrence = psi
        return record

    def predict_occurrence(self, df: pd.DataFrame) -> np.ndarray:
        """Probability that an avalanche occurs (psi), whether or not it is reported."""
        return self._predict_parts(df)[0]

    def _linear_parts(self, df: pd.DataFrame, s: int, eta: np.ndarray):
        """Occurrence logit without the region-day effect, and detection probability, for posterior draw s."""
        X, B, polygon, _, weekend = self._design(df)
        post = self._post
        logit_psi = post["a0"][s] + B @ post["w_season"][s] + X @ post["beta"][s]
        d = expit(post["delta"][s][polygon] + eta + post["lambda_weekend"][s] * weekend)
        return logit_psi, d

    def _season_effect(self, df: pd.DataFrame, s: int, rng: np.random.Generator) -> np.ndarray:
        """Season effect for draw s: fitted for a season in the training data, else one draw from the prior."""
        season = self._seasons.get_indexer(df["season"])
        return np.where(season >= 0, self._post["eta"][s][season], rng.normal(0.0, SEASON_EFFECT_SD))

    def predict_draws(self, df: pd.DataFrame):
        """
        Record and occurrence probabilities for each posterior draw.

        The storm effect of a region-day is weather, not something the model is
        unsure about, so each draw averages over it (Gauss-Hermite quadrature).
        The spread across draws is then the model's own uncertainty, including
        the reporting level of a season it hasn't seen.

        Returns:
            Tuple[np.ndarray, np.ndarray]: record and occurrence probabilities,
                                           each (draws, rows).
        """
        nodes, weights = np.polynomial.hermite_e.hermegauss(STORM_QUADRATURE_NODES)
        weights = weights / weights.sum()
        rng = np.random.default_rng(self.seed + 2)
        n = len(self._post["a0"])
        record, occurrence = np.empty((n, len(df))), np.empty((n, len(df)))
        for s in range(n):
            logit_psi, d = self._linear_parts(df, s, self._season_effect(df, s, rng))
            psi = sum(w * expit(logit_psi + self._post["sigma_gamma"][s] * z) for z, w in zip(nodes, weights))
            occurrence[s], record[s] = psi, psi * d
        return record, occurrence

    def predict_interval(self, df: pd.DataFrame, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
        """Posterior mean and quantiles of the record and occurrence probabilities, one row per polygon-day."""
        record, occurrence = self.predict_draws(df)
        out = {"p_record": record.mean(axis=0), "p_occurrence": occurrence.mean(axis=0)}
        for q in quantiles:
            out[f"p_record_q{int(q * 100):02d}"] = np.quantile(record, q, axis=0)
            out[f"p_occurrence_q{int(q * 100):02d}"] = np.quantile(occurrence, q, axis=0)
        return pd.DataFrame(out, index=df.index)

    def simulate_records(self, df: pd.DataFrame, group: np.ndarray, n_groups: int) -> np.ndarray:
        """
        Posterior predictive number of records per group, for checking the model against what was recorded.

        Each draw takes new storm effects for every region-day and, for an
        unseen season, a new season effect, then simulates the records.

        Returns:
            np.ndarray: Simulated record counts, (draws, n_groups).
        """
        _, _, _, region, _ = self._design(df)
        region_day, region_days = pd.factorize(pd.MultiIndex.from_arrays([region, df["date"].to_numpy()]))
        rng = np.random.default_rng(self.seed + 3)
        n = len(self._post["a0"])
        counts = np.empty((n, n_groups))
        for s in range(n):
            logit_psi, d = self._linear_parts(df, s, self._season_effect(df, s, rng))
            gamma = rng.normal(0.0, self._post["sigma_gamma"][s], size=len(region_days))[region_day]
            y = rng.uniform(size=len(df)) < expit(logit_psi + gamma) * d
            counts[s] = np.bincount(group, weights=y, minlength=n_groups)
        return counts

    def driver_effects(self) -> pd.DataFrame:
        """Posterior summary of the occurrence coefficients, labeled by driver."""
        names = list(self._drivers.get_feature_names_out())
        beta = self.summary.loc[[i for i in self.summary.index if i.startswith("beta[")]].copy()
        beta.index = names
        return beta
