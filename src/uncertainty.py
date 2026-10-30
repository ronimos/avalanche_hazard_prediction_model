# -*- coding: utf-8 -*-
"""
================================================================================
Prediction Intervals and Posterior Predictive Checks
================================================================================

Fits the occupancy-detection model with recent activity on all seasons but one
and, for the held-out season:
    - saves the posterior mean and 10/50/90% quantiles of the record and
      occurrence probabilities for every polygon-day;
    - simulates the number of records per polygon group and month from the
      posterior predictive and compares it with what was recorded.

If the intervals are honest, recorded counts fall inside the central 80%
of simulated counts about 80% of the time, and their position in the simulated
distribution (the PIT, from 0 to 1) is spread evenly rather than piled at
the ends.

Usage:
    python uncertainty.py --seasons 2023
    python uncertainty.py --summarize

Author: Ron Simenhois
"""
import argparse
import logging

import numpy as np
import pandas as pd

import config
import evaluation as ev
import likelihood_data as ld

OUTPUT_DIR = config.RESULTS_DIR / "uncertainty"
COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def polygon_groups(train: pd.DataFrame, polygons: pd.Series) -> pd.Series:
    """Polygon group (by events in the training seasons) for each polygon, as in the evaluation harness."""
    events = train.groupby("polygon")["y"].sum().reindex(polygons.unique(), fill_value=0)
    return polygons.map(pd.cut(events, ev.GROUP_EDGES, labels=ev.GROUP_NAMES)).astype(str)


def predictive_check(simulated: np.ndarray, cells: pd.DataFrame) -> pd.DataFrame:
    """Recorded count, simulated quantiles and PIT for each cell (one column of `simulated`)."""
    out = cells.copy()
    for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975):
        out[f"sim_q{q * 100:g}"] = np.quantile(simulated, q, axis=0)
    recorded = out["recorded"].to_numpy()
    # Randomized PIT for counts: below, plus a uniform share of ties
    below = (simulated < recorded).mean(axis=0)
    ties = (simulated == recorded).mean(axis=0)
    out["pit"] = below + np.random.default_rng(0).uniform(size=len(out)) * ties
    return out


def run_season(table: pd.DataFrame, season: int) -> None:
    import bayes_model
    train, test = table[table["season"] != season], table[table["season"] == season].reset_index(drop=True)
    model = bayes_model.OccupancyDetectionModel(drivers=ld.DRIVERS + ld.ACTIVITY_FEATURES).fit(train)
    logging.info(f"Held-out {season}: {model.diagnostics}")

    test = test.assign(polygon_group=polygon_groups(train, test["polygon"]))
    intervals = model.predict_interval(test)
    pd.concat([test[["polygon", "date", "season", "y", "polygon_group"]], intervals], axis=1).to_csv(
        OUTPUT_DIR / f"intervals_{season}.csv", index=False)

    keys = test[["polygon_group"]].assign(month=test["date"].dt.month)
    cell, cells = pd.factorize(pd.MultiIndex.from_frame(keys))
    simulated = model.simulate_records(test, cell, len(cells))
    cells = cells.to_frame(index=False, name=list(keys.columns)).assign(
        season=season, recorded=np.bincount(cell, weights=test["y"], minlength=len(cells)))
    predictive_check(simulated, cells).to_csv(OUTPUT_DIR / f"predictive_check_{season}.csv", index=False)


def summarize() -> None:
    checks = pd.concat([pd.read_csv(p) for p in sorted(OUTPUT_DIR.glob("predictive_check_*.csv"))], ignore_index=True)
    bounds = {0.5: ("sim_q25", "sim_q75"), 0.8: ("sim_q10", "sim_q90"), 0.95: ("sim_q2.5", "sim_q97.5")}
    rows = []
    for (group,), g in checks.groupby(["polygon_group"]):
        row = {"polygon_group": group, "cells": len(g), "mean_pit": g["pit"].mean()}
        for level, (lo, hi) in bounds.items():
            row[f"coverage_{int(level * 100)}"] = g["recorded"].between(g[lo], g[hi]).mean()
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
    print(coverage.round(2).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction intervals and posterior predictive checks.")
    parser.add_argument("--seasons", nargs="+", type=int, help="Held-out seasons to run (default: all).")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize()
        return
    table = ld.load_modeling_table(ld.DEFAULT_LABEL)
    for season in args.seasons or sorted(table["season"].unique()):
        run_season(table, season)


if __name__ == "__main__":
    main()
