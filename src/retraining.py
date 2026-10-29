# -*- coding: utf-8 -*-
"""
================================================================================
Mid-Season Retraining Backtest
================================================================================

Replays each season as if it were being forecast live. The model starts from
the other seasons only and is retrained on a schedule (weekly or monthly from
Dec 15). Each retrain adds the current season's polygon-days up to
`GAP_DAYS` before the retrain date, since the latest reports are still coming
in. The retrained model then predicts the days until the next retrain. The
other seasons stand in for past seasons, as in leave-one-season-out.

Variants:
    static         fit on the other seasons once, never retrained (reference)
    refit          logistic regression refit on the other seasons plus the
                   season so far
    weighted       as refit, with the season so far weighted CURRENT_WEIGHT times
    recalibrate    static model, with an offset on the logit scale refit to
                   the season so far (corrects the season's overall level only)
    bayes_static   occupancy-detection model fit on the other seasons once
    bayes_refit    occupancy-detection model refit with the season so far; the
                   season's own effect is then estimated, not drawn from the prior

Every variant scores the same polygon-days (Dec 15 to the end of the season),
so differences are paired.

Usage:
    python retraining.py --variants static refit weighted recalibrate --schedule weekly
    python retraining.py --variants bayes_static bayes_refit --schedule monthly --seasons 2023
    python retraining.py --summarize

Author: Ron Simenhois
"""
import argparse
import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

import config
import evaluation as ev
import likelihood_data as ld

RETRAIN_START = "12-15"
SEASON_END = "05-31"
# Days of labels left out before each retrain, for reports that haven't arrived yet
GAP_DAYS = 3
# Weight of the current season's polygon-days in the weighted refit
CURRENT_WEIGHT = 3.0
SCHEDULES = {"weekly": pd.DateOffset(days=7), "monthly": pd.DateOffset(months=1)}

OUTPUT_DIR = config.RESULTS_DIR / "retraining"


def retrain_dates(season: int, schedule: str) -> List[pd.Timestamp]:
    """Retrain dates for a season, from Dec 15 to the end of the season."""
    return list(pd.date_range(f"{season}-{RETRAIN_START}", f"{season + 1}-{SEASON_END}", freq=SCHEDULES[schedule]))


def _bayes(**kwargs):
    import bayes_model
    return bayes_model.OccupancyDetectionModel(drivers=ld.DRIVERS + ld.ACTIVITY_FEATURES, **kwargs)


def _recalibrated(p_seen: np.ndarray, y_seen: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Shifts p_test on the logit scale by the offset that best fits the season so far."""
    if y_seen.sum() == 0:
        return p_test
    z = logit(np.clip(p_seen, 1e-9, 1 - 1e-9))

    def log_loss(a):
        q = np.clip(expit(z + a), 1e-12, 1 - 1e-12)
        return -np.mean(y_seen * np.log(q) + (1 - y_seen) * np.log(1 - q))

    offset = minimize_scalar(log_loss, bounds=(-5, 5), method="bounded").x
    return expit(logit(np.clip(p_test, 1e-9, 1 - 1e-9)) + offset)


def replay_season(table: pd.DataFrame, season: int, variant: str, schedule: str,
                  gap_days: int = GAP_DAYS, current_weight: float = CURRENT_WEIGHT) -> pd.DataFrame:
    """
    Predictions for one season from Dec 15 on, retraining on the schedule.

    Args:
        table (pd.DataFrame): Modeling table from `likelihood_data`.
        season (int): Season replayed as the live one.
        variant (str): One of the variants in the module docstring.
        schedule (str): 'weekly' or 'monthly'.
        gap_days (int): Days of the current season's labels left out before each retrain.
        current_weight (float): Weight of the season so far, for the 'weighted' variant.

    Returns:
        pd.DataFrame: polygon, date, season, y and p for the scored days, with
                      'retrained' (the retrain date that produced each prediction).
    """
    other = table[table["season"] != season]
    current = table[table["season"] == season]
    dates = retrain_dates(season, schedule)
    static = None
    if variant in ("static", "recalibrate"):
        static = ev.activity_logistic_model().fit(other)
    elif variant == "bayes_static":
        static = _bayes().fit(other)

    folds = []
    for start, end in zip(dates, dates[1:] + [pd.Timestamp.max]):
        test = current[(current["date"] >= start) & (current["date"] < end)]
        if test.empty:
            continue
        seen = current[current["date"] < start - pd.Timedelta(days=gap_days)]
        train = pd.concat([other, seen])
        if variant in ("static", "bayes_static"):
            p = static.predict(test)
        elif variant == "recalibrate":
            p = _recalibrated(static.predict(seen), seen["y"].to_numpy(), static.predict(test))
        elif variant == "refit":
            p = ev.activity_logistic_model().fit(train).predict(test)
        elif variant == "weighted":
            weight = np.where(train["season"] == season, current_weight, 1.0)
            p = ev.activity_logistic_model().fit(train, sample_weight=weight).predict(test)
        elif variant == "bayes_refit":
            model = _bayes().fit(train)
            logging.info(f"Season {season}, retrained {start.date()}: {model.diagnostics}")
            p = model.predict(test)
        else:
            raise ValueError(f"Unknown variant '{variant}'.")
        folds.append(test[["polygon", "date", "season", "y"]].assign(p=p, retrained=start))
        logging.info(f"{variant}, season {season}: retrained {start.date()} with {seen['y'].sum()} events so far, "
                     f"{len(test)} polygon-days scored.")
    return pd.concat(folds, ignore_index=True)


def monthly_calibration(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Predicted over recorded events per season and month, for each variant."""
    rows = []
    for name, df in predictions.items():
        g = df.groupby(["season", df["date"].dt.month]).agg(recorded=("y", "sum"), predicted=("p", "sum"))
        rows.append(g.assign(model=name))
    out = pd.concat(rows).rename_axis(["season", "month"]).reset_index()
    out["ratio"] = out["predicted"] / out["recorded"].replace(0, np.nan)
    return out


def summarize(label: str, n_boot: int) -> None:
    """Bootstrap comparison against the static model, and monthly calibration, over every saved run."""
    predictions = {}
    for path in sorted(OUTPUT_DIR.glob(f"*/predictions_*_{label}_*.csv")):
        variant = path.stem.removeprefix("predictions_").rsplit(f"_{label}_", 1)[0]
        name = variant if variant.endswith("static") else f"{variant} ({path.parent.name})"
        predictions.setdefault(name, []).append(pd.read_csv(path, parse_dates=["date"]))
    # The static variants don't depend on the schedule, so they may be saved under both
    predictions = {name: pd.concat(parts, ignore_index=True).drop_duplicates(["polygon", "date"])
                   for name, parts in predictions.items()}
    logging.info(f"Summarizing {sorted(predictions)}.")

    intervals = ev.block_bootstrap(predictions, reference="static", n_boot=n_boot)
    intervals.to_csv(OUTPUT_DIR / f"bootstrap_{label}.csv", index=False)
    monthly_calibration(predictions).to_csv(OUTPUT_DIR / f"monthly_calibration_{label}.csv", index=False)
    print(intervals[intervals["model"].str.contains(" - ")].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest of mid-season retraining.")
    parser.add_argument("--label", choices=sorted(ld.LABELS), default="d2plus")
    parser.add_argument("--variants", nargs="+", default=["static", "refit", "weighted", "recalibrate"])
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), default="weekly")
    parser.add_argument("--seasons", nargs="+", type=int, help="Seasons to replay (default: all).")
    parser.add_argument("--summarize", action="store_true", help="Compare all saved runs instead of running.")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    if args.summarize:
        summarize(args.label, args.n_boot)
        return

    table = ld.load_modeling_table(args.label)
    output_dir = OUTPUT_DIR / args.schedule
    output_dir.mkdir(parents=True, exist_ok=True)
    for season in args.seasons or sorted(table["season"].unique()):
        for variant in args.variants:
            replay_season(table, season, variant, args.schedule).to_csv(
                output_dir / f"predictions_{variant}_{args.label}_{season}.csv", index=False)


if __name__ == "__main__":
    main()
