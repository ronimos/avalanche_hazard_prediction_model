# Avalanche Likelihood Model — Methods

Oct 28, 2026 · Ron Simenhois

## Objective

We estimate, for each of the 147 backcountry forecast polygons and each day, the probability that at least one avalanche occurs. The model treats avalanche reports as incomplete: a report confirms an avalanche, but no report does not rule one out.

This replaces the event model in the current pipeline, a positive-unlabeled RandomForest. Its reported test AUC of 0.90 came from a random split of polygon-days. Holding out a whole season, its AUC fell to 0.48 and 0.55, below a day-of-season plus polygon baseline (0.68 and 0.58). It was also trained on only 7 polygons and applied to all 147.

Danger-rating prediction is out of scope until the likelihood model shows skill on held-out seasons.

## Data

Three seasons have both avalanche observations and SNOWPACK output: 2023-24, 2024-25 and 2025-26. A season runs October to September and is labeled by its starting year (2025 = October 2025 to September 2026).

| Source | Location | Coverage |
| --- | --- | --- |
| Avalanche observations | `data/raw/caic-avalanche-records-2026-10-28.csv` (CAIC export) | 11,199 rows, 2023-10-15 to 2026-10-27 |
| Forecast polygons | `data/raw/BC_Polygons.geojson` | 147 polygons |
| Snowpack and weather drivers | `/ssd/snowpack/fcst/<season>/zone<NNN>/<station>/<station>_res.smet` | One SNOWPACK station per polygon, nearest to its centroid; 107,900 polygon-days over three seasons |

Zone folders changed in 2025 (90 zones before, 134 after), so the same station sits in a different zone folder by season; stations are looked up by ID.

Each row of the export is one avalanche, or a group of similar avalanches whose count is in the `#` column. Rows that share an observation ID come from the same report, so they are kept as separate records.

### Assigning avalanches to polygons

Each record is assigned to the polygon that contains its latitude and longitude (`src/avalanche_observations.py`).

1. Drop exact duplicate rows: 118 of 11,199.
2. Take the occurrence date from `Date`, converted to Mountain time. `Date Known` says whether that date is known (30% of records) or estimated (70%).
3. Test each point against the polygon shapes and keep the polygon that contains it.
4. Drop records outside every polygon: 99 of 11,081. Of these, 88 are outside Colorado and 11 lie 1 to 5 km outside a polygon edge.

The result is 10,982 records in 108 of the 147 polygons, summarized as 4,438 polygon-days with at least one recorded avalanche.

| Season | Records | Avalanches | Polygon-days with activity |
| --- | --- | --- | --- |
| 2023-24 | 4,030 | 6,029 | 1,728 |
| 2024-25 | 4,166 | 5,850 | 1,611 |
| 2025-26 | 2,756 | 4,029 | 1,085 |
| 2026-27 (to Oct 27) | 30 | 72 | 14 |

Records are very uneven across polygons. The median polygon has 32 avalanches over three seasons, the busiest has 2,007, and 39 have none. That spread reflects where observers travel as much as where avalanches happen.

## Labels and the positive-unlabeled problem

The unit of analysis is a polygon-day: 147 polygons over November to May of each season. A polygon-day is labeled 1 when at least one D2 or larger avalanche is recorded in the polygon on that day, and 0 otherwise.

| Season | D2+ polygon-days | Natural-trigger polygon-days | Any-avalanche polygon-days |
| --- | --- | --- | --- |
| 2023-24 | 814 | 1,083 | 1,728 |
| 2024-25 | 714 | 1,029 | 1,611 |
| 2025-26 | 404 | 709 | 1,085 |

A 0 means no avalanche was reported, not that none occurred. An avalanche enters the record only if it happens and someone sees and reports it. Reporting depends on where observers travel, the weather, weekends, and how much effort the forecast center puts into field work in a given season. The model therefore separates two probabilities: that an avalanche occurred, and that it was reported given that it occurred (see Model).

Label choices that differ from the previous pipeline:

- **Occurrence date, not report date.** The previous pipeline used the report timestamp. The export's `Date` is the estimated or known occurrence date.
- **No forward smearing.** The previous label marked a day positive if an event fell anywhere from the day before to two days after. That leaks future observations into today's label and multiplied positives about 2.4 times.
- **Date uncertainty is kept explicit.** 70% of dates are estimated. A sensitivity run will let an estimated-date avalanche count on its date or the day before.
- **D2+ as the main label.** D2+ avalanches can bury, injure or kill a person, and their larger debris is easier to see from a distance than D1. Natural-trigger and any-size labels are kept as alternatives.

## Features

The model uses 14 drivers chosen for known avalanche physics, down from the previous pipeline's 595 (3,624 feature pairs with |r| > 0.95, against 256 event days). All come from SNOWPACK's `.smet` summaries, read with [xsnow](https://gitlab.com/avacollabra/postprocessing/xsnow) (`src/snowpack_drivers.py`), so none need layer-by-layer parsing of the `.pro` profile. The `.smet` timestamps are UTC every 6 hours; they are converted to Mountain time and grouped by local date.

| Driver | `.smet` fields | Daily summary |
| --- | --- | --- |
| New snow | `HN24` | Depth over the last 24 h, and summed over 3 days (m) |
| Load | `SWE` | Change over 24 h and over 3 days (kg m-2) |
| Wind | `VW` | Daily mean and maximum wind speed (m s-1) |
| Warming and wetting | `TA`, `MS_Rain`, `MS_Water` | Max air temperature and its 24 h change (°C); daily rain; liquid water in the snowpack |
| Natural stability | `Sn`, `zSn` | Daily minimum of Sn38 over the four slopes, and its depth |
| Skier stability | `Ss`, `zSs` | Daily minimum of Sk38 over the four slopes, and its depth |
| Snow depth | `HS_mod` | Modeled snow height (m) |
| Season | (date) | Day of season, as a smooth term |

Some fields can't be used as they are in these runs:

- `wind_trans24` is always 0 and `VW_drift` always equals `VW`, so wind loading is represented by wind speed alone.
- `HN72_24` equals `HN24` in all but 23 of 1,018 timesteps, so 72 h new snow is summed from daily `HN24`.
- The 24 h precipitation field changed between seasons (`PSUM24`, total precipitation, before 2025; `HNW24`, new-snow water equivalent, from 2025), so load comes from SWE change, which is the same in every season.
- Stability indices are capped at 6, the value SNOWPACK writes when it finds no weak layer to evaluate. The cap appears almost only where snow is under 0.5 m deep: 22%, 13% and 46% of December to March polygon-days for Sk38 in 2023-24, 2024-25 and 2025-26. Capped values get an indicator rather than being treated as measured stability.

Each SNOWPACK station runs five simulations: a flat field (`<station>_res`) and four 38° slopes facing north, east, south and west (`<station>1_res` to `<station>4_res`), as stated in the `SlopeAngle` and `SlopeAzi` file headers. Loading and weather drivers come from the flat field. Stability drivers take the least stable of the four slopes, because one unstable aspect is enough to produce an avalanche.

Missing values stay missing. The previous pipeline replaced them with 0, which made "no weak layer found" and "no data" look like a measured zero. An explicit missing indicator is used where a driver can be absent, such as a stability minimum before the snowpack forms (4% of November to May polygon-days for Sn38, 2% for Sk38). Drivers are standardized using the training seasons only.

## Model

The model is a Bayesian hierarchical occupancy-detection model. Ecologists use this structure to estimate where a species lives from surveys that can miss it. Here the "species" is a D2+ avalanche and the "survey" is the observer network. A record on polygon p and day t appears only when an avalanche occurred (probability ψ) and was reported (probability d):

```math
\begin{aligned}
z_{p,t} &\sim \mathrm{Bernoulli}(\psi_{p,t}) && \text{avalanche occurred}\\
y_{p,t} \mid z_{p,t} &\sim \mathrm{Bernoulli}(z_{p,t}\, d_{p,t}) && \text{avalanche recorded}\\
\operatorname{logit} \psi_{p,t} &= a_0 + f(\text{day of season}) + \boldsymbol{\beta}^{\top} \mathbf{x}_{p,t} + \gamma_{r[p],t}\\
\operatorname{logit} d_{p,t} &= \delta_{p} + \eta_{s} + \lambda\, \text{weekend or holiday}_{t}
\end{aligned}
```

![Occupancy-detection model: snowpack drivers feed occurrence, observer effort feeds detection, and a record needs both](figures/model_structure.png)

Snowpack drivers only move ψ and observer effort only moves d, which is what lets the model tell a quiet day from an unwatched one.

**Occurrence** (ψ) depends on the snowpack and weather drivers x from the Features table:

- a\_0 is a statewide intercept. Occurrence differs between polygons only through the drivers (see below for why).
- f is a smooth seasonal curve.
- γ\_{r,t} is a shared effect for forecast region r (the 11 CAIC regions) on day t. A storm cycle raises avalanche probability across a whole region at once, so days within a region are not treated as independent. Its value on a future day is unknown, so forecasts integrate over it.

**Detection** (d) depends on observer-effort covariates w, which are kept separate from the snowpack drivers:

- δ\_p is a polygon reporting level, drawn from its region's mean, which is drawn from a statewide mean μ\_d. Polygons with few records are pulled toward their region, and every polygon gets a value. Highway corridors and ski-area boundaries are watched daily and remote polygons rarely, so this is where most of the difference between polygons belongs.
- η\_s is a season-level reporting effort with a fixed N(0, 0.5) prior. Two training seasons can't estimate the spread of season effects (estimating it produced a funnel and divergent transitions). A new season draws from the same prior.
- λ is the effect of weekends, federal holidays and December 24 to January 1. Visibility is not included: the only storm measures are snowfall and wind, which are occurrence drivers.

**What the model can and cannot separate.** With one report stream per polygon-day and records this rare, logit(ψ·d) is close to logit ψ + logit d. A polygon effect on occurrence and one on detection can't be told apart, so the polygon baseline sits in detection only. Until access data separates them, differences in avalanche terrain between polygons are absorbed into δ\_p. The statewide detection level μ\_d has a prior with median 0.2 (normal on the logit scale, sd 0.75), which anchors the occurrence intercept. Rankings of days don't depend on that prior; absolute probabilities do. A better detection prior will come from polygons with near-complete monitoring, such as highway avalanche paths.

**Fitting.** The model (`src/bayes_model.py`) is fit with NUTS in PyMC using the NumPyro sampler: 4 chains, 1,000 tuning and 500 kept draws, target acceptance 0.95. Coefficients get normal priors with sd 1 on standardized drivers, and group standard deviations get half-normal priors. A fit passes with R-hat below 1.01, bulk effective sample size above 400 and no divergent transitions. One fold takes about 8 minutes on 62,000 polygon-days.

## Evaluation

The model is scored by leave-one-season-out validation: fit on two seasons, predict the third, and repeat for all three. Random splits of polygon-days are not used. Neighboring days share storms and rolling-window drivers, and in the previous pipeline that inflated AUC from about 0.5 to 0.90.

All fitting choices are made inside the training seasons, including standardization, priors and model variants. The held-out season is scored once.

| Metric | What it measures | Notes |
| --- | --- | --- |
| ROC AUC, per season | Ranking of all polygon-days by likelihood | Mostly rewards knowing which polygons get reports at all |
| Within-polygon AUC, per season | Ranking of each polygon's days against each other, averaged with weights by events | The main skill measure: it does not depend on detection differences between polygons |
| Average precision, per season | Ranking quality among the rare positives | Compared with that season's base rate |
| Brier score of ψ·d, per season | Calibration of the probability of a record | ψ alone cannot be checked directly, because true occurrence is never observed |
| Skill by polygon group | Whether pooling helps | Polygons grouped by number of records: none, few, many |

The model has to beat two baselines on held-out seasons:

1. **Climatology:** a smooth day-of-season curve plus a fixed effect per polygon.
2. **Driver logistic regression:** climatology plus the 14 drivers and the stability-cap indicators, without pooling or a detection model. This shows what the hierarchical structure adds.

The harness is `src/evaluation.py`; any model with `fit(table)` and `predict(table)` can be scored.

### Baseline results

Held-out season results for the D2+ label (`results/evaluation/season_metrics_d2plus.csv`):

| Held-out season | Recorded D2+ polygon-days | Climatology within-polygon AUC | Drivers within-polygon AUC | Climatology average precision | Drivers average precision |
| --- | --- | --- | --- | --- | --- |
| 2023-24 | 803 (2.6%) | 0.72 | 0.79 | 0.19 | 0.27 |
| 2024-25 | 708 (2.3%) | 0.71 | 0.77 | 0.21 | 0.29 |
| 2025-26 | 404 (1.3%) | 0.80 | 0.83 | 0.19 | 0.29 |

The drivers add skill in every held-out season. Within-polygon AUC rises by 0.03 to 0.07, and average precision rises by 0.08 to 0.10. The natural-trigger and any-size labels show the same pattern, with within-polygon AUC gains of 0.04 to 0.07.

Pooled AUC is 0.89 to 0.94 for both baselines. Climatology alone reaches it, which confirms that pooled AUC mostly measures where reports come from rather than when avalanches happen.

Two caveats:

- 2023-24 and 2024-25 drivers come from the `output/` SNOWPACK runs, and 2025-26 from the operational forecast runs. If the earlier runs used analyzed rather than forecast weather, their driver skill would overstate what a forecast can achieve. The 2025-26 forecast-run season scores highest, so there is no sign of this yet.
- With three seasons, these numbers have no confidence intervals yet. A block bootstrap over storm cycles is the next addition to the harness.

Posterior predictive checks compare simulated and observed records by season, by polygon group and by month. Reporting effort is expected to differ between seasons, so the check confirms that η\_s absorbs it.

### Bayesian model results

The occupancy-detection model scores the same as the driver logistic regression on held-out seasons (`results/evaluation/`):

| Held-out season | Within-polygon AUC, drivers | Within-polygon AUC, Bayesian | Paired difference (95% interval) | Average precision, Bayesian | Max R-hat |
| --- | --- | --- | --- | --- | --- |
| 2023-24 | 0.789 | 0.789 | 0.000 (−0.004 to 0.004) | 0.27 | 1.007 |
| 2024-25 | 0.774 | 0.771 | −0.003 (−0.010 to 0.003) | 0.28 | 1.011 |
| 2025-26 | 0.830 | 0.830 | 0.000 (−0.010 to 0.011) | 0.28 | 1.009 |

All three fits had no divergent transitions. The 2024-25 fold misses the R-hat bar slightly (1.011, on polygon detection levels), so it needs longer chains.

The structure doesn't improve forecasts yet, for two reasons:

- **Ranking comes from the same drivers.** Within a polygon, both models rank days by the same linear combination of drivers. The detection side adds only a weekend effect, which matters little for ranking.
- **The storm-day effect can't be forecast.** γ has a large spread (sd 1.4 to 1.8 on the logit scale), so regional avalanche cycles explain much of what the drivers miss. But γ for a future day is unknown, and forecasts integrate it out.

What the model does provide: an occurrence probability separate from reporting, uncertainty on every prediction, and an estimate for all 147 polygons, including the 39 with no records.

Driver effects (posterior means on standardized drivers) that are consistent across all three folds:

| Driver | Effect on occurrence, by held-out fold |
| --- | --- |
| Lowest natural stability Sn38 | −0.65 to −1.00: lower stability, more avalanches |
| New snow over 72 h | +0.54 to +0.98 |
| Daily maximum wind | +0.11 to +0.37 |
| Daily maximum air temperature | −0.16 to −0.46 |
| 24 h warming | +0.14 to +0.23 |
| Natural stability at the cap of 6 | +0.96 to +1.22, offsetting the linear Sn38 term where no weak layer exists |

Skier stability Sk38 and its depth change sign between folds, so their effects are not reliable.

### Recent regional activity

Avalanche cycles last several days and cover whole regions, so recorded activity from the days before a forecast could carry information the snowpack drivers miss. Three features are added (`src/likelihood_data.py`), each over a 2-day window ending `lag` days before the forecast day:

- whether the polygon itself had any record;
- the number of D2+ avalanches recorded in its forecast region (log of 1 + count);
- the number of avalanches of any size recorded in its region (log of 1 + count).

The lag decides what is available in practice. The forecast runs around 08:16 and predicts the next day. At lag 1, the window ends on the day the forecast is issued, so it would use reports from that same morning and day, which aren't in yet. Lag 2, through the previous day, is the earliest realistic setting, and even then some of that day's reports arrive later. The export has no report timestamps to measure the delay.

Gain from the activity features over the snowpack drivers alone, D2+ label (paired 95% intervals from the block bootstrap):

| Held-out season | Within-polygon AUC gain, lag 1 | Within-polygon AUC gain, lag 2 | Average precision gain, lag 1 | Average precision gain, lag 2 |
| --- | --- | --- | --- | --- |
| 2023-24 | +0.014 (0.000 to 0.021) | +0.003 (−0.012 to 0.009) | +0.039 (0.018 to 0.054) | +0.017 (0.003 to 0.029) |
| 2024-25 | +0.030 (0.008 to 0.055) | +0.016 (0.001 to 0.034) | +0.040 (0.020 to 0.061) | +0.019 (−0.001 to 0.040) |
| 2025-26 | +0.035 (0.006 to 0.069) | +0.015 (−0.004 to 0.034) | +0.052 (0.015 to 0.085) | +0.018 (−0.012 to 0.042) |

At lag 1 the gain is clear in every season. At the operational lag 2 it is about half as large, and only two of the six comparisons exclude zero. Activity is worth most if reports from the previous day arrive before the forecast runs.

The Bayesian model with the same features (lag 2) converged in all three folds (R-hat at most 1.010, no divergent transitions) and again matches the logistic regression: paired within-polygon AUC differences are 0.000 in every season. Regional D2+ activity has the largest effect (posterior mean +0.32 to +0.47 on the standardized count), the polygon's own recent record adds +0.16 to +0.21, and all-size regional activity adds nothing once D2+ is in. The spread of the storm-day effect barely shrinks (sd 1.38 to 1.71, against 1.41 to 1.77 without activity), so 2-day-old reports explain little of the regional cycles.

## Status and open questions

Work happens on the `dev` branch. The production forecast keeps running the current models until the new one beats both baselines on held-out seasons.

- [x] Load the CAIC export and assign records to polygons (`src/avalanche_observations.py`, with tests)
- [x] Build the daily driver table from `.smet` files for 2023-24 to 2025-26 (`src/snowpack_drivers.py`, with tests)
- [x] Build the evaluation harness and both baselines (`src/evaluation.py`, with tests)
- [x] Fit the occupancy-detection model (`src/bayes_model.py`, with tests)
- [ ] Run posterior predictive checks, and rerun the 2024-25 fold with longer chains
- [x] Add recent regional activity as a forecastable feature (lag 2 by default)
- [ ] Connect the model to `predict.py`, keeping the current output format

Open questions:

- **Observer access by polygon.** Highway paths, ski-area boundaries and trailhead locations would give the detection model its strongest covariates. Is there a layer for these?
- **Detection prior.** Which polygons can be treated as near-fully monitored?
- **Committing the export.** The CSV includes free-text comments and descriptions. Should it go into the public GitHub repo, or stay on the server only?
- **Report delay.** Are most of the previous day's avalanche reports in by about 8 am? If so, lag 1 activity (the larger gain) could be used by running the forecast later in the morning.
- **Thin 2025-26 snowpack.** Median November to May modeled snow depth was 0.27 m, against 0.59 m in 2023-24 and 0.48 m in 2024-25. Confirm this matches observations and is not a change in the forecast runs.
