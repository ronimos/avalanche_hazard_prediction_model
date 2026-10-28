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
| Snowpack and weather drivers | `/ssd/snowpack/fcst/<season>/zone<NNN>/<station>/<station>_res.pro` and `.smet` | One SNOWPACK station per polygon, nearest to its centroid |

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

The model uses about 12 drivers chosen for known avalanche physics, down from the previous pipeline's 595 (3,624 feature pairs with |r| > 0.95, against 256 event days). Most come straight from SNOWPACK's daily `.smet` summaries, so they need no layer-by-layer parsing of the `.pro` profile.

| Driver | `.smet` fields | Daily summary |
| --- | --- | --- |
| New snow | `HN24`, `HN72_24` | Depth over 24 h and 72 h (cm) |
| Load | `HNW24`, `SWE` | 24 h water equivalent; 3-day change in SWE (kg m-2) |
| Wind loading | `wind_trans24`, `VW`, `VW_drift` | 24 h wind transport; mean and drift wind speed (m s-1) |
| Warming and wetting | `TA`, `MS_Rain`, `MS_Water` | Max air temperature and 24 h change (°C); rain; liquid water in the snowpack |
| Natural stability | `Sn`, `zSn` | Minimum natural stability index Sn38 and its depth |
| Skier stability | `Ss`, `zSs` | Minimum skier stability index Sk38 and its depth |
| Snow depth | `HS_mod` | Modeled snow height (cm) |
| Season | (date) | Day of season, as a smooth term |

Each SNOWPACK station runs five simulations: a flat field (`<station>_res`) and four 38° slopes facing north, east, south and west (`<station>1_res` to `<station>4_res`), as stated in the `SlopeAngle` and `SlopeAzi` file headers. Loading and weather drivers come from the flat field. Stability drivers take the least stable of the four slopes, because one unstable aspect is enough to produce an avalanche.

Missing values stay missing. The previous pipeline replaced them with 0, which made "no weak layer found" and "no data" look like a measured zero. The model will use an explicit missing indicator where a driver can be absent, such as a stability minimum before the snowpack forms. Drivers are standardized using the training seasons only.

## Model

The model is a Bayesian hierarchical occupancy-detection model. Ecologists use this structure to estimate where a species lives from surveys that can miss it. Here the "species" is a D2+ avalanche and the "survey" is the observer network. A record on polygon p and day t appears only when an avalanche occurred (probability ψ) and was reported (probability d):

```math
\begin{aligned}
z_{p,t} &\sim \mathrm{Bernoulli}(\psi_{p,t}) && \text{avalanche occurred}\\
y_{p,t} \mid z_{p,t} &\sim \mathrm{Bernoulli}(z_{p,t}\, d_{p,t}) && \text{avalanche recorded}\\
\operatorname{logit} \psi_{p,t} &= \alpha_{p} + f(\text{day of season}) + \boldsymbol{\beta}^{\top} \mathbf{x}_{p,t} + \gamma_{r[p],t}\\
\operatorname{logit} d_{p,t} &= \delta_{p} + \eta_{s} + \boldsymbol{\lambda}^{\top} \mathbf{w}_{p,t}
\end{aligned}
```

![Occupancy-detection model: snowpack drivers feed occurrence, observer effort feeds detection, and a record needs both](figures/model_structure.png)

Snowpack drivers only move ψ and observer effort only moves d, which is what lets the model tell a quiet day from an unwatched one.

**Occurrence** (ψ) depends on the snowpack and weather drivers x from the Features table:

- α\_p is a polygon baseline, drawn from its zone's mean, which is drawn from the state mean. Polygons with few records are pulled toward their zone, and polygons with none get the zone value. This lets the model predict all 147 polygons.
- f is a smooth seasonal curve.
- γ\_{r,t} is a shared effect for mountain range r on day t. A storm cycle raises avalanche probability across a whole range at once, so days within a range are not treated as independent.

**Detection** (d) depends on observer-effort covariates w, which are kept separate from the snowpack drivers:

- δ\_p is a polygon access level: highway corridors and ski-area boundaries are watched daily, remote polygons rarely.
- η\_s is a season-level reporting effort, which absorbs changes in field programs between seasons.
- w holds day-level factors: weekend or holiday, and visibility (storm days hide debris).

**What the model can and cannot separate.** With one report stream per polygon-day, ψ and d are only separately estimable because they depend on different covariates and because detection gets an informative prior. Without that, the model still ranks days and polygons by avalanche likelihood correctly, but the absolute probability is uncertain. Priors on detection will come from polygons with near-complete monitoring, such as highway avalanche paths.

**Fitting.** The model is fit with NUTS in PyMC, with weakly informative priors on coefficients (normal, sd 1 on standardized drivers) and half-normal priors on group standard deviations. Fits pass only with R-hat below 1.01, adequate effective sample size and no divergent transitions.

## Evaluation

The model is scored by leave-one-season-out validation: fit on two seasons, predict the third, and repeat for all three. Random splits of polygon-days are not used. Neighboring days share storms and rolling-window drivers, and in the previous pipeline that inflated AUC from about 0.5 to 0.90.

All fitting choices are made inside the training seasons, including standardization, priors and model variants. The held-out season is scored once.

| Metric | What it measures | Notes |
| --- | --- | --- |
| ROC AUC, per season | Ranking of polygon-days by likelihood | Does not depend on detection rate, so it is valid under positive-unlabeled labels |
| Average precision, per season | Ranking quality among the rare positives | Compared with that season's base rate |
| Brier score of ψ·d, per season | Calibration of the probability of a record | ψ alone cannot be checked directly, because true occurrence is never observed |
| Skill by polygon group | Whether pooling helps | Polygons grouped by number of records: none, few, many |

The model has to beat two baselines on held-out seasons:

1. **Climatology:** day of season plus polygon. This scored AUC 0.68 and 0.58 on the previous pipeline's two seasons.
2. **Logistic regression** on the same drivers without the hierarchy or the detection model. This shows what the extra structure adds.

Posterior predictive checks compare simulated and observed records by season, by polygon group and by month. Reporting effort is expected to differ between seasons, so the check confirms that η\_s absorbs it.

## Status and open questions

Work happens on the `dev` branch. The production forecast keeps running the current models until the new one beats both baselines on held-out seasons.

- [x] Load the CAIC export and assign records to polygons (`src/avalanche_observations.py`, with tests)
- [ ] Build the daily driver table from `.smet` files for 2023-24 to 2025-26
- [ ] Build the evaluation harness and both baselines
- [ ] Fit the occupancy-detection model and run posterior checks
- [ ] Connect the model to `predict.py`, keeping the current output format

Open questions:

- **Observer access by polygon.** Highway paths, ski-area boundaries and trailhead locations would give the detection model its strongest covariates. Is there a layer for these?
- **Detection prior.** Which polygons can be treated as near-fully monitored?
- **Committing the export.** The CSV includes free-text comments and descriptions. Should it go into the public GitHub repo, or stay on the server only?
- **Reading SNOWPACK output.** [xsnow](https://gitlab.com/avacollabra/postprocessing/xsnow) could replace the custom `.pro` reader. It is licensed AGPL-3.0, while this repo is MIT.
