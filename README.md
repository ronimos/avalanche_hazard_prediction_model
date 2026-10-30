# Machine Learning Pipeline for Avalanche Hazard Forecasting
## 1. Overview
This project forecasts daily avalanche conditions for the 147 Colorado backcountry forecast polygons from simulated snowpack data (SNOWPACK driven by numerical weather predictions), so it doesn't depend on local field observations. Each daily run produces, for every polygon:

- **Dry slab likelihood:** the probability that a D2 or larger dry slab avalanche occurs, whether or not it is reported, with an 80% interval. This comes from the new likelihood model (section 3).
- **Danger rating:** a predicted rating (Low to High) with a confidence score, from the original two-stage pipeline (section 4).

The core difficulty is incomplete data (a positive-unlabeled problem). A recorded avalanche confirms that one happened, but no record doesn't mean none occurred. The likelihood model handles this by modeling occurrence and reporting separately, and reports the probability of occurrence. Its level rests on an assumed reporting rate (a prior of about 20% of D2+ avalanches reported), which the records alone can't pin down; the 80% interval includes that uncertainty.

The output is a CSV, a JSON file and an interactive map for the forecasters.

## 2. Status
The likelihood model is being rebuilt on the `dev` branch; the methods, results and open questions are in [`docs/methods.md`](docs/methods.md).

- The original event model (positive-unlabeled `RandomForestClassifier`) showed no skill on held-out seasons: AUC 0.48 and 0.55, against 0.90 on a random split. It is replaced by the dry slab likelihood model in the output.
- The danger rating model still uses the original event score as an input. Rebuilding it is deferred.
- Wet avalanches are outside the scope of the likelihood model.

## 3. Dry Slab Likelihood Model
A Bayesian hierarchical occupancy-detection model (`bayes_model.py`, PyMC with the NumPyro sampler):

- **Occurrence** depends on daily snowpack drivers from the SNOWPACK `.smet` output (new snow, SWE gain, wind, air temperature, rain, liquid water, natural and skier stability and their depths, snow depth), a smooth seasonal curve, and a regional storm-day effect.
- **Reporting** depends on the polygon (pooled within its forecast region), the season and weekends or holidays.
- On held-out seasons (2023-24 to 2025-26) it ranks days within a polygon with AUC 0.82 / 0.82 / 0.89, against 0.74 / 0.73 / 0.84 for a seasonal-curve baseline.
- It is refit once before each season on all past seasons. Mid-season retraining didn't help.
- Recent recorded activity is not used in operation, because no live source of avalanche records is confirmed and it added little at the operational 2-day lag.

Scripts (in `src/`):

| Script | Purpose |
| --- | --- |
| `avalanche_observations.py` | Cleans the CAIC avalanche records export and assigns records to polygons |
| `snowpack_drivers.py` | Builds daily drivers per polygon from SNOWPACK `.smet` files (read with xsnow) |
| `likelihood_data.py` | Joins drivers and records into the polygon-day modeling table |
| `evaluation.py` | Leave-one-season-out evaluation with a block bootstrap |
| `uncertainty.py`, `retraining.py` | Prediction intervals, predictive checks, retraining backtest |
| `likelihood_forecast.py` | Trains the operational model and predicts one day |

## 4. Danger Rating Pipeline
The original two-stage pipeline, orchestrated by `run_pipeline.py`:

1. **Generate training data** (`make_avalanche_event_training_dataset.py`): reads SNOWPACK `.pro` files with `snowpack_reader.py` and computes features, including 3 to 9-day statistics.
2. **Train event model** (`train_avalanche_event_model.py`): positive-unlabeled `RandomForestClassifier` with a Spy+Bootstrap algorithm. Its score is an input to the next stage.
3. **Train hazard model** (`train_avalanche_hazard_model.py`): `LGBMClassifier` tuned with `GridSearchCV`, with isotonic calibration.
4. **Generate prediction data** (`make_prediction_dataset.py`): features for the target date.
5. **Predict** (`predict.py`, `run_prediction.py`): runs both stages, adds the dry slab likelihood, and optionally draws the map (`plot_results.py`). SHAP values explain the hazard model's predictions.

## 5. Installation
Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Dependencies, including xsnow (installed from GitLab at a pinned commit), PyMC and NumPyro, are locked in `uv.lock`.

```Bash
git clone https://github.com/ronimos/avalanche_hazard_prediction_model.git
cd avalanche_hazard_prediction_model
uv sync --locked --extra dev --python /usr/bin/python3.12
```

Use the same lock in every environment. The saved models are pickled, so a model trained under one version of pandas or scikit-learn may not load under another.

## 6. Usage
Set `SNOWPACK_DATA_ROOT` to the SNOWPACK output folder (e.g. `/ssd/snowpack/fcst`). Otherwise the scripts read `data/external/snowpack`. **With `SNOWPACK_DATA_ROOT` set, the JSON output is written to the live web folder** (`config.PATHS["ARTIFACTS"]["all_predictions"]`), so test runs outside production should redirect it.

**Daily forecast** (after the models are trained). Predicts the given date. The production run (`src/run_predictions.sh`) forecasts tomorrow:
```Bash
python src/run_prediction.py --date YYYY-MM-DD --plot
```
Predictions go to `results/final_hazard_predictions.csv` and the JSON file. The map is saved as `results/prediction_map_YYYY-MM-DD.html`. The Dry Slab Likelihood layer is on by default; its 80% range is in the tooltip and in two optional layers.

Output fields:

| Field | Meaning |
| --- | --- |
| `dry_slab_probability` | Probability that a D2+ dry slab avalanche occurs in the polygon, reported or not |
| `dry_slab_probability_q10`, `dry_slab_probability_q90` | 80% interval of that probability |
| `dry_slab_record_probability` | Probability that one occurs and is recorded; the part that can be checked against records |
| `predicted_hazard`, `confidence` | Danger rating (1 to 4) and its calibrated probability |
| `calibrated_proba_hazard_1` .. `_4` | Calibrated probability of each rating |
| `avalanche_event_propability` | Original event score (input to the danger rating) |

If the likelihood step fails, the other fields are still written.

**Train the likelihood model** (once before each season, about 20 minutes):
```Bash
python src/avalanche_observations.py     # needs the CAIC export in data/raw
python src/snowpack_drivers.py           # about 6 minutes
python src/likelihood_forecast.py --train
```

**Evaluate models on held-out seasons:**
```Bash
python src/evaluation.py --models climatology driver_logistic bayes --reference driver_logistic
```

**Retrain the danger rating pipeline:**
```Bash
python src/run_pipeline.py --date YYYY-MM-DD
```
