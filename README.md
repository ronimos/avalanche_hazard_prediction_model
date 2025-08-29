# Machine Learning Pipeline for Avalanche Hazard Forecasting
## 1. Overview
This project provides a complete, end-to-end machine learning pipeline for forecasting daily avalanche hazard levels in data-sparse regions. The core challenge in avalanche forecasting is the lack of complete data; while avalanche occurrences are reliably recorded (positive labels), the absence of a report does not guarantee safety (unlabeled data). This pipeline is specifically designed to address this Positive-Unlabeled (PU) learning problem.

The system uses simulated snowpack data from the SNOWPACK model and numerical weather predictions as its primary inputs, removing the dependency on direct, localized field observations. It employs a two-stage modeling process to first predict the likelihood of an avalanche event and then uses that information to forecast a final, multi-level hazard rating.

The final output is a daily, spatially continuous map of avalanche hazard predictions, complete with confidence scores to aid human forecasters in their decision-making process.

## 2. Key Features
- **Two-Stage Modeling Pipeline:**

    1. **Avalanche Event Model:** A Positive-Unlabeled (PU) learning model (`RandomForestClassifier`) using a custom Spy+Bootstrap algorithm to predict the probability of an avalanche event.

    2. **Avalanche Hazard Model:** A multi-class classifier (`LGBMClassifier`) that uses the output from the event model as a key feature to predict the final hazard rating (Low, Moderate, Considerable, High).

- **Data-Driven:** Relies on simulated SNOWPACK data and weather models, making it applicable to regions without extensive manual observations.

- **Automated Workflow:** The entire process, from data fetching and feature engineering to model training and prediction, is orchestrated by a single command.

- **Model Interpretability:** Uses **SHAP (SHapley Additive exPlanations)** to provide insights into which features are driving the model's predictions.

- **Probabilistic Output:** Provides not just a hazard rating, but also a **confidence** score representing the model's certainty in its prediction.

- **Visualization:** Automatically generates daily interactive forecast maps showing hazard levels, event likelihood, and model confidence.

## 3. Pipeline Workflow
The pipeline is executed by the `run_pipeline.py` script, which orchestrates the following steps in sequence:

1. **Generate Training Data** (`make_avalanche_event_training_dataset.py`):

    - Reads raw SNOWPACK `.pro` files using the hardware-adaptive `snowpack_reader.py`.

    - Calculates hundreds of features, including temporal statistics (3 to 9-day means/standard deviations) for snowpack and weather variables.

    - Creates the master feature and target datasets.

2. **Train Event Model (**`train_avalanche_event_model.py`**):**

    - Trains the PU learning model to distinguish between days with known avalanches and unlabeled days.

    - Saves the trained model, scaler, and a file of its predictions on the historical data, which will be used as a feature in the next stage.

3. **Train Hazard Model (**`train_avalanche_hazard_model.py`**):**

    - Trains the multi-class `LGBMClassifier`.

    - Performs hyperparameter tuning using `GridSearchCV`.

    - Saves the final model, scaler, and calibrators.

4. **Generate Prediction Data (**`make_prediction_dataset.py`**):**

    - Prepares the feature set for a specific target date (e.g., tomorrow) required for inference.

5. **Generate Daily Predictions (**`predict.py` **&** `run_prediction.py`**):**

    - Loads all trained artifacts (models, scalers, calibrators).

    - Runs the two-stage inference process to generate the final hazard rating and confidence score for the target date.

    - Optionally generates a daily forecast map (`plot_results.py`).

## 4. Installation
1. **Clone the repository:**
``` Bash
git clone https://github.com/ronimos/avalanche_hazard_prediction_model.git
cd avalanche_hazard_prediction_model
```

2. **Create and activate a virtual environment:**
```Bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. **Install the required packages:**

The project uses a `pyproject.toml` file to manage dependencies. To install the core requirements, run:
```Bash
pip install .
```
To install optional dependencies for development, visualization, or GPU support, you can specify them in brackets. For example, to install everything needed for development and visualization:
```Bash
pip install .[dev,visualization]
```
## 5. Usage
**To Run the Full Training Pipeline and Generate a Forecast**
Execute the main orchestration script. You can specify a date for the forecast; otherwise, it will default to the current date.

```Bash
python src/run_pipeline.py --date YYYY-MM-DD
```

*Example:*
```Bash
python src/run_pipeline.py --date 2024-01-14
```

**To Generate a Forecast for a New Date (after models are trained)**
Execute the prediction script. Use the `--plot` flag to generate the interactive map.
```Bash
python src/run_prediction.py --date YYYY-MM-DD --plot
```

*Example:*
```Bash
python src/run_prediction.py --date 2024-01-15 --plot
```

The final predictions will be saved as a CSV file in the `results/` directory, and the map will be saved as an HTML file in the same location.