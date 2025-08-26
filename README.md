# Avalanche Hazard Forecasting Pipeline
This project provides a complete, end-to-end machine learning pipeline for forecasting daily avalanche hazard ratings. It leverages a two-stage modeling approach, using detailed snowpack data, weather conditions, and historical avalanche observations to produce daily predictions.
## Overview
The core of this project is a two-stage model:
1. **Avalanche Event Model:** A Positive-Unlabeled (PU) learning model that first predicts the likelihood of a significant avalanche event occurring.
2. **Hazard Rating Model:** A multi-class classification model that uses the output from the event model, along with a rich set of engineered features, to predict the final hazard rating (e.g., 1-Low, 2-Moderate, 3-Considerable, 4-High).
The pipeline is designed to be modular, configurable, and reproducible, handling everything from data fetching to final prediction.
## Features
- **Automated Data Ingestion:** Downloads snowpack (.pro), weather (.smet), and avalanche observation data from public sources.
- **Advanced Feature Engineering:** Parses complex SNOWPACK .pro files to extract detailed features about weak layers, slabs, and overall snowpack structure.
- **Robust Training Pipeline:** Includes scripts to generate training data and train both stages of the model.
- **Daily Prediction Workflow:** A streamlined pipeline to generate features and run predictions for any given day.
- **Flexible Configuration:** Use environment variables (SNOWPACK_DATA_ROOT) to switch between downloading data and using a local, pre-existing data source.
## 📂 Project Structurehazard-forecasting/
```
├── data/
│   ├── raw/          # Static raw data (e.g., polygon GeoJSON)
│   ├── external/     # Downloaded data (e.g., .pro, .smet files)
│   └── processed/    # Intermediate and final datasets
├── models/           # Trained model artifacts (.joblib, .json)
├── results/          # Prediction outputs (.csv, .json)
├── src/              # All Python source code
│   ├── __init__.py
│   ├── config.py     # Central configuration for paths and parameters
│   ├── dataset_utils.py # Low-level data processing functions
│   ├── snowpack_reader.py # .pro file parser
│   ├── run_pipeline.py    # Main script to run the full training pipeline
│   └── run_prediction.py  # Main script to generate daily predictions
├── .gitignore
├── pyproject.toml    # Project metadata and dependencies
└── README.md
```
## Setup and Installation
### Prerequisites
- Python 3.10 or newer
- Git
###Installation Steps
1. Clone the repository:
```
git clone [<your-repository-url>](https://github.com/ronimos/avalanche_hazard_prediction_model)
cd hazard-forecasting
```
2. Create and activate a virtual environment:# Create the virtual environment
```
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
.\.venv\Scripts\activate
```
3. Install dependencies:
The project uses `pyproject.toml` to manage dependencies. Install the project in editable mode, which will also install all required packages.
```
pip install -e .
```

To install optional dependencies for visualization or development, you can specify them:
```
pip install -e .[visualization,dev]
```
## UsageTraining
###the Models
To run the entire data generation and model training pipeline from scratch, execute the `run_pipeline.py` script. This will create all the necessary datasets and save the trained models to the `/models` directory.
```
python src/run_pipeline.py
```
### Generating Daily Predictions
To generate a forecast for a specific date, use the `run_prediction.py` script.
```
# Run for a specific date
python src/run_prediction.py --date YYYY-MM-DD

# Example:
python src/run_prediction.py --date 2024-02-20
```

The final predictions will be saved in the `/results` directory.

### Using a Local Data Source
If you have the snowpack and weather data stored locally (e.g., on a high-speed drive), you can tell the scripts to use it instead of downloading the data. Set the `SNOWPACK_DATA_ROOT` environment variable before running your script.
#### On macOS/Linux:
```
export SNOWPACK_DATA_ROOT="/path/to/your/snowpack/data/"
python src/run_prediction.py --date 2024-02-20
```
#### On Windows:
```
set SNOWPACK_DATA_ROOT="C:\path\to\your\snowpack\data"
python src\run_prediction.py --date 2024-02-20
```
## Scripts Overvie
- `wrun_pipeline.py`: The main entry point for training. It executes all data generation and model training scripts in the correct order.
- `run_prediction.py`: The main entry point for inference. It runs the data generation and prediction scripts for a specific date.
- `make_..._dataset.py`: Scripts responsible for preparing the feature sets for training and prediction.
- `train_..._model.py`: Scripts that handle the training and evaluation of the two models.
- `predict.py`: Loads the trained models and runs inference on a new feature set.
- `config.py`: Centralizes all file paths, model parameters, and API settings.
- `dataset_utils.py`: Contains all shared, low-level functions for data fetching, processing, and feature engineering.
- `snowpack_reader.py`: A specialized parser for reading and extracting features from SNOWPACK .pro files.

## Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
