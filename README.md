# Avalanche Hazard Forecasting Pipeline

An end-to-end machine learning pipeline to forecast daily avalanche hazard ratings using a **two-stage modeling approach** that combines detailed snowpack data, weather conditions, and historical avalanche observations.

## Overview

The pipeline consists of two core models:

1. **Avalanche Event Model:** A Positive-Unlabeled (PU) learning model that first estimates the likelihood of a significant avalanche event.

2. **Hazard Rating Model:** A multi-class classification model that uses the event model’s output, along with engineered features, to predict the final hazard rating (1-Low, 2-Moderate, 3-Considerable, 4-High). 

The pipeline is modular, configurable, and fully reproducible—covering everything from data ingestion to final predictions.

## Features

- **Automated Data Ingestion**: Downloads snowpack (`.pro`), weather (`.smet`), and avalanche observation data from public sources.  
- **Advanced Feature Engineering**: Extracts snowpack features from SNOWPACK `.pro` files, including weak layers, slabs, and structure.  
- **Robust Training Pipeline**: Scripts for data preparation, training, and evaluation of both models.  
- **Daily Prediction Workflow**: Generate hazard forecasts for any given day.  
- **Flexible Configuration**: Use the `SNOWPACK_DATA_ROOT` environment variable to work with either downloaded or local high-speed data.

## 📂 Project Structure

```
hazard-forecasting/
├── data/
│   ├── raw/          # Static raw data (e.g., polygon GeoJSON)
│   ├── external/     # Downloaded data (.pro, .smet files)
│   └── processed/    # Intermediate and final datasets
├── models/           # Trained model artifacts (.joblib, .json)
├── results/          # Prediction outputs (.csv, .json)
├── src/              # Python source code
│   ├── __init__.py
│   ├── config.py           # Paths and parameters
│   ├── dataset_utils.py    # Data processing utilities
│   ├── snowpack_reader.py  # .pro file parser
│   ├── run_pipeline.py     # Train models end-to-end
│   └── run_prediction.py   # Generate daily predictions
├── .gitignore
├── pyproject.toml    # Project metadata and dependencies
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.10+
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ronimos/avalanche_hazard_prediction_model
   cd hazard-forecasting
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv

   # macOS/Linux
   source .venv/bin/activate

   # Windows
   .\.venv\Scripts\ctivate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

   To include optional dependencies for visualization or development:
   ```bash
   pip install -e .[visualization,dev]
   ```

### Training the Models

Run the full pipeline (data generation + training):
```bash
python src/run_pipeline.py
```

Trained models will be stored in the `/models` directory.

### Generating Daily Predictions

Run the prediction script for a specific date:
```bash
python src/run_prediction.py --date YYYY-MM-DD

# Example
python src/run_prediction.py --date 2024-02-20
```

Results will be saved to the `/results` directory.

### Using a Local Data Source

If snowpack and weather data are available locally, set the `SNOWPACK_DATA_ROOT` environment variable:

- **macOS/Linux**
  ```bash
  export SNOWPACK_DATA_ROOT="/path/to/your/snowpack/data/"
  python src/run_prediction.py --date 2024-02-20
  ```

- **Windows**
  ```bash
  set SNOWPACK_DATA_ROOT="C:\path	o\your\snowpack\data"
  python src
un_prediction.py --date 2024-02-20
  ```

## Scripts Overview

- **run_pipeline.py** — End-to-end training workflow.  
- **run_prediction.py** — Generate daily predictions.  
- **make_..._dataset.py** — Build feature datasets for training/prediction.  
- **train_..._model.py** — Train and evaluate models.  
- **predict.py** — Load trained models and run inference.  
- **config.py** — Centralized parameters and paths.  
- **dataset_utils.py** — Shared utilities for data processing and feature engineering.  
- **snowpack_reader.py** — Specialized SNOWPACK `.pro` parser.  

## Contributing

Contributions are welcome! To contribute:

1. Fork the project  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit changes (`git commit -m 'Add YourFeature'`)  
4. Push to your branch (`git push origin feature/YourFeature`)  
5. Open a pull request  

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
