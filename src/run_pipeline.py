#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Orchestration Script for the Avalanche Hazard Forecasting Pipeline
================================================================================

Description:
------------
This script is the main entry point for executing the entire avalanche hazard
forecasting pipeline. It orchestrates a sequence of data processing, model
training, and prediction generation tasks to produce a daily avalanche hazard
forecast.

The pipeline is structured as a two-stage modeling process:
1. An avalanche event prediction model (PU learning).
2. An avalanche hazard rating prediction model (multi-class classification),
   which uses the output from the event model as a key feature.

Workflow Pipeline:
------------------
The script executes the following five phases in order:

1.  **Generate Training Data**: Prepares historical features and targets for the
    avalanche event model.
    (Source: `make_avalanche_event_training_dataset.py`)

2.  **Train Event Model**: Trains the PU learning model for avalanche events and
    generates historical predictions to be used as features in the next stage.
    (Source: `train_avalanche_event_model.py`)

3.  **Train Hazard Model**: Trains the multi-class classifier to predict hazard
    ratings, using the event model's predictions as input features.
    (Source: `train_avalanche_hazard_model.py`)

4.  **Generate Prediction Data**: Prepares the feature set for a specific target
    date (e.g., today) required for inference.
    (Source: `make_prediction_dataset.py`)

5.  **Generate Daily Predictions**: Executes the two-stage inference process to
    predict the final avalanche hazard ratings for the target date.
    (Source: `predict.py`)

CLI Usage:
----------
- To run the entire pipeline for the current date's forecast:
  $ python run_pipeline.py

- To run the pipeline for a specific historical or future date:
  $ python run_pipeline.py --date YYYY-MM-DD

Example:
  $ python run_pipeline.py --date 2024-02-15

"""

import argparse
import logging
from datetime import datetime

# Import the main functions from each pipeline script
import make_avalanche_event_training_dataset
import train_avalanche_event_model
import train_avalanche_hazard_model
import make_prediction_dataset
import predict

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_full_pipeline(prediction_date: datetime):
    """
    Executes the complete avalanche forecasting pipeline.

    Args:
        prediction_date (datetime): The date for which to generate new predictions.
    """
    logging.info("================================================================================")
    logging.info(">>> Starting Full Avalanche Hazard Forecasting Pipeline Orchestration <<<")
    logging.info("================================================================================")
    logging.info(f"Target prediction date: {prediction_date.date()}")
    logging.info("-" * 80)

    # Step 1: Generate Training Data
    logging.info("Phase 1/5: Generating training dataset (make_avalanche_event_training_dataset.py)")
    make_avalanche_event_training_dataset.main_pipeline()
    logging.info("Phase 1/5: Training data generation complete.")
    logging.info("-" * 80)

    # Step 2: Train Avalanche Event Model
    logging.info("Phase 2/5: Training Avalanche Event Model (train_avalanche_event_model.py)")
    train_avalanche_event_model.main()
    logging.info("Phase 2/5: Avalanche Event Model training complete.")
    logging.info("-" * 80)

    # Step 3: Train Avalanche Hazard Model
    logging.info("Phase 3/5: Training Avalanche Hazard Model (train_avalanche_hazard_model.py)")
    train_avalanche_hazard_model.main()
    logging.info("Phase 3/5: Avalanche Hazard Model training complete.")
    logging.info("-" * 80)

    # Step 4: Generate Prediction Data for the specified date
    logging.info(f"Phase 4/5: Generating prediction dataset for {prediction_date.date()} (make_prediction_dataset.py)")
    make_prediction_dataset.build_prediction_dataset(prediction_date=prediction_date)
    logging.info(f"Phase 4/5: Prediction data generation for {prediction_date.date()} complete.")
    logging.info("-" * 80)

    # Step 5: Generate Daily Predictions for the specified date
    logging.info(f"Phase 5/5: Generating daily avalanche predictions for {prediction_date.date()} (predict.py)")
    predict.predict_avalanche_hazards(prediction_date=prediction_date)
    logging.info(f"Phase 5/5: Daily avalanche predictions for {prediction_date.date()} complete.")
    logging.info("-" * 80)

    logging.info("================================================================================")
    logging.info(">>> Full Avalanche Hazard Forecasting Pipeline Orchestration Completed <<<")
    logging.info("================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full avalanche hazard forecasting pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--date',
        type=str,
        help="The date for which to generate predictions, in YYYY-MM-DD format.\n"
             "If not provided, the script will use the current date for prediction.",
        default="2024-01-14"  # Default to a specific date for testing; can be changed to None for current date
    )
    args = parser.parse_args()

    # Determine the prediction date
    run_date = datetime.now()
    if args.date:
        try:
            run_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")
            exit(1) # Exit with an error code

    run_full_pipeline(prediction_date=run_date)