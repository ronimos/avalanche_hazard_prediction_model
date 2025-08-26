# -*- coding: utf-8 -*-
"""
================================================================================
End-to-End Prediction Pipeline
================================================================================

Purpose:
--------
This script orchestrates the entire prediction pipeline, from data generation
to making the final hazard predictions. It serves as the single entry point
for running daily (or on-demand) forecasts.

Workflow:
---------
1.  **Generate Prediction Dataset:** It first calls the functions in
    `make_prediction_dataset.py` to download the latest data and generate
    the feature set for a specified date.
2.  **Run Inference:** It then calls the functions in `predict.py` to load the
    trained model and use it to make predictions on the newly generated data.
3.  **Save Results:** The final predictions are saved to the `results/` directory.

CLI Usage:
----------
To run predictions for the current date:
    python run_prediction.py

To run predictions for a specific historical date:
    python run_prediction.py --date YYYY-MM-DD

Example:
    python run_prediction.py --date 2024-02-20

"""

import logging
import argparse
from datetime import datetime

# Import the main functions from the other pipeline scripts
import make_prediction_dataset
import predict

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    """
    Main function to orchestrate the prediction pipeline.
    """
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run the full prediction pipeline: generate features and make predictions for a specific date.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--date',
        type=str,
        help="The date to generate predictions for, in YYYY-MM-DD format.\nIf not provided, the script will use the current date.",
        default='2024-01-14',
    )
    args = parser.parse_args()

    # Default to the current time if no date is provided
    run_date = datetime.now()
    if args.date:
        try:
            # Normalize to the start of the day
            run_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")
            exit(1)

    logging.info("="*60)
    logging.info(f"STARTING PREDICTION PIPELINE FOR: {run_date.date()}")
    logging.info("="*60)

    # --- Step 1: Generate Prediction Dataset ---
    logging.info("STEP 1: Generating prediction dataset...")
    try:
        make_prediction_dataset.build_prediction_dataset(prediction_date=run_date)
        logging.info("...Prediction dataset generation complete.")
    except Exception as e:
        logging.error(f"Failed during dataset generation: {e}", exc_info=True)
        return

    # --- Step 2: Run Inference ---
    logging.info("STEP 2: Running inference on the new dataset...")
    try:
        predict.predict_avalanche_hazards(args.date)
        logging.info("...Inference complete.")
    except Exception as e:
        logging.error(f"Failed during model inference: {e}", exc_info=True)
        return

    logging.info("="*60)
    logging.info("PREDICTION PIPELINE FINISHED SUCCESSFULLY")
    logging.info("="*60)


if __name__ == "__main__":
    main()
