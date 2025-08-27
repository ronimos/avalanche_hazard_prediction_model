# -*- coding: utf-8 -*-
"""
================================================================================
End-to-End Prediction Pipeline
================================================================================

Purpose:
--------
This script orchestrates the entire prediction pipeline, from data generation
to making the final hazard predictions and optionally visualizing the results on a map.

Workflow:
---------
1.  **Generate Prediction Dataset:** Generates the feature set for a specified date.
2.  **Run Inference:** Loads trained models to make predictions.
3.  **Plot Results (Optional):** Creates an interactive Folium map of the predictions.

CLI Usage:
----------
    # Run predictions without creating a map
    python run_prediction.py --date YYYY-MM-DD

    # Run predictions AND create a map
    python run_prediction.py --date YYYY-MM-DD --plot

Example:
    python run_prediction.py --date 2024-02-20 --plot

"""

import logging
import argparse
from datetime import datetime

# Import the main functions from the other pipeline scripts
import make_prediction_dataset
import predict
import plot_results # Import the plotting script

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
        description="Run the full prediction pipeline: generate features, make predictions, and optionally create a map.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--date',
        type=str,
        help="The date to generate predictions for, in YYYY-MM-DD format.",
        default='2024-01-14',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False,
        help="If included, a Folium map of the predictions will be generated."
    )
    args = parser.parse_args()

    try:
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
        predict.predict_avalanche_hazards(prediction_date=run_date)
        logging.info("...Inference complete.")
    except Exception as e:
        logging.error(f"Failed during model inference: {e}", exc_info=True)
        return

    # --- Step 3: Plot Results on a Map (Conditional) ---
    if args.plot:
        logging.info("STEP 3: Generating visualization map...")
        try:
            plot_results.create_prediction_map(prediction_date=run_date)
            logging.info("...Map generation complete.")
        except Exception as e:
            logging.error(f"Failed during map generation: {e}", exc_info=True)
            return
    else:
        logging.info("STEP 3: Skipping map generation as --plot flag was not provided.")

    logging.info("="*60)
    logging.info("PREDICTION PIPELINE FINISHED SUCCESSFULLY")
    logging.info("="*60)


if __name__ == "__main__":
    main()
