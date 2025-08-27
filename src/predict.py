# -*- coding: utf-8 -*-
"""
================================================================================
Inference Script for Avalanche Hazard Forecasting
================================================================================

Purpose:
--------
This script loads the final, pre-trained models to make predictions on new data.
It can be run directly from the command line for a specific date or its main
function can be imported and called by orchestration scripts.

Workflow:
---------
1.  Loads the feature set for inference for a specific date.
2.  **Avalanche Event Prediction:**
    a. Loads the event model, scaler, and feature list.
    b. Generates raw and adjusted scores for avalanche events.
3.  **Avalanche Hazard Prediction:**
    a. Loads the hazard model, scaler, and feature list.
    b. Uses the event predictions as input features for the hazard model.
    c. Generates final hazard ratings and calibrated probabilities.
4.  Saves the comprehensive predictions to the results directory.

CLI Usage:
----------
    python predict.py --date YYYY-MM-DD

Example:
    python predict.py --date 2024-02-15

"""
import pandas as pd
import numpy as np
import joblib
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union

# All paths and configurations are managed by the config file.
import config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def predict_avalanche_hazards(prediction_date: Union[str, datetime]):
    """
    Orchestrates the prediction process for both avalanche events and hazards.

    This is the main functional entry point. It loads models, makes predictions
    for a specific date, and saves the final hazard forecasts.

    Args:
        prediction_date (Union[str, datetime]): The specific date for which to
            generate predictions, either as a 'YYYY-MM-DD' string or a
            datetime object.
    """
    # --- Input Validation and Conversion ---
    if isinstance(prediction_date, str):
        try:
            prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d")
        except ValueError:
            logging.error(f"Invalid date string format: '{prediction_date}'. Please use YYYY-MM-DD.")
            return
    elif not isinstance(prediction_date, datetime):
        logging.error(f"Invalid type for prediction_date: {type(prediction_date)}. Must be a datetime object or a 'YYYY-MM-DD' string.")
        return

    logging.info(f"Starting comprehensive avalanche hazard prediction process for date: {prediction_date.date()}...")

    # --- 1. Load Base Inference Data ---
    try:
        inference_data_path = config.PATHS["PROCESSED_DATA"]["inference_features"]
        base_inference_df = pd.read_csv(inference_data_path)
        
        base_inference_df['date'] = pd.to_datetime(base_inference_df['date'])
        base_inference_df['polygon'] = base_inference_df['polygon'].astype(str)

        logging.info(f"Loaded {len(base_inference_df)} new data points for prediction from '{inference_data_path}'.")

        predictions_df = base_inference_df[base_inference_df['date'].dt.date == prediction_date.date()].copy()

        if predictions_df.empty:
            logging.error(f"No inference data found for the specified date: {prediction_date.date()}. "
                          "Please ensure make_prediction_dataset.py was run for this date.")
            return
        
        logging.info(f"Filtered to {len(predictions_df)} data points for prediction on {prediction_date.date()}.")

    except FileNotFoundError as e:
        logging.error(f"Could not load inference data file: {e}. Please run make_prediction_dataset.py first.")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading base inference data: {e}", exc_info=True)
        return

    # --- 2. Avalanche Event Prediction ---
    logging.info("--- Performing Avalanche Event Prediction ---")
    try:
        def load_event_model_and_scaler(config):
            event_model = joblib.load(config.PATHS["ARTIFACTS"]["event_model"])
            event_scaler = joblib.load(config.PATHS["ARTIFACTS"]["event_scaler"])
            with open(config.PATHS["ARTIFACTS"]["event_final_features"], 'r') as f:
                event_feature_cols = json.load(f)
            with open(config.PATHS["ARTIFACTS"]["event_model_params"], "r") as f:
                event_model_meta = json.load(f)
                event_optimal_threshold = event_model_meta.get('best_threshold', 0.5)
            return event_model, event_scaler, event_feature_cols, event_optimal_threshold

        event_model, event_scaler, event_feature_cols, event_optimal_threshold = load_event_model_and_scaler(config)
        logging.info(f"Successfully loaded event model. Optimal threshold: {event_optimal_threshold}")

        for col in event_feature_cols:
            if col not in predictions_df.columns:
                predictions_df[col] = 0
                logging.warning(f"Event feature '{col}' not found in new data. Filling with 0.")

        X_event = predictions_df[event_feature_cols]
        X_event_scaled = event_scaler.transform(X_event)
        event_raw_scores = event_model.predict_proba(X_event_scaled)[:, 1]

        event_adjusted_scores = np.zeros_like(event_raw_scores)
        low_mask = event_raw_scores < event_optimal_threshold
        if np.any(low_mask):
            denominator = max(event_optimal_threshold, 1e-8)
            event_adjusted_scores[low_mask] = 0.5 * (event_raw_scores[low_mask] / denominator)

        high_mask = ~low_mask
        if np.any(high_mask):
            denominator = max(1.0 - event_optimal_threshold, 1e-8)
            event_adjusted_scores[high_mask] = 0.5 + 0.5 * ((event_raw_scores[high_mask] - event_optimal_threshold) / denominator)

        predictions_df['event_raw_score'] = event_raw_scores
        predictions_df['event_adjusted_score'] = event_adjusted_scores
        predictions_df['event_predicted_label'] = (event_adjusted_scores >= 0.5).astype(int)
        logging.info("Avalanche event predictions generated.")

    except FileNotFoundError as e:
        logging.error(f"Could not load event model artifact: {e}. Please run train_avalanche_event_model.py first.")
        return
    except Exception as e:
        logging.error(f"An error occurred during event prediction: {e}", exc_info=True)
        return

    # --- 3. Avalanche Hazard Prediction ---
    logging.info("--- Performing Avalanche Hazard Prediction ---")
    try:
        hazard_model = joblib.load(config.PATHS["ARTIFACTS"]["hazard_model"])
        hazard_scaler = joblib.load(config.PATHS["ARTIFACTS"]["hazard_scaler"])
        with open(config.PATHS["ARTIFACTS"]["hazard_final_features"], 'r') as f:
            hazard_feature_cols = json.load(f)
        
        logging.info(f"Successfully loaded hazard model with {len(hazard_feature_cols)} features.")

        for col in hazard_feature_cols:
            if col not in predictions_df.columns:
                predictions_df[col] = 0
                logging.warning(f"Hazard feature '{col}' not found. Filling with 0.")

        X_hazard = predictions_df[hazard_feature_cols]
        X_hazard_scaled = hazard_scaler.transform(X_hazard)

        hazard_labels_0_indexed = hazard_model.predict(X_hazard_scaled)
        predictions_df['predicted_hazard'] = hazard_labels_0_indexed + 1

        if hasattr(hazard_model, 'predict_proba'):
            probabilities = hazard_model.predict_proba(X_hazard_scaled)
            for i, class_label in enumerate(hazard_model.classes_):
                predictions_df[f'calibrated_proba_hazard_{class_label + 1}'] = probabilities[:, i]
            logging.info("Calibrated hazard probabilities generated.")
        
        logging.info("Avalanche hazard predictions generated.")

    except FileNotFoundError as e:
        logging.error(f"Could not load hazard model artifact: {e}. Please run train_avalanche_hazard_model.py first.")
        return
    except Exception as e:
        logging.error(f"An error occurred during hazard prediction: {e}", exc_info=True)
        return

    # --- 4. Format and Save Final Results ---
    output_cols = ['date', 'polygon', 'event_raw_score', 'event_adjusted_score', 'event_predicted_label', 'predicted_hazard']
    prob_cols = [col for col in predictions_df.columns if col.startswith('calibrated_proba_hazard_')]
    final_predictions_df = predictions_df[output_cols + prob_cols].copy()

    # Save the comprehensive predictions to CSV and JSON
    csv_path = config.PATHS["ARTIFACTS"]["hazard_predictions_csv"]
    json_path = config.PATHS["ARTIFACTS"]["hazard_predictions_json"]
    
    final_predictions_df.to_csv(csv_path, index=False)
    final_predictions_df.to_json(json_path, orient='records', indent=4)
    
    logging.info(f"Comprehensive predictions saved to '{csv_path}' and '{json_path}'")

    # Create and save the summarized predictions JSON
    summary_df = final_predictions_df[[
        "polygon",
        "event_adjusted_score",
        "predicted_hazard"
    ]].copy()
    
    summary_df.rename(columns={
        "event_adjusted_score": "event_likelihood_score",
        "predicted_hazard": "hazard_rating_prediction"
    }, inplace=True)
    
    summary_json_path = config.PATHS["ARTIFACTS"]["all_predictions"]
    summary_df.to_json(summary_json_path, orient='records', indent=4)
    logging.info(f"Summarized predictions saved to '{summary_json_path}'")

    print("\n--- FINAL PREDICTION RESULTS (HEAD) ---")
    print(final_predictions_df.head())


if __name__ == "__main__":
    """
    This block allows the script to be run directly from the command line.
    It parses the --date argument and calls the main prediction function.
    """
    parser = argparse.ArgumentParser(
        description="Generate avalanche event and hazard predictions for a specific date.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--date',
        type=str,
        help="The date to generate predictions for, in YYYY-MM-DD format.\nIf not provided, the script will use the current date.",
        default="2024-01-14"
    )
    args = parser.parse_args()

    try:
        prediction_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        exit(1)
    
    # Call the main function with the parsed date
    predict_avalanche_hazards(prediction_date=prediction_date)
