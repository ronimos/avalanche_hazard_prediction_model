# -*- coding: utf-8 -*-
"""
predict.py
==========

This script runs the inference part of the two-stage avalanche forecasting
pipeline for a specific date.

Workflow:
---------
1.  **Load Prediction Data**: Reads the pre-generated feature set for the
    target prediction date.
2.  **Load Event Model Artifacts**: Loads the trained first-stage (event) model,
    scaler, and feature list.
3.  **Predict Avalanche Event**: Generates the `adjusted_score` for an
    avalanche event, which is a key input for the next stage.
4.  **Load Hazard Model Artifacts**: Loads the trained second-stage (hazard)
    model, its scaler, feature list, and the crucial Isotonic Regression
    calibrators.
5.  **Predict Hazard (with Calibration)**:
    a. The hazard model makes raw probability predictions.
    b. The Isotonic Regression models are applied to these raw probabilities
       to correct for calibration issues.
    c. The calibrated probabilities are re-normalized to sum to 1.
    d. The final hazard level and its associated confidence are determined.
6.  **Save Results**: Saves the final predictions, including the confidence
    score, to a CSV file.
"""

import logging
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union

# Import project-wide configurations
import config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def predict_avalanche_hazards(prediction_date: Union[str, datetime]) -> None:
    """
    Orchestrates the full two-stage prediction process for a given date.

    Args:
        prediction_date (Union[str, datetime]): The date for which to generate predictions,
                                                accepting either a 'YYYY-MM-DD' string
                                                or a datetime object.
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

    logging.info(f"Starting avalanche hazard prediction for date: {prediction_date.date()}")

    try:
        # --- 1. Load All Necessary Artifacts ---
        logging.info("Loading trained models and artifacts...")

        # Event model artifacts
        event_model = joblib.load(config.PATHS["ARTIFACTS"]["event_model"])
        event_scaler = joblib.load(config.PATHS["ARTIFACTS"]["event_scaler"])
        with open(config.PATHS["ARTIFACTS"]["event_final_features"], 'r') as f:
            event_features = json.load(f)
        with open(config.PATHS["ARTIFACTS"]["event_model_params"], 'r') as f:
            event_params = json.load(f)
        event_threshold = event_params.get('best_threshold', 0.5)

        # Hazard model artifacts
        hazard_model = joblib.load(config.PATHS["ARTIFACTS"]["hazard_model"])
        hazard_scaler = joblib.load(config.PATHS["ARTIFACTS"]["hazard_scaler"])
        hazard_calibrators = joblib.load(config.PATHS["ARTIFACTS"]["hazard_calibrators"])
        with open(config.PATHS["ARTIFACTS"]["hazard_final_features"], 'r') as f:
            hazard_features = json.load(f)

        # --- 2. Load Inference Features ---
        logging.info(f"Loading inference features for {prediction_date.date()}...")
        prediction_features_path = config.PATHS["PROCESSED_DATA"]["inference_features"]
        inference_df = pd.read_csv(prediction_features_path)
        inference_df['date'] = pd.to_datetime(inference_df['date'])

        # Filter for the specific prediction date
        daily_features_df = inference_df[inference_df['date'].dt.date == prediction_date.date()].copy()
        if daily_features_df.empty:
            logging.warning(f"No inference features found for {prediction_date.date()}. Aborting prediction.")
            return

        polygons_for_prediction = daily_features_df['polygon']

        # --- 3. Stage 1: Predict Avalanche Event Likelihood ---
        logging.info("Executing Stage 1: Predicting event likelihood...")
        X_event = daily_features_df[event_features]
        X_event_scaled = event_scaler.transform(X_event)
        
        raw_scores = event_model.predict_proba(X_event_scaled)[:, 1]

        # Apply the same threshold adjustment logic from training
        adjusted_scores = np.zeros_like(raw_scores)
        low_mask = raw_scores < event_threshold
        high_mask = ~low_mask
        denominator = 1.0 - event_threshold
        
        if np.any(low_mask):
            adjusted_scores[low_mask] = 0.5 * (raw_scores[low_mask] / event_threshold) if event_threshold > 0 else 0
        if np.any(high_mask) and denominator > 0:
            adjusted_scores[high_mask] = 0.5 + 0.5 * ((raw_scores[high_mask] - event_threshold) / denominator)

        # --- 4. Stage 2: Predict Hazard Rating ---
        logging.info("Executing Stage 2: Predicting hazard rating...")
        # Prepare feature set for hazard model by adding the event score
        X_hazard_base = daily_features_df.drop(columns=['date', 'polygon'], errors='ignore')
        X_hazard_base['adjusted_score'] = adjusted_scores
        
        # Ensure column order matches the training features
        X_hazard = X_hazard_base[hazard_features]
        X_hazard_scaled_np = hazard_scaler.transform(X_hazard)
        X_hazard_scaled = pd.DataFrame(X_hazard_scaled_np, columns=hazard_features)
        
        # Get raw (uncalibrated) probabilities from the hazard model
        raw_hazard_probs = hazard_model.predict_proba(X_hazard_scaled)

        # --- 5. Apply Calibration ---
        logging.info("Applying Isotonic Regression calibration...")
        calibrated_probs = np.zeros_like(raw_hazard_probs)
        for i, calibrator in enumerate(hazard_calibrators):
            calibrated_probs[:, i] = calibrator.predict(raw_hazard_probs[:, i])

        # Re-normalize probabilities to ensure they sum to 1
        prob_sum = calibrated_probs.sum(axis=1)[:, np.newaxis]
        final_calibrated_probs = calibrated_probs / prob_sum
        
        # Determine final prediction and confidence
        predicted_hazard_0_indexed = np.argmax(final_calibrated_probs, axis=1)
        confidence = np.max(final_calibrated_probs, axis=1)
        
        # Convert prediction back to 1-indexed hazard level
        predicted_hazard_1_indexed = predicted_hazard_0_indexed + 1

        # --- 6. Assemble and Save Final Predictions ---
        logging.info("Assembling and saving final prediction files...")
        final_predictions_df = pd.DataFrame({
            'date': prediction_date,
            'polygon': polygons_for_prediction,
            'predicted_hazard': predicted_hazard_1_indexed,
            'confidence': confidence,
        })
        # Add event score and individual calibrated probabilities
        final_predictions_df['avalanche_event_propability'] = adjusted_scores
        for i, class_label_0 in enumerate(hazard_model.classes_):
            class_label_1 = class_label_0 + 1
            final_predictions_df[f'hazard_{class_label_1}_prob_calibrated'] = final_calibrated_probs[:, i]

        # Save to standard CSV
        csv_output_path = config.PATHS["RESULTS"]["hazard_predictions_csv"]
        final_predictions_df.to_csv(csv_output_path, index=False)
        logging.info(f"Standard CSV predictions saved to '{csv_output_path}'")
        
        # --- 7. Create and Save Custom JSON Output (Model Predictions Only) ---
        logging.info("Creating custom JSON output with model predictions only...")
        
        # Start with the final predictions DataFrame
        json_data_df = final_predictions_df.copy()
        
        # Rename probability columns to match the desired JSON format
        prob_rename_map = {
            f'hazard_{i}_prob_calibrated': f'calibrated_proba_hazard_{i}'
            for i in range(1, len(hazard_model.classes_) + 1)
        }
        json_data_df.rename(columns=prob_rename_map, inplace=True)
        
        # Define the exact columns needed for the JSON output (model predictions only)
        json_feature_cols = ['polygon', 'predicted_hazard', 'confidence', 'avalanche_event_propability'] + list(prob_rename_map.values())

        # Convert the relevant part of the DataFrame to a list of dictionaries
        features_list = json_data_df[json_feature_cols].to_dict(orient='records')
        
        # Construct the final JSON object
        json_output = {
            "type": "polyGeom",
            "time": prediction_date.strftime("%Y-%m-%d"),
            "features": features_list
        }
        
        # Save the JSON file
        json_output_path = config.PATHS["ARTIFACTS"]["all_predictions"]
        with open(json_output_path, 'w') as f:
            json.dump(json_output, f, indent=4)
        logging.info(f"Custom JSON predictions saved to '{json_output_path}'")

    except FileNotFoundError as e:
        logging.error(f"A required model artifact or data file was not found: {e}. Please run the full training pipeline first.", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)

    logging.info(f"Prediction process for {prediction_date.date()} completed.")

# =============================================================================
# 3. SCRIPT EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Example of how to run this script directly
    # In a real run, this would be called from run_prediction.py or run_pipeline.py
    example_date = "2024-01-14"
    predict_avalanche_hazards(prediction_date=example_date)
