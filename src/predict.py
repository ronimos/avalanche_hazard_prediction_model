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

def predict_avalanche_hazards(prediction_date: Union[str, datetime]):
    """
    Runs the full two-stage prediction pipeline for a given date.

    Args:
        prediction_date (Union[str, datetime]): The date for which to generate
            predictions, either as a 'YYYY-MM-DD' string or a datetime object.
    """
    if isinstance(prediction_date, str):
        prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d")

    logging.info(f"--- Starting Prediction for {prediction_date.date()} ---")

    # --- STAGE 1: PREDICT AVALANCHE EVENT PROBABILITY ---
    try:
        # Load prediction features
        prediction_features_path = config.PATHS["PROCESSED_DATA"]["inference_features"]
        features_df = pd.read_csv(prediction_features_path)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        # Filter for the specific prediction date
        predict_df = features_df[features_df['date'].dt.date == prediction_date.date()].copy()
        if predict_df.empty:
            logging.warning(f"No prediction data found for {prediction_date.date()}. Aborting.")
            return

        # Load event model artifacts
        event_model = joblib.load(config.PATHS["ARTIFACTS"]["event_model"])
        event_scaler = joblib.load(config.PATHS["ARTIFACTS"]["event_scaler"])
        with open(config.PATHS["ARTIFACTS"]["event_model_params"], 'r') as f:
            event_params = json.load(f)
        with open(config.PATHS["ARTIFACTS"]["event_final_features"], 'r') as f:
            event_features = json.load(f)

    except FileNotFoundError as e:
        logging.error(f"Missing artifact for Stage 1 (Event Model): {e}. Cannot proceed.")
        return

    X_event = predict_df[event_features]
    X_event_scaled = event_scaler.transform(X_event)

    # Generate raw scores and adjusted scores
    raw_scores = event_model.predict_proba(X_event_scaled)[:, 1]
    optimal_threshold = event_params.get('best_threshold', 0.5)
    
    adjusted_scores = np.zeros_like(raw_scores)
    low_mask = raw_scores < optimal_threshold
    high_mask = ~low_mask
    
    if np.any(low_mask):
        adjusted_scores[low_mask] = 0.5 * (raw_scores[low_mask] / optimal_threshold) if optimal_threshold > 0 else 0.0
    if np.any(high_mask):
        denominator = (1.0 - optimal_threshold)
        adjusted_scores[high_mask] = 0.5 + 0.5 * ((raw_scores[high_mask] - optimal_threshold) / denominator) if denominator > 0 else 1.0

    predict_df['raw_score'] = raw_scores
    predict_df['adjusted_score'] = adjusted_scores
    predict_df['predicted_label'] = (adjusted_scores >= 0.5).astype(int)

    # --- STAGE 2: PREDICT FINAL HAZARD RATING (WITH CALIBRATION) ---
    try:
        # Load hazard model artifacts
        hazard_model = joblib.load(config.PATHS["ARTIFACTS"]["hazard_model"])
        hazard_calibrators = joblib.load(config.PATHS["ARTIFACTS"]["hazard_calibrators"])
        hazard_scaler = joblib.load(config.PATHS["ARTIFACTS"]["hazard_scaler"])
        with open(config.PATHS["ARTIFACTS"]["hazard_final_features"], 'r') as f:
            hazard_features = json.load(f)

    except FileNotFoundError as e:
        logging.error(f"Missing artifact for Stage 2 (Hazard Model): {e}. Cannot proceed.")
        return

    X_hazard = predict_df[hazard_features]
    X_hazard_scaled = hazard_scaler.transform(X_hazard)

    # Get raw probabilities from the base model
    raw_hazard_probs = hazard_model.predict_proba(X_hazard_scaled)

    # Apply the trained calibrators to the raw probabilities
    logging.info("Applying Isotonic Regression calibrators to raw probabilities...")
    calibrated_probs = np.zeros_like(raw_hazard_probs)
    n_classes = raw_hazard_probs.shape[1]
    for i in range(n_classes):
        # Ensure calibrators list is long enough
        if i < len(hazard_calibrators):
            calibrated_probs[:, i] = hazard_calibrators[i].predict(raw_hazard_probs[:, i])
        else:
            logging.warning(f"Mismatch between number of classes ({n_classes}) and calibrators ({len(hazard_calibrators)}). Using raw probability for class {i}.")
            calibrated_probs[:, i] = raw_hazard_probs[:, i]

    # Re-normalize the probabilities to ensure they sum to 1
    prob_sum = calibrated_probs.sum(axis=1, keepdims=True)
    prob_sum[prob_sum == 0] = 1 # Avoid division by zero
    final_probs = calibrated_probs / prob_sum

    # The final prediction is the class with the highest *calibrated* probability
    final_prediction = np.argmax(final_probs, axis=1)
    # The confidence is the probability of that predicted class
    confidence = np.max(final_probs, axis=1)

    # --- 6. Save Results ---
    output_df = predict_df[['date', 'polygon']].copy()
    output_df['predicted_hazard'] = final_prediction + 1 # Convert back to 1-4 scale
    output_df['confidence'] = confidence # Add the confidence score
    output_df['event_adjusted_score'] = predict_df['adjusted_score']
    
    # Add individual calibrated probabilities for each hazard level
    for i in range(n_classes):
        output_df[f'hazard_{i+1}_prob_calibrated'] = final_probs[:, i]

    output_path = config.PATHS["ARTIFACTS"]["hazard_predictions_csv"]
    output_df.to_csv(output_path, index=False)
    logging.info(f"Final calibrated predictions saved to: {output_path}")
    logging.info("--- Prediction Pipeline Finished Successfully ---")


if __name__ == "__main__":
    # Example of how to run this script directly
    # In a real run, this would be called from run_prediction.py or run_pipeline.py
    example_date = "2024-01-14"
    predict_avalanche_hazards(prediction_date=example_date)
