# -*- coding: utf-8 -*-
"""
================================================================================
Avalanche Hazard Forecasting Model Training Pipeline
================================================================================

Purpose:
--------
This script implements a comprehensive pipeline for training and evaluating
machine learning models to predict daily avalanche hazard ratings. It leverages
features derived from snowpack profiles, weather data, and past avalanche
activity, including the adjusted predictions from a preceding avalanche event
forecasting model.

The pipeline performs the following key steps:
1.  **Data Loading and Preparation**: Loads features, raw targets, and
    adjusted avalanche event predictions, merging them into a unified dataset.
    The 'hazard' column is set as the primary target variable.
2.  **Model Training and Hyperparameter Tuning**: Compares and tunes multiple
    classification algorithms (RandomForest, XGBoost, LightGBM) using
    cross-validation to find the optimal model and hyperparameters.
3.  **Model Evaluation**: Evaluates the performance of each tuned model based
    on relevant classification metrics (e.g., F1-score, AUC).
4.  **Artifact Saving**: Saves the best-performing model, its scaler, and
    training results for future inference and analysis.

Dependencies:
-   config.py: For managing file paths and general configurations.
-   make_avalanche_event_training_dataset.py: Must have been run to generate
    'training_features.csv' and 'training_targets.csv'.
-   train_avalanche_event_model.py: Must have been run to generate
    'event_adjusted_predictions.csv'.

Usage:
------
Run this script from the project root:
    python src/train_avalanche_hazard_model.py

"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from typing import List, Dict, Optional, Any, Tuple, Union

# Scikit-learn models and utilities
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, make_scorer, mean_absolute_error,
                             cohen_kappa_score, classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve # Import calibration_curve
import shap # SHAP for model interpretability

# Gradient Boosting Libraries
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt # Import for plotting
from matplotlib.patches import Patch # Import for legend patches
import seaborn as sns # Import for heatmap

from tqdm import tqdm # For progress bars

# Assuming a config.py file with necessary paths
import config

# --- Configuration ---
# Use paths from config.py
OUTPUT_DIR: Path = config.RESULTS_DIR
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the directory exists

RANDOM_STATE: int = config.MODELING_CONFIG["random_state"]

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# 2. DATA LOADING AND PREPARATION
# =============================================================================
def load_and_prepare_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Loads features, targets, and adjusted avalanche event predictions, then
    merges them into a unified dataset for hazard model training.

    The target variable for this model is 'hazard'. Features include original
    snowpack/weather data and the raw/adjusted scores and predicted labels
    from the avalanche event model.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame]]:
            - X (pd.DataFrame): DataFrame containing the features for hazard prediction.
            - y_true (pd.Series): Series containing the true 'hazard' labels.
            - original_merged_df (pd.DataFrame): DataFrame containing 'date', 'polygon',
                                                 'hazard', and 'avalanche_event' for saving predictions later.
            Returns (None, None, None) if data loading or preparation fails.
    """
    logging.info("Loading and preparing data for hazard model training...")
    try:
        features_path: Path = config.PATHS["PROCESSED_DATA"]["training_features"]
        targets_path: Path = config.PATHS["PROCESSED_DATA"]["training_targets"]
        # Correctly read the adjusted predictions from the event model
        adjusted_predictions_path: Path = config.PATHS["ARTIFACTS"]["event_adjusted_predictions"]

        X_df_features: pd.DataFrame = pd.read_csv(features_path)
        y_df_targets: pd.DataFrame = pd.read_csv(targets_path)
        adjusted_preds_df: pd.DataFrame = pd.read_csv(adjusted_predictions_path)

        logging.info(f"Loaded features from '{features_path}'. Shape: {X_df_features.shape}")
        logging.info(f"Loaded targets from '{targets_path}'. Shape: {y_df_targets.shape}")
        logging.info(f"Loaded adjusted predictions from '{adjusted_predictions_path}'. Shape: {adjusted_preds_df.shape}")

        # Ensure 'date' and 'polygon' columns are datetime/string for merging
        X_df_features['date'] = pd.to_datetime(X_df_features['date'])
        y_df_targets['date'] = pd.to_datetime(y_df_targets['date'])
        adjusted_preds_df['date'] = pd.to_datetime(adjusted_preds_df['date'])
        
        X_df_features['polygon'] = X_df_features['polygon'].astype(str)
        y_df_targets['polygon'] = y_df_targets['polygon'].astype(str)
        adjusted_preds_df['polygon'] = adjusted_preds_df['polygon'].astype(str)

        # Merge features with raw targets
        # Use an inner merge to ensure all dates/polygons align
        base_merged_df: pd.DataFrame = pd.merge(X_df_features, y_df_targets, on=['date', 'polygon'], how='inner')
        logging.info(f"Merged features and raw targets. Shape: {base_merged_df.shape}")

        # Merge with adjusted predictions from the event model
        # The suffixes are important to distinguish columns if names overlap
        final_merged_df: pd.DataFrame = pd.merge(base_merged_df, adjusted_preds_df, on=['date', 'polygon'], how='left', suffixes=('', '_event_pred'))
        logging.info(f"Merged with adjusted event predictions. Shape: {final_merged_df.shape}")

        # Define the target variable for hazard prediction
        # IMPORTANT: Subtract 1 to make hazard ratings 0-indexed for models like XGBoost
        y_true: pd.Series = final_merged_df['hazard'] - 1 # Convert to 0-indexed

        # Define columns to exclude from features to prevent data leakage
        # This includes all original target-related columns from y_df_targets
        # and identifiers. The event model outputs are now *included* as features.
        target_and_prediction_cols_to_exclude: List[str] = [
            'date', 'polygon', 'hazard', # Identifiers and current target (original 1-indexed 'hazard')
            'avalanche_event', # Target from event model
            'daily_AAI', 'max_destructive_size', 'num_daily_avalanches', # Raw avalanche data
            'daily_AAI_d-1', 'max_destructive_size_d-1', 'num_daily_avalanches_d-1',
            'daily_AAI_d-1_mean_3d', 'daily_AAI_d-1_std_3d',
            'max_destructive_size_d-1_mean_3d', 'max_destructive_size_d-1_std_3d',
            'num_daily_avalanches_d-1_mean_3d', 'num_daily_avalanches_d-1_std_3d',
            'hazard-1', 'hazard-2', # Lagged hazard from original targets
        ]
        
        # Create the feature DataFrame by dropping all excluded columns
        X: pd.DataFrame = final_merged_df.drop(columns=target_and_prediction_cols_to_exclude, errors='ignore')

        # Ensure all feature columns are numeric, coercing errors and filling NaNs
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                except ValueError:
                    logging.warning(f"Could not convert column '{col}' to numeric. Filling with 0.")
                    X[col] = 0
            else:
                X[col] = X[col].fillna(0) # Fill any remaining numeric NaNs

        # Keep a copy of merged_df with date, polygon, original 1-indexed hazard, AND avalanche_event for saving predictions later
        original_merged_df = final_merged_df[['date', 'polygon', 'hazard', 'avalanche_event']].copy()

        return X, y_true, original_merged_df

    except FileNotFoundError as e:
        logging.error(f"Required data file not found: {e}. Please ensure "
                      "make_avalanche_event_training_dataset.py and train_avalanche_event_model.py "
                      "have been run successfully to generate all necessary input files.")
        return None, None, None
    except KeyError as e:
        logging.error(f"Missing expected column in data: {e}. Check data generation scripts and ensure "
                      "all required columns are present after merging.")
        return None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return None, None, None

# =============================================================================
# 3. MODEL TRAINING AND EVALUATION
# =============================================================================
def train_and_evaluate_model(
    model_name: str,
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scorer: Any
) -> Tuple[Dict[str, Any], Any]:
    """
    Trains and evaluates a given model using GridSearchCV for hyperparameter tuning.

    Args:
        model_name (str): Name of the model (e.g., "RandomForest", "XGBoost").
        estimator (Any): The scikit-learn estimator or compatible model.
        param_grid (Dict[str, List[Any]]): Dictionary of hyperparameters to tune.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
        scorer (Any): Scorer object (e.g., from sklearn.metrics.make_scorer).

    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing:
            - metrics (Dict[str, Any]): Dictionary of evaluation metrics for the best model.
            - best_estimator (Any): The best trained model.
    """
    logging.info(f"Starting training and tuning for {model_name}...")

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scorer, # Primary scoring for GridSearchCV
        cv=3,
        n_jobs=multiprocessing.cpu_count(),
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test) # Direct predicted labels for metrics
    
    # Calculate comprehensive metrics
    metrics = {
        "Model": model_name,
        "Best Params": grid_search.best_params_,
        "F1 Score (weighted)": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "Precision (weighted)": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "MAE": mean_absolute_error(y_test, y_pred),
        "Quadratic Kappa": cohen_kappa_score(y_test, y_pred, weights='quadratic'),
        "Classification Report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization
    }
    
    logging.info(f"Finished tuning {model_name}. Best params: {grid_search.best_params_}")
    logging.info(f"Test Metrics for {model_name}:")
    logging.info(f"  F1 (weighted): {metrics['F1 Score (weighted)']:.4f}")
    logging.info(f"  MAE: {metrics['MAE']:.4f}")
    logging.info(f"  Quadratic Kappa: {metrics['Quadratic Kappa']:.4f}")
    logging.info(f"  Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    logging.info(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics, best_estimator

def compute_and_save_shap(model: Any, X_df: pd.DataFrame, model_name: str, plot_save_shap_values: bool = False, classes: List[int] = None) -> pd.DataFrame:
    """
    Computes SHAP (SHapley Additive exPlanations) values for a given model and
    feature set, and optionally saves a plot of feature importance.

    Args:
        model (Any): The trained machine learning model.
        X_df (pd.DataFrame): The DataFrame of features used for SHAP explanation.
        model_name (str): A name for the model, used in plot titles and logging.
        plot_save_shap_values (bool): If True, generates and saves a SHAP summary plot.
        classes (List[int], optional): List of class labels (0-indexed) for plotting.

    Returns:
        pd.DataFrame: A DataFrame containing features and their mean absolute SHAP values,
                      sorted by importance.
    """
    logging.info(f"Generating SHAP values for {model_name}...")
    try:
        def get_shap_sample(df: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
            """Return a sample of the DataFrame if it exceeds max_samples, else return the DataFrame itself."""
            return shap.utils.sample(df, max_samples) if df.shape[0] > max_samples else df

        X_sample_df = get_shap_sample(X_df, 1000)

        # Convert the sample DataFrame to a NumPy array for SHAP explainer
        X_sample_np = X_sample_df.values

        # For tree-based models, TreeExplainer is generally preferred
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback for other models, might be slower
            explainer = shap.KernelExplainer(model.predict_proba, X_sample_np)
        
        shap_values = explainer.shap_values(X_sample_np)

        # Calculate mean_abs_shap based on the structure of shap_values
        if isinstance(shap_values, list):
            # Multi-class case: shap_values is a list of arrays, each (n_samples, n_features)
            # Stack them into a 3D array: (n_classes, n_samples, n_features)
            stacked_abs_shap = np.stack([np.abs(s) for s in shap_values], axis=0)
            # Average across samples (axis=1) and then across classes (axis=0)
            # Resulting shape should be (n_features,)
            mean_abs_shap = np.mean(stacked_abs_shap, axis=(0, 1))
        else:
            # This branch is hit if shap_values is a single numpy array.
            # For multi-class problems with TreeExplainer, this is typically (n_samples, n_features, n_classes).
            # We need to average over samples (axis=0) and classes (axis=2) to get (n_features,).
            if shap_values.ndim == 3:
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else: # Fallback for binary or regression (samples, features)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Ensure mean_abs_shap is a 1D array
        mean_abs_shap = mean_abs_shap.flatten()

        # It's crucial that mean_abs_shap has the same length as X_sample_df.columns
        if len(mean_abs_shap) != len(X_sample_df.columns):
            logging.error(f"SHAP calculation resulted in {len(mean_abs_shap)} values, but there are {len(X_sample_df.columns)} features in sample. "
                          "This indicates an issue with SHAP value computation or feature alignment.")
            # If a mismatch occurs here, it means the SHAP library's output is not as expected.
            # Raising an error is appropriate as the data is fundamentally misaligned.
            raise ValueError("Calculated SHAP feature importance array has inconsistent length with DataFrame columns.")

        shap_df = pd.DataFrame({
            "feature": X_sample_df.columns, # Use X_sample_df.columns for direct alignment
            "mean_abs_shap_value": mean_abs_shap
        })

        shap_df = shap_df.sort_values(by="mean_abs_shap_value", ascending=False)


        if plot_save_shap_values:
            # Save feature names to the specified path in config
            shap_df.to_csv(config.PATHS["ARTIFACTS"]["hazard_feature_names"], index=False)
            # Use X_df.columns here for the plot, as it represents the full feature set
            # Ensure mean_abs_shap has the correct length before creating this Series
            if len(mean_abs_shap) == len(X_df.columns):
                shap_series_for_plot = pd.Series(mean_abs_shap, index=X_df.columns)
            else:
                logging.warning("Skipping SHAP plot due to feature mismatch between sample and full dataset.")
                return shap_df # Exit early if plot cannot be made reliably

            top_features = shap_series_for_plot.sort_values(ascending=False).head(20)

            colors = []
            edge_colors = []
            for feature in top_features.sort_values(ascending=True).index:
                color = 'gray' # Default color
                if feature.startswith('weak_layer'):
                    color = 'blue'
                elif feature.startswith('slab'):
                    color = 'red'
                elif feature.startswith('upper_snowpack'):
                    color = 'yellow'
                elif feature.startswith('wind') or feature.startswith('snow_drift'):
                    color = 'green'
                elif '_event_pred' in feature: # Color for event model outputs
                    color = 'purple'
                colors.append(color)

                if '_2d' in feature or '_3d' in feature or '_4d' in feature or '_5d' in feature or '_6d' in feature or '_7d' in feature or '_lag' in feature:
                    edge_colors.append('black')
                else:
                    edge_colors.append(color)

            plt.figure(figsize=(12, 10))
            top_features.sort_values(ascending=True).plot(kind='barh', color=colors, edgecolor=edge_colors, linewidth=1.5)

            legend_elements = [
                Patch(facecolor='blue', label='Weak Layer'),
                Patch(facecolor='red', label='Slab'),
                Patch(facecolor='yellow', label='Upper Snowpack'),
                Patch(facecolor='green', label='Weather/Other'),
                Patch(facecolor='purple', label='Event Model Output'), # Added legend for event model output
                Patch(facecolor='white', edgecolor='black', label='Temporal/Lag Feature')
            ]
            plt.legend(handles=legend_elements, loc='lower right')

            plt.title(f'Feature Importance for {model_name} (Top 20)')
            plt.xlabel("Mean Absolute SHAP Value (Impact on model output)")
            plt.tight_layout()
            # Save SHAP plot to the specified path in config
            plt.savefig(config.PATHS["RESULTS"]["hazard_shap_plot"].parent / f"shap_avalanche_hazard_feature_importance_{model_name}.png")
            plt.close()

        return shap_df
    except Exception as e:
        logging.error(f"Error computing or saving SHAP values for {model_name}: {e}", exc_info=True)
        return pd.DataFrame(columns=["feature", "mean_abs_shap_value"])


def select_top_k_and_prune(X_df: pd.DataFrame, shap_df: pd.DataFrame, top_k: int = 60, correlation_threshold: float = 0.99) -> List[str]:
    """
    Selects a subset of top-k features based on SHAP importance and then prunes
    highly correlated features from this subset.

    Args:
        X_df (pd.DataFrame): The full feature DataFrame.
        shap_df (pd.DataFrame): DataFrame containing SHAP feature importances.
        top_k (int): The initial number of top features to select based on SHAP values.
        correlation_threshold (float): The maximum allowed Pearson correlation coefficient
                                       between features; features above this threshold are pruned.

    Returns:
        List[str]: A list of selected and pruned feature names.
    """
    if shap_df.empty or "feature" not in shap_df.columns or "mean_abs_shap_value" not in shap_df.columns:
        logging.warning("SHAP DataFrame is empty or malformed. Cannot perform feature selection.")
        return []

    top_features = shap_df.sort_values(by="mean_abs_shap_value", ascending=False)["feature"].tolist()
    top_features = [f for f in top_features if f in X_df.columns]
    top_features = top_features[:top_k]

    if len(top_features) < 2:
        return top_features

    corr_matrix = X_df[top_features].corr().abs()
    corr_matrix.fillna(0, inplace=True)

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    return [f for f in top_features if f not in to_drop]

def plot_reliability_diagrams(y_true: pd.Series, calibrated_probabilities: np.ndarray, classes: List[int], model_name: str, save_path: Path) -> None:
    """
    Generates and saves a single plot with subplots for Reliability Diagrams
    (Calibration Plots) for each hazard class.

    Args:
        y_true (pd.Series): True labels (0-indexed).
        calibrated_probabilities (np.ndarray): Calibrated probabilities for all classes.
        classes (List[int]): List of unique 0-indexed class labels.
        model_name (str): Name of the best model.
        save_path (Path): Directory to save the plot.
    """
    logging.info(f"Generating Reliability Diagrams for {model_name}...")
    num_classes = len(classes)
    # Changed to 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

    for i, class_label_0_indexed in enumerate(classes):
        class_label_1_indexed = class_label_0_indexed + 1
        prob_true, prob_pred = calibration_curve(
            y_true == class_label_0_indexed, # True binary labels for this class
            calibrated_probabilities[:, i], # Calibrated probabilities for this class
            n_bins=10
        )
        ax = axes[i]
        ax.plot(prob_pred, prob_true, marker='o', label='Calibrated Model')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='orange')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives') # Set ylabel for each subplot for clarity
        ax.set_title(f'Hazard {class_label_1_indexed}')
        ax.legend(loc='upper left')
        ax.grid(True)
    
    # Hide any unused subplots if num_classes < 4
    for j in range(num_classes, 4):
        fig.delaxes(axes[j])

    fig.suptitle(f'Reliability Diagrams for Best Model ({model_name})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(save_path / f"reliability_diagrams_{model_name}.png")
    plt.close()
    logging.info(f"Reliability Diagrams plot saved to '{save_path / f'reliability_diagrams_{model_name}.png'}'")

def plot_probability_histograms(y_true: pd.Series, predictions_df: pd.DataFrame, classes: List[int], model_name: str, save_path: Path) -> None:
    """
    Generates and saves a single plot with subplots for Histograms of Probabilities
    by Outcome for each hazard class.

    Args:
        y_true (pd.Series): True labels (0-indexed).
        predictions_df (pd.DataFrame): DataFrame containing calibrated probabilities.
        classes (List[int]): List of unique 0-indexed class labels.
        model_name (str): Name of the best model.
        save_path (Path): Directory to save the plot.
    """
    logging.info(f"Generating Probability Histograms by Outcome for {model_name}...")
    num_classes = len(classes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True) # Adjusted figsize for 2x2
    axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

    for i, class_label_0_indexed in enumerate(classes):
        class_label_1_indexed = class_label_0_indexed + 1
        
        calibrated_probs_for_class = predictions_df[f'calibrated_proba_hazard_{class_label_1_indexed}']
        
        true_positives_mask = (y_true == class_label_0_indexed)
        true_negatives_mask = (y_true != class_label_0_indexed)

        ax = axes[i]
        sns.histplot(calibrated_probs_for_class[true_positives_mask], color='green', label=f'True Hazard {class_label_1_indexed}', bins=20, kde=True, stat='density', alpha=0.6, ax=ax)
        sns.histplot(calibrated_probs_for_class[true_negatives_mask], color='red', label=f'Not True Hazard {class_label_1_indexed}', bins=20, kde=True, stat='density', alpha=0.6, ax=ax)
        
        ax.set_xlabel(f'Calibrated Probability for Hazard {class_label_1_indexed}')
        ax.set_ylabel('Density') # Set ylabel for each subplot for clarity
        ax.set_title(f'Hazard {class_label_1_indexed} Probabilities')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6) # Reduced alpha for grid to avoid clutter

    # Hide any unused subplots if num_classes < 4
    for j in range(num_classes, 4):
        fig.delaxes(axes[j])

    fig.suptitle(f'Calibrated Probability Distributions for Best Model ({model_name})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.savefig(save_path / f"prob_histograms_{model_name}.png")
    plt.close()
    logging.info(f"Probability Histograms plot saved to '{save_path / f'prob_histograms_{model_name}.png'}'")


# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================
def main() -> None:
    """
    Main function to orchestrate the hazard model training pipeline.
    """
    logging.info(">>> Starting Avalanche Hazard Model Training Pipeline <<<")

    X, y_true, original_merged_df = load_and_prepare_data()
    if X is None or y_true is None or original_merged_df is None or X.empty or y_true.empty:
        logging.error("Data loading failed or resulted in empty datasets. Aborting training.")
        return

    logging.info(f"Loaded data with {X.shape[1]} features and {len(y_true)} samples.")
    logging.info(f"Target variable ('hazard') distribution (0-indexed): \n{y_true.value_counts()}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.3, random_state=RANDOM_STATE, stratify=y_true
    )
    logging.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Define models and their parameter grids
    models_to_train: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {
        # "RandomForestClassifier": (
        #     RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        #     {
        #         'n_estimators': [100, 200],
        #         'max_depth': [10, 20, None],
        #         'min_samples_leaf': [1, 5]
        #     }
        # ),
        # "XGBoostClassifier": (
        #     # For multi-class, XGBoost expects objective='multi:softmax' and eval_metric='mlogloss'
        #     # num_class should be set to the number of unique classes (e.g., 4 for 0,1,2,3)
        #     xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss',  
        #                       num_class=len(y_true.unique()), random_state=RANDOM_STATE, n_jobs=-1),
        #     {
        #         'n_estimators': [100, 200],
        #         'learning_rate': [0.05, 0.1],
        #         'max_depth': [3, 5]
        #     }
        # ),
        "LGBMClassifier": (
            # For multi-class, LightGBM expects objective='multiclass' and num_class
            lgb.LGBMClassifier(objective='multiclass', num_class=len(y_true.unique()), random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [20, 31],
            }
        ),
        # "LogisticRegression": (
        #     # Logistic Regression with 'ovr' (One-vs-Rest) strategy for multi-class
        #     LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', multi_class='ovr', class_weight='balanced', max_iter=1000),
        #     {
        #         'C': [0.1, 1.0, 10.0]
        #     }
        # )
    }

    # Define a scorer. For multi-class, 'f1_weighted' is a good general choice for GridSearchCV.
    scorer = make_scorer(f1_score, average='weighted', zero_division=0)

    all_results: List[Dict[str, Any]] = []
    best_overall_f1: float = -np.inf
    best_overall_model: Optional[Any] = None
    best_overall_model_name: Optional[str] = None
    best_overall_params: Optional[Dict[str, Any]] = None

    for model_name, (estimator, param_grid) in models_to_train.items():
        metrics, trained_model = train_and_evaluate_model(
            model_name, estimator, param_grid,
            X_train_scaled_df, y_train, X_test_scaled_df, y_test, scorer
        )
        all_results.append(metrics)

        # For selecting the best model, we can still use F1-weighted or choose another primary metric
        if metrics["F1 Score (weighted)"] > best_overall_f1:
            best_overall_f1 = metrics["F1 Score (weighted)"]
            best_overall_model = trained_model
            best_overall_model_name = model_name
            best_overall_params = metrics["Best Params"]

    logging.info("\n" + "="*50)
    logging.info("Training and Evaluation Summary")
    logging.info("="*50)
    for res in all_results:
        logging.info(f"Model: {res['Model']}")
        logging.info(f"  Best Params: {res['Best Params']}")
        logging.info(f"  F1 (weighted): {res['F1 Score (weighted)']:.4f}")
        logging.info(f"  MAE: {res['MAE']:.4f}")
        logging.info(f"  Quadratic Kappa: {res['Quadratic Kappa']:.4f}")
        # Log classification report and confusion matrix for each model
        logging.info(f"  Classification Report:\n{json.dumps(res['Classification Report'], indent=4)}")
        logging.info(f"  Confusion Matrix:\n{json.dumps(res['Confusion Matrix'], indent=4)}")
        logging.info("-" * 30)

    logging.info(f"\nBest overall model: {best_overall_model_name} with F1 (weighted): {best_overall_f1:.4f}")
    logging.info("="*50)

    # Save the best model and scaler
    if best_overall_model:
        joblib.dump(best_overall_model, config.PATHS["ARTIFACTS"]["hazard_model"])
        joblib.dump(scaler, config.PATHS["ARTIFACTS"]["hazard_scaler"])
        logging.info(f"Best model ({best_overall_model_name}) saved to '{config.PATHS['ARTIFACTS']['hazard_model']}'")
        logging.info(f"Scaler saved to '{config.PATHS['ARTIFACTS']['hazard_scaler']}'")

        # Save the final feature list for the hazard model
        # This is crucial for inference to ensure feature consistency
        final_features_list = X.columns.tolist()
        with open(config.PATHS["ARTIFACTS"]["hazard_final_features"], 'w') as f:
            json.dump(final_features_list, f, indent=4)
        logging.info(f"Hazard model final feature list saved to '{config.PATHS['ARTIFACTS']['hazard_final_features']}'")

        # Generate and save predictions of the best model on the full dataset
        full_scaled_X = scaler.transform(X)
        full_scaled_X_df = pd.DataFrame(full_scaled_X, columns=X.columns, index=X.index)
        
        # Predict labels and convert back to 1-indexed hazard ratings
        final_predictions_0_indexed = best_overall_model.predict(full_scaled_X_df)
        final_predictions_1_indexed = final_predictions_0_indexed + 1 # Convert back to 1-indexed
        
        predictions_df = original_merged_df.copy()
        predictions_df['predicted_hazard'] = final_predictions_1_indexed # Save 1-indexed predictions
        
        # --- Probability Calibration and Saving ---
        logging.info(f"Calibrating probabilities for the best model ({best_overall_model_name})...")
        if hasattr(best_overall_model, 'predict_proba'):
            calibrated_model = CalibratedClassifierCV(
                best_overall_model, method='isotonic', cv=10 # Using 10-fold CV for calibration
            )
            calibrated_model.fit(X_train_scaled_df, y_train) # Fit calibration on training data

            final_calibrated_probabilities = calibrated_model.predict_proba(full_scaled_X_df)
            
            for i, class_label_0_indexed in enumerate(best_overall_model.classes_):
                class_label_1_indexed = class_label_0_indexed + 1 # Convert class label to 1-indexed for column name
                predictions_df[f'calibrated_proba_hazard_{class_label_1_indexed}'] = final_calibrated_probabilities[:, i]
            logging.info("Calibrated probabilities added to predictions.")
        else:
            logging.warning(f"Model {best_overall_model_name} does not support predict_proba. Skipping probability calibration.")

        # Save predictions to CSV
        predictions_output_csv_path = config.PATHS["ARTIFACTS"]["hazard_predictions_csv"]
        predictions_df.to_csv(predictions_output_csv_path, index=False)
        logging.info(f"Final hazard predictions (CSV) saved to '{predictions_output_csv_path}'")

        # Save predictions to JSON
        predictions_output_json_path = config.PATHS["ARTIFACTS"]["hazard_predictions_json"]
        # Convert DataFrame to JSON, orient='records' makes it a list of objects
        predictions_df.to_json(predictions_output_json_path, orient='records', indent=4)
        logging.info(f"Final hazard predictions (JSON) saved to '{predictions_output_json_path}'")


        # Calculate final metrics on the full dataset for the report
        final_y_pred = best_overall_model.predict(full_scaled_X_df)
        final_metrics_report = {
            "F1 Score (weighted)": f1_score(y_true, final_y_pred, average='weighted', zero_division=0),
            "Precision (weighted)": precision_score(y_true, final_y_pred, average='weighted', zero_division=0),
            "Recall (weighted)": recall_score(y_true, final_y_pred, average='weighted', zero_division=0),
            "MAE": mean_absolute_error(y_true, final_y_pred),
            "Quadratic Kappa": cohen_kappa_score(y_true, final_y_pred, weights='quadratic'),
            "Classification Report": classification_report(y_true, final_y_pred, zero_division=0, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_true, final_y_pred).tolist()
        }

        # Create a summary dictionary of all results
        summary_results = {
            "best_overall_model_name": best_overall_model_name,
            "best_overall_model_params": best_overall_params,
            "final_features_list": final_features_list,
            "final_metrics_on_full_dataset": final_metrics_report,
            "all_model_comparison": all_results
        }
        
        # Save the summary to a JSON file
        report_path = config.RESULTS_DIR / "avalanche_hazard_model_report.json"
        with open(report_path, "w") as f:
            json.dump(summary_results, f, indent=4)
        logging.info(f"Final report for hazard model saved to '{report_path}'.")

        # Plot and save Confusion Matrix (counts) for the best model on the full dataset
        cm = confusion_matrix(y_true, final_predictions_0_indexed) # Use 0-indexed true and predicted labels
        
        # Get unique classes from y_true (0-indexed), then convert to 1-indexed for labels
        class_labels_0_indexed = np.unique(y_true)
        class_labels_1_indexed = [str(int(label) + 1) for label in class_labels_0_indexed]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels_1_indexed, yticklabels=class_labels_1_indexed)
        plt.xlabel('Predicted Hazard (1-4)')
        plt.ylabel('True Hazard (1-4)')
        plt.title(f'Confusion Matrix (Counts) for Best Model ({best_overall_model_name})')
        plt.savefig(config.PATHS["RESULTS"]["hazard_confusion_matrix_plot_base"] / f"confusion_matrix_counts_{best_overall_model_name}.png")
        plt.close()
        logging.info(f"Confusion Matrix (Counts) plot saved to '{config.PATHS['RESULTS']['hazard_confusion_matrix_plot_base'] / f'confusion_matrix_counts_{best_overall_model_name}.png'}'")

        # Plot and save Confusion Matrix (percentages) for the best model on the full dataset
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize by row (true labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', 
                    xticklabels=class_labels_1_indexed, yticklabels=class_labels_1_indexed)
        plt.xlabel('Predicted Hazard (1-4)')
        plt.ylabel('True Hazard (1-4)')
        plt.title(f'Confusion Matrix (Percentages) for Best Model ({best_overall_model_name})')
        plt.savefig(config.PATHS["RESULTS"]["hazard_confusion_matrix_plot_base"] / f"confusion_matrix_percentages_{best_overall_model_name}.png")
        plt.close()
        logging.info(f"Confusion Matrix (Percentages) plot saved to '{config.PATHS['RESULTS']['hazard_confusion_matrix_plot_base'] / f'confusion_matrix_percentages_{best_overall_model_name}.png'}'")

        # --- SHAP Feature Importance Plot for the Best Model ---
        logging.info(f"Generating SHAP feature importance for the best model ({best_overall_model_name})...")
        compute_and_save_shap(
            best_overall_model,
            X, # Use the full feature set X
            best_overall_model_name,
            plot_save_shap_values=True,
            classes=list(best_overall_model.classes_) # Pass 0-indexed classes for SHAP plotting
        )

        # --- Reliability Diagrams (Calibration Plots) ---
        if hasattr(best_overall_model, 'predict_proba'):
            plot_reliability_diagrams(
                y_true,
                final_calibrated_probabilities,
                list(best_overall_model.classes_), # Pass 0-indexed classes
                best_overall_model_name,
                config.PATHS["RESULTS"]["hazard_confusion_matrix_plot_base"] # Save to figures directory
            )
        else:
            logging.warning(f"Model {best_overall_model_name} does not support predict_proba. Skipping Reliability Diagrams.")

        # --- Histograms of Probabilities by Outcome ---
        if hasattr(best_overall_model, 'predict_proba'):
            plot_probability_histograms(
                y_true,
                predictions_df, # Use predictions_df which contains calibrated probabilities
                list(best_overall_model.classes_), # Pass 0-indexed classes
                best_overall_model_name,
                config.PATHS["RESULTS"]["hazard_confusion_matrix_plot_base"] # Save to figures directory
            )
        else:
            logging.warning(f"Model {best_overall_model_name} does not support predict_proba. Skipping Probability Histograms.")


    # --- Summary Table ---
    logging.info("\n" + "="*50)
    logging.info("Model Comparison Summary Table")
    logging.info("="*50)

    summary_data = [{
        "Model": res['Model'],
        "F1 (weighted)": f"{res['F1 Score (weighted)']:.4f}",
        "MAE": f"{res['MAE']:.4f}",
        "Quadratic Kappa": f"{res['Quadratic Kappa']:.4f}",
        "Best Params": str(res['Best Params']) # Convert dict to string for table display
    } for res in all_results]
    
    summary_df = pd.DataFrame(summary_data)
    logging.info("\n" + summary_df.to_string(index=False))
    logging.info("="*50)


    logging.info(">>> Avalanche Hazard Model Training Pipeline Completed <<<")

if __name__ == "__main__":
    main()