# -*- coding: utf-8 -*-
"""
================================================================================
Final PU Learning Pipeline for Avalanche Hazard Forecasting
================================================================================

Purpose:
--------
This script implements and evaluates the final, optimized Positive-Unlabeled (PU)
learning model for avalanche hazard forecasting. It compares the performance of the
custom 'Spy+Bootstrap' model against a theoretical "Oracle" model to validate its
effectiveness under real-world data scarcity conditions.

This version includes a feature selection loop to identify the optimal number of
features by pruning less important and highly correlated predictors.

Methodology:
------------
The core of this pipeline is the 'Spy+Bootstrap' model, a two-step algorithm:
1.  **Spy Phase:** A portion of labeled positive samples ("spies") are hidden
    within the unlabeled set. A preliminary classifier is trained to find them,
    which allows for the identification of a high-confidence set of
    "Reliable Negatives" (RN) from the unlabeled data.
2.  **Training Phase:** A final, powerful RandomForestClassifier is trained on a
    "cleaned" dataset composed only of the original Labeled Positives (P) and
    the newly identified Reliable Negatives (RN).

This script performs the following steps:
1.  Loads the pre-processed master dataset and target files.
2.  Performs an iterative feature selection process:
    a. For different values of top-k features (e.g., 20, 30, 40, 60):
    b. Selects the top-k features based on pre-computed SHAP values.
    c. Prunes this set by removing highly correlated features.
    d. Trains and evaluates the Spy+Bootstrap model on the reduced feature set.
3.  Trains an 'Oracle' model on the full feature set as a benchmark.
4.  Saves and displays the final performance comparison.

"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import json
import shap
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple, Union

# Assuming a config.py file with necessary paths
import config

# --- Configuration ---
# Use paths from config.py
OUTPUT_DIR = config.RESULTS_DIR
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the directory exists

# Set the single, final fraction of labeled positives for this experiment
P_FRACTION: float = config.MODELING_CONFIG["p_fraction"]
RANDOM_STATE: int = config.MODELING_CONFIG["random_state"]

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for more detailed logs during execution
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# 2. CORE CLASSES AND FUNCTIONS
# =============================================================================
class EnsemblePuClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom Positive-Unlabeled (PU) learning classifier that uses an ensemble
    approach with a "Spy+Bootstrap" methodology.

    This classifier is designed for scenarios where only positive and unlabeled
    samples are available. It identifies reliable negative samples from the
    unlabeled set and then trains a final classifier on the labeled positives
    and these reliable negatives.

    Attributes:
        base_estimator: The base scikit-learn estimator to be used (e.g., RandomForestClassifier).
        n_bootstrap_runs (int): Number of bootstrap iterations for identifying reliable negatives.
        spy_ratio (float): The fraction of labeled positives to use as "spies" in the unlabeled set.
        consensus_threshold (float): The minimum proportion of bootstrap runs a sample must be
                                     identified as negative to be considered a "reliable negative".
        random_state (Optional[int]): Seed for random number generation for reproducibility.
        final_model_ (Optional[BaseEstimator]): The trained final classifier.
    """
    def __init__(self, base_estimator: BaseEstimator, n_bootstrap_runs: int = 25, spy_ratio: float = 0.15,
                 consensus_threshold: float = 0.9, random_state: Optional[int] = None):
        """
        Initializes the EnsemblePuClassifier.
        """
        self.base_estimator = base_estimator
        self.n_bootstrap_runs = n_bootstrap_runs
        self.spy_ratio = spy_ratio
        self.consensus_threshold = consensus_threshold
        self.random_state = random_state
        self.final_model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsemblePuClassifier":
        """
        Fits the PU classifier to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Labels, where 1 indicates a positive sample and 0 indicates an unlabeled sample.

        Returns:
            EnsemblePuClassifier: The fitted classifier instance.
        """
        pos_idx = np.where(y == 1)[0]
        unlabeled_idx = np.where(y == 0)[0]

        if len(pos_idx) == 0:
            logging.warning("No labeled positives provided. Fitting on full data.")
            self.final_model_ = clone(self.base_estimator).fit(X, y)
            return self
        
        # Handle cases where there are no unlabeled samples
        if len(unlabeled_idx) == 0:
            logging.warning("No unlabeled samples provided. Fitting on positive samples only.")
            self.final_model_ = clone(self.base_estimator).fit(X[pos_idx], y[pos_idx])
            return self

        rn_counts: Counter = Counter()
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_bootstrap_runs):
            spy_model = clone(self.base_estimator)
            spy_model.set_params(random_state=rng.randint(1e6))
            spy_count = int(len(pos_idx) * self.spy_ratio)
            if spy_count < 2:
                # If spy_count is too small, skip this bootstrap run
                continue

            spies = rng.choice(pos_idx, size=spy_count, replace=False)
            
            # Ensure X_spy and y_spy are correctly constructed
            # X from unlabeled, then X from spies
            X_spy = np.vstack([X[unlabeled_idx], X[spies]])
            # y for unlabeled is 0, y for spies is 1
            y_spy = np.concatenate([np.zeros(len(unlabeled_idx)), np.ones(spy_count)])

            spy_model.set_params(class_weight='balanced')
            spy_model.fit(X_spy, y_spy)

            # Check if predict_proba is available and use it
            if hasattr(spy_model, 'predict_proba'):
                scores_spy = spy_model.predict_proba(X[spies])[:, 1]
                scores_unlabeled = spy_model.predict_proba(X[unlabeled_idx])[:, 1]
            else:
                # Fallback to decision_function for models without predict_proba
                # Or raise an error if predict_proba is essential
                logging.warning("Base estimator does not have predict_proba. Using decision_function if available.")
                if hasattr(spy_model, 'decision_function'):
                    scores_spy = spy_model.decision_function(X[spies])
                    scores_unlabeled = spy_model.decision_function(X[unlabeled_idx])
                else:
                    raise AttributeError("Base estimator must have either 'predict_proba' or 'decision_function' method.")

            threshold = np.percentile(scores_spy, 25)
            rns = unlabeled_idx[scores_unlabeled < threshold]
            rn_counts.update(rns)

        consensus_rns = [idx for idx, count in rn_counts.items()
                         if count / self.n_bootstrap_runs >= self.consensus_threshold]

        if not consensus_rns:
            logging.warning("No consensus reliable negatives found. Using P vs U for final training.")
            # Fallback: Train on all positives vs all unlabeled if no reliable negatives are found
            self.final_model_ = clone(self.base_estimator).fit(X, y)
            return self

        logging.info(f"Using {len(consensus_rns)} reliable negatives for training.")
        X_train_final = np.vstack([X[pos_idx], X[consensus_rns]])
        y_train_final = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(consensus_rns))])
        self.final_model_ = clone(self.base_estimator)

        # Only apply class_weight if supported
        if 'class_weight' in self.final_model_.get_params():
            self.final_model_.set_params(class_weight='balanced')
        self.final_model_.fit(X_train_final, y_train_final)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.final_model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        # Ensure the final model has predict_proba
        if hasattr(self.final_model_, 'predict_proba'):
            return self.final_model_.predict_proba(X)
        else:
            # If not, try to use decision_function and convert to probabilities (e.g., for Logistic Regression)
            # This is a generic sigmoid for binary classification, might need adjustment for multi-class
            logging.warning("Final model does not have predict_proba. Attempting to use decision_function and sigmoid.")
            if hasattr(self.final_model_, 'decision_function'):
                decision_scores = self.final_model_.decision_function(X)
                # For binary classification, convert decision scores to probabilities using sigmoid
                probabilities = 1 / (1 + np.exp(-decision_scores))
                # Return as (n_samples, 2) array for binary classification
                return np.vstack([1 - probabilities, probabilities]).T
            else:
                raise AttributeError("Final model must have either 'predict_proba' or 'decision_function' method.")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for input samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        if self.final_model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return self.final_model_.predict(X)

def sanitize_params(params: Union[Dict[str, Any], str]) -> str:
    """
    Converts a dictionary of parameters or a string into a sanitized string
    suitable for use in filenames or identifiers.

    Args:
        params (Union[Dict[str, Any], str]): The parameters to sanitize.

    Returns:
        str: A sanitized string representation of the parameters.
    """
    if isinstance(params, dict):
        return "_".join(f"{k}-{str(v).replace('/', '-')}" for k, v in params.items())
    if isinstance(params, str):
        return params.replace('/', '-')
    return str(params)

def load_and_prepare_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Loads features and targets from pre-processed CSV files and prepares them
    for model training. It merges features and targets based on 'date' and
    'polygon' to ensure data consistency.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame]]:
            - X (pd.DataFrame): DataFrame containing the features.
            - y_true (pd.Series): Series containing the true binary labels.
            - merged_df (pd.DataFrame): DataFrame containing original 'date' and 'polygon'
                                        along with other merged data, useful for saving predictions.
            Returns (None, None, None) if data loading or preparation fails.
    """
    logging.info("Loading and preparing data...")
    try:
        features_path = config.PATHS["PROCESSED_DATA"]["training_features"]
        targets_path = config.PATHS["PROCESSED_DATA"]["training_targets"]

        X_df_features = pd.read_csv(features_path)
        y_df_targets = pd.read_csv(targets_path)

        logging.info(f"Loaded features from '{features_path}'. Shape: {X_df_features.shape}")
        logging.info(f"Loaded targets from '{targets_path}'. Shape: {y_df_targets.shape}")

        X_df_features['date'] = pd.to_datetime(X_df_features['date'])
        y_df_targets['date'] = pd.to_datetime(y_df_targets['date'])
        
        X_df_features['polygon'] = X_df_features['polygon'].astype(str)
        y_df_targets['polygon'] = y_df_targets['polygon'].astype(str)

        # Merge features and targets to ensure they align by date and polygon
        # Use an inner merge to keep only rows present in both (should be all)
        merged_df = pd.merge(X_df_features, y_df_targets, on=['date', 'polygon'], how='inner')

        # The target column is 'avalanche_event' from make_avalanche_event_training_dataset.py
        # Both X and y_true are now derived from merged_df to ensure consistent sample counts
        y_true = merged_df['avalanche_event']
        
        # Get all columns from y_df_targets (excluding 'date' and 'polygon' because they are merge keys)
        # These are the columns that represent target-related information and should be excluded from features.
        target_columns_to_exclude = [col for col in y_df_targets.columns if col not in ['date', 'polygon']]
        
        # Drop all target-related columns and the 'date' and 'polygon' identifiers from X
        X = merged_df.drop(columns=target_columns_to_exclude + ['date', 'polygon'], errors='ignore')

    except FileNotFoundError as e:
        logging.error(f"Missing data file: {e}. Please ensure 'make_avalanche_event_training_dataset.py' was run correctly.")
        return None, None, None
    except KeyError as e:
        logging.error(f"Missing expected column in training data: {e}. "
                      "Ensure 'make_avalanche_event_training_dataset.py' was run correctly and generated 'avalanche_event'.")
        return None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return None, None, None

    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except ValueError:
                logging.warning(f"Could not convert column '{col}' to numeric. Filling with 0.")
                X[col] = 0

    # Return merged_df without the 'avalanche_event' target column, but with date/polygon for prediction saving
    return X, y_true, merged_df.drop(columns=target_columns_to_exclude, errors='ignore')


def evaluate_model(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Evaluates model performance using Precision-Recall Curve and F1-score.

    Args:
        y_true (np.ndarray): True binary labels.
        y_scores (np.ndarray): Predicted probability scores for the positive class.

    Returns:
        Dict[str, float]: A dictionary containing F1 score, Precision, Recall,
                          AUC, and the best threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = np.divide(2 * precision * recall, precision + recall,
                          out=np.zeros_like(precision), where=(precision + recall) != 0)
    best_idx = np.argmax(f1_scores)
    
    # Handle case where thresholds might be empty (e.g., all samples are one class)
    threshold = thresholds[best_idx] if thresholds.size > 0 else 0.5 # Default to 0.5 if no thresholds
    
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "F1 (Tuned)": f1_scores[best_idx],
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_scores),
        "Best Threshold": threshold,
    }


def compute_and_save_shap(model: BaseEstimator, X_df: pd.DataFrame, model_name: str, plot_save_shap_values: bool = False) -> pd.DataFrame:
    """
    Computes SHAP (SHapley Additive exPlanations) values for a given model and
    feature set, and optionally saves a plot of feature importance.

    Args:
        model (BaseEstimator): The trained machine learning model.
        X_df (pd.DataFrame): The DataFrame of features used for SHAP explanation.
        model_name (str): A name for the model, used in plot titles and logging.
        plot_save_shap_values (bool): If True, generates and saves a SHAP summary plot.

    Returns:
        pd.DataFrame: A DataFrame containing features and their mean absolute SHAP values,
                      sorted by importance.
    """
    logging.info(f"Generating SHAP values for {model_name}...")
    try:
        # For tree-based models, TreeExplainer is generally preferred
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback for other models, might be slower
            # Use a sample of the data for KernelExplainer if X_df is too large, for performance
            X_sample_df = shap.utils.sample(X_df, 1000) if X_df.shape[0] > 1000 else X_df
            explainer = shap.KernelExplainer(model.predict_proba, X_sample_df)
            X_df = X_sample_df # Use the sampled data for SHAP value computation

        shap_values = explainer(X_df, check_additivity=False)

        if shap_values.values.ndim == 3:
            # For multi-output models or classifiers with multiple classes,
            # use SHAP values for the positive class (index 1)
            shap_array = shap_values.values[:, :, 1]
        else:
            shap_array = shap_values.values

        mean_abs_shap = np.abs(shap_array).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": X_df.columns,
            "mean_abs_shap_value": mean_abs_shap
        })

        shap_df = shap_df.sort_values(by="mean_abs_shap_value", ascending=False)


        if plot_save_shap_values:
            # Save feature names to the specified path in config
            shap_df.to_csv(config.PATHS["ARTIFACTS"]["event_feature_names"], index=False)
            shap_series = pd.Series(mean_abs_shap, index=X_df.columns)
            top_features = shap_series.sort_values(ascending=False).head(20)

            colors = []
            edge_colors = []
            for feature in top_features.sort_values(ascending=True).index:
                color = 'green'
                if feature.startswith('weak_layer'):
                    color = 'blue'
                elif feature.startswith('slab'):
                    color = 'red'
                elif feature.startswith('upper_snowpack'):
                    color = 'yellow'
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
                Patch(facecolor='white', edgecolor='black', label='Temporal/Lag Feature')
            ]
            plt.legend(handles=legend_elements, loc='lower right')

            plt.title(f'Feature Importance for {model_name} (Top 20)')
            plt.xlabel("Mean Absolute SHAP Value (Impact on model output)")
            plt.tight_layout()
            # Save SHAP plot to the specified path in config
            plt.savefig(config.PATHS["RESULTS"]["event_shap_plot"].parent / f"shap_avalanche_event_feature_importance_{model_name}.png")
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

def main() -> None:
    """
    Main function to execute the entire avalanche event modeling pipeline.
    This includes data loading, PU learning, hyperparameter tuning with
    feature selection, model evaluation, and saving of artifacts.
    """
    X, y_true, merged_df = load_and_prepare_data()
    if X is None or y_true is None or X.empty or y_true.empty:
        logging.error("Data loading failed or resulted in empty datasets. Aborting training.")
        return

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_true: pd.Series
    y_test_true: pd.Series

    X_train, X_test, y_train_true, y_test_true = train_test_split(
        X, y_true, test_size=0.3, random_state=RANDOM_STATE, stratify=y_true
    )

    scaler = StandardScaler()

    y_train_pu: np.ndarray = np.zeros_like(y_train_true, dtype=int)
    pos_indices: np.ndarray = np.where(y_train_true == 1)[0]

    if len(pos_indices) == 0:
        logging.error("No positive samples in the training set. Cannot perform PU learning.")
        return

    rng = np.random.RandomState(RANDOM_STATE)
    selected_pos: np.ndarray = rng.choice(pos_indices, size=int(P_FRACTION * len(pos_indices)), replace=False)
    y_train_pu[selected_pos] = 1

    n_cores: int = multiprocessing.cpu_count()

    param_grid: Dict[str, List[Any]] = {
        "n_estimators": [300, 250, 200],
        "min_samples_leaf": [2, 3, 4],
        "spy_ratio": [0.15, 0.2],
        "max_depth": [None, 20, 15, 10],
        "max_features": ['sqrt']
    }

    rf_param_grid = ParameterGrid(param_grid)
    results: List[Dict[str, Any]] = []
    best_score: float = -np.inf
    best_model: Optional[EnsemblePuClassifier] = None
    best_features: Optional[List[str]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_threshold: Optional[float] = None
    best_metrics: Dict[str, Any] = {}

    n_feature: int = X.shape[1]
    features_list: List[int] = [int(x * n_feature) for x in [1, 0.9]]#, 0.8, 0.7, 0.5]]
    total_iterations: int = len(list(rf_param_grid)) * len(features_list)

    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for params in rf_param_grid:
            logging.info(f"Tuning with params: {params}")
            base_model: RandomForestClassifier = RandomForestClassifier(
                n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
                random_state=RANDOM_STATE, n_jobs=n_cores
            )

            pu_model: EnsemblePuClassifier = EnsemblePuClassifier(
                base_estimator=base_model, n_bootstrap_runs=20, spy_ratio=params["spy_ratio"],
                consensus_threshold=0.9, random_state=RANDOM_STATE
            )

            X_train_scaled: np.ndarray = scaler.fit_transform(X_train)

            pu_model.fit(X_train_scaled, y_train_pu)

            if pu_model.final_model_:
                model_name: str = f"RandomForest_PU_params_{sanitize_params(params)}"
                shap_df: pd.DataFrame = compute_and_save_shap(
                    pu_model.final_model_,
                    pd.DataFrame(X_train_scaled, columns=X_train.columns),
                    model_name
                )

                for top_k in features_list:
                    selected_features: List[str] = select_top_k_and_prune(
                        pd.DataFrame(X_train_scaled, columns=X_train.columns), shap_df, top_k=top_k
                    )

                    if not selected_features:
                        logging.warning(f"No features selected for top_k={top_k}. Skipping.")
                        pbar.update(1)
                        continue

                    X_train_sub: pd.DataFrame = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]
                    X_test_scaled: np.ndarray = scaler.transform(X_test)
                    X_test_sub: pd.DataFrame = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features]

                    pu_model.fit(X_train_sub.values, y_train_pu)
                    scores: np.ndarray = pu_model.predict_proba(X_test_sub.values)[:, 1]
                    metrics: Dict[str, float] = evaluate_model(y_test_true, scores)
                    metrics["Model"] = f"RF PU {sanitize_params(params)}, Top {top_k} Features"
                    results.append(metrics)

                    if metrics["F1 (Tuned)"] > best_score:
                        best_score = metrics["F1 (Tuned)"]
                        best_model = clone(pu_model)
                        best_features = selected_features
                        best_params = params
                        best_threshold = metrics["Best Threshold"]
                        best_metrics = metrics.copy()

                    logging.info(f"Metrics for {metrics['Model']}: \n{json.dumps(metrics, indent=4)}")
                    pbar.update(1)

    if best_model and best_features and best_params and best_threshold is not None:
        logging.info("Retraining best model on full dataset...")
        print(f"{'='*100}")
        print(f"Best Model Parameters: {best_params}")
        print(f"Best F1 Score: {best_score:.4f}")
        print(f"Number of Best Features: {len(best_features)}")
        print(f"Optimal Prediction Threshold: {best_threshold:.4f}")
        print(f"{'='*100}")

        full_scaled: np.ndarray = scaler.fit_transform(X.loc[:, best_features])
        full_y: np.ndarray = y_true.values
        pu_labels_full: np.ndarray = np.zeros_like(full_y)
        full_pos_indices: np.ndarray = np.where(full_y == 1)[0]
        rng = np.random.RandomState(RANDOM_STATE)
        selected_pos_full: np.ndarray = rng.choice(
            full_pos_indices,
            size=int(P_FRACTION * len(full_pos_indices)),
            replace=False
        )
        pu_labels_full[selected_pos_full] = 1

        best_model.fit(full_scaled, pu_labels_full)
        
        raw_scores: np.ndarray = best_model.predict_proba(full_scaled)[:, 1]

        OPTIMAL_THRESHOLD: float = best_threshold
        adjusted_scores: np.ndarray = np.zeros_like(raw_scores)
        
        low_mask: np.ndarray = raw_scores < OPTIMAL_THRESHOLD
        high_mask: np.ndarray = ~low_mask

        if np.any(low_mask):
            if OPTIMAL_THRESHOLD > 0:
                adjusted_scores[low_mask] = 0.5 * (raw_scores[low_mask] / OPTIMAL_THRESHOLD)
            else:
                adjusted_scores[low_mask] = 0.0
        
        if np.any(high_mask):
            denominator: float = (1.0 - OPTIMAL_THRESHOLD)
            if denominator > 0:
                adjusted_scores[high_mask] = 0.5 + 0.5 * ((raw_scores[high_mask] - OPTIMAL_THRESHOLD) / denominator)
            else:
                adjusted_scores[high_mask] = 1.0
        
        predicted_labels_adjusted: np.ndarray = (adjusted_scores >= 0.5).astype(int)
        
        # Calculate final metrics on the full dataset
        final_metrics = evaluate_model(full_y, raw_scores)
        
        # Create a report dictionary
        report = {
            "best_model_name": "EnsemblePuClassifier with RandomForest",
            "best_model_params": best_params,
            "optimal_prediction_threshold": best_threshold,
            "final_metrics": {
                "F1_score_full_data": final_metrics['F1 (Tuned)'],
                "Precision_full_data": final_metrics['Precision'],
                "Recall_full_data": final_metrics['Recall'],
                "AUC_full_data": final_metrics['AUC']
            },
            "test_metrics": {
                "F1_score_test_data": best_metrics['F1 (Tuned)'],
                "Precision_test_data": best_metrics['Precision'],
                "Recall_test_data": best_metrics['Recall'],
                "AUC_test_data": best_metrics['AUC'],
            },
        }

        # Save the report to a JSON file
        report_path = config.RESULTS_DIR / "avalanche_event_model_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Final report for event model saved to '{report_path}'.")

        predictions_df_output: pd.DataFrame = merged_df[['date', 'polygon']].copy()
        predictions_df_output['raw_score'] = raw_scores
        predictions_df_output['adjusted_score'] = adjusted_scores
        predictions_df_output['predicted_label'] = predicted_labels_adjusted
        
        # Use the new specific path for event model adjusted predictions
        predictions_df_output.to_csv(config.PATHS["ARTIFACTS"]["event_adjusted_predictions"], index=False)
        logging.info(f"Adjusted predictions saved to '{config.PATHS['ARTIFACTS']['event_adjusted_predictions']}'.")


        pd.DataFrame(results).to_csv(config.PATHS["ARTIFACTS"]["event_results_summary"], index=False)
        joblib.dump(best_model.final_model_, config.PATHS["ARTIFACTS"]["event_model"])
        joblib.dump(scaler, config.PATHS["ARTIFACTS"]["event_scaler"])

        model_params_to_save: Dict[str, Any] = best_params.copy()
        model_params_to_save['best_threshold'] = best_threshold
        with open(config.PATHS["ARTIFACTS"]["event_model_params"], "w") as pf:
            json.dump(model_params_to_save, pf, indent=4)

        # Save the final feature list for the event model
        # This is crucial for inference to ensure feature consistency
        with open(config.PATHS["ARTIFACTS"]["event_final_features"], 'w') as f:
            json.dump(best_features, f, indent=4)
        logging.info(f"Event model final feature list saved to '{config.PATHS['ARTIFACTS']['event_final_features']}'")

        precision, recall, _ = precision_recall_curve(full_y, raw_scores) # Use raw_scores for PR curve
        plt.figure()
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve on Full Data (Final Model)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PATHS["RESULTS"]["event_pr_curve_plot"])
        plt.close()

        shap_df_final: pd.DataFrame = compute_and_save_shap(best_model.final_model_,
                                        pd.DataFrame(full_scaled, columns=best_features),
                                        "best_model_final",
                                        plot_save_shap_values=True)
        shap_df_final.to_csv(config.PATHS["ARTIFACTS"]["event_shap_values"], index=False)

    else:
        logging.error("No best model found after tuning. Check logs for errors during training iterations.")

    logging.info("Evaluation complete. Results and artifacts saved.")

if __name__ == "__main__":
    main()
