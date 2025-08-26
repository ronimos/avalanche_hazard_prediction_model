# -*- coding: utf-8 -*-
"""
================================================================================
Avalanche Hazard Forecasting Package
================================================================================

This package contains the complete pipeline for avalanche hazard forecasting,
including data processing, model training, and prediction.

The primary functions from each module are exposed at the package level for
easy access and programmatic execution of the pipeline steps.

Main Modules:
-------------
- run_pipeline: Orchestrates the entire training process.
- run_prediction: Orchestrates the end-to-end prediction process for a given day.
- make_avalanche_event_training_dataset: Generates the training dataset.
- train_avalanche_event_model: Trains the first-stage avalanche event model.
- train_avalanche_hazard_model: Trains the final avalanche hazard model.
- make_prediction_dataset: Prepares a feature set for a new prediction date.
- predict: Runs inference using the trained models to generate final predictions.
- plot_results: Contains functions for visualizing model predictions and results.

"""

# Expose the main orchestration functions for easy access
from .run_pipeline import run_full_pipeline
from .run_prediction import main as run_prediction_pipeline

# Expose the core function from each step of the pipeline
from .make_avalanche_event_training_dataset import main_pipeline as generate_training_data
from .train_avalanche_event_model import main as train_event_model
from .train_avalanche_hazard_model import main as train_hazard_model
from .make_prediction_dataset import build_prediction_dataset
from .predict import predict_avalanche_hazards
from .plot_results import create_prediction_map

# Define the public API of the package
# This specifies what is imported when a user runs `from src import *`
__all__ = [
    'run_full_pipeline',
    'run_prediction_pipeline',
    'generate_training_data',
    'train_event_model',
    'train_hazard_model',
    'build_prediction_dataset',
    'predict_avalanche_hazards',
    'create_prediction_map',
]

__version__ = "1.0.0"