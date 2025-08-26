# -*- coding: utf-8 -*-
"""
================================================================================
Prediction Data Generation Pipeline
================================================================================

Purpose:
--------
This script prepares a feature dataset for making new predictions with the
trained avalanche hazard model. It is a streamlined version of the training
data pipeline, designed to be run on a daily or as-needed basis to generate
features for a specific day.

Methodology:
------------
This script runs for ALL forecast polygons. It automatically determines the
necessary date range (the last 10 days from the prediction date) to calculate
all required temporal features (e.g., 3, 5, and 7-day means, standard
deviations, trends, and lags). This ensures that the feature set for inference
is created identically to the training set.

Workflow:
----------
1.  **Manifest Creation**: Identifies all forecast polygons and determines the
    required 10-day date range for data collection based on the prediction date.
2.  **Data Fetching**: Downloads the necessary snowpack (.pro) and weather
    (.smet) files for the recent period using centralized utility functions.
3.  **Data Processing & Feature Engineering**:
    a. For each polygon, processes the raw snowpack and weather data using the
       shared functions from `dataset_utils`. The snowpack data is sliced to
       the specific 10-day window for efficiency.
    b. Calculates all temporal and engineered features, ensuring the methodology
       is identical to the training pipeline.
4.  **Finalization**: Merges all data, performs final feature engineering,
    and filters the dataset to only the specified prediction date. It saves the
    complete, analysis-ready feature set to `data/processed/`.

CLI Usage:
----------
To run for the current date:
    python make_prediction_dataset.py

To run for a specific date:
    python make_prediction_dataset.py --date YYYY-MM-DD

Example:
    python make_prediction_dataset.py --date 2024-02-15

"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import shared, centralized functions and project-wide configurations
import dataset_utils as dsu
import config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# 2. CONSTANTS & CONFIGURATION
# =============================================================================
# Set to 10 days to provide a safe buffer for all rolling feature
# calculations (up to 7 days) and lags (up to 2 days).
DAYS_OF_DATA_NEEDED = 10

# =============================================================================
# 3. CORE FUNCTIONS FOR PREDICTION DATA
# =============================================================================

def create_prediction_manifest(end_date: datetime, save_manifest: bool=True) -> Optional[pd.DataFrame]:
    """
    Creates a download manifest for all polygons for the recent date range
    needed to generate features for the specified end_date.

    Args:
        end_date (datetime): The final date for which features are needed.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the download manifest for
                                the prediction period, or None if a critical error occurs.
    """
    logging.info("--- Creating Prediction Download Manifest ---")
    
    start_date = end_date - timedelta(days=DAYS_OF_DATA_NEEDED)
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))

    # Load all polygons to generate predictions for every forecast area
    all_polygons_df = dsu.get_polygon_centroids(config.PATHS["RAW_DATA"]["polygons"], config.PROCESSING_CONFIG["geoprojection_epsg"])
    if all_polygons_df is None:
        logging.error("Failed to load polygon centroids. Aborting manifest creation.")
        return None
        
    snowpack_locations = pd.read_csv(config.PATHS["RAW_DATA"]["snowpack_locations"])
    station_map_df = dsu.find_closest_stations(all_polygons_df, snowpack_locations)
    polygons_with_stations = all_polygons_df.join(station_map_df)

    manifest_list = []
    for _, polygon_row in polygons_with_stations.iterrows():
        manifest_list.extend([
            {
                'date': date, 'polygon': int(polygon_row['title']),
                'title': int(polygon_row['title']), 'st_id': int(polygon_row['st_id']),
                'zone': int(polygon_row['zone']), 'centroid_lat': polygon_row['centroid_lat'],
                'centroid_lon': polygon_row['centroid_lon']
            }
            for date in date_range
        ])
            
    if not manifest_list:
        logging.error("No manifest entries could be created.")
        return None
        
    final_df = pd.DataFrame(manifest_list)
    final_df['water_year'] = np.where(final_df['date'].dt.month >= 10, final_df['date'].dt.year, final_df['date'].dt.year - 1)

    st_id_str = final_df['st_id'].astype(int).astype(str).str.zfill(6)
    zone_str = final_df['zone'].astype(int).astype(str).str.zfill(3)
    water_year_str = final_df['water_year'].astype(str)

    # Construct file paths and download URLs for dataset_utils
    # Note: This relative path is used by dsu.download_snowpack_files
    final_df['URL'] = "../data/snowpack/output/" + water_year_str + "/zone" + zone_str + "/" + st_id_str
    final_df['download_URL'] = "https://nwp.mtnweather.info/ron/ssd/snowpack/output/" + water_year_str + "/zone" + zone_str + "/" + st_id_str + "/" + st_id_str + "_res.pro"
    
    logging.info(f"Created manifest for {len(all_polygons_df)} polygons for the period {start_date.date()} to {end_date.date()}.")
    
    if save_manifest:
        # Construct a date-stamped filename to avoid overwriting other manifests.
        output_dir = config.PATHS["PROCESSED_DATA"]["download_manifest"].parent
        date_str = end_date.strftime('%Y-%m-%d')
        filename = f"prediction_manifest_{date_str}.csv"
        output_path = output_dir / filename
        
        # Save the DataFrame to a CSV file.
        final_df.to_csv(output_path, index=False)
        logging.info(f"Prediction manifest saved to '{output_path}'")
    
    return final_df


def build_prediction_dataset(prediction_date: datetime):
    """
    Main function to execute the entire prediction data generation pipeline for a given date.
    
    Args:
        prediction_date (datetime): The target date for which to generate features.
    """
    logging.info(f">>> Starting Prediction Data Generation Pipeline for date: {prediction_date.date()} <<<")
    
    manifest_df = create_prediction_manifest(end_date=prediction_date)
    if manifest_df is None or manifest_df.empty:
        logging.error("Manifest creation failed or is empty. Aborting pipeline.")
        return

    # --- CONDITIONAL DATA DOWNLOAD ---
    # Check the flag from the config file. If a local data source is specified,
    # skip the download steps.
    if not config.IS_LOCAL_DATA_SOURCE:
        logging.info("Downloading required snowpack and weather files...")
        dsu.download_snowpack_files(manifest_df)
        dsu.download_smet_files(manifest_df)
    else:
        logging.info("Skipping data download because a local data source is being used.")
    
    snowpack_locations_df = pd.read_csv(config.PATHS["RAW_DATA"]["snowpack_locations"])
    unique_polygons = manifest_df[['polygon', 'title', 'centroid_lat', 'centroid_lon']].drop_duplicates('polygon').reset_index(drop=True)

    # Define the specific date range for processing
    start_date_str = (prediction_date - timedelta(days=DAYS_OF_DATA_NEEDED)).strftime('%Y-%m-%d')
    end_date_str = prediction_date.strftime('%Y-%m-%d')

    all_polygons_data = []
    for _, polygon_row in tqdm(unique_polygons.iterrows(), desc="Processing Polygons for Prediction", total=len(unique_polygons)):
        # ... (The rest of the data processing logic remains the same)
        polygon_id = int(polygon_row['polygon'])
        
        polygon_dates = manifest_df[manifest_df['polygon'] == polygon_id]['date']
        water_years_array = np.where(polygon_dates.dt.month >= 10, polygon_dates.dt.year, polygon_dates.dt.year - 1)
        water_years_needed = np.unique(water_years_array)
            
        yearly_snowpack_dfs, yearly_weather_dfs = [], []
        for year in water_years_needed:
            snowpack_df = dsu.process_snowpack_data_for_polygon_and_year(
                polygon_info=polygon_row.to_dict(),
                year=year,
                snowpack_locations_df=snowpack_locations_df,
                start_date=start_date_str,
                end_date=end_date_str
            )
            if snowpack_df is not None: yearly_snowpack_dfs.append(snowpack_df)
                
            weather_df = dsu.process_weather_data_for_polygon_and_year(
                polygon_info=polygon_row.to_dict(), year=year, weather_locations_df=snowpack_locations_df
            )
            if weather_df is not None: yearly_weather_dfs.append(weather_df)
        
        if not yearly_snowpack_dfs or not yearly_weather_dfs:
            logging.warning(f"No snowpack or weather data could be processed for polygon '{polygon_id}'. Skipping.")
            continue

        combined_snowpack_df = pd.concat(yearly_snowpack_dfs, ignore_index=True)
        combined_weather_df = pd.concat(yearly_weather_dfs, ignore_index=True)
        
        if not combined_snowpack_df.empty:
            combined_snowpack_df['date'] = pd.to_datetime(combined_snowpack_df['date'])
        if not combined_weather_df.empty:
            combined_weather_df['date'] = pd.to_datetime(combined_weather_df['date'])
        
        merged_df = pd.merge(combined_snowpack_df, combined_weather_df, on='date', how='outer')
        merged_df['polygon'] = polygon_id
        all_polygons_data.append(merged_df)

    if all_polygons_data:
        # Filter out any empty DataFrames from the list before concatenation
        non_empty_dfs = [df for df in all_polygons_data if not df.empty]
        
        if non_empty_dfs:
            final_df = pd.concat(non_empty_dfs, ignore_index=True)
            final_df.sort_values(['polygon', 'date'], inplace=True)
            final_df.fillna(0, inplace=True) # Fill NaNs before feature engineering

            # --- Final Feature Engineering (must match the training script exactly) ---
            logging.info("Engineering final features for the prediction dataset...")

            # Calculate wind_loading_index directly on final_df first
            final_df['wind_loading_index'] = final_df['wind_speed'] * final_df['HNS_24h']

            # Create new features using .assign() or compute them and then concat
            # This avoids DataFrame fragmentation warnings
            engineered_features = pd.DataFrame(index=final_df.index)

            engineered_features['relative_weak_layer_depth'] = final_df['weak_layer_depth'] / (final_df['height-max'] + 1e-6)

            lag_features_to_process = ['weak_layer_stress', 'weak_layer_ssi', 'weak_layer_viscosity', 'weak_layer_rc_flat']
            for col in lag_features_to_process:
                if col in final_df.columns:
                    engineered_features[f'{col}_lag_1'] = final_df.groupby('polygon')[col].shift(1).fillna(0)
                    engineered_features[f'{col}_lag_2'] = final_df.groupby('polygon')[col].shift(2).fillna(0)
            
            # Now, wind_loading_index is guaranteed to be in final_df for rolling calculations
            engineered_features['wind_loading_index_mean_3d'] = final_df.groupby('polygon')['wind_loading_index'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
            engineered_features['wind_loading_index_std_3d'] = final_df.groupby('polygon')['wind_loading_index'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)


            # Concatenate all newly engineered features to the original DataFrame
            final_df = pd.concat([final_df, engineered_features], axis=1)
            final_df.fillna(0, inplace=True) # Fill any NaNs that might have appeared after concat

        else:
            # Handle the case where no data was processed at all
            final_df = pd.DataFrame()        
        
        # Filter to only the specified day for the final prediction file
        logging.info(f"Filtering final dataset to the prediction date: {prediction_date.date()}")
        final_prediction_day_df = final_df[final_df['date'].dt.date == prediction_date.date()].copy()

        if final_prediction_day_df.empty:
            logging.warning(f"No data available for the specified prediction date: {prediction_date.date()}. Cannot generate features.")
            return

        # --- Final Column Alignment ---
        logging.info("Aligning columns to match the features expected by the model.")
        # Load the final feature list saved by the hazard model training script
        try:
            with open(config.PATHS["ARTIFACTS"]["hazard_final_features"], 'r') as f:
                feature_cols = json.load(f)
        except FileNotFoundError:
            logging.error(f"Hazard model final feature list not found at {config.PATHS['ARTIFACTS']['hazard_final_features']}. Please run train_avalanche_hazard_model.py first.")
            return
        
        # Add identifier columns to the final dataframe
        final_cols = ['date', 'polygon'] + feature_cols
        
        for col in final_cols:
            if col not in final_prediction_day_df.columns:
                final_prediction_day_df[col] = 0
        
        # Reorder columns to exactly match the training data schema using .loc
        final_prediction_day_df = final_prediction_day_df.loc[:, final_cols]

        # Save the final dataset to the processed data directory
        output_path = config.PATHS["PROCESSED_DATA"]["inference_features"]
        final_prediction_day_df.to_csv(output_path, index=False)
        logging.info(f"Prediction-ready feature set saved to '{output_path}'")
        logging.info(f"Generated predictions for {len(final_prediction_day_df)} polygons for {prediction_date.date()}.")
    else:
        logging.warning("No prediction data was generated. Check for data availability or processing errors.")


if __name__ == "__main__":
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Generate a prediction feature set for a specific date. By default, it runs for the current day.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--date',
        type=str,
        help="The date to generate predictions for, in YYYY-MM-DD format.\\nIf not provided, the script will use the current date.",
        default="2024-01-14"
    )
    args = parser.parse_args()

    # Default to the current time if no date is provided
    run_date = datetime.now()
    if args.date:
        try:
            # We only care about the date part, so we normalize to the start of the day
            run_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")
            exit(1) # Exit with an error code
    
    build_prediction_dataset(prediction_date=run_date)
