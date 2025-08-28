# -*- coding: utf-8 -*-
"""
================================================================================
Training Data Generation Pipeline
================================================================================

Purpose:
--------
This script orchestrates the complete data generation process for the training
dataset, from fetching raw data to creating a final, merged feature set ready
for machine learning. It imports all low-level processing functions from the
`dataset_utils.py` module, keeping this script focused on the high-level
pipeline orchestration.

It uses the centralized paths defined in `config.py` to read from `data/raw`
and `data/external` and write to `data/processed`.

"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np # Import numpy for calculations

# Import shared functions and project-wide configurations
import dataset_utils as dsu
import config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# 2. MAIN EXECUTION BLOCK
# =============================================================================
def main_pipeline():
    """
    Main function to execute the entire training data generation pipeline.
    """
    logging.info(">>> Starting Training Data Generation Pipeline <<<")

    # --- Step 1: Create Download Manifest ---
    manifest_df = dsu.create_download_manifest()
    if manifest_df is None:
        logging.error("Manifest creation failed. Aborting pipeline.")
        return

    # --- Step 2: Download All Necessary Data to `data/external` ---
    dsu.download_snowpack_files(manifest_df)
    dsu.download_smet_files(manifest_df)
    raw_avalanche_records = dsu.load_or_fetch_avalanche_data(
        config.PATHS["EXTERNAL_DATA"]["raw_avalanche_records"]
    )

    # --- Step 3: Process Raw Data into Daily Summaries ---
    daily_avalanche_summary_df = dsu.process_daily_avalanche_data(
        raw_avalanche_records,
        config.PATHS["RAW_DATA"]["polygons"],
        config.PATHS["PROCESSED_DATA"]["daily_avalanche_data"]
    )
    
    all_hazard_data_df = pd.read_csv(config.PATHS["RAW_DATA"]["danger_ratings"])
    danger_cols = ['atl', 'ntl', 'btl']
    all_hazard_data_df['hazard'] = all_hazard_data_df[danger_cols].max(axis=1)
    all_hazard_data_df['polygon'] = all_hazard_data_df['polygon'].astype(str)
    
    snowpack_locations_df = pd.read_csv(config.PATHS["RAW_DATA"]["snowpack_locations"])

    # --- Step 4: Process Data for Each Polygon and Merge ---
    unique_polygons = manifest_df[['polygon', 'title', 'centroid_lat', 'centroid_lon']].drop_duplicates('polygon').reset_index(drop=True)
    
    if daily_avalanche_summary_df is not None and not daily_avalanche_summary_df.empty:
        polygon_counts = daily_avalanche_summary_df.groupby('polygon')['num_daily_avalanches'].sum()
        threshold = polygon_counts.quantile(0.90)
        high_activity_polygons = polygon_counts[polygon_counts >= threshold].index.tolist()
        unique_polygons = unique_polygons[unique_polygons['polygon'].isin(high_activity_polygons)]
        logging.info(f"Filtered to the top 10% of avalanche activity. Processing {len(unique_polygons)} polygons with at least {round(threshold)} avalanches.")
    
    all_polygons_data = []
    for _, polygon_row in tqdm(unique_polygons.iterrows(), desc="Processing All Polygons", total=len(unique_polygons)):
        polygon_id = str(int(polygon_row['polygon']))
        polygon_name = polygon_row['title']
        
        logging.info("-" * 50)
        logging.info(f"Processing data for polygon: '{polygon_name}' (ID: {polygon_id})")

        yearly_snowpack_dfs, yearly_weather_dfs = [], []
        for year in config.PROCESSING_CONFIG["target_years"]:
            snowpack_year_df = dsu.process_snowpack_data_for_polygon_and_year(
                polygon_info=polygon_row.to_dict(), year=year, snowpack_locations_df=snowpack_locations_df
            )
            if snowpack_year_df is not None: yearly_snowpack_dfs.append(snowpack_year_df)
                
            weather_year_df = dsu.process_weather_data_for_polygon_and_year(
                polygon_info=polygon_row.to_dict(), year=year, weather_locations_df=snowpack_locations_df
            )
            if weather_year_df is not None: yearly_weather_dfs.append(weather_year_df)   
                
        if not yearly_snowpack_dfs or not yearly_weather_dfs:
            logging.warning(f"No snowpack or weather data for polygon '{polygon_name}'. Skipping.")
            continue

        combined_snowpack_df = pd.concat(yearly_snowpack_dfs, ignore_index=True)
        combined_weather_df =  pd.concat(yearly_weather_dfs, ignore_index=True)  
        hazard_df = all_hazard_data_df[all_hazard_data_df['polygon'] == polygon_id].copy()
        if hazard_df.empty:
            logging.warning(f"No hazard data for polygon '{polygon_name}'. Skipping.")
            continue

        master_df = dsu.create_master_dataset_for_polygon(
            polygon_name, daily_avalanche_summary_df, combined_snowpack_df, combined_weather_df, hazard_df
        )
        
        if master_df is not None:
            all_polygons_data.append(master_df)

    # --- Step 5: Finalize and Save Master Datasets ---
    if all_polygons_data:
        final_master_df = pd.concat(all_polygons_data, ignore_index=True)
        logging.info("="*50)
        logging.info("Successfully created master DataFrame for all polygons.")

        # Final Feature Engineering (from build_training_dataset.py)
        logging.info("Engineering final features...")
        
        # Create a copy to avoid fragmentation warnings during feature engineering
        final_master_df_copy = final_master_df.copy()

        # Calculate features that are used in subsequent calculations directly on the copy
        final_master_df_copy['wind_loading_index'] = final_master_df_copy['wind_speed'] * final_master_df_copy['HNS_24h']
        final_master_df_copy['relative_weak_layer_depth'] = final_master_df_copy['weak_layer_depth'] / (final_master_df_copy['height-max'] + 1e-6)

        # Create a temporary DataFrame for other newly engineered features
        engineered_features_temp = pd.DataFrame(index=final_master_df_copy.index)
        
        lag_features = ['weak_layer_stress', 'weak_layer_ssi', 'weak_layer_viscosity', 'weak_layer_rc_flat']
        for col in lag_features:
            if col in final_master_df_copy.columns:
                engineered_features_temp[f'{col}_lag_1'] = final_master_df_copy.groupby('polygon')[col].shift(1).fillna(0)
                engineered_features_temp[f'{col}_lag_2'] = final_master_df_copy.groupby('polygon')[col].shift(2).fillna(0)
        
        # Now, wind_loading_index is guaranteed to be in final_master_df_copy for rolling calculations
        engineered_features_temp['wind_loading_index_mean_3d'] = final_master_df_copy.groupby('polygon')['wind_loading_index'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
        engineered_features_temp['wind_loading_index_std_3d'] = final_master_df_copy.groupby('polygon')['wind_loading_index'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)


        # Concatenate all newly engineered features to the original DataFrame
        final_master_df = pd.concat([final_master_df_copy, engineered_features_temp], axis=1)
        final_master_df.fillna(0, inplace=True) # Fill any NaNs that might have appeared after concat
        
        # --- Add 'avalanche_event' target column ---
        logging.info("Adding 'avalanche_event' target column...")
        final_master_df['date'] = pd.to_datetime(final_master_df['date'])
        final_master_df.sort_values(by=['polygon', 'date'], inplace=True)

        # Initialize the 'avalanche_event' column to 0
        final_master_df['avalanche_event'] = 0

        # Create temporary columns for checking conditions across the 4-day window
        temp_df = final_master_df[['polygon', 'date', 'max_destructive_size', 'num_daily_avalanches']].copy()
        temp_df['max_destructive_size'] = temp_df['max_destructive_size'].fillna(0)
        temp_df['num_daily_avalanches'] = temp_df['num_daily_avalanches'].fillna(0)

        # Iterate through the 4-day window: day before (-1), current day (0), next day (+1), day after next (+2)
        for i in range(-1, 3):
            shifted_max_size = temp_df.groupby('polygon')['max_destructive_size'].shift(-i)
            shifted_num_avalanches = temp_df.groupby('polygon')['num_daily_avalanches'].shift(-i)

            # Condition 1: max_destructive_size >= D2 (which is 2)
            condition_d2 = (shifted_max_size >= 2)

            # Condition 2: num_daily_avalanches > 5
            condition_count_gt_5 = (shifted_num_avalanches > 5)

            # Update 'avalanche_event' if either condition is met in any part of the window
            final_master_df['avalanche_event'] = final_master_df['avalanche_event'] | (condition_d2 | condition_count_gt_5).fillna(False).astype(int)

        logging.info(f"Calculated 'avalanche_event'. Total events: {final_master_df['avalanche_event'].sum()}")

        # --- Separate features and targets ---
        # Define columns that are direct outcomes or would cause data leakage.
        # These will be separated into the target file.
        target_markers = [
            'daily_AAI', 'max_destructive_size', 'num_daily_avalanches',
            'mean_destructive_size', 'hazard', 'avalanche_event'
        ]
        
        target_related_cols = [
            col for col in final_master_df.columns 
            if any(marker in col for marker in target_markers)
        ]        
        
        # target_related_cols = [col for col in target_related_cols if col in final_master_df.columns]

        # All other columns are considered features for the initial training run.
        # The training script will then perform the final pruning.
        # Explicitly exclude 'date' and 'polygon' from features as requested.
        feature_cols_to_keep = [
            col for col in final_master_df.columns
            if col not in target_related_cols
        ]
        
        # Create the two final dataframes
        # Feature_df should NOT have 'date' or 'polygon'
        feature_df = final_master_df[feature_cols_to_keep]
        target_df = final_master_df[['date', 'polygon'] + [col for col in target_related_cols if col in final_master_df.columns]]

        # Save to the `processed` data directory
        feature_output_path = config.PATHS["PROCESSED_DATA"]["training_features"]
        target_output_path = config.PATHS["PROCESSED_DATA"]["training_targets"]
        
        feature_df.to_csv(feature_output_path, index=False)
        logging.info(f"Master feature dataset with {len(feature_cols_to_keep)} features saved to '{feature_output_path}'")
        
        target_df.to_csv(target_output_path, index=False)
        logging.info(f"Master target dataset saved to '{target_output_path}'")
    else:
        logging.warning("No data was processed. Please check file paths and previous error messages.")

if __name__ == "__main__":
    main_pipeline()
