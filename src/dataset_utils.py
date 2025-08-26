# -*- coding: utf-8 -*-
"""
================================================================================
Dataset Utilities for Avalanche Hazard Forecasting
================================================================================

Purpose:
--------
This script contains a collection of shared, reusable functions for the
avalanche hazard forecasting pipeline. It handles all low-level data fetching,
parsing, and processing tasks, allowing the main training and prediction
scripts to be clean, high-level orchestrators.

By centralizing this logic, we ensure that both the training and prediction
pipelines use the exact same data processing steps, which is critical for
model consistency and performance.

Functions:
----------
- Geographic and Utility Functions: Helpers for common calculations like
  Haversine distance and circular averaging for wind directions.
- Data Fetching: Functions to download snowpack and weather data, and to
  fetch avalanche observation records from the CAIC API.
- Data Processing: Core functions to transform raw data into model-ready
  features, such as processing daily avalanche activity and extracting
  complex features from snowpack profiles.
- Manifest Creation: Logic to determine which files need to be downloaded
  based on the project's configuration.

"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

# Local import for the .pro file reader
from snowpack_reader import SnowpackProfile, read_snowpack
# Import project-wide configurations
import config

# --- Utility Functions ---

def _safe_float(value: Any, default: float = np.nan) -> float:
    """
    Safely convert a value to a float, returning a default value on failure.

    This helper function is used to prevent errors when processing data that
    may contain non-numeric or missing values, ensuring the pipeline continues
    to run smoothly.

    Args:
        value (Any): The value to convert to a float.
        default (float, optional): The value to return if the conversion fails.
                                   Defaults to np.nan.

    Returns:
        float: The converted float value or the specified default.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def haversine_distance(lat1: float, lon1: float, df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    """
    Calculate the great-circle distance between a single point and multiple
    other points in a DataFrame using the Haversine formula.

    This is essential for finding the closest weather or snowpack station to a
    given forecast polygon centroid.

    Args:
        lat1 (float): Latitude of the origin point (in decimal degrees).
        lon1 (float): Longitude of the origin point (in decimal degrees).
        df (pd.DataFrame): DataFrame containing the destination points.
        lat_col (str): The name of the latitude column in the DataFrame.
        lon_col (str): The name of the longitude column in the DataFrame.

    Returns:
        pd.Series: A pandas Series containing the calculated distances in kilometers,
                   with the same index as the input DataFrame.
    """
    radius_km = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(df[lat_col]), np.radians(df[lon_col])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c

def average_directions(degrees: list[float]) -> float:
    """
    Computes the circular average of a list of angles in degrees.

    This function is critical for correctly averaging wind directions, where a
    simple arithmetic mean would be incorrect (e.g., the average of 350° and
    10° should be 0°, not 180°). It converts the angles to vectors, averages
    the vectors, and converts the resultant vector back to an angle.

    Args:
        degrees (list[float]): A list or pandas Series of angles in degrees (0-360).

    Returns:
        float: The average angle in degrees, wrapped to the range [0, 360).
    """
    radians = np.radians(degrees)
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))
    avg_angle_rad = np.arctan2(sin_sum, cos_sum)
    return np.degrees(avg_angle_rad) % 360

# --- Data Preparation & Manifest Creation ---

def get_polygon_centroids(file_path: Path, projected_crs: str) -> Optional[gpd.GeoDataFrame]:
    """
    Reads a GeoJSON file of forecast polygons, calculates their geometric
    centroids, and returns a GeoDataFrame with key location information.

    Using a projected CRS is important for accurate area and centroid
    calculations, as geographic coordinates (lat/lon) can be distorted.

    Args:
        file_path (Path): The path to the input GeoJSON file.
        projected_crs (str): The EPSG code for a projected coordinate reference
                             system (e.g., "EPSG:3310") suitable for accurate
                             centroid calculation in the region of interest.

    Returns:
        Optional[gpd.GeoDataFrame]: A GeoDataFrame containing the polygon ID,
                                    title, and centroid coordinates (lat/lon).
                                    Returns None if a critical error occurs.
    """
    logging.info(f"Reading and processing GeoJSON from: {file_path}")
    try:
        def _validate_geojson_columns(gdf):
            # Ensure GeoJSON contains required columns
            return 'geometry' in gdf.columns and 'id' in gdf.columns and 'title' in gdf.columns

        gdf = gpd.read_file(file_path)
        if not _validate_geojson_columns(gdf):
            logging.error("GeoJSON must contain 'id', 'title', and 'geometry' properties.")
            return None
        
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        if gdf.empty:
            logging.warning("No valid Polygon or MultiPolygon geometries found.")
            return gdf

        # Project to a suitable CRS for accurate centroid calculation
        projected_centroids = gdf.geometry.to_crs(projected_crs).centroid
        # Convert the centroid back to the original CRS (e.g., WGS84) for lat/lon
        gdf['centroid'] = projected_centroids.to_crs(gdf.crs)
        gdf['centroid_lat'] = gdf['centroid'].y
        gdf['centroid_lon'] = gdf['centroid'].x
        
        logging.info(f"Successfully calculated centroids for {len(gdf)} polygons.")
        return gdf[['id', 'title', 'centroid_lat', 'centroid_lon']]
    except Exception as e:
        logging.error(f"An error occurred while processing the GeoJSON file: {e}", exc_info=True)
        return None

def find_closest_stations(points_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the closest snowpack station for each point in a DataFrame.

    This function uses a vectorized Haversine distance calculation for efficiency,
    computing a distance matrix between all points and all stations to find the
    minimum distance for each point in a single operation.

    Args:
        points_df (pd.DataFrame): DataFrame of points (e.g., polygon centroids)
                                  with 'centroid_lat' and 'centroid_lon' columns.
        stations_df (pd.DataFrame): DataFrame of station locations with 'lat',
                                    'lon', and 'id' columns.

    Returns:
        pd.DataFrame: A DataFrame with the 'st_id' and 'zone' of the closest
                      station for each input point, preserving the original index.
    """
    logging.info(f"Finding closest stations for {len(points_df)} unique points.")
    points_rad = np.radians(points_df[['centroid_lat', 'centroid_lon']].values)
    stations_rad = np.radians(stations_df[['lat', 'lon']].values)

    lat1, lon1 = points_rad[:, 0][:, np.newaxis], points_rad[:, 1][:, np.newaxis]
    lat2, lon2 = stations_rad[:, 0], stations_rad[:, 1]
    
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_matrix_km = 6371.0 * c

    closest_indices = np.argmin(distance_matrix_km, axis=1)

    return pd.DataFrame({
        'st_id': stations_df.loc[closest_indices, 'id'].values,
        'zone': stations_df.loc[closest_indices, 'zone'].values
    }, index=points_df.index)

def create_download_manifest() -> Optional[pd.DataFrame]:
    """
    Creates a manifest of all snowpack and weather files that need to be
    downloaded for the training period.

    This function is a key part of the data preparation pipeline. It merges
    polygon, danger rating, and station location data to determine which
    snowpack files are relevant for each polygon and date, then constructs
    the appropriate download URLs for later use.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the download manifest,
                                including dates, polygon info, station info,
                                and download URLs. Returns None if a critical
                                error occurs.
    """
    manifest_path = config.PATHS["PROCESSED_DATA"]["download_manifest"]
    if manifest_path.exists():
        logging.info(f"Loading existing download manifest from '{manifest_path}'")
        try:
            return pd.read_csv(manifest_path)
        except Exception as e:
            logging.warning(f"Could not read existing manifest file: {e}. Recreating it.")

    logging.info("--- Creating Training Download Manifest ---")

    try:
        polygons_path = config.PATHS["RAW_DATA"]["polygons"]
        danger_path = config.PATHS["RAW_DATA"]["danger_ratings"]
        locations_path = config.PATHS["RAW_DATA"]["snowpack_locations"]

        centroids_df = get_polygon_centroids(polygons_path, config.PROCESSING_CONFIG["geoprojection_epsg"])
        danger_df = pd.read_csv(danger_path)
        snowpack_locations = pd.read_csv(locations_path)
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Could not load required raw data file for manifest creation: {e}")
        return None

    if any(df is None for df in [centroids_df, danger_df, snowpack_locations]):
        logging.critical("Failed to load one or more necessary files for manifest creation.")
        return None

    danger_df['polygon'] = pd.to_numeric(danger_df['polygon'], errors='coerce')
    centroids_df['title'] = pd.to_numeric(centroids_df['title'], errors='coerce')

    danger_df = danger_df.dropna(subset=['polygon'])
    centroids_df.dropna(subset=['title'], inplace=True)

    danger_df['polygon'] = danger_df['polygon'].astype(int)
    centroids_df['title'] = centroids_df['title'].astype(int)

    merged_df = danger_df.merge(centroids_df, left_on='polygon', right_on='title', how='left')

    unique_polygons = merged_df.dropna(
        subset=['centroid_lat', 'centroid_lon']
    ).drop_duplicates(subset=['polygon']).set_index('polygon')

    station_map_df = find_closest_stations(unique_polygons, snowpack_locations)

    final_df = merged_df.merge(station_map_df, on='polygon', how='left')
    final_df = final_df.dropna(subset=['centroid_lat', 'st_id'])

    dates = pd.to_datetime(final_df['date'])
    final_df['water_year'] = np.where(dates.dt.month >= 10, dates.dt.year, dates.dt.year - 1)

    final_df = final_df[final_df['water_year'].isin(config.PROCESSING_CONFIG["target_years"])].copy()

    st_id_str = final_df['st_id'].astype(int).astype(str).str.zfill(6)
    zone_str = final_df['zone'].astype(int).astype(str).str.zfill(3)
    water_year_str = final_df['water_year'].astype(str)

    # This URL is the relative path used for constructing local file paths
    final_df['URL'] = water_year_str + "/zone" + zone_str + "/" + st_id_str
    final_df['download_URL'] = "https://nwp.mtnweather.info/ron/ssd/snowpack/output/" + water_year_str + "/zone" + zone_str + "/" + st_id_str + "/" + st_id_str + "_res.pro"

    manifest_path = config.PATHS["PROCESSED_DATA"]["download_manifest"]
    final_df.to_csv(manifest_path, index=False)
    logging.info(f"Download manifest created and saved to '{manifest_path}'.")

    return final_df

# --- Data Fetching and Processing ---

def download_snowpack_files(info_df: pd.DataFrame):
    """
    Reads a manifest DataFrame and downloads any missing snowpack .pro files
    to the `data/external/snowpack` directory.

    Args:
        info_df (pd.DataFrame): The download manifest DataFrame.
    """
    # 1. Validate the input manifest DataFrame.
    required_cols = ['download_URL', 'water_year', 'zone', 'st_id']
    if any(col not in info_df.columns for col in required_cols):
        logging.error(f"Manifest is missing required columns for downloading. Needs: {required_cols}")
        return

    # 2. Get a list of unique files to check/download.
    download_tasks = info_df[required_cols].drop_duplicates().dropna()
    logging.info(f"Found {len(download_tasks)} unique snowpack files to check.")

    for _, row in tqdm(download_tasks.iterrows(), total=len(download_tasks), desc="Checking/Downloading Snowpack Files"):
        download_url = row['download_URL']
        
        # --- ROBUST PATH RECONSTRUCTION ---
        # Build the destination path from scratch to ensure it's always correct
        # and consistent with the data processing functions.
        year_str = str(int(row['water_year']))
        zone_str = f"zone{int(row['zone']):03d}"
        st_id_str = f"{int(row['st_id']):06d}"
        
        full_dest_path = config.PATHS["EXTERNAL_DATA"]["snowpack_output"] / year_str / zone_str / st_id_str / f"{st_id_str}_res.pro"
        
        # 3. Create the parent directory if it doesn't exist.
        full_dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 4. Skip if the file already exists.
        if full_dest_path.exists():
            continue

        # 5. Download the file with error handling.
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(full_dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            logging.error(f"Failed to download from {download_url}: {e}")            

def download_smet_files(info_df: pd.DataFrame):
    """
    Reads a manifest DataFrame and downloads any missing weather .smet files
    to the `data/external/snowpack` directory.

    Args:
        info_df (pd.DataFrame): The download manifest DataFrame.
    """
    # 1. Validate the input manifest DataFrame.
    required_cols = ['download_URL', 'water_year', 'zone', 'st_id']
    if any(col not in info_df.columns for col in required_cols):
        logging.error(f"Manifest is missing required columns for downloading. Needs: {required_cols}")
        return

    # 2. Get a list of unique files to check/download.
    download_tasks = info_df[required_cols].drop_duplicates().dropna()
    logging.info(f"Found {len(download_tasks)} unique weather files to check.")

    for _, row in tqdm(download_tasks.iterrows(), total=len(download_tasks), desc="Checking/Downloading Weather Files"):
        # The download URL for smet files needs to be derived from the pro file URL
        download_url = row['download_URL'].replace('_res.pro', '_res.smet')
        
        # Build the destination path from scratch to ensure it's always correct.
        year_str = str(int(row['water_year']))
        zone_str = f"zone{int(row['zone']):03d}"
        st_id_str = f"{int(row['st_id']):06d}"
        
        full_dest_path = config.PATHS["EXTERNAL_DATA"]["snowpack_output"] / year_str / zone_str / st_id_str / f"{st_id_str}_res.smet"

        # 3. Create the parent directory if it doesn't exist.
        full_dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if full_dest_path.exists():
            continue

        # 4. Download the file with error handling.
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(full_dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            logging.error(f"Failed to download from {download_url}: {e}")
            
            
def load_or_fetch_avalanche_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads raw avalanche observation records from a local JSON file. If the file
    does not exist, it fetches the data from the CAIC API and saves it.

    Args:
        file_path (Path): The path to the local JSON file where records are
                          cached (e.g., `data/external/avalanche_records.json`).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a raw avalanche observation record.
    """
    if file_path.exists():
        logging.info(f"Loading raw avalanche data from '{file_path}'.")
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    all_records = []
    session = requests.Session()
    logging.info(f"Fetching avalanche data from {config.API_CONFIG['base_url']}...")
    for page in tqdm(range(1, config.API_CONFIG['max_pages'] + 1), desc="Fetching API Pages"):
        for attempt in range(config.API_CONFIG['max_retries']):
            try:
                response = session.get(
                    config.API_CONFIG['base_url'], 
                    params={"page": page, "page_size": config.API_CONFIG['page_size']}, 
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                if not data: break
                all_records.extend(data)
                time.sleep(0.1)
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                logging.warning(f"Attempt {attempt + 1} failed for page {page}: {e}. Retrying...")
                time.sleep(config.API_CONFIG['retry_delay_sec'])
        else:
            logging.error(f"Failed to fetch page {page} after {config.API_CONFIG['max_retries']} attempts. Stopping.")
            break
        if not data: break

    with file_path.open('w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)
    logging.info(f"Successfully downloaded and saved {len(all_records)} records to '{file_path}'.")
    return all_records

def process_daily_avalanche_data(raw_records: List[Dict[str, Any]], 
                                 polygons_geojson_path: Path, 
                                 output_path: Path
) -> Optional[pd.DataFrame]:
    """
    Processes raw avalanche records into a daily summary DataFrame.

    This function assigns each avalanche to a forecast polygon using a spatial
    join, then aggregates the data by day to calculate activity metrics like
    the Avalanche Activity Index (AAI), max destructive size, and number of
    avalanches. It also creates lagged features for modeling.

    Args:
        raw_records (List[Dict[str, Any]]): The list of raw avalanche records
                                            from `load_or_fetch_avalanche_data`.
        polygons_geojson_path (Path): Path to the forecast polygons GeoJSON file.
        output_path (Path): Path to save the processed daily data CSV.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the daily aggregated
                                avalanche data per polygon. Returns None on failure.
    """
    if not polygons_geojson_path.exists():
        logging.error(f"BC Polygons GeoJSON not found at: {polygons_geojson_path}")
        return None

    logging.info("Assigning avalanches to polygons using spatial join...")
    
    polygons_gdf = gpd.read_file(polygons_geojson_path)
    if 'title' in polygons_gdf.columns:
        polygons_gdf.rename(columns={'title': 'polygon'}, inplace=True)
    else:
        logging.error("GeoJSON is missing the 'title' property in its features.")
        return None
    
    processed_records = [
        {
            'observed_at': record.get('observed_at'),
            'destructive_size': _safe_float(str(record.get('destructive_size', '')).replace('D', '')),
            'latitude': _safe_float(record.get('latitude')),
            'longitude': _safe_float(record.get('longitude'))
        }
        for record in raw_records
        if record.get('destructive_size') and record.get('latitude') and record.get('longitude')
    ]

    if not processed_records:
        logging.error("No valid avalanche records with coordinates and size found.")
        return None

    df = pd.DataFrame(processed_records).dropna()
    avalanches_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    merged_gdf = gpd.sjoin(avalanches_gdf, polygons_gdf[['polygon', 'geometry']], how="inner", predicate="within")
    logging.info(f"Successfully assigned {len(merged_gdf)} avalanches to polygons.")
    df_assigned = pd.DataFrame(merged_gdf.drop(columns=['geometry', 'index_right']))

    df_assigned['observed_at'] = pd.to_datetime(df_assigned['observed_at'], errors='coerce').dt.date
    df_assigned['avalanche_power'] = 10 ** (df_assigned['destructive_size'] - 1)

    daily_aai_df = df_assigned.groupby(['polygon', 'observed_at']).agg(
        daily_AAI=('avalanche_power', 'sum'),
        max_destructive_size=('destructive_size', 'max'),
        mean_destructive_size=('destructive_size', 'mean'),
        num_daily_avalanches=('observed_at', 'count')
    ).reset_index()
    
    daily_aai_df.rename(columns={'observed_at': 'date'}, inplace=True)
    daily_aai_df['polygon'] = pd.to_numeric(daily_aai_df['polygon'], errors='coerce')
    
    daily_aai_df.sort_values(['polygon', 'date'], inplace=True)

    avalanche_cols = ['daily_AAI', 'max_destructive_size', 'num_daily_avalanches']
    
    for col in avalanche_cols:
        daily_aai_df[f'{col}_d-1'] = daily_aai_df.groupby('polygon')[col].shift(1)

    lagged_rolling_features = [f'{col}_d-1' for col in avalanche_cols]
    for feat in lagged_rolling_features:
        daily_aai_df[f'{feat}_mean_3d'] = daily_aai_df.groupby('polygon')[feat].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        daily_aai_df[f'{feat}_std_3d'] = daily_aai_df.groupby('polygon')[feat].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)

    daily_aai_df.to_csv(output_path, index=False)
    logging.info(f"Daily avalanche data processed and saved to '{output_path}'.")
    
    return daily_aai_df

def process_snowpack_data_for_polygon_and_year(polygon_info: dict, 
                                               year: int, 
                                               snowpack_locations_df: pd.DataFrame,
                                               start_date: Optional[str] = None,
                                               end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Processes snowpack data for a given polygon and year to extract features.

    This function reads a snowpack profile for a given water year, slices it to
    a specified date range (or the full season if no range is given), and then
    identifies weak layers, analyzes the slab, and computes temporal features.

    Args:
        polygon_info (dict): A dictionary containing the polygon's metadata.
        year (int): The water year to process (used to locate the file).
        snowpack_locations_df (pd.DataFrame): DataFrame of snowpack station locations.
        start_date (Optional[str]): The start date for analysis (YYYY-MM-DD).
                                    Defaults to the beginning of the water year.
        end_date (Optional[str]): The end date for analysis (YYYY-MM-DD).
                                  Defaults to the end of the water year season.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing daily time-series features
                                for the snowpack, or None if data cannot be processed.
    """
    polygon_name = polygon_info['title']
    dists = haversine_distance(polygon_info['centroid_lat'], polygon_info['centroid_lon'], snowpack_locations_df, 'lat', 'lon')
    closest_station = snowpack_locations_df.loc[dists.idxmin()]
    zone_id = f"zone{int(closest_station['zone']):03d}"
    profile_id = f"{int(closest_station['id']):06d}"
    pro_path = config.PATHS["EXTERNAL_DATA"]["snowpack_output"] / str(year) / zone_id / profile_id / f"{profile_id}_res.pro"

    if not pro_path.exists():
        logging.warning(f"Snowpack .pro file not found for polygon '{polygon_name}' at '{pro_path}'.")
        return None

    try:
        reader = read_snowpack(str(pro_path))
        if reader is None or len(reader) == 0:
            logging.warning(f"No profiles found or read for file: {pro_path}")
            return None
    except Exception as e:
        logging.error(f"Failed to initialize SnowpackProfile reader for {pro_path}: {e}", exc_info=True)
        return None

    if start_date is None: start_date = f"{year}-10-01"
    if end_date is None: end_date = f"{year+1}-04-30"
    
    sliced_profile = reader.slice(start_date=start_date, end_date=end_date)
    if len(sliced_profile) == 0: return None

    weak_layers_df = sliced_profile.find_layer_by_criteria(
        criteria=config.SNOWPACK_CONFIG["weak_layer_criteria"], search_from='bottom'
    )
    if weak_layers_df.empty: return None
    
    # Extract gs_difference and hardness_difference from matching_parameters
    # The 'matching_parameters' column contains a dictionary of the criteria matched.
    # We need to safely extract these specific keys.
    weak_layers_df['weak_layer_gs_difference'] = weak_layers_df['matching_parameters'].apply(
        lambda x: x.get('gs_difference') if isinstance(x, dict) else np.nan
    )
    weak_layers_df['weak_layer_hardness_difference'] = weak_layers_df['matching_parameters'].apply(
        lambda x: x.get('hardness_difference') if isinstance(x, dict) else np.nan
    )

    weak_layers_df = weak_layers_df.drop(columns=['matching_parameters']).rename(columns=lambda c: f'weak_layer_{c}')
    
    slab_results = []
    for date, wl_row in weak_layers_df.iterrows():
        wl_height = wl_row['weak_layer_height']
        if pd.isna(wl_height): continue
        
        slab_summary = sliced_profile.get_profile_summary(
            parameters_to_calculate=config.SNOWPACK_CONFIG["slab_parameters"],
            start_date=date.strftime('%Y-%m-%d'), end_date=date.strftime('%Y-%m-%d'),
            from_height=wl_height, above_or_below='above'
        )
        if not slab_summary.empty: slab_results.append(slab_summary)

    # Filter out any potentially empty DataFrames before concatenation to avoid FutureWarning
    slab_results = [df for df in slab_results if not df.empty]
    # Check if slab_results is empty *after* filtering
    if not slab_results:
        return weak_layers_df.reset_index()

    slab_df = pd.concat(slab_results)
    combined_df = weak_layers_df.join(slab_df, how='left').reset_index()
    
    general_summary_df = sliced_profile.get_profile_summary(
        parameters_to_calculate={'height-max': 'max', 'temperature-mean': 'mean'}
    )
    
    upper_snowpack_params = config.SNOWPACK_CONFIG.get("upper_snowpack_parameters", {})
    
    upper_snowpack_results = []
    for date, row in general_summary_df.iterrows():
        total_height = row['height-max']
        if pd.notna(total_height) and total_height >= 15:
            from_height = total_height - 15
            summary = sliced_profile.get_profile_summary(
                parameters_to_calculate=upper_snowpack_params,
                start_date=date.strftime('%Y-%m-%d'), end_date=date.strftime('%Y-%m-%d'),
                from_height=from_height, above_or_below='above'
            )
            if not summary.empty: upper_snowpack_results.append(summary)

    # Filter out any potentially empty DataFrames before concatenation to avoid FutureWarning
    non_empty_upper_snowpack_results = [df for df in upper_snowpack_results if not df.empty]
    # Check if non_empty_upper_snowpack_results is empty *after* filtering
    upper_snowpack_df = pd.concat(non_empty_upper_snowpack_results) if non_empty_upper_snowpack_results else pd.DataFrame()
    
    general_summary_df.reset_index(inplace=True)
    final_df = pd.merge(combined_df, general_summary_df, on='date', how='left')
    
    if not upper_snowpack_df.empty:
        upper_snowpack_df.reset_index(inplace=True)
        final_df = pd.merge(final_df, upper_snowpack_df, on='date', how='left')
    
    if 'height-max' in final_df.columns and 'weak_layer_height' in final_df.columns:
        final_df['slab_thickness'] = final_df['height-max'] - final_df['weak_layer_height']
        final_df['weak_layer_depth'] = final_df['height-max'] - final_df['weak_layer_height']
    
    if 'height-max' in final_df.columns:
        final_df.sort_values('date', inplace=True)
        final_df['24_hrs_HS_delta'] = final_df['height-max'].diff().fillna(0)
        new_snow_24h = final_df['24_hrs_HS_delta'].clip(lower=0)
        final_df['HNS_24h'] = new_snow_24h
        final_df['HNS_5d'] = new_snow_24h.rolling(window=5, min_periods=1).sum()
        
    rolling_features = [
        'weak_layer_matching_criteria_count', 'weak_layer_depth', 'weak_layer_rc_flat',
        'weak_layer_density', 'weak_layer_hand_hardness', 'weak_layer_grain_size',
        'weak_layer_sphericity', 'weak_layer_stress', 'weak_layer_viscosity',
        'weak_layer_sn38', 'weak_layer_ssi', 'weak_layer_sk38',
        'weak_layer_gs_difference', 'weak_layer_hardness_difference',
        'slab_log_hardness_mean', 'slab_density_mean', 'slab_density_weighted_mean',
        'slab_hardness_weighted_mean', 'slab_load', 
        'height-max', 'temperature-mean',
        'upper_snowpack_density_mean', 'upper_snowpack_sphericity_mean',
        'upper_snowpack_grain_size_mean', 'upper_snowpack_hand_hardness_mean',
    ]
    
    # Ensure all columns for rolling calculations are numeric.
    for feat in rolling_features:
        if feat in final_df.columns:
            final_df[feat] = pd.to_numeric(final_df[feat], errors='coerce')

    new_cols = {}
    for feat in rolling_features:
        if feat in final_df.columns:
            for window in [2, 3, 4, 5, 6, 9]: # Added 7-day window
                new_cols[f'{feat}_mean_{window}d'] = final_df[feat].rolling(window=window, min_periods=1).mean()
                new_cols[f'{feat}_std_{window}d'] = final_df[feat].rolling(window=window, min_periods=1).std()
                new_cols[f'{feat}_trend_{window}d'] = final_df[feat].diff(periods=window).fillna(0) / window
    
    if new_cols:
        final_df = pd.concat([final_df, pd.DataFrame(new_cols)], axis=1)
    
    lag_features = [
        'weak_layer_matching_criteria_count', 'weak_layer_depth', 'weak_layer_rc_flat',
        'weak_layer_density', 'weak_layer_hand_hardness', 'weak_layer_grain_size',
        'weak_layer_sphericity', 'weak_layer_stress', 'weak_layer_viscosity',
        'weak_layer_sn38', 'weak_layer_ssi', 'weak_layer_sk38',
        'weak_layer_gs_difference', 'weak_layer_hardness_difference'
        ]

    for col in lag_features:
        if col in final_df.columns:
            final_df[f'{col}_lag_1'] = final_df[col].shift(1).fillna(0)
            final_df[f'{col}_lag_2'] = final_df[col].shift(2).fillna(0)
            final_df[f'{col}_lag_3'] = final_df[col].shift(3).fillna(0)
            
    return final_df

def process_weather_data_for_polygon_and_year(polygon_info: dict, year: int, weather_locations_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Processes weather data for a given polygon and year from a .smet file.

    This function reads a .smet file, extracts key weather variables, aggregates
    them to a daily resolution (e.g., mean wind speed, average wind direction),
    and calculates rolling statistical features.

    Args:
        polygon_info (dict): A dictionary containing the polygon's metadata.
        year (int): The water year to process.
        weather_locations_df (pd.DataFrame): DataFrame of weather station locations.

    Returns:
        Optional[pd.DataFrame]: A DataFrame of daily weather features, or None
                                if the data cannot be processed.
    """
    def read_smet_file(smet_path: Path) -> pd.DataFrame:
        """Nested helper to parse a .smet file."""
        data, data_section = [], False
        with open(smet_path) as f:
            for line in f:
                if line.startswith('fields'):
                    columns = line.split('=')[1].strip().split(' ')
                elif '[DATA]' in line:
                    data_section = True
                elif data_section:
                    data.append(np.array(line.strip().split()))
        
        df = pd.DataFrame(data, columns=columns).rename(columns={'timestamp': 'date', 'VW': 'wind_speed', 'DW': 'wind_dir', 'VW_drift': 'snow_drift'})  
        return df[['date', 'wind_speed', 'wind_dir', 'snow_drift']]       
    
    polygon_name = polygon_info['title']
    dists = haversine_distance(polygon_info['centroid_lat'], polygon_info['centroid_lon'], weather_locations_df, 'lat', 'lon')
    closest_station = weather_locations_df.loc[dists.idxmin()]
    zone_id = f"zone{int(closest_station['zone']):03d}"
    profile_id = f"{int(closest_station['id']):06d}"
    smet_path = config.PATHS["EXTERNAL_DATA"]["snowpack_output"] / str(year) / zone_id / profile_id / f"{profile_id}_res.smet"

    if not smet_path.exists(): return None

    try:
        weather = read_smet_file(smet_path)
    except Exception as e:
        logging.error(f"Failed to process file {smet_path}: {e}", exc_info=True)
        return None

    if weather.empty: return None
    weather['date'] = pd.to_datetime(weather['date'])
    for col in ['wind_speed', 'wind_dir', 'snow_drift']:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')
            
    weather = weather.groupby(weather['date'].dt.date).agg({
        'wind_speed': 'mean', 'wind_dir': average_directions, 'snow_drift': 'mean'
    }).reset_index()
    
    rolling_features = ['wind_speed', 'wind_dir']
    new_cols = {}
    for feat in rolling_features:
        if feat in weather.columns:
            for window in [2, 3, 4, 5, 6, 9]: # Added 7-day window
                new_cols[f'{feat}_mean_{window}d'] = weather[feat].rolling(window=window, min_periods=1).mean()
                new_cols[f'{feat}_std_{window}d'] = weather[feat].rolling(window=window, min_periods=1).std()
                new_cols[f'{feat}_trend_{window}d'] = weather[feat].diff(periods=window).fillna(0) / window
    
    if new_cols:
        weather = pd.concat([weather, pd.DataFrame(new_cols)], axis=1)
           
    return weather

def create_master_dataset_for_polygon(polygon_name: str, daily_avalanche_df: pd.DataFrame, snowpack_df: pd.DataFrame, weather_df: pd.DataFrame, hazard_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merges all processed data sources for a single polygon into one master
    feature set for a given time period.

    Args:
        polygon_name (str): The name/ID of the polygon being processed.
        daily_avalanche_df (pd.DataFrame): Processed daily avalanche activity.
        snowpack_df (pd.DataFrame): Processed daily snowpack features.
        weather_df (pd.DataFrame): Processed daily weather features.
        hazard_df (pd.DataFrame): DataFrame containing daily hazard ratings.

    Returns:
        Optional[pd.DataFrame]: A single, merged DataFrame containing all
                                features for the polygon, or None on failure.
    """
    for df in [daily_avalanche_df, snowpack_df, weather_df, hazard_df]:
        if 'date' not in df.columns:
             logging.error(f"A DataFrame for polygon '{polygon_name}' is missing a 'date' column.")
             return None
        df['date'] = pd.to_datetime(df['date'])

    polygon_avalanche_data = daily_avalanche_df[daily_avalanche_df['polygon'] == polygon_name].copy()
    
    base_df = hazard_df[['date', 'hazard']].copy()
    base_df = base_df.sort_values('date').reset_index(drop=True)
    
    merged_df = pd.merge(base_df, polygon_avalanche_data, on='date', how='left')
    
    final_df = pd.merge(merged_df, snowpack_df, on='date', how='left')
    final_df = pd.merge(final_df, weather_df, on='date', how='left')

    final_df['polygon'] = final_df['polygon'].fillna(polygon_name)
    final_df = final_df.sort_values(by='date').reset_index(drop=True)
    
    # Add hazard-1 and hazard-2 as they are present in build_training_dataset.py
    final_df['hazard-1'] = final_df['hazard'].shift(1)
    final_df['hazard-2'] = final_df['hazard'].shift(2)

    final_df.fillna(0, inplace=True) 
    
    return final_df
