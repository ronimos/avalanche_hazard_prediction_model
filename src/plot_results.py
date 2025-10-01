# -*- coding: utf-8 -*-
"""
plot_results.py
===================

This script provides functions to visualize the output of the avalanche hazard
forecasting model on an interactive map using Folium.

It creates a multi-layered map that allows for the comparison of different
prediction outputs against various basemaps.
"""

import logging
from datetime import datetime
from typing import Union

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import numpy as np

# Import project-wide configurations for consistent path management
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_prediction_map(prediction_date: Union[str, datetime]):
    """
    Reads prediction results and creates an interactive Folium map with multiple layers.

    The map includes:
    - Basemaps: Street, Topo, and Hybrid Satellite imagery.
    - A choropleth layer for the event likelihood score.
    - A standard choropleth layer for the predicted hazard rating.
    - A new choropleth layer where hazard color opacity is tied to model confidence.
    - A layer showing only the polygon outlines.

    Args:
        prediction_date (Union[str, datetime]): The specific date for which to plot predictions,
                                                accepting either a 'YYYY-MM-DD' string
                                                or a datetime object.
    """
    def get_confidence(row):
        # Get the predicted hazard level for this row (e.g., 3)
        predicted_level = row['predicted_hazard']
        # Construct the name of the corresponding probability column (e.g., 'calibrated_proba_hazard_3')
        prob_col_name = f'calibrated_proba_hazard_{predicted_level}'
        
        # Check if that column exists and return its value
        return row[prob_col_name] if prob_col_name in row else 0.0 # Return a default value if the column isn't found

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

    logging.info(f"Generating prediction map for date: {prediction_date.date()}...")

    try:
        # Use centralized paths from the config file
        output_dir = config.RESULTS_DIR
        
        # 1. Load the polygon geometries
        polygons_gdf = gpd.read_file(config.PATHS["RAW_DATA"]["polygons"])
        # Ensure the polygon ID column is an integer for merging
        polygons_gdf['id'] = polygons_gdf['title'].astype(int)

        # 2. Load the LATEST prediction results from the correct CSV file
        predictions_df = pd.read_csv(config.PATHS["ARTIFACTS"]["hazard_predictions_csv"])
        
        # Convert the 'date' column from strings back to datetime objects
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        # Ensure the polygon ID column is an integer for merging
        predictions_df['polygon'] = predictions_df['polygon'].astype(int)

    except FileNotFoundError as e:
        logging.error(f"Could not read a required data file: {e}. Please ensure paths are correct and predictions exist.")
        return

    # 3. Filter the predictions to get only the data for the requested date
    date_str = prediction_date.strftime('%Y-%m-%d')
    day_specific_preds = predictions_df[predictions_df['date'] == date_str].copy()

    if day_specific_preds.empty:
        logging.warning(f"No predictions found for {prediction_date.date()}. Cannot create map.")
        return

    # 4. Merge the polygon geometries with the daily prediction data
    # This step combines the data into the final GeoDataFrame for the map.
    merged_gdf = polygons_gdf.merge(day_specific_preds, left_on='id', right_on='polygon', how='inner')    
    numeric_cols = merged_gdf.select_dtypes(include=np.number).columns
    merged_gdf[numeric_cols] = merged_gdf[numeric_cols].fillna(0)

    if 'date' in merged_gdf.columns:
        merged_gdf['date'] = merged_gdf['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A')

    # --- Create the Map ---
    map_center = [merged_gdf.unary_union.centroid.y, merged_gdf.unary_union.centroid.x]
    # Initialize map with OpenStreetMap as the default
    m = folium.Map(location=map_center, zoom_start=7, tiles='OpenStreetMap')

    # Add a title to the map
    title_html = f'''
                 <h3 align="center" style="font-size:20px; font-family: Arial, sans-serif;">
                   <b>Avalanche Hazard Forecast: {prediction_date.strftime('%Y-%m-%d')}</b>
                 </h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html)) # type: ignore

    # --- Add Tile Layers (Basemaps) ---
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://opentopomap.org'>OpenTopoMap</a>",
        name="OpenTopoMap",
        ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Hybrid Satellite (Labels)',
        overlay=True
    ).add_to(m)

    # --- Define Colormaps ---
    likelihood_colormap = cm.LinearColormap(
        colors=['#3b82f6', '#facc15', '#ef4444', '#000000'],
        vmin=0, vmax=1,
        caption='Event Likelihood Score'
    )
    m.add_child(likelihood_colormap)

    # --- Create Data Layers ---
    hazard_colors = {
        1: '#22c55e',  # Green for Low (1)
        2: '#facc15',  # Yellow for Moderate (2)
        3: '#f97316',  # Orange for Considerable (3)
        4: '#ef4444',  # Red for High (4)
        0: '#e5e7eb'   # Gray for no data
    }
    hazard_levels = {
        1: 'Low', 2: 'Moderate', 3: 'Considerable', 4: 'High', 0: 'No Data'
    }
    # Create formatted strings for the tooltip
    merged_gdf['hazard_level_str'] = merged_gdf['predicted_hazard'].apply(lambda x: hazard_levels.get(int(x), 'Unknown'))
    merged_gdf['confidence'] = merged_gdf.apply(get_confidence, axis=1)
    merged_gdf['confidence_str'] = (merged_gdf['confidence'] * 100).map('{:.1f}%'.format)

    # 1. Polygon Outlines Layer
    outline_layer = folium.FeatureGroup(name='Polygon Outlines', show=False)
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            'fillOpacity': 0,
            'color': '#ffffff', # White outlines
            'weight': 1.5,
        },
        tooltip=folium.GeoJsonTooltip(fields=['title'], aliases=['Region:'])
    ).add_to(outline_layer)
    outline_layer.add_to(m)

    # 2. Event Likelihood Score Layer
    likelihood_layer = folium.FeatureGroup(name='Event Likelihood Score', show=True)
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            'fillColor': likelihood_colormap(feature['properties']['confidence']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['title', 'confidence_str'],
            aliases=['Region:', 'Hazard Confidence:'],
            localize=True
        )
    ).add_to(likelihood_layer)
    likelihood_layer.add_to(m)

    # 3. Hazard Rating Prediction Layer (Standard)
    hazard_layer = folium.FeatureGroup(name='Hazard Rating Prediction', show=True)
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            'fillColor': hazard_colors.get(int(feature['properties']['predicted_hazard']), '#e5e7eb'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6 # A standard opacity
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['title', 'predicted_hazard', 'hazard_level_str', 'confidence_str'],
            aliases=['Region:', 'Predicted Rating:', 'Hazard Level:', 'Confidence:'],
            localize=True
        )
    ).add_to(hazard_layer)
    hazard_layer.add_to(m)

    # 4. NEW: Hazard Rating Prediction Layer (Confidence as Opacity)
    hazard_confidence_layer = folium.FeatureGroup(name='Hazard Rating (by Confidence)', show=False)
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            'fillColor': hazard_colors.get(int(feature['properties']['predicted_hazard']), '#e5e7eb'),
            'color': 'black',
            'weight': 1,
            # Opacity is now dynamic based on the confidence score
            # A base opacity of 0.15 ensures even low-confidence zones are visible
            'fillOpacity': 0.15 + (feature['properties']['confidence'] * 0.75)
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['title', 'predicted_hazard', 'hazard_level_str', 'confidence_str'],
            aliases=['Region:', 'Predicted Rating:', 'Hazard Level:', 'Confidence:'],
            localize=True
        )
    ).add_to(hazard_confidence_layer)
    hazard_confidence_layer.add_to(m)


    # --- Add Layer Control ---
    folium.LayerControl().add_to(m)

    # --- Save the Map ---
    output_filename = f"prediction_map_{prediction_date.strftime('%Y-%m-%d')}.html"
    output_path = output_dir / output_filename
    m.save(str(output_path))
    logging.info(f"Successfully created and saved map to: {output_path}")

if __name__ == '__main__':
    # Example of how to run this script directly
    # In the main pipeline, the date will be passed from run_prediction.py
    example_date = "2024-01-14"
    create_prediction_map(prediction_date=example_date)
