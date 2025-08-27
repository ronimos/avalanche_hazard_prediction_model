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
from shapely.ops import union_all

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
    - A choropleth layer for the final predicted hazard rating.
    - A layer showing only the polygon outlines.

    Args:
        prediction_date (Union[str, datetime]): The specific date for which to plot predictions,
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

    logging.info(f"Generating prediction map for date: {prediction_date.date()}...")

    try:
        # Use centralized paths from the config file
        geojson_path = config.PATHS["RAW_DATA"]["polygons"]
        predictions_path = config.PATHS["ARTIFACTS"]["hazard_predictions_csv"]
        output_dir = config.RESULTS_DIR

        gdf = gpd.read_file(geojson_path)
        preds_df = pd.read_csv(predictions_path)
    except FileNotFoundError as e:
        logging.error(f"Could not read a required data file: {e}. Please ensure paths are correct and predictions exist.")
        return

    # Prepare and merge data
    preds_df['date'] = pd.to_datetime(preds_df['date'])
    day_specific_preds = preds_df[preds_df['date'].dt.date == prediction_date.date()].copy()

    if day_specific_preds.empty:
        logging.warning(f"No predictions found for {prediction_date.date()}. Cannot create map.")
        return

    # The GeoJSON 'title' should match the 'polygon' ID in the predictions
    gdf['polygon'] = gdf['title'].astype(int)
    day_specific_preds['polygon'] = day_specific_preds['polygon'].astype(int)
    merged_gdf = gdf.merge(day_specific_preds, on='polygon', how='left')
    
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
    m.get_root().html.add_child(folium.Element(title_html))

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
            'fillColor': likelihood_colormap(feature['properties']['event_adjusted_score']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['title', 'event_adjusted_score'],
            aliases=['Region:', 'Likelihood Score:'],
            localize=True
        )
    ).add_to(likelihood_layer)
    likelihood_layer.add_to(m)

    # 3. Hazard Rating Prediction Layer
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
    merged_gdf['hazard_level_str'] = merged_gdf['predicted_hazard'].apply(lambda x: hazard_levels.get(int(x), 'Unknown'))

    hazard_layer = folium.FeatureGroup(name='Hazard Rating Prediction', show=True)
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            'fillColor': hazard_colors.get(int(feature['properties']['predicted_hazard']), '#e5e7eb'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['title', 'predicted_hazard', 'hazard_level_str'],
            aliases=['Region:', 'Predicted Rating:', 'Hazard Level:'],
            localize=True
        )
    ).add_to(hazard_layer)
    hazard_layer.add_to(m)

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
    example_date = "2024-01-14"#datetime.now()
    create_prediction_map(prediction_date=example_date)
