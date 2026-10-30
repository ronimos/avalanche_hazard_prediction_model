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

# Occurrence probabilities above this get the top color (about the 95th percentile of winter polygon-days)
DRY_SLAB_COLOR_MAX = 0.5

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
        predictions_path = config.PATHS["RESULTS"]["hazard_predictions_csv"]
        output_dir = config.RESULTS_DIR

        gdf = gpd.read_file(geojson_path)
        preds_df = pd.read_csv(predictions_path)
    except FileNotFoundError as e:
        logging.error(f"Could not read a required data file: {e}. Please ensure paths are correct and predictions exist.")
        return

    # Prepare and merge data
    preds_df['date'] = pd.to_datetime(preds_df['date'])
    
    # Filter for the specific date
    day_specific_preds = preds_df[preds_df['date'].dt.date == prediction_date.date()].copy()

    if day_specific_preds.empty:
        logging.warning(f"No predictions found for {prediction_date.date()}. Cannot create map.")
        return

    # FIX: Convert the date column to string immediately. 
    # This prevents Timestamp objects from propagating into the GeoDataFrame and crashing Folium/JSON.
    day_specific_preds['date'] = day_specific_preds['date'].dt.strftime('%Y-%m-%d')

    # The GeoJSON 'polygon_number' should match the 'polygon' ID in the predictions
    gdf['polygon'] = gdf['polygon_number'].astype(int)
    day_specific_preds['polygon'] = day_specific_preds['polygon'].astype(int)
    
    merged_gdf = gdf.merge(day_specific_preds, on='polygon', how='left')
    
    numeric_cols = merged_gdf.select_dtypes(include=np.number).columns
    merged_gdf[numeric_cols] = merged_gdf[numeric_cols].fillna(0)

    # FIX: Robust check to ensure NO Timestamp objects remain in the GeoDataFrame columns
    for col in merged_gdf.columns:
        # Check for datetime dtypes
        if pd.api.types.is_datetime64_any_dtype(merged_gdf[col]):
            merged_gdf[col] = merged_gdf[col].dt.strftime('%Y-%m-%d').fillna('')
        # Check for object columns that might contain Timestamps (like 'date' if not fixed above)
        elif merged_gdf[col].dtype == 'object':
             merged_gdf[col] = merged_gdf[col].astype(str).replace({'NaT': '', 'nan': '', 'None': ''})
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
    has_dry_slab = 'dry_slab_probability' in merged_gdf.columns
    if has_dry_slab:
        dry_slab_colormap = cm.LinearColormap(
            colors=['#3b82f6', '#facc15', '#ef4444', '#000000'],
            vmin=0, vmax=DRY_SLAB_COLOR_MAX,
            caption=f'Probability of a D2+ Dry Slab Avalanche, Natural or Triggered (reported or not; capped at {DRY_SLAB_COLOR_MAX:.0%})'
        )
        m.add_child(dry_slab_colormap)

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
        tooltip=folium.GeoJsonTooltip(fields=['polygon_number'], aliases=['Region:'])
    ).add_to(outline_layer)
    outline_layer.add_to(m)

    # 2. Dry Slab Likelihood Layers: the probability that a D2+ dry slab avalanche occurs (reported or not),
    # and the low and high ends of its 80% interval
    if has_dry_slab:
        merged_gdf['dry_slab_str'] = merged_gdf['dry_slab_probability'].map(
            lambda p: f"{p:.1%}" if pd.notna(p) else 'No data')
        merged_gdf['dry_slab_range_str'] = merged_gdf.apply(
            lambda r: f"{r['dry_slab_probability_q10']:.1%} to {r['dry_slab_probability_q90']:.1%}"
            if pd.notna(r['dry_slab_probability']) else 'No data', axis=1)
        for column, name, show in [('dry_slab_probability', 'Dry Slab Likelihood', True),
                                   ('dry_slab_probability_q10', 'Dry Slab Likelihood, Low End of 80% Range', False),
                                   ('dry_slab_probability_q90', 'Dry Slab Likelihood, High End of 80% Range', False)]:
            layer = folium.FeatureGroup(name=name, show=show)
            folium.GeoJson(
                merged_gdf,
                style_function=lambda feature, column=column: {
                    'fillColor': dry_slab_colormap(min(feature['properties'][column], DRY_SLAB_COLOR_MAX))
                    if feature['properties'][column] is not None else '#e5e7eb',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['polygon_number', 'dry_slab_str', 'dry_slab_range_str'],
                    aliases=['Region:', 'D2+ Dry Slab Probability:', '80% Range:'],
                    localize=True
                )
            ).add_to(layer)
            layer.add_to(m)

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
            fields=['polygon_number', 'predicted_hazard', 'hazard_level_str', 'confidence_str'],
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
            fields=['polygon_number', 'predicted_hazard', 'hazard_level_str', 'confidence_str'],
            aliases=['Region:', 'Predicted Rating:', 'Hazard Level:', 'Confidence:'],
            localize=True
        )
    ).add_to(hazard_confidence_layer)
    hazard_confidence_layer.add_to(m)


    # --- Link to the methods doc ---
    m.get_root().html.add_child(folium.Element(
        '<div style="position: fixed; bottom: 24px; left: 12px; z-index: 1000; background: white; padding: 6px 10px; '
        'border-radius: 4px; font: 13px sans-serif; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);">'
        f'<a href="{config.METHODS_DOC_URL}" target="_blank" rel="noopener">How to read the dry slab probability (methods)</a></div>'
    ))

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
