# -*- coding: utf-8 -*-
"""
snowpack_reader.py
===================

A hardware-adaptive reader for SNOWPACK .pro files using xarray.

This module provides the `SnowpackProfile` class for parsing, processing, 
and analyzing snow profile data from SNOWPACK `.pro` files. It is designed 
to seamlessly leverage available hardware—CPU or GPU—for optimal performance.

Key Features
------------
- Parses station metadata and profile time series from SNOWPACK `.pro` files.
- Stores data as an `xarray.Dataset` backed by either NumPy (CPU) or CuPy (GPU).
- Slices data by date range using the `slice()` method.
- Calculates critical snowpack properties such as `rc_flat` using vectorized methods.
- Supports flexible profile summarization (min, max, weighted mean, custom functions).
- Enables analysis of specific snowpack sections (e.g., slabs above weak layers).

Hardware Acceleration
---------------------
- **GPU Support:** If an NVIDIA GPU and `cupy` are available, numerical computations 
  are offloaded to the GPU for faster processing.
- **CPU Fallback:** Falls back to NumPy if no GPU is detected.

Typical Workflow
----------------
1.  Use the `read_snowpack()` function to load a .pro file, which returns a
    `SnowpackProfile` object.
2.  Access parsed data via the `.data` attribute (an xarray Dataset).
3.  Use `slice()` to select a date range, then chain analysis methods like
    `get_profile_summary()` or `find_layer_by_criteria()`.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

# --- Hardware-Adaptive Array Library ---
try:
    import cupy as xp
    _ = xp.arange(1)
    GPU_AVAILABLE = True
    print("✅ GPU detected. Using cupy for accelerated calculations.")
except (ImportError, RuntimeError):
    import numpy as xp
    GPU_AVAILABLE = False
    print("ℹ️ No GPU or cupy found. Falling back to CPU using numpy.")

import pandas as pd
import xarray as xr
from tqdm import tqdm
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

logger = logging.getLogger(__name__)

# --- Static Mappings ---
HEADER_MAP = {
    'Altitude=': 'altitude', 'Latitude=': 'latitude', 'Longitude=': 'longitude',
    'SlopeAngle=': 'slopeAngle', 'SlopeAzi=': 'slopeAzi', 'StationName=': 'stationName'
}
PARAM_CODES = {
    "0500": "timestamp", "0501": "height", "0504": "element_ID", "0502": "density",
    "0503": "temperature", "0506": "lwc", "0508": "dendricity", "0509": "sphericity",
    "0510": "coord_number", "0511": "bond_size", "0512": "grain_size",
    "0513": "grain_type", "0515": "ice_content", "0516": "air_content",
    "0517": "stress", "0518": "viscosity", "0520": "temperature_gradient",
    "0523": "viscous_deformation_rate", "0531": "stab_deformation_rate",
    "0532": "sn38", "0533": "sk38", "0534": "hand_hardness",
    "0535": "opt_equ_grain_size", "0601": "shear_strength", "0602": "gs_difference",
    "0603": "hardness_difference", "0604": "ssi", "1501": "height_nodes",
    "1532": "sn38_nodes", "1533": "sk38_nodes", "0540": "date_of_birth",
    "0607": "accumulated_temperature_gradient", "9998": "depth", "9999": "rc_flat"
}

class SnowpackProfile:
    """
    Represents a single SNOWPACK .pro file, containing its metadata and time-series data.

    This class serves as the primary interface for all interactions with a snowpack
    profile. It handles the entire lifecycle of a .pro file, from reading and
    parsing to performing hardware-accelerated numerical computations and
    high-level analysis. The class is designed to be hardware-aware, automatically
    using a GPU with `cupy` if available, or falling back to the CPU with `numpy`.

    Attributes:
        filename (str): The full path to the input .pro file being represented.
        save_netcdf (bool): A flag indicating whether to create a NetCDF (.nc)
            cache file upon first read. This can significantly accelerate
            subsequent loads of the same file.
        metadata (Dict): A dictionary containing station parameters parsed from
            the file header, such as latitude, longitude, and altitude.
        data (Optional[xr.Dataset]): An xarray Dataset that holds all the time-series
            profile data. The data is organized by timestamp and layer index.
            The underlying arrays will be `cupy` arrays if a GPU is available,
            otherwise they will be standard `numpy` arrays. This attribute is
            None if the file fails to load or contains no valid data.
    """

    def __init__(self, filename: str, save_netcdf: bool = False):
        """
        Initializes an empty SnowpackProfile object without loading data.

        This constructor sets up the object but does not perform any file I/O.
        Data loading is deferred to the `read()` classmethod or the top-level
        `read_snowpack()` function to provide a cleaner and more explicit API.

        Args:
            filename (str): The full path to the .pro file that this object
                will represent.
            save_netcdf (bool): A flag to indicate if a .nc cache file should be
                created upon the first successful read. This is useful for
                speeding up subsequent loads of the same file. Defaults to False.
        """
        self.filename: str = filename
        self.save_netcdf: bool = save_netcdf
        self.metadata: Dict = {}
        self.data: Optional[xr.Dataset] = None

    @classmethod
    def read(cls, filename: str, save_netcdf: bool = False) -> Optional['SnowpackProfile']:
        """
        Factory method to read and parse a .pro file, returning a new, populated
        SnowpackProfile instance.

        This is the primary and recommended method for loading data into a
        SnowpackProfile object. It encapsulates the entire file reading and
        parsing logic, handling file validation, data structuring into an
        xarray Dataset, and computation of derived properties like depth and rc_flat.

        Args:
            filename (str): The full path to the .pro file to be read.
            save_netcdf (bool): If True, instructs the instance to create a .nc
                cache file in the same directory after a successful parse, which
                will be used for faster loading in the future.

        Returns:
            A new SnowpackProfile instance containing the loaded data. If the
            file cannot be read, is empty, or contains no valid data profiles,
            this method returns None.
        """
        instance = cls(filename, save_netcdf)
        instance._read_profile()
        return instance if instance.data is not None else None

    def __len__(self) -> int:
        """
        Returns the number of profiles (i.e., unique timestamps) in the dataset.

        This allows for a simple way to check if the profile contains data,
        e.g., `if snowpack_profile: ...`

        Returns:
            The total number of timestamps in the loaded data. Returns 0 if no
            data has been loaded or the file was empty.
        """
        return len(self.data.timestamp) if self.data is not None else 0

    def __repr__(self) -> str:
        """
        Provides a concise, developer-friendly string representation of the object.

        This representation is useful for debugging and logging, as it quickly
        communicates the source file, the amount of data it contains, and the
        compute device (CPU or GPU) being used for its calculations.

        Returns:
            A string summarizing the object's state, including the filename,
            number of profiles, and the active compute device.
        """
        device = "GPU" if GPU_AVAILABLE else "CPU"
        return f"<SnowpackProfile(filename='{self.filename}', profiles={len(self)}, device='{device}')>"

    def _read_profile(self):
        """
        Internal method to orchestrate the reading and parsing of the entire .pro file.

        This method implements a simple state machine to parse the different
        sections of the .pro file format ([STATION_PARAMETERS], [DATA]). It is
        designed to be robust against common file format errors, such as
        malformed lines, and will log warnings rather than crashing. After
        parsing, it triggers the creation of the xarray Dataset and the
        computation of derived variables.
        """
        file_path = Path(self.filename)
        if not file_path.exists():
            logger.error(f"File not found: {self.filename}")
            return

        section = None
        temp_profiles: List[Dict] = []
        current_ts_data: Dict = {}

        try:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line: continue

                    if line.startswith('[') and line.endswith(']'):
                        section = line
                        continue

                    if section == '[STATION_PARAMETERS]':
                        self._parse_header_line(line)
                    elif section == '[DATA]':
                        try:
                            is_new_ts, timestamp_key = self._is_new_timestamp_line(line)
                            if is_new_ts:
                                if current_ts_data:
                                    temp_profiles.append(current_ts_data)
                                current_ts_data = {'timestamp': timestamp_key}
                            else:
                                self._parse_data_line(line, current_ts_data)
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Skipping malformed data line {i+1} in {self.filename}. Error: {e}. Line: '{line}'")
        except Exception as e:
            logger.error(f"Failed to read or process file {self.filename}: {e}", exc_info=True)
            return

        if current_ts_data:
            temp_profiles.append(current_ts_data)

        if not temp_profiles:
            logger.warning(f"No valid data profiles were parsed from file: {self.filename}")
            return

        self._create_dataset_from_profiles(temp_profiles)
        if self.data is not None:
            self._compute_and_add_depth()
            self._compute_and_add_rc_flat_vectorized()


    def _create_dataset_from_profiles(self, profiles: List[Dict]):
        """
        Internal method to convert the parsed list of profile dictionaries into
        a structured and hardware-accelerated xarray.Dataset.

        This method takes the raw parsed data, determines the dimensions
        (number of timestamps and maximum number of layers), creates NumPy arrays
        to hold the data, and then transfers them to the GPU as CuPy arrays if
        hardware acceleration is available. The final result is stored in the
        `self.data` attribute.

        Args:
            profiles: A list of dictionaries, where each dictionary represents
                a single timestamp and contains the data for all its layers.
        """
        timestamps = pd.to_datetime([p['timestamp'] for p in profiles], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        valid_indices = ~pd.isna(timestamps)
        profiles = [p for i, p in enumerate(profiles) if valid_indices[i]]
        timestamps = timestamps.dropna()
        if not profiles: return

        all_params = sorted({key for p in profiles for key in p if key != 'timestamp'})
        max_layers = max((len(p.get('height', [])) for p in profiles), default=0)
        data_vars = {param: (("timestamp", "layer_index"), np.full((len(profiles), max_layers), np.nan, dtype=np.float32)) for param in all_params}

        for i, profile in enumerate(profiles):
            num_layers = len(profile.get('height', []))
            for param, (dims, arr) in data_vars.items():
                if param in profile:
                    values = profile.get(param)
                    if values is not None:
                        arr[i, :num_layers] = np.array(values)[:num_layers]

        if GPU_AVAILABLE:
            for param, (dims, arr) in data_vars.items():
                data_vars[param] = (dims, xp.asarray(arr))
        
        self.data = xr.Dataset(data_vars, coords={'timestamp': timestamps, 'layer_index': np.arange(max_layers)})
        self.data = self.data.sortby('timestamp')

    def _parse_header_line(self, line: str):
        """Helper function to parse a single line from the [STATION_PARAMETERS] section."""
        for key, value in HEADER_MAP.items():
            if line.startswith(key):
                self.metadata[value] = line.split('=', 1)[1].strip()

    def _is_new_timestamp_line(self, line: str) -> Tuple[bool, Optional[str]]:
        """Helper function to check if a data line marks the beginning of a new profile."""
        parts = line.split(',', 1)
        if len(parts) < 2: raise IndexError("Line does not contain a comma.")
        return (True, parts[1]) if parts[0] == "0500" else (False, None)

    def _parse_data_line(self, line: str, current_ts_data: Dict):
        """Helper function to parse a single data line containing layer data."""
        parts = line.split(',')
        if len(parts) < 3: return # Not enough data to be a valid parameter line
        if (param_name := PARAM_CODES.get(parts[0])):
            current_ts_data[param_name] = np.array(parts[2:], dtype=float)

    def _compute_and_add_depth(self):
        """
        Calculates the depth of each layer relative to the snow surface (top-down)
        and adds it as a new variable named 'depth' to the dataset.
        """
        if self.data is None or 'height' not in self.data.data_vars:
            logger.warning("Cannot calculate depth without 'height' variable.")
            return
        
        height = self.data['height']
        total_height = height.max(dim='layer_index', skipna=True)
        depth = total_height - height
        
        if GPU_AVAILABLE:
            final_depth_data = xp.where(height.notnull().data, depth.data, xp.nan)
            self.data['depth'] = xr.DataArray(final_depth_data, dims=depth.dims, coords=depth.coords)
        else:
            self.data['depth'] = depth.where(height.notnull())           
        
    def _compute_and_add_rc_flat_vectorized(self):
        """
        Calculates the `rc_flat` stability parameter for all layers and profiles
        in a single, efficient vectorized operation and adds it to the dataset.
        """
        if self.data is None: return
        required_vars = {'density', 'grain_size', 'shear_strength', 'height'}
        if not required_vars.issubset(self.data.data_vars):
            logger.warning("Skipping rc_flat calculation due to missing variables.")
            return

        RHO_ICE, GS_0, G, A, B = 917.0, 0.00125, 9.81, 4.6e-9, -2.0
        height, density, grain_size, shear_strength = (
            self.data['height'], self.data['density'],
            self.data['grain_size'], self.data['shear_strength']
        )

        height_of_bottom = height.shift(layer_index=1, fill_value=0)
        thick = height - height_of_bottom
        layer_load = (density * thick * G)
        load = layer_load.reindex(layer_index=layer_load.layer_index[::-1]).cumsum(dim='layer_index').reindex(layer_index=layer_load.layer_index)
        total_thick_above = self.data['height'].max(dim='layer_index', skipna=True) - height
        
        rho_sl_raw = load / (total_thick_above * G)
        
        if GPU_AVAILABLE:
            rho_sl_numpy = rho_sl_raw.to_numpy()
            rho_sl_cpu_da = xr.DataArray(rho_sl_numpy, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
            rho_sl_filled_cpu = rho_sl_cpu_da.bfill(dim='layer_index').ffill(dim='layer_index')
            rho_sl_gpu = xp.asarray(rho_sl_filled_cpu.values)
            rho_sl = xr.DataArray(rho_sl_gpu, dims=rho_sl_raw.dims, coords=rho_sl_raw.coords)
        else:
            rho_sl = rho_sl_raw.bfill(dim='layer_index').ffill(dim='layer_index')

        tau_p = shear_strength * 1000.0
        gs = grain_size * 0.001
        e_prime = 5.07e9 * (rho_sl / RHO_ICE)**5.13 / (1 - 0.2**2)
        dsl_over_sigman = 1.0 / (G * rho_sl)
        term1_under = A * (density / RHO_ICE * gs / GS_0)**B
        term2_under = 2 * tau_p * e_prime * dsl_over_sigman

        def _process_term_data(term_under):
            term_data = term_under.data.copy()
            term_data[term_data <= 0] = xp.nan
            return xr.DataArray(xp.sqrt(term_data), dims=term_under.dims, coords=term_under.coords)

        term1 = _process_term_data(term1_under)
        term2 = _process_term_data(term2_under)

        rc_flat_da = xr.DataArray(
            xp.nan_to_num((term1 * term2).data, nan=9999.0),
            dims=term1.dims,
            coords=term1.coords
        )

        max_heights = height.max(dim='layer_index', skipna=True)
        is_not_surface = height < max_heights
        
        if GPU_AVAILABLE:
            final_data = xp.where(is_not_surface.data, rc_flat_da.data, 9999.0)
            rc_flat_da = xr.DataArray(final_data, dims=rc_flat_da.dims, coords=rc_flat_da.coords)
        else:
            rc_flat_da = rc_flat_da.where(is_not_surface, 9999.0)

        self.data['rc_flat'] = rc_flat_da.transpose('timestamp', 'layer_index')

    def slice(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> 'SnowpackProfile':
        """
        Creates a new, independent SnowpackProfile object containing a temporal
        slice of the data for a specified date range.

        This method enables a clean, chainable API for analysis (e.g.,
        `profile.slice('2023-01-01', '2023-01-31').get_profile_summary(...)`).
        It returns a new SnowpackProfile instance, ensuring that the original
        object remains unmodified and can be reused.

        Args:
            start_date: The start date for the slice in 'YYYY-MM-DD' format.
                If None, the slice starts from the beginning of the available data.
            end_date: The end date for the slice in 'YYYY-MM-DD' format.
                If None, the slice extends to the end of the available data.

        Returns:
            A new SnowpackProfile instance containing only the data within the
            specified date range. If no data is found in the range, it returns
            a new instance with an empty data attribute.
        """
        if self.data is None or self.data.timestamp.size == 0: return self
        timestamps = pd.to_datetime(self.data.timestamp.values).normalize()
        start_mask = timestamps >= pd.to_datetime(start_date).normalize() if start_date else np.ones(len(timestamps), dtype=bool)
        end_mask = timestamps <= pd.to_datetime(end_date).normalize() if end_date else np.ones(len(timestamps), dtype=bool)
        combined_mask = start_mask & end_mask
        
        if not np.any(combined_mask):
            logger.warning(f"No data found in the date range {start_date} to {end_date} for file {self.filename}")
            sliced_data = self.data.isel(timestamp=slice(0, 0))
        else:
            sliced_data = self.data.isel(timestamp=np.where(combined_mask)[0])

        new_profile = SnowpackProfile(self.filename)
        new_profile.data = sliced_data
        new_profile.metadata = self.metadata
        return new_profile

    def save_as_netcdf(self, output_path: str):
        """
        Saves the profile's entire xarray.Dataset to a NetCDF (.nc) file.

        NetCDF is an efficient, binary format that allows for much faster data
        loading compared to re-parsing the original .pro text file. This method
        is particularly useful for caching large datasets. It automatically
        handles the conversion from GPU (CuPy) to CPU (NumPy) arrays before
        saving, as this is a requirement of the underlying NetCDF library.

        Args:
            output_path: The destination file path where the .nc file will be saved.
                The necessary parent directories will be created if they do not exist.
        """
        if self.data is None or self.data.timestamp.size == 0:
            logger.warning(f"No data to save for NetCDF file: {output_path}")
            return
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            data_to_save = self.data.as_numpy() if GPU_AVAILABLE else self.data
            data_to_save.to_netcdf(output_path)
            logger.debug(f"Successfully saved profile to NetCDF: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save NetCDF file to {output_path}: {e}", exc_info=True)

    def _prepare_daily_profiles(self, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Internal helper to slice data and select the single most representative
        profile (closest to noon) for each day.

        This method is a crucial preprocessing step for daily summary analyses.
        It ensures that comparisons between days are consistent by using data
        from approximately the same time each day.

        Args:
            start_date: The start date for the selection ('YYYY-MM-DD').
            end_date: The end date for the selection ('YYYY-MM-DD').

        Returns:
            A pandas DataFrame containing only the noon profiles for each day
            in the specified range.
        """
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()

        # Transfer data to CPU if needed and convert to a pandas DataFrame
        data_in_range = sliced_profile.data
        full_df = (data_in_range.as_numpy().to_dataframe().reset_index()
                   if GPU_AVAILABLE
                   else data_in_range.to_dataframe().reset_index())

        # Find the profile closest to noon for each day
        noon_time = full_df['timestamp'].dt.normalize() + pd.Timedelta(hours=12)
        full_df['time_from_noon'] = (full_df['timestamp'] - noon_time).abs()
        
        # Get the index of the closest profile for each day and filter
        closest_indices = full_df.loc[full_df.groupby(full_df['timestamp'].dt.date)['time_from_noon'].idxmin()]
        
        return closest_indices.copy()

    def _calculate_layer_thickness(self, profile_layers: pd.DataFrame, from_height: Optional[float], above_or_below: str) -> pd.DataFrame:
        """Internal helper to calculate the thickness of each layer in a profile DataFrame."""
        if profile_layers.empty:
            return profile_layers
        
        profile_layers = profile_layers.sort_values('height').copy()
        profile_layers['thickness'] = profile_layers['height'].diff()
        
        # Calculate thickness for the first layer in the slice
        base_h = from_height if (from_height is not None and above_or_below == 'above') else 0
        if not profile_layers.empty:
            first_layer_index = profile_layers.index[0]
            first_layer_height = profile_layers.loc[first_layer_index, 'height']
            profile_layers.loc[first_layer_index, 'thickness'] = first_layer_height - base_h
            
        return profile_layers

    def _calculate_layer_summaries(self, profile_layers: pd.DataFrame, parameters: Dict[str, Any]) -> pd.Series:
        """Internal helper to perform all statistical calculations for a single profile's layers."""
        summary_data = {}
        for name, calc in parameters.items():
            if callable(calc):
                try: summary_data[name] = calc(profile_layers)
                except Exception: summary_data[name] = np.nan
                continue

            param, calc_type = (calc if isinstance(calc, tuple) else (name.split('-')[0], calc))
            if not param or param not in profile_layers: continue

            series = profile_layers[param].dropna()
            if series.empty: continue
            if param == 'hand_hardness': series = series.abs()

            if calc_type == 'min':
                idx = series.idxmin()
                summary_data[name] = series.min()
                if 'height' in profile_layers.columns: summary_data[f"{name}-height"] = profile_layers.loc[idx, 'height']
            elif calc_type == 'max':
                idx = series.idxmax()
                summary_data[name] = series.max()
                if 'height' in profile_layers.columns: summary_data[f"{name}-height"] = profile_layers.loc[idx, 'height']
            elif calc_type == 'mean':
                summary_data[name] = series.mean()
            elif calc_type == 'weighted_mean':
                weights = profile_layers.loc[series.index, 'thickness']
                if weights.sum() > 0:
                    summary_data[name] = np.average(series, weights=weights)
        
        return pd.Series(summary_data)

    def _summarize_layers(self, profile_layers: pd.DataFrame, parameters: Dict[str, Any], from_height: Optional[float], above_or_below: str) -> pd.Series:
        """Internal helper to orchestrate the summarization of a single day's snow profile."""
        # Filter layers based on height criteria
        if from_height is not None:
            if above_or_below == 'above':
                profile_layers = profile_layers[profile_layers['height'] > from_height]
            else: # 'below'
                profile_layers = profile_layers[profile_layers['height'] <= from_height]
        
        if profile_layers.empty:
            return pd.Series(dtype=float)

        # Add a 'thickness' column to the layers
        profile_with_thickness = self._calculate_layer_thickness(profile_layers, from_height, above_or_below)
        
        # Perform all statistical calculations
        return self._calculate_layer_summaries(profile_with_thickness, parameters)

    def get_profile_summary(self, parameters_to_calculate: Dict[str, Any], from_height: Optional[float] = None, above_or_below: str = 'above', start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts daily summary statistics for specified snowpack parameters.

        This is a high-level analysis function that computes aggregate statistics
        (e.g., mean, max, weighted mean) for the snow profile on each day within
        a given date range. It can analyze the entire snowpack or a specific
        vertical section (e.g., the slab above a certain height). This method
        uses a robust looping mechanism to process each day individually,
        ensuring reliable results even with inconsistent data.

        Args:
            parameters_to_calculate: A dictionary that defines the calculations
                to perform. Keys are the desired output column names, and values
                specify the parameter and calculation type.
                Examples:
                - `{'slab_density_mean': ('density', 'mean')}`
                - `{'max_temp': ('temperature', 'max')}`
                - `{'custom_metric': lambda df: df['col'].sum()}`
            from_height: An optional height in meters used to slice the snowpack
                vertically for analysis (e.g., analyze only layers above 0.5m).
                If None, the entire profile is used.
            above_or_below: Specifies which part of the snowpack to analyze
                relative to `from_height`. Must be 'above' or 'below'.
            start_date: The start date for the summary ('YYYY-MM-DD'). Defaults
                to the start of the profile data.
            end_date: The end date for the summary ('YYYY-MM-DD'). Defaults to
                the end of the profile data.

        Returns:
            A pandas DataFrame with a 'date' index and a column for each
            requested summary statistic. Returns an empty DataFrame if no data
            is available in the specified range.
        """
        # Step 1: Prepare the data by selecting the single most representative
        # profile (closest to noon) for each day in the date range.
        sliced_profile = self.slice(start_date, end_date)
        if sliced_profile.data is None or sliced_profile.data.timestamp.size == 0:
            return pd.DataFrame()
        
        data_in_range = sliced_profile.data
        full_df = (data_in_range.as_numpy().to_dataframe().reset_index() 
                   if GPU_AVAILABLE 
                   else data_in_range.to_dataframe().reset_index())

        noon_time = full_df['timestamp'].dt.normalize() + pd.Timedelta(hours=12)
        full_df['time_from_noon'] = (full_df['timestamp'] - noon_time).abs()
        closest_indices = full_df.loc[full_df.groupby(full_df['timestamp'].dt.date)['time_from_noon'].idxmin()]
        noon_df = closest_indices.copy()
        
        if noon_df.empty:
            return pd.DataFrame()
        
        # Step 2: Use a robust loop to iterate through each day's profile
        summary_list = []
        for ts in noon_df['timestamp']:
            summary_row = {'date': ts.normalize()}
            profile_layers = full_df[full_df['timestamp'] == ts].copy()

            if from_height is not None:
                if above_or_below == 'above':
                    profile_layers = profile_layers[profile_layers['height'] > from_height]
                else:
                    profile_layers = profile_layers[profile_layers['height'] <= from_height]

            if not profile_layers.empty:
                profile_layers = self._calculate_layer_thickness(profile_layers, from_height, above_or_below)

            # Calculate summaries for the current profile
            for name, calc in parameters_to_calculate.items():
                if callable(calc):
                    try:
                        summary_row[name] = calc(profile_layers)
                    except Exception as e:
                        logger.warning(f"Custom function for '{name}' failed: {e}")
                    continue

                param, calc_type = (calc if isinstance(calc, tuple) else (name.split('-')[0], calc))
                if not param or param not in profile_layers: continue

                series = profile_layers[param].dropna()
                if series.empty: continue
                if param == 'hand_hardness': series = series.abs()
                
                if calc_type == 'min':
                    summary_row[name] = series.min()
                    if 'height' in profile_layers.columns:
                        summary_row[f"{name}-height"] = profile_layers.loc[series.idxmin()]['height']
                elif calc_type == 'max':
                    summary_row[name] = series.max()
                    if 'height' in profile_layers.columns:
                        summary_row[f"{name}-height"] = profile_layers.loc[series.idxmax()]['height']
                elif calc_type == 'mean':
                    summary_row[name] = series.mean()
                elif calc_type == 'weighted_mean':
                    weights = profile_layers.loc[series.index]['thickness']
                    if weights.sum() > 0:
                        summary_row[name] = np.average(series, weights=weights)
            
            summary_list.append(summary_row)

        if not summary_list:
            return pd.DataFrame()
            
        return pd.DataFrame(summary_list).set_index('date')

    def _get_condition_mask(self, df: pd.DataFrame, param: str, condition: str) -> pd.Series:
        """Parses a condition string and returns a boolean mask for a DataFrame column."""
        if param not in df.columns:
            logger.warning(f"Parameter '{param}' not found in profile. Skipping criterion.")
            return pd.Series(False, index=df.index)

        if ' to ' in condition:
            try:
                low, high = map(float, condition.split(' to ', 1))
                return (df[param] >= low) & (df[param] <= high)
            except (ValueError, IndexError):
                logger.warning(f"Invalid 'between' format for '{param}': '{condition}'.")
                return pd.Series(False, index=df.index)

        op_pattern = re.compile(r'([<>=!]+)\s*(\S+)')
        match = op_pattern.match(condition)
        if not match:
            logger.warning(f"Invalid condition format for '{param}': '{condition}'.")
            return pd.Series(False, index=df.index)

        op, value_str = match.groups()
        try: value = float(value_str)
        except ValueError:
            logger.warning(f"Could not convert value '{value_str}' to float.")
            return pd.Series(False, index=df.index)

        if op == '<': return df[param] < value
        if op == '>': return df[param] > value
        if op == '<=': return df[param] <= value
        if op == '>=': return df[param] >= value
        if op == '==': return df[param] == value
        if op == '!=': return df[param] != value
        
        logger.warning(f"Unsupported operator '{op}' in criteria.")
        return pd.Series(False, index=df.index)

    def _find_best_layer(self, daily_profile: pd.DataFrame, criteria: Dict[str, str], search_from: str) -> pd.Series:
        """Finds the single best-matching layer in a daily profile based on weighted criteria."""
        if daily_profile.empty: return pd.Series(dtype=object)

        if 'depth' in criteria and 'height' in daily_profile.columns:
            daily_profile['depth'] = daily_profile['height'].max() - daily_profile['height']
        if 'hand_hardness' in criteria and 'hand_hardness' in daily_profile.columns:
            daily_profile['hand_hardness'] = daily_profile['hand_hardness'].abs()

        criteria_masks = {}
        score = pd.Series(0, index=daily_profile.index, dtype=int)
        num_criteria = len(criteria)

        for i, (param, condition) in enumerate(criteria.items()):
            weight = 2**(num_criteria - 1 - i)
            mask = self._get_condition_mask(daily_profile, param, condition).fillna(False)
            criteria_masks[param] = mask
            score += mask.astype(int) * weight

        max_score = score.max()
        if max_score == 0 or pd.isna(max_score):
            result = {'height': np.nan, 'matching_criteria_count': 0, 'matching_parameters': {}}
            for param in criteria: result[param] = np.nan
            return pd.Series(result)

        best_matching_layers = daily_profile[score == max_score]
        target_layer = best_matching_layers.iloc[-1] if search_from == 'top' else best_matching_layers.iloc[0]

        matched_params = {
            param: target_layer[param] for param, mask in criteria_masks.items()
            if param in target_layer and mask.get(target_layer.name, False)
        }
        
        result = {
            'height': target_layer.get('height'),
            'matching_criteria_count': len(matched_params),
            'matching_parameters': matched_params
        }
        for param in criteria:
            result[param] = target_layer.get(param)

        return pd.Series(result)

    def find_layer_by_criteria(self, criteria: Dict[str, str], search_from: str = 'top', start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Finds the single layer that best matches a set of prioritized criteria for each day.

        This powerful analysis method searches through each daily snow profile to
        identify the single most relevant layer (e.g., a potential weak layer)
        based on a user-defined, ordered set of conditions.

        Methodology:
        1.  **Daily Profile Selection**: For each day in the specified date range,
            it selects the single profile closest to noon to ensure consistency.
        2.  **Weighted Scoring**: It evaluates every layer in the daily profile
            against the provided `criteria`. Each layer receives a score based on
            how many criteria it meets.
        3.  **Prioritization**: The `criteria` dictionary is ordered. Criteria listed
            earlier have a higher weight in the scoring, acting as a tie-breaker.
            For example, if two layers match 3 criteria each, but one of them
            matches a higher-priority criterion, it will be ranked higher.
        4.  **Tie-Breaking**: If multiple layers have the exact same top score, the
            `search_from` parameter is used as the final tie-breaker, selecting
            either the highest ('top') or lowest ('bottom') layer in the snowpack.

        Args:
            criteria (Dict[str, str]): An ordered dictionary where keys are
                parameter names (e.g., 'density', 'depth', 'grain_size') and
                values are condition strings. Supported conditions include:
                - Comparison: '< 150', '>= 0.2'
                - Range: '1.0 to 2.0'
                The order of items in the dictionary defines their priority.
            search_from (str, optional): If a tie in score occurs, this direction
                is used to select the final layer. Can be 'top' (highest layer,
                closest to the snow surface) or 'bottom' (lowest layer).
                Defaults to 'top'.
            start_date (Optional[str], optional): The start date for the search
                period ('YYYY-MM-DD'). Defaults to the start of the profile data.
            end_date (Optional[str], optional): The end date for the search period
                ('YYYY-MM-DD'). Defaults to the end of the profile data.

        Returns:
            pd.DataFrame: A DataFrame with the date as the index and columns
            summarizing the best-matching layer for each day, including:
            - `height`: The height of the best-matching layer.
            - `matching_criteria_count`: How many criteria the layer met.
            - `matching_parameters`: A dictionary of the specific criteria that
              were met and their values in the layer.
            - A column for each parameter in the input `criteria`, showing its
              value in the found layer.

        Raises:
            ValueError: If an invalid `search_from` direction is provided.
        """
        if search_from not in ['top', 'bottom']:
            raise ValueError("Argument 'search_from' must be either 'top' or 'bottom'.")

        daily_profiles_df = self._prepare_daily_profiles(start_date, end_date)
        if daily_profiles_df.empty:
            return pd.DataFrame()

        results = daily_profiles_df.groupby(daily_profiles_df['timestamp'].dt.normalize()).apply(
            self._find_best_layer,
            criteria=criteria,
            search_from=search_from
        )
        results.index.name = 'date'
        return results
        
def read_snowpack(pro_file_path: str, save_netcdf: bool = False) -> Optional[SnowpackProfile]:
    """
    Reads snowpack data, prioritizing a cached NetCDF file over the raw .pro file.

    This function serves as the main entry point for loading snowpack data.
    It abstracts away the caching logic. If a .nc file corresponding to the
    .pro file exists, it is loaded directly for speed. If not, the .pro file
    is parsed, and if `save_netcdf` is True, a new .nc cache file is created
    for future use.

    Args:
        pro_file_path (str): The full path to the raw .pro file.
        save_netcdf (bool): If True, creates a .nc cache file after parsing.
            Defaults to False.

    Returns:
        Optional[SnowpackProfile]: A SnowpackProfile object with the loaded data,
                                   or None if both reading methods fail.
    """
    pro_path = Path(pro_file_path)
    nc_path = pro_path.with_suffix('.nc')

    if nc_path.exists():
        try:
            data = xr.open_dataset(nc_path)
            profile = SnowpackProfile(str(pro_path))
            profile.data = data
            logger.debug(f"Loaded snowpack data from cached NetCDF: {nc_path}")
            return profile
        except Exception as e:
            logger.warning(f"Could not read cached NetCDF file {nc_path}, falling back to .pro. Error: {e}")

    try:
        profile = SnowpackProfile.read(str(pro_path), save_netcdf=save_netcdf)
        if profile and profile.data is not None and profile.data.timestamp.size > 0 and save_netcdf:
            profile.save_as_netcdf(str(nc_path))
            logging.info(f"Created NetCDF cache file at: {nc_path}")
        return profile
    except Exception as e:
        logger.error(f"Failed to read and process .pro file {pro_path}: {e}")
        return None
