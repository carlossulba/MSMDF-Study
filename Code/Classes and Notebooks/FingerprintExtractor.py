# FingerprintExtractor.py

import os
import numpy as np
import pandas as pd
import logging
import re
import warnings
import json
import pickle
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass, field, asdict, is_dataclass


class FingerprintSetting(Enum):
    """Available data collection settings"""
    ON_HAND = "on hand"
    ON_HAND_AUDIO = "on hand audio"
    ON_DESK = "on desk"
    ON_DESK_AUDIO = "on desk audio"
    WALKING = "walking" 
    
class FingerprintSensor(Enum):
    """Supported data stream types"""
    ACCELEROMETER = "accelerometer"
    GRAVITY = "gravity"
    TOTAL_ACCELERATION = "total_acceleration"
    GYROSCOPE = "gyroscope"

class FingerprintDataStream(Enum):
    """Supported data stream types"""
    X = 'x' 
    Y = 'y'
    Z = 'z'
    MAGNITUDE = 'magnitude'
    AZIMUTH = 'azimuth'
    INCLINATION = 'inclination'
    INTER_SAMPLE_TIME = 'inter_sample_time'

class FingerprintFeature(Enum):
    """Available features for fingerprint extraction"""
    # time domain features
    AVERAGE_DEVIATION = "average_deviation", "time"
    KURTOSIS = "kurtosis", "time"
    MAXIMUM = "maximum", "time"
    MEAN = "mean", "time"
    MINIMUM = "minimum", "time"
    MODE = "mode", "time"
    NON_NEGATIVE_COUNT = "non_negative_count", "time"
    RANGE = "range", "time"
    RMS = "rms", "time"
    SKEWNESS = "skewness", "time"
    STD_DEV = "std_dev", "time"
    VARIANCE = "variance", "time"
    ZERO_CROSSING_RATE = "zero_crossing_rate", "time"
    
    # frequency domain features
    DC = ("dc", "frequency")
    IRREGULARITY_J = "irregularity_j", "frequency"
    IRREGULARITY_K = "irregularity_k", "frequency"
    LOW_ENERGY_RATE = "low_energy_rate", "frequency"
    SMOOTHNESS = "smoothness", "frequency"
    ATTACK_SLOPE = "attack_slope", "frequency"  # not implemented yet
    ATTACK_TIME = "attack_time", "frequency"    # not implemented yet
    SPECTRAL_BRIGHTNESS = "spectral_brightness", "frequency"
    SPECTRAL_CENTROID = "spectral_centroid", "frequency"
    SPECTRAL_CREST = "spectral_crest", "frequency"
    SPECTRAL_ENTROPY = "spectral_entropy", "frequency"
    SPECTRAL_FLATNESS = "spectral_flatness", "frequency"
    SPECTRAL_FLUX = "spectral_flux", "frequency"
    SPECTRAL_IRREGULARITY = "spectral_irregularity", "frequency"
    SPECTRAL_KURTOSIS = "spectral_kurtosis", "frequency"
    SPECTRAL_RMS = "spectral_rms", "frequency"
    SPECTRAL_ROLLOFF = "spectral_rolloff", "frequency"
    SPECTRAL_ROUGHNESS = "spectral_roughness", "frequency"
    SPECTRAL_SKEWNESS = "spectral_skewness", "frequency"
    SPECTRAL_SPREAD = "spectral_spread", "frequency"
    SPECTRAL_STD = "spectral_std", "frequency"
    SPECTRAL_VARIANCE = "spectral_variance", "frequency"
    
    @property
    def name(self):
        return self.value[0]

    @property
    def domain(self):
        return self.value[1]

    
@dataclass
class FingerprintConfig:
    """Configuration for feature extraction"""
    data_location: str = "."
    
    fingerprint_length: float = 9.0 # in seconds
    sampling_rate: int  = 100 # in Hz
    
    enabled_settings: set = field(default_factory=lambda: set(FingerprintSetting))
    enabled_sensors: set = field(default_factory=lambda: set(FingerprintSensor))
    enabled_streams: set = field(default_factory=lambda: set(FingerprintDataStream))
    enabled_features: set = field(default_factory=lambda: set(FingerprintFeature))
    
    # Adjustable parameters for feature extraction
    spectral_brightness_threshold: float = 1500  # in Hz (as in Das et al. (2015))
    spectral_rolloff_threshold: float = 0.85  # Percentage (as in Das et al. (2015))
    frame_duration: float = 0.05  # in seconds (as in Das et al. (2015))
    
    def __post_init__(self):
        """Validate configuration parameters"""
        def validate_items(items: set, field_name: str, enum_type: type):
            """Validate that all elements in a set are valid members or values of an Enum."""
            valid_members = set(enum_type)  # Get all valid values (strings)
            if not items.issubset(valid_members):
                invalid_items = items - valid_members
                raise ValueError(
                    f"Invalid values in '{field_name}': {invalid_items}. "
                    f"Valid options are: {valid_members}."
                )
        
        if not os.path.exists(self.data_location):
            raise ValueError(f"The path '{self.data_location}' does not exist.")
        elif not (os.path.isdir(self.data_location) or os.path.isfile(self.data_location)):
            raise ValueError(f"The path '{self.data_location}' is neither a file nor a directory.")
        
        if not 1 <= self.fingerprint_length <= 10:
            raise ValueError("The fingerprint length must be between 1 and 10.")
        if not 1 <= self.sampling_rate <= 200:
            raise ValueError("The sampling rate must be between 1 and 200.")
        
        validate_items(self.enabled_settings, "enabled_settings", FingerprintSetting)
        validate_items(self.enabled_streams, "enabled_streams", FingerprintDataStream)
        validate_items(self.enabled_features, "enabled_features", FingerprintFeature)
        
        if not 0 < self.spectral_brightness_threshold < 20000:
            raise ValueError("The spectral brightness threshold must be between 0 and 20000.")
        if not 0 < self.spectral_rolloff_threshold < 1:
            raise ValueError("The spectral rolloff threshold must be between 0 and 1.")
        if not 0 < self.frame_duration < self.fingerprint_length:
            raise ValueError(f"The frame duration must be between 0 and {self.fingerprint_length}.")

    
    def __str__(self) -> str:
        """Return a JSON string representation of the configuration."""
        # Convert Enums to their values for JSON serialization
        config_dict = asdict(self)
        config_dict['enabled_settings'] = [item.value for item in self.enabled_settings]
        config_dict['enabled_sensors'] = [item.value for item in self.enabled_sensors]
        config_dict['enabled_streams'] = [item.value for item in self.enabled_streams]
        config_dict['enabled_features'] = [item.value[0] for item in self.enabled_features]
        return json.dumps(config_dict, indent=4)
    
    @classmethod
    def from_str(cls, config_str: str) -> 'FingerprintConfig':
        """Create a FingerprintConfig instance from a JSON string."""
        config_dict = json.loads(config_str)
        
        # Reconstruct Enums from their string values
        config_dict['enabled_settings'] = set(FingerprintSetting(item) for item in config_dict['enabled_settings'])
        config_dict['enabled_sensors'] = set(FingerprintSensor(item) for item in config_dict['enabled_sensors'])
        config_dict['enabled_streams'] = set(FingerprintDataStream(item) for item in config_dict['enabled_streams'])
        config_dict['enabled_features'] = set(FingerprintFeature[item.upper()] for item in config_dict['enabled_features'])
        
        return cls(**config_dict)
      
class FingerprintExtractor:
    """Main class for extracting fingerprints from sensor data"""
    
    def __init__(self, config: FingerprintConfig, log_level: str = 'INFO'):
        """
        Initialize the fingerprint extractor.
        
        Args:
            config: Configuration of the fingerprints to extract
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        self.config = config
        self.logger = self._setup_logger(log_level)
        
    def _setup_logger(self, log_level: str):
        """Set up logging for the fingerprint extractor."""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
        )
        return logging.getLogger(__name__)
    
    def extract_fingerprints(self) -> Dict[str, List[np.ndarray]]:
        """
        Extract fingerprints for the given configuration.
        
        Returns:
        
        """
        extracted_fingerprints = {}
        
        for setting in self.config.enabled_settings:
            print(f"Extracting fingerprints for setting '{setting.value}'...")
            try:
                extracted_fingerprints[setting.value] = self.extract_fingerprints_from_setting(setting)
            except Exception as e:
                self.logger.error(f"Error processing setting {setting.value}: {str(e)}")
                extracted_fingerprints[setting.value] = []
        
        return extracted_fingerprints

    def extract_fingerprints_from_setting(self, setting: FingerprintSetting) -> Dict[str, List[np.ndarray]]:
        """
        Extract fingerprints for a single data colection setting.
        
        Args:
            setting: The data collection setting from which to extract fingerprints
        
        Returns:
            
        """
        extracted_fingerprints_per_setting = {}
        
        # Get the directory paths for the given setting
        if setting.value == "walking":
            setting_dirs = [
                os.path.join(self.config.data_location, "walking 1"),
                os.path.join(self.config.data_location, "walking 2")
            ]
        else:
            setting_dirs = [os.path.join(self.config.data_location, setting.value)]
        
        # Extract fingerprints for each recording directory within the setting directory
        for setting_dir in setting_dirs:
            if os.path.exists(setting_dir):
                
                # Extract fingerprint for a single recording directory
                for recording_directory in os.listdir(setting_dir):
                    
                    # Determine the device ID
                    recording_path = os.path.join(setting_dir, recording_directory)
                    recording_name = os.path.basename(recording_path)
                    match = re.search(r'-\s*([a-zA-Z0-9\s]+)_([a-f0-9\-]+)-', recording_name)
                    if match:
                        device_id = f"{match.group(1).strip()}_{match.group(2).strip()}"
                    else:
                        self.logger.error(f"Invalid recording name format: {recording_name}")
                        device_id = f"unknown_device_{recording_name}"
                    
                    try:
                        # Extract fingerprints for the current recording
                        fingerprints_from_recording, fingerprints_from_recording_dict = self.extract_fingerprints_from_recording(recording_path)
                    except Exception as e:
                        self.logger.error(f"Error processing recording {recording_directory}: {str(e)}")
                        fingerprints_from_recording = []
                        fingerprints_from_recording_dict = {}
                    
                    # Accumulate fingerprints for the same device ID
                    if device_id in extracted_fingerprints_per_setting:
                        extracted_fingerprints_per_setting[device_id].append((fingerprints_from_recording, fingerprints_from_recording_dict))
                    else:
                        extracted_fingerprints_per_setting[device_id] = [(fingerprints_from_recording, fingerprints_from_recording_dict)]
                
            else:
                self.logger.error(f"Error: Directory not found: {setting_dir}")
                continue
                
        return extracted_fingerprints_per_setting
        
    def extract_fingerprints_from_recording(self, recording_path: str) -> Tuple[List[np.ndarray], Dict[int, Dict[str, Any]]]:
        """
        Extract fingerprints for a single recording directory.
        
        Args:
            recording_path: Path to the recording directory
        
        Returns:
            Tuple containing the list of fingerprints and the dictionary of features for each window
        """

        # Load and resample the sensor data for this recording
        resampled_data = self._load_and_resample_data(recording_path)
        
        # Calculate number of fingerprints we can extract from the available data
        samples_per_fingerprint = int(self.config.fingerprint_length * self.config.sampling_rate)
        min_stream_length = min(len(data) for data in resampled_data.values())
        num_fingerprints = min_stream_length // samples_per_fingerprint
        
        if num_fingerprints == 0:
            if "walking" not in recording_path.lower():
                self.logger.warning(
                    f"Insufficient samples for full fingerprint extraction in recording {recording_path}. "
                    f"Using available data length {min_stream_length} instead of required {samples_per_fingerprint}."
                )
            else:
                self.logger.info(
                    f"Insufficient samples for full fingerprint extraction due to walking recording. "
                    f"Using available data length {min_stream_length} instead of required {samples_per_fingerprint}."
                )
            num_fingerprints = 1
            samples_per_fingerprint = min_stream_length  # Use the whole available data length
        
        # Extract fingerprint for each window
        fingerprints_in_recording = []
        fingerprints_in_recording_dict = {}
        
        for window_idx in range(num_fingerprints):
            start_idx = window_idx * samples_per_fingerprint
            end_idx = start_idx + samples_per_fingerprint
            
            window_fingerprint_array, window_fingerprint_dict = self.extract_fingerprint_from_window(
                window_idx, 
                start_idx, 
                end_idx, 
                resampled_data
            )
            
            if len(window_fingerprint_array) > 0:  # Only add if features were extracted successfully
                fingerprints_in_recording.append(window_fingerprint_array)
                fingerprints_in_recording_dict[window_idx] = window_fingerprint_dict
                
        return fingerprints_in_recording, fingerprints_in_recording_dict

    def _load_and_resample_data(self, recording_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load and resample sensor data for a single recording, considering only the enabled sensors and streams.
        
        Args:
            recording_path: Path to the recording directory
        
        Returns:
            Dictionary of resampled stream data
        """
        resampled_data = {}
        
        # Get a list of CSV files in the recording directory
        csv_files = [f for f in os.listdir(recording_path) if f.endswith(".csv")]
        
        # Iterate through the list of files
        for file in csv_files:
            file_path = os.path.join(recording_path, file)
            
            # Determine the sensor from the file name
            sensor_name = os.path.splitext(file)[0].lower()
            
            # Validate that sensor_name is a known FingerprintSensor
            if sensor_name not in [sensor.value for sensor in FingerprintSensor]:
                self.logger.warning(f"Unknown sensor '{sensor_name}' in file '{file}'. Skipping this file.")
                continue
            
            # Check if the sensor is enabled
            if sensor_name not in [sensor.value for sensor in self.config.enabled_sensors]:
                self.logger.info(f"Sensor '{sensor_name}' is not enabled in configuration. Skipping this file.")
                continue
            
            try:
                # Define the columns to load, including 'seconds_elapsed'
                columns_to_load = ['seconds_elapsed'] + [stream.value for stream in self.config.enabled_streams]
                
                if "gyroscope" in file_path.lower():
                    columns_to_load = [col for col in columns_to_load if col not in ['azimuth', 'inclination']]

                # Load the CSV file into a pandas DataFrame, only with the columns we need
                df = pd.read_csv(file_path, usecols=columns_to_load)
            except ValueError as e:
                self.logger.error(f"Error reading file '{file_path}': {e}")
                continue
            
            # Check if 'seconds_elapsed' is present
            if 'seconds_elapsed' not in df.columns:
                self.logger.error(f"Missing 'seconds_elapsed' column in file '{file_path}'. Skipping this file.")
                continue
            
            # Check for missing columns
            missing_columns = [col for col in columns_to_load if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns {missing_columns} in file '{file_path}'. Proceeding with available columns.")

            # Resample the DataFrame
            resampled_df = self._resample_dataframe(df, self.config.sampling_rate)
            resampled_data[sensor_name] = resampled_df
        
        return resampled_data
    
    def _resample_dataframe(self, df: pd.DataFrame, target_sampling_rate: float) -> pd.DataFrame:
        """
        Resample a DataFrame to the target sampling rate.

        Args:
            df: DataFrame containing timestamp and sensor data
            target_sampling_rate: Desired sampling rate in Hz

        Returns:
            Resampled DataFrame
        """
        try:
            # Work on a copy to avoid modifying the original DataFrame
            df = df.copy()

            # Validate input DataFrame
            if 'seconds_elapsed' not in df.columns:
                raise ValueError("Error: timestamp column 'seconds_elapsed' not found in data")

            if df['seconds_elapsed'].isna().any():
                raise ValueError("Error: 'seconds_elapsed' column contains NaN values")

            # Calculate current sampling rate from timestamps
            time_diffs = df['seconds_elapsed'].diff().dropna()
            time_diff_mean = time_diffs.mean()
            if time_diff_mean <= 0 or np.isnan(time_diff_mean):
                raise ValueError("Error: Invalid or non-positive time differences found in 'seconds_elapsed'")

            current_sampling_rate = 1 / time_diff_mean

            if target_sampling_rate <= 0:
                raise ValueError("Error: target_sampling_rate must be a positive value")

            if current_sampling_rate > target_sampling_rate * 10:
                self.logger.warning("Current sampling rate is significantly higher than the target sampling rate")

            # Create new time index at desired sampling rate
            new_time_index = np.arange(
                df['seconds_elapsed'].iloc[0],
                df['seconds_elapsed'].iloc[-1],
                1 / target_sampling_rate
            )

            # Resample each numerical column
            resampled_df = pd.DataFrame(index=new_time_index)
            for column in df.select_dtypes(include=[np.number]).columns:
                if column != 'seconds_elapsed':  # Skip timestamp column
                    # Handle NaN values
                    if df[column].isna().any():
                        df[column] = df[column].interpolate(method='linear', limit_direction='both')
                        df[column] = df[column].bfill().ffill()

                    # Perform interpolation
                    resampled_df[column] = np.interp(
                        new_time_index,
                        df['seconds_elapsed'],
                        df[column]
                    )

            resampled_df['seconds_elapsed'] = new_time_index
            return resampled_df.reset_index(drop=True)

        except ValueError as e:
            self.logger.error(e)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return pd.DataFrame()
   
    def extract_fingerprint_from_window(self, 
                                        window_idx: int, 
                                        start_idx: int, 
                                        end_idx: int, 
                                        resampled_data: Dict[str, pd.DataFrame]
                                        ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Process a single window across all streams and extract features, considering only the enabled features.
        
        Args:
            window_idx: Index of the current window
            start_idx: Start index of the window
            end_idx: End index of the window
            resampled_data: Dictionary of resampled stream data
        
        Returns:
            Tuple containing the feature array and dictionary (for debug) for this window
        """
        window_fingerprint = []  # List to hold all features for the current window
        window_fingerprint_dict = {}  # Dictionary to hold features for each sensor stream
        
        # Loop over each sensor stream 
        for sensor_stream in self.config.enabled_sensors:
            if sensor_stream.value not in resampled_data:
                self.logger.error(f"Error: Sensor_stream {sensor_stream.value} not found in sensor data")
                return None, None
            
             # Extract the relevant window of data for the current sensor stream
            stream_df = resampled_data[sensor_stream.value]
            
            # Ensure indices are within bounds
            if end_idx > len(stream_df):
                self.logger.warning(f"End index {end_idx} exceeds data length {len(stream_df)} for stream '{sensor_stream.value}'. Adjusting end_idx.")
                end_idx = len(stream_df)
            
            stream_data = stream_df.iloc[start_idx:end_idx]
            
            if stream_data.empty:
                self.logger.error(f"No data available in the specified window [{start_idx}:{end_idx}] for stream '{sensor_stream.value}'")
                return None, None
            
            window_stream_features = []
            window_stream_features_dict = {}
            
            # Iterate through each column in the stream data
            for column in stream_data.columns:
                # Skip the timestamp column
                if column == 'seconds_elapsed':
                    continue
                
                column_data = stream_data[column].values
                
                if len(column_data) == 0:
                    self.logger.warning(f"Column '{column}' has no data in window {window_idx}")
                    continue
                
                # Compute time-domain features
                time_domain_features, time_domain_features_dict = self._compute_time_domain_features(column_data)
                                
                # Compute frequency-domain features
                frequency_domain_features, frequency_domain_features_dict = self._compute_frequency_domain_features(column_data)
                
                # Combine features
                column_features = np.concatenate((time_domain_features, frequency_domain_features))
                column_features_dict = {**time_domain_features_dict, **frequency_domain_features_dict}
                prefixed_feature_dict = {f"{column}_{k}": v for k, v in column_features_dict.items()}
                
                # Append column features to the stream's features
                window_stream_features.extend(column_features)
                window_stream_features_dict[column] = prefixed_feature_dict
                
            if not window_stream_features:
                self.logger.error(f"No features extracted for stream '{sensor_stream.value}' in window {window_idx}")
                continue  # Proceed to next stream instead of returning None
                
            # Accumulate features to create the single fingerprint
            window_fingerprint.extend(window_stream_features)
            window_fingerprint_dict[sensor_stream.value] = window_stream_features_dict 
        
        if not window_fingerprint:
            self.logger.error(f"No features extracted in window {window_idx}")
            return None, None
        
        # Convert the fingerprint list into a NumPy array
        fingerprint_array = np.array(window_fingerprint)
        
        return fingerprint_array, window_fingerprint_dict
    
    def _compute_time_domain_features(self, x: np.ndarray):
        """Compute time-domain features."""
        try:
            # Ensure x is 1D
            if x.ndim != 1:
                raise ValueError("Input x must be one-dimensional.")
            
            mean = np.mean(x).item()
            max = np.max(x).item()
            min = np.min(x).item()
            mode = stats.mode(x, nan_policy="raise").mode.item()
            std = np.std(x).item()
            var = np.var(x).item()
            rms = np.sqrt(np.mean(np.square(x))).item()
            range = (max - min)
            avg_dev = np.mean(np.abs(x - mean)).item()
            zero_crossing_rate = (np.sum(np.abs(np.diff(np.signbit(x)))) / len(x)).item()
            non_neg_count = np.sum(x >= 0).item()
            
            if var < 1e-30:
                self.logger.warning(f"Variance is close to zero ({var}). Setting kurtosis and skewness to nan.")
                kurtosis = np.nan
                skewness = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)  # Convert RuntimeWarnings to exceptions
                    try:
                        kurtosis = stats.kurtosis(x, nan_policy="raise").item()
                        skewness = stats.skew(x, nan_policy="raise").item()
                    except RuntimeWarning as e:
                        self.logger.warning(
                            f"RuntimeWarning during skewness/kurtosis computation: {e}. Setting kurtosis and skewness to nan."
                        )
                        kurtosis = np.nan
                        skewness = np.nan
                    except Exception as e:
                        self.logger.error(
                            f"Error computing skewness/kurtosis: {e}. Setting kurtosis and skewness to nan."
                            )
                        kurtosis = np.nan
                        skewness = np.nan
            
            feature_dict = {
            'mean': mean,
                'max': max,
                'min': min,
                'mode': mode,
                'std': std,
                'var': var,
                'rms': rms,
                'range': range,
                'avg_dev': avg_dev,
                'zero_crossing_rate': zero_crossing_rate,
                'non_neg_count': non_neg_count,
                'kurtosis': kurtosis,
                'skewness': skewness
            }
            
            # Collect features into a NumPy array (order corresponds to the dictionary)
            feature_array = np.array(list(feature_dict.values()))
            
            return feature_array, feature_dict
        except Exception as e:
            self.logger.error(f"Error computing time-domain features: {str(e)}")
    
    def _compute_frequency_domain_features(self, x: np.ndarray):
        """Compute frequency-domain features."""
        try:
            # Ensure x is 1D
            if x.ndim != 1:
                raise ValueError("Input x must be one-dimensional.")
            
            # Center the signal
            x_centered = x - np.mean(x)
            
            # Compute FFT and related variables
            fft_result = np.fft.fft(x_centered)
            fft_magnitude = np.abs(fft_result)
            fft_magnitude_squared = fft_magnitude ** 2
            fft_magnitude_rms = np.sqrt(np.mean(fft_magnitude_squared)).item()
            fft_magnitude_log = 20 * np.log10(fft_magnitude + 1e-10)  # Avoid log(0)
            fft_magnitude_cumulative = np.cumsum(fft_magnitude)
            fft_magnitude_geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10))).item()
            fft_magnitude_arithmetric_mean = np.mean(fft_magnitude).item()
            fft_freqs = np.fft.fftfreq(len(fft_result), d=1 / self.config.sampling_rate)
            
            # Probability mass function (PMF) (Das et al., 2015)
            fft_magnitude_pmf = (                                               
                (fft_magnitude / fft_magnitude_cumulative[-1]) 
                if fft_magnitude_cumulative[-1] > 0 else np.zeros_like(fft_magnitude)
            )
            
            # Define frames
            frame_size = int(self.config.sampling_rate * self.config.frame_duration)
            frames = (
                np.lib.stride_tricks.sliding_window_view(fft_magnitude, window_shape=frame_size) 
                if len(fft_magnitude) >= frame_size else []
            )
        
            # Das et al. (2015) - Exploring Ways To Mitigate Sensor-Based Smartphone Fingerprinting
            spectral_rms = fft_magnitude_rms
            
            low_energy_rate = (
                np.sum(np.sqrt(np.mean(frames ** 2, axis=1)) < fft_magnitude_rms) / len(frames)
            ) if len(frames) > 0 else 0
                
            spectral_centroid = (
                np.sum(fft_freqs * fft_magnitude) / fft_magnitude_cumulative[-1]
            ) if fft_magnitude_cumulative[-1] > 0 else 0

            # spectral_entropy = -np.sum(fft_magnitude_pmf * np.log2(fft_magnitude_pmf)) if fft_magnitude_cumulative[-1] > 0 else 0
            if fft_magnitude_cumulative[-1] > 0:
                pmf_nonzero = fft_magnitude_pmf[fft_magnitude_pmf > 0]
                spectral_entropy = -np.sum(pmf_nonzero * np.log2(pmf_nonzero))
            else:
                spectral_entropy = 0

            
            
            spectral_spread = np.sqrt(
                np.sum(np.power((fft_freqs - spectral_centroid), 2) * fft_magnitude_pmf)
            ).item()
            
            spectral_skewness = stats.skew(fft_magnitude).item()
            
            spectral_kurtosis = stats.kurtosis(fft_magnitude).item()
            
            spectral_flatness = (
                fft_magnitude_geometric_mean / fft_magnitude_arithmetric_mean
                if fft_magnitude_arithmetric_mean > 0 else 0
            )
            
            spectral_brightness = np.sum(
                fft_magnitude[fft_freqs >= self.config.spectral_brightness_threshold]
            ).item()
            
            rolloff_threshold = self.config.spectral_rolloff_threshold * fft_magnitude_cumulative[-1]
            indices_above_threshold = np.where(fft_magnitude_cumulative >= rolloff_threshold)[0]
            spectral_roll_off = (
                fft_freqs[indices_above_threshold[0]]
                if len(indices_above_threshold) > 0
                else 0
            )
            
            a = np.append(fft_magnitude, 0) # Amplitude = Magnitude (ambiguous description in Das et al. (2015))
            spectral_irregularity = (
                np.sum((a[:-1] - a[1:]) ** 2) / np.sum(a ** 2) 
                if fft_magnitude_cumulative[-1] != 0 else 0
            )
            
            spectral_flux = (
                np.mean(np.sqrt(np.sum(np.diff(frames, axis=0) ** 2, axis=1))) 
                if len(frames) > 0 else 0
            )
            
            spectral_attack_time = 1 # Not implemented yet

            spectral_attack_slope = 1 # Not implemented yet
            
            spectral_roughness = self._spectral_roughness(fft_magnitude, fft_freqs) 
            
            
            # Dey et al. (2014) - AccelPrint            
            irregularity_j = (
                np.sum((fft_magnitude[:-1] - fft_magnitude[1:]) ** 2) /
                np.sum(fft_magnitude[:-1] ** 2)  
            ) if np.sum(fft_magnitude[:-1] ** 2) > 0 else 0
            
            irregularity_k = (
                np.sum(np.abs(fft_magnitude[1:-1] - (fft_magnitude[:-2] + fft_magnitude[1:-1] + fft_magnitude[2:]) / 3))
            ).item()
            
            smoothness = np.sum(
                np.abs(
                    fft_magnitude_log[1:-1] - (fft_magnitude_log[:-2] + fft_magnitude_log[1:-1] + fft_magnitude_log[2:]) / 3
                )
            )
            
            spectral_crest = (
                np.max(fft_magnitude) / spectral_centroid 
                if spectral_centroid > 0 else 0
            )
            
            
            # Ding and Ming (2019) - Accelerometer-Based Mobile Device Identification System for the Realistic Environment
            dc = fft_magnitude[0]
            spectral_mean = np.mean(fft_magnitude).item()
            spectral_std = np.std(fft_magnitude).item()
            spectral_var = np.var(fft_magnitude).item()
            
            
            # Collect features into a dictionary
            feature_dict = {
                'spectral_rms': spectral_rms,
                'low_energy_rate': low_energy_rate,
                'spectral_centroid': spectral_centroid,
                'spectral_entropy': spectral_entropy,
                'spectral_spread': spectral_spread,
                'spectral_skewness': spectral_skewness,
                'spectral_kurtosis': spectral_kurtosis,
                'spectral_flatness': spectral_flatness,
                'spectral_brightness': spectral_brightness,
                'spectral_roll_off': spectral_roll_off,
                'spectral_irregularity': spectral_irregularity,
                'spectral_flux': spectral_flux,
                'spectral_attack_time': spectral_attack_time,
                'spectral_attack_slope': spectral_attack_slope,
                'spectral_roughness': spectral_roughness,
                'irregularity_j': irregularity_j,
                'irregularity_k': irregularity_k,
                'smoothness': smoothness,
                'spectral_crest': spectral_crest,
                'dc': dc,
                'spectral_mean': spectral_mean,
                'spectral_std': spectral_std,
                'spectral_var': spectral_var
            }
            
            # Collect features into a NumPy array (order corresponds to the dictionary)
            feature_array = np.array(list(feature_dict.values()))
            
            return feature_array, feature_dict
        except Exception as e:
            self.logger.error(f"Error computing frequency-domain features: {str(e)}")
    
    def _spectral_roughness(self, fft_magnitude, fft_freqs):
        peaks_indices, _ = find_peaks(fft_magnitude)
        peaks_freqs = fft_freqs[peaks_indices]
        peaks_amplitudes = fft_magnitude[peaks_indices]

        num_peaks = len(peaks_indices)
        if num_peaks < 2:
            return 0.0

        # Constants from Sethares' model
        alpha = 3.5
        beta = 5.75

        dissonance_values = []
        for i in range(num_peaks):
            for j in range(i + 1, num_peaks):
                a1 = peaks_amplitudes[i]
                a2 = peaks_amplitudes[j]
                f1 = peaks_freqs[i]
                f2 = peaks_freqs[j]

                # Calculate s based on critical bandwidth
                s = 0.24 / (0.021 * min(f1, f2) + 19)

                d = abs(f2 - f1)
                dissonance = a1 * a2 * (np.exp(-alpha * s * d) - np.exp(-beta * s * d))
                dissonance_values.append(dissonance)

        spectral_roughness_value = np.mean(dissonance_values)
        return spectral_roughness_value

    def print_extraction_summary(self, extracted_fingerprints) :
        """Print a summary of the extraction process and the structure of fingerprints_dict."""

        total_fingerprints = 0
        total_devices = set()
        summary = {}
        all_shapes = set()
        all_setting_totals = set()

        for setting, devices in extracted_fingerprints.items():
            setting_total = 0
            setting_shapes = {}
            setting_device_totals = set()
            setting_devices = {}
            setting_device_ids = set()
            device_shapes_in_setting = set()

            for device_id, recordings in devices.items():
                setting_device_ids.add(device_id)
                total_devices.add(device_id)
                device_total = 0
                device_shapes = {}
                
                for fingerprints_list, _ in recordings:
                    for fp in fingerprints_list:
                        shape = fp.shape
                        device_shapes[shape] = device_shapes.get(shape, 0) + 1
                        device_total += 1
                        setting_total += 1
                        total_fingerprints += 1
                        all_shapes.add(shape)
                        device_shapes_in_setting.add(shape)
                        
                setting_devices[device_id] = {
                    'total': device_total,
                    'shapes': device_shapes
                }
                setting_device_totals.add(device_total)

                # Aggregate shapes at setting level
                for shape, count in device_shapes.items():
                    setting_shapes[shape] = setting_shapes.get(shape, 0) + count

            # Check if all devices in setting have the same shape
            if len(device_shapes_in_setting) == 1:
                setting_shape_uniform = True
                common_shape = next(iter(device_shapes_in_setting))
            else:
                setting_shape_uniform = False

            summary[setting] = {
                'total': setting_total,
                'devices': setting_devices,
                'shapes': setting_shapes,
                'device_totals': setting_device_totals,
                'device_ids': setting_device_ids,
                'shape_uniform': setting_shape_uniform,
                'common_shape': common_shape if setting_shape_uniform else None
            }
            all_setting_totals.add(setting_total)

        # Print the summary
        print("\nExtraction Summary:")
        print("===================")
        num_total_devices = len(total_devices)
        print(f"Total number of devices: {len(total_devices)}")
        num_total_fingerprints = total_fingerprints
        print(f"Total fingerprints extracted: {total_fingerprints}\n")

        for setting, data in summary.items():
            print(f"Setting: {setting}")
            print(f"  Total fingerprints in setting: {data['total']}")
            print(f"  Number of devices in setting: {len(data['device_ids'])}")

            device_totals = data['device_totals']
            if len(device_totals) == 1:
                device_total = next(iter(device_totals))
                print(f"  All devices have the same number of fingerprints: {device_total}")
            else:
                print(f"  Devices have different numbers of fingerprints:")
                for device_id, device_data in data['devices'].items():
                    print(f"    Device '{device_id}': {device_data['total']} fingerprints")
            if data['shape_uniform']:
                print(f"  All devices have the same fingerprint shape: {data['common_shape']}")
            else:
                print(f"  Devices have different fingerprint shapes:")
                for device_id, device_data in data['devices'].items():
                    device_shapes = device_data['shapes']
                    if len(device_shapes) == 1:
                        device_shape = next(iter(device_shapes))
                        count = device_shapes[device_shape]
                        print(f"    Device '{device_id}': All fingerprints have shape {device_shape} ({count} fingerprints)")
                    else:
                        print(f"    Device '{device_id}' has fingerprints with different shapes:")
                        for shape, count in device_shapes.items():
                            print(f"      Shape {shape}: {count} fingerprints")
        print(f"\nTotal fingerprints extracted: {total_fingerprints}")

        # Check if all fingerprints have the same shape
        if len(all_shapes) == 1:
            print(f"\nAll fingerprints have the same shape: {all_shapes.pop()}")
        else:
            print(f"\nWarning: Fingerprints have different shapes: {all_shapes}")

        # Check if all settings have the same number of fingerprints
        if len(all_setting_totals) == 1:
            setting_total = next(iter(all_setting_totals))
            print(f"\nAll settings have the same number of fingerprints: {setting_total}")
        else:
            print(f"\nSettings have different numbers of fingerprints: {all_setting_totals}")
        
        return num_total_devices, num_total_fingerprints

    def save_extracted_fingerprints(self, 
                                    extracted_fingerprints: Dict[str, Dict[str, List[Tuple[List[np.ndarray], Dict[int, Dict[str, Any]]]]]], 
                                    output_dir: str, 
                                    format: str = 'pickle'):
        """
        Save the extracted fingerprints to disk.
        
        Args:
            extracted_fingerprints: Dictionary of extracted fingerprints
            output_dir: Directory to save the fingerprints
            format: Output format ('pickle' or 'json')
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        current_date = datetime.now().strftime('%Y-%m-%d__%H-%M')
        
        data_to_save = {
            'fingerprints': extracted_fingerprints,
            'config': self.config  # Convert the configuration to a serializable dictionary
        }
        
        if format == 'pickle':
            file_path = os.path.join(output_dir, f'extracted_fingerprints_{current_date}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Extracted fingerprints saved to {file_path} in pickle format.")
        elif format == 'json':
            raise NotImplementedError("JSON serialization is not yet implemented.")
            # Since JSON does not support certain data types, we need to convert them to serializable formats
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif is_dataclass(obj):
                    return asdict(obj)
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, Enum):
                    return obj.value
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(v) for v in obj]
                else:
                    return obj  # Assuming it's a basic data type
            
            json_data = convert_to_serializable(data_to_save)
            file_path = os.path.join(output_dir, f'extracted_fingerprints_{current_date}.json')
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Extracted fingerprints saved to {file_path} in JSON format.")
        else:
            self.logger.error(f"Unsupported format '{format}'. Supported formats are 'pickle' and 'json'.")        

# local main
if __name__ == "__main__":
    # Configure fingerprint extraction
    fingerprint_config = FingerprintConfig(
        data_location="../Data/raw data/separated by setting",
        
        fingerprint_length=9.0,
        sampling_rate=100,
        
        enabled_settings=set([FingerprintSetting.ON_HAND, FingerprintSetting.ON_DESK]),
        enabled_sensors=set([FingerprintSensor.ACCELEROMETER, FingerprintSensor.GYROSCOPE]),
        enabled_streams=set([FingerprintDataStream.X]),
        enabled_features=set([FingerprintFeature.MEAN, FingerprintFeature.RMS, FingerprintFeature.STD_DEV])
    )
    
    # Initialize extractor
    extractor = FingerprintExtractor(fingerprint_config, log_level='INFO')
    
    # Extract fingerprints
    fingerprints = extractor.extract_fingerprints()

    # Print extraction summary
    extractor.print_extraction_summary(fingerprints)
    print("Fingerprint extraction complete.")
    
    # Save extracted fingerprints
    extractor.save_extracted_fingerprints(fingerprints, output_dir='../Data/extracted fingerprints', format='pickle')
    