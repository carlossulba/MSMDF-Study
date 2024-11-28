"""
Data Processing Script
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.ticker as ticker
import tkinter as tk
import json
import shutil
import traceback
from scipy import stats
from scipy.fft import fft
from typing import Any, List, Optional, Tuple, Dict
from matplotlib.widgets import SpanSelector
from IPython import display
from tkinter import messagebox, simpledialog


class StepConditions:
    def __init__(self):
        # Define base thresholds for each step
        self.thresholds = {
            'step1': {
                'acc_mag': (0, 1.0),
                'gyro_mag': (0, 0.3),
                'gravity_x': (-2, 2),
                'gravity_y': (1, 10),
                'gravity_z': (0, 10)
            },
            'step2': {
                'acc_mag': (0, 0.28),
                'gyro_mag': (0, 0.1),
                'gravity_x': (-1, 1),
                'gravity_y': (-1, 1),
                'gravity_z': (8, 10)
            },
            'step3': {
                'acc_mag': (0, 1),
                'gyro_mag': (0, 4.2),
                'gravity_x': (-2, 2),
                'gravity_y': (1, 10),
                'gravity_z': (0, 10)
            },
            'step4': {
                'acc_mag': (0, 0.3),
                'gyro_mag': (0, 4.2),
                'gravity_x': (-1, 1),
                'gravity_y': (-1, 1),
                'gravity_z': (8, 10)
            },
            'step5': {
                'acc_mag': (0, 20),
                'gyro_mag': (0, 10),
                'gravity_x': (-10, 10),
                'gravity_y': (-10, 6),
                'gravity_z': (-6, 6)
            },
            'step6': {  # Same as step5
                'acc_mag': (0, 20),
                'gyro_mag': (0, 10),
                'gravity_x': (-10, 10),
                'gravity_y': (-10, 6),
                'gravity_z': (-6, 6)
            }
        }

    def get_adjusted_thresholds(self, step_name: str, relax_factor: float = 1.0) -> dict:
        """Get thresholds for a step, adjusted by relax factor if needed."""
        base = self.thresholds[step_name]
        adjusted = {}
        
        for measure, (lower, upper) in base.items():
            # For lower bounds, divide by relax_factor to make condition more lenient
            # For upper bounds, multiply by relax_factor to make condition more lenient
            adjusted[measure] = (lower / relax_factor, upper * relax_factor)
            
        return adjusted

def check_files(directory_path: str) -> bool:
    """
    Check for the presence of all required files in a directory.
    """

    # List of required files that need to be present in the directory.
    required_files = [
        "Accelerometer.csv",
        "Gravity.csv",
        "Gyroscope.csv",
        "Metadata.csv",
        "Orientation.csv",
        "Pedometer.csv",
        "Annotation.csv",
        "StudyMetadata.json",
    ]

    # Check each required file
    missing_files = [
        file for file in required_files if not os.path.exists(os.path.join(directory_path, file))
    ]

    # If there are any missing files, notify the user and return False
    if missing_files:
        print(f"The following required files are missing in {directory_path}:")
        for file in missing_files:
            print(f"- {file}")
        return False

    # If all files are present, print confirmation and return True.
    print("All required files are present.")
    return True


def load_metadata(recording_path: str) -> Dict[str, Any]:
    """
    Load and process metadata from a specified recording directory.
    """

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(recording_path, "*.csv"))

    # Load Metadata.csv
    metadata_df = pd.read_csv(os.path.join(recording_path, "Metadata.csv"))

    # Extract Device Name
    device_name = metadata_df.loc[0, "device name"]

    # Open and load the StudyMetadata.json file.
    with open(os.path.join(recording_path, "StudyMetadata.json"), "r") as f:
        study_metadata = json.load(f)

    # Extract the pseudonym
    pseudonym = study_metadata[0].get("value", study_metadata[0].get("title"))

    # Extract the number of recordings
    number_of_recordings = study_metadata[1].get("value", study_metadata[1].get("title"))

    # Return
    return {
        "csv_files": csv_files,
        "device_name": device_name,
        "pseudonym": pseudonym,
        "number_of_recordings": number_of_recordings,
    }


def load_data(files: List[str], recording_path: str) -> List[Tuple[pd.DataFrame, str]]:
    """
    Load CSV files into DataFrames and preprocess the data.

    Parameters:
    - files: List of CSV file names
    - recording_path: The directory where the CSV files are located

    Returns:
    - List of tuples containing preprocessed DataFrames and their file names
    """

    data = []

    # Iterate through the list of files
    for file in files:
        # Load each CSV file into a pandas DataFrame
        df = pd.read_csv(os.path.join(recording_path, file))

        # If columns "x", "y", and "z" exist, calculate the magnitude.
        if all(col in df.columns for col in ["x", "y", "z"]):
            df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

        data.append((df, file))

    return data


def plot_sensor_data(data: List[Tuple[pd.DataFrame, str]]) -> None:
    """
    Plot sensor data for each data stream of each sensor in a grid of subplots.

    Parameters:
    - data: List containing tuples of a DataFrame and its corresponding file name.
    """

    # Set up colors for each component
    colors = {
        "x": "#3498DB",  # Blue for x-axis (cool tone)
        "y": "#E74C3C",  # Red for y-axis (warm tone)
        "z": "#2ECC71",  # Green for z-axis (natural tone)
        "roll": "#3498DB",  # Blue for roll (same as x)
        "pitch": "#E74C3C",  # Red for pitch (same as y)
        "yaw": "#2ECC71",  # Green for yaw (same as z)
    }

    # Create a figure with subplots arranged in a 5x3 grid (5 sensors, 3 components each)
    num_sensors = len(data)
    fig, axs = plt.subplots(num_sensors, 3, figsize=(18, 4 * num_sensors))

    # Ensure axs is a 2D array even if there's only one sensor
    if num_sensors == 1:
        axs = [axs]

    # Plot each data stream in a separate subplot
    for sensor_idx, (df, file) in enumerate(data):
        components_to_plot = [
            col for col in ["x", "y", "z", "roll", "pitch", "yaw"] if col in df.columns
        ]

        for component_idx, component in enumerate(components_to_plot[:3]):
            ax = axs[sensor_idx][component_idx]
            ax.plot(
                df["seconds_elapsed"],
                df[component],
                label=f'{component.capitalize()} ({file.split(".")[0]})',
                color=colors.get(component, "#000000")  # Default to black if color not found
            )

            # Set labels, title, and legend for each subplot
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel(f"{component.capitalize()} Values")
            ax.set_title(f"Sensor Data: {file.split(".")[0]} - {component.capitalize()}")
            ax.grid(True, which="both", linestyle="--", alpha=0.7)
            ax.legend(loc="upper right")

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def quality_control_sensors(loaded_data: List[Tuple[pd.DataFrame, str]], 
                             min_sampling_rate: float = 10.0,
                             min_duration: float = 60.0) -> Tuple[bool, float]:
    """
    Check if sensor data meets minimum sampling rate requirement.
    
    Parameters:
    -----------
    loaded_data : List[Tuple[pd.DataFrame, str]]
        List of tuples containing DataFrames and their file names
    min_sampling_rate : float, optional
        Minimum acceptable sampling rate in Hz (default 10 Hz)
    min_duration : float, optional
        Minimum duration of data to check in seconds (default 60 seconds)
    
    Returns:
    --------
    Tuple[bool, float]
        - First value: True if all sensors meet the sampling rate requirement, False otherwise
        - Second value: Maximum time gap found across all sensors
    """
    max_gap_overall = 0.0
    max_acceptable_interval = 1 / min_sampling_rate
    
    for df, _ in loaded_data:
        # Sort by seconds_elapsed to ensure chronological order
        time_values = df['seconds_elapsed']
        
        # Check recording duration
        recording_duration = time_values.max() - time_values.min()
        if recording_duration < min_duration:
            return False, max_gap_overall
        
        # Calculate time differences
        time_diffs = time_values.diff().dropna()
        
        # Find maximum gap for this sensor
        max_gap = time_diffs.max()
        max_gap_overall = max(max_gap_overall, max_gap)
        
        # Check if any time gap exceeds the maximum acceptable interval
        if max_gap > max_acceptable_interval:
            return False, max_gap_overall
    
    return True, max_gap_overall


def detect_steps(
    data: List[Tuple[pd.DataFrame, str]], 
    search_start_time: float, 
    steps_to_relax: List[int] = None, 
    relax_factor: float = 1.0
) -> Dict[str, Tuple[float, float]]:
    """
    Detect periods of Steps 1 to 6 in accelerometer, gyroscope, and gravity data.

    Parameters:
    - data: List of tuples containing preprocessed DataFrames and their file names
    - search_start_time: Start time (in seconds) to begin searching for steps
    - steps_to_relax: List of step numbers to relax detection conditions
    - relax_factor: Factor to relax the detection conditions

    Returns:
    - Dictionary with keys 'step1' to 'step6', each containing a tuple with start and end times
    """
    
    # Extract accelerometer, gyroscope and gravity data
    acc_data = next(df for df, name in data if "Accelerometer" in name)
    gyro_data = next(df for df, name in data if "Gyroscope" in name)
    gravity_data = next(df for df, name in data if "Gravity" in name)

    # Sort data by time
    for df in [acc_data, gyro_data, gravity_data]:
        df.sort_values("seconds_elapsed", inplace=True)

    # Interpolate data to a common time scale
    common_time = np.union1d(
        np.union1d(acc_data["seconds_elapsed"], gyro_data["seconds_elapsed"]),
        gravity_data["seconds_elapsed"],
    )
    common_time = common_time[common_time >= search_start_time]

    # Interpolate all signals
    acc_mag = np.interp(common_time, acc_data["seconds_elapsed"], acc_data["magnitude"])
    gyro_mag = np.interp(common_time, gyro_data["seconds_elapsed"], gyro_data["magnitude"])
    gravity_x = np.interp(common_time, gravity_data["seconds_elapsed"], gravity_data["x"])
    gravity_y = np.interp(common_time, gravity_data["seconds_elapsed"], gravity_data["y"])
    gravity_z = np.interp(common_time, gravity_data["seconds_elapsed"], gravity_data["z"])

    # Smooth signals
    acc_mag_smooth = np.convolve(acc_mag, np.ones(5) / 5, mode='same')
    gyro_mag_smooth = np.convolve(gyro_mag, np.ones(5) / 5, mode='same')

    # Initialize step conditions
    step_conditions = StepConditions()
    step_masks = {}
        
    # Create masks for each step
    for step_num in range(1, 7):
        step_name = f'step{step_num}'
        current_relax_factor = relax_factor if steps_to_relax and step_num in steps_to_relax else 1.0
        
        # Get adjusted thresholds for current step
        thresholds = step_conditions.get_adjusted_thresholds(step_name, current_relax_factor)
        
        # Print thresholds for this step
        # print(f"\nStep {step_num} Thresholds:")
        # for measure, (lower, upper) in thresholds.items():
        #     print(f"  {measure}: [{lower:.3f}, {upper:.3f}]")

        # Create mask using adjusted thresholds
        step_masks[step_name] = (
            (acc_mag >= thresholds['acc_mag'][0]) & (acc_mag < thresholds['acc_mag'][1]) &
            (gyro_mag >= thresholds['gyro_mag'][0]) & (gyro_mag < thresholds['gyro_mag'][1]) &
            (gravity_x > thresholds['gravity_x'][0]) & (gravity_x < thresholds['gravity_x'][1]) &
            (gravity_y > thresholds['gravity_y'][0]) & (gravity_y < thresholds['gravity_y'][1]) &
            (gravity_z > thresholds['gravity_z'][0]) & (gravity_z < thresholds['gravity_z'][1])
        )
    
    def find_periods(mask: np.ndarray, min_duration: float) -> List[Tuple[int, int]]:
        """
        Find continuous periods where the mask is True for at least min_duration seconds.

        Parameters:
        - mask: Boolean array indicating where the condition is met
        - min_duration: Minimum duration of step period in seconds

        Returns:
        - List of tuples, each containing start and end indices of a period
        """
        periods = []
        start_idx = None

        for i, is_valid in enumerate(mask):
            if is_valid and start_idx is None:
                # Start of a new period
                start_idx = i
            elif not is_valid and start_idx is not None:
                # End of a period
                if common_time[i] - common_time[start_idx] >= min_duration:
                    periods.append((start_idx, i))
                start_idx = None

        # Handle case where valid period continues to the end
        if start_idx is not None and common_time[-1] - common_time[start_idx] >= min_duration:
            periods.append((start_idx, len(common_time) - 1))

        return periods
    
    def find_best_window_stillness(data: np.ndarray, start_idx: int, end_idx: int, window_size: float, min_start_time: float = None) -> Tuple[int, int]:
        """Find the best window of exactly window_size seconds with the lowest variance in data within the given period."""
        best_start = None
        min_variance = float("inf")
        
        # If min_start_time is provided, adjust start_idx
        if min_start_time is not None:
            start_idx = max(start_idx, np.searchsorted(common_time, min_start_time, side="right"))

        for i in range(start_idx, end_idx):
            end_time = common_time[i] + window_size
            window_end = np.searchsorted(common_time, end_time, side="right")
            if window_end > end_idx or (common_time[window_end - 1] - common_time[i]) != window_size:
                continue

            window_data = data[i:window_end]
            variance = np.var(window_data)
            if variance < min_variance:
                min_variance = variance
                best_start = i

        if best_start is None:
            return start_idx, start_idx + int(window_size * (len(common_time) / (common_time[-1] - common_time[0])))

        best_end = np.searchsorted(common_time, common_time[best_start] + window_size, side="right")
        return best_start, int(min(best_end, end_idx))

    def find_best_window_variance_range(
        data: np.ndarray, start_idx: int, end_idx: int, window_size: float, 
        min_var: float = 0.2, max_var: float = 5.0, min_start_time: float = None
    ) -> Tuple[int, int]:
        """Find a window with variance within a specified range in the given period."""
        # If min_start_time is provided, adjust start_idx
        if min_start_time is not None:
            start_idx = max(start_idx, np.searchsorted(common_time, min_start_time, side="right"))
        
        best_start = start_idx
        valid_window_found = False

        for i in range(start_idx, end_idx):
            end_time = common_time[i] + window_size
            window_end = np.searchsorted(common_time, end_time, side="right")
            if window_end > end_idx:
                break

            window_data = data[i:window_end]
            variance = np.var(window_data)
            
            if min_var <= variance <= max_var:
                best_start = i
                valid_window_found = True
                break

        if not valid_window_found:
            return start_idx, end_idx
        
        best_end = np.searchsorted(common_time, common_time[best_start] + window_size, side="right")
        return best_start, int(min(best_end, end_idx))

    def find_best_window_gyro_high_accel_low(
        gyro_data: np.ndarray, accel_data: np.ndarray, start_idx: int, end_idx: int, 
        window_size: float, min_start_time: float = None
    ) -> Tuple[int, int]:
        """Find the best window where gyroscope variance is high and accelerometer variance is low."""
        # If min_start_time is provided, adjust start_idx
        if min_start_time is not None:
            start_idx = max(start_idx, np.searchsorted(common_time, min_start_time, side="right"))
        
        gyro_min_variance = 2.0
        accel_max_variance = 0.5
        best_start = None

        for i in range(start_idx, end_idx):
            end_time = common_time[i] + window_size
            window_end = np.searchsorted(common_time, end_time, side="right")
            if window_end > end_idx or (common_time[window_end - 1] - common_time[i]) != window_size:
                continue

            gyro_window_data = gyro_data[i:window_end]
            accel_window_data = accel_data[i:window_end]
            gyro_variance = np.var(gyro_window_data)
            accel_variance = np.var(accel_window_data)

            if gyro_variance >= gyro_min_variance and accel_variance <= accel_max_variance:
                best_start = i
                break

        if best_start is None:
            return start_idx, start_idx + int(window_size * (len(common_time) / (common_time[-1] - common_time[0])))

        best_end = np.searchsorted(common_time, common_time[best_start] + window_size, side="right")
        return best_start, int(min(best_end, end_idx))

    # Find periods for each step
    step1_periods = find_periods(step_masks["step1"], min_duration=10.0)
    step2_periods = find_periods(step_masks["step2"] , min_duration=10.0)
    step3_periods = find_periods(step_masks["step3"], min_duration=10.0)
    step4_periods = find_periods(step_masks["step4"], min_duration=10.0)
    step5_periods = find_periods(step_masks["step5"], min_duration=4.0)
    step6_periods = find_periods(step_masks["step6"], min_duration=4.0)
    
    selected_periods = {}
    last_end_time = 0  # Track the end time of the last selected window
    
    # Process steps sequentially
    step_configs = [
        (1, step1_periods, acc_mag, 10.0, find_best_window_stillness),
        (2, step2_periods, acc_mag, 10.0, find_best_window_stillness),
        (3, step3_periods, (gyro_mag, acc_mag), 10.0, lambda d, s, e, w, m: find_best_window_gyro_high_accel_low(d[0], d[1], s, e, w, m)),
        (4, step4_periods, (gyro_mag, acc_mag), 10.0, lambda d, s, e, w, m: find_best_window_gyro_high_accel_low(d[0], d[1], s, e, w, m)),    
        (5, step5_periods, acc_mag_smooth, 10.0, lambda d, s, e, w, m: find_best_window_variance_range(d, s, e, w, min_start_time=m)),
        (6, step6_periods, gyro_mag_smooth, 10.0, lambda d, s, e, w, m: find_best_window_variance_range(d, s, e, w, min_start_time=m))
    ]

    for step, periods, data, window_size, find_func in step_configs:
        best_period = None
        min_start_time = last_end_time  # Use the last end time as minimum start time

        for start, end in periods:
            # Skip periods that end before the minimum start time
            if common_time[end] <= min_start_time:
                continue
                
            # Find the best window in this period, considering the minimum start time
            best_start, best_end = find_func(data, start, end, window_size, min_start_time)
            
            # Only consider this window if it starts after the last selected window
            if common_time[best_start] > last_end_time:
                best_period = (common_time[best_start], common_time[best_end])
                break  # Take the first valid window we find

        if best_period is not None:
            selected_periods[f"step{step}"] = best_period
            last_end_time = best_period[1]
        else:
            selected_periods[f"step{step}"] = None

    return selected_periods
"""
    # Find best windows within each period
    all_periods = []
    for step, periods, data, window_size, find_func in [
        (1, step1_periods, acc_mag, 10.0, find_best_window_stillness),
        (2, step2_periods, acc_mag, 10.0, find_best_window_stillness),
        (3, step3_periods, (gyro_mag, acc_mag), 10.0, lambda d, s, e, w: find_best_window_gyro_high_accel_low(d[0], d[1], s, e, w)),
        (4, step4_periods, (gyro_mag, acc_mag), 10.0, lambda d, s, e, w: find_best_window_gyro_high_accel_low(d[0], d[1], s, e, w)),    
        (5, step5_periods, acc_mag_smooth, 10.0, lambda d, s, e, w: find_best_window_variance_range(d, s, e, w)),
        (6, step6_periods, gyro_mag_smooth, 10.0, lambda d, s, e, w: find_best_window_variance_range(d, s, e, w)),
    ]:
        for start, end in periods:
            best_start, best_end = find_func(data, start, end, window_size)
            all_periods.append((step, common_time[best_start], common_time[best_end]))

    # Sort periods by start time
    all_periods.sort(key=lambda x: x[1])

    # Select one period per step, ensuring correct order and no overlap
    selected_periods = {}
    current_step = 1
    last_end_time = 0

    for step, start, end in all_periods:
        if step == current_step and f"step{step}" not in selected_periods and start > last_end_time:
            selected_periods[f"step{step}"] = (start, end)
            last_end_time = end
            current_step += 1
        if current_step > 6:
            break

    # Ensure all steps are present, even if some are missing
    for step in range(1, 7):
        if f"step{step}" not in selected_periods:
            selected_periods[f"step{step}"] = None

    return selected_periods
"""


def plot_steps_data(data: List[Tuple[pd.DataFrame, str]], steps: Dict[str, Tuple[float, float]]):
    """
    Plot the sensor data and highlight Step 1 to Step 6 periods.

    Parameters:
    - data: List of tuples containing preprocessed DataFrames and their file names
    - steps: Dictionary with 'step1' to 'step6' keys, each containing a tuple with start and end times of detected periods
    """
    print("-" * 35)
    print("Step      Start    End")
    print("-" * 35)
    for step, period in steps.items():
        if period:
            print(f"{step:9} {period[0]:7.2f} {period[1]:7.2f}")
        else:
            print(f"{step:9} {'-':7} {'-':7}")
    print("-" * 35)

    acc_data = next(df for df, name in data if "Accelerometer" in name)
    gyro_data = next(df for df, name in data if "Gyroscope" in name)
    gravity_data = next(df for df, name in data if "Gravity" in name)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot accelerometer data
    ax1.plot(acc_data["seconds_elapsed"], acc_data["magnitude"], label="Magnitude")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Accelerometer Data")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot gyroscope data
    ax2.plot(gyro_data["seconds_elapsed"], gyro_data["magnitude"], label="Magnitude")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Gyroscope Data")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Plot gravity data
    ax3.plot(gravity_data["seconds_elapsed"], gravity_data["x"], label="X")
    ax3.plot(gravity_data["seconds_elapsed"], gravity_data["y"], label="Y")
    ax3.plot(gravity_data["seconds_elapsed"], gravity_data["z"], label="Z")
    ax3.set_xlabel("Seconds Elapsed")
    ax3.set_ylabel("Gravity")
    ax3.set_title("Gravity Data")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Adjust x-axis ticks to have more frequent marks
    tick_interval = 5.0  # Set your desired interval here
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

    # Highlight Step periods
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#FF5733", "#C70039"]
    for i, (step, period) in enumerate(steps.items()):
        if period:
            start, end = period
            color = colors[i]
            ax1.axvspan(start, end, alpha=0.3, color=color)
            ax2.axvspan(start, end, alpha=0.3, color=color)
            ax3.axvspan(start, end, alpha=0.3, color=color)

            # Annotate start and end times
            ax1.text(
                start,
                acc_data["magnitude"].max(),
                f"{start:.1f}s",
                color=color,
                verticalalignment="bottom",
            )
            ax1.text(
                end,
                acc_data["magnitude"].max(),
                f"{end:.1f}s",
                color=color,
                verticalalignment="bottom",
            )
            ax2.text(
                start,
                gyro_data["magnitude"].max(),
                f"{start:.1f}s",
                color=color,
                verticalalignment="bottom",
            )
            ax2.text(
                end,
                gyro_data["magnitude"].max(),
                f"{end:.1f}s",
                color=color,
                verticalalignment="bottom",
            )
            ax3.text(
                start,
                gravity_data[["x", "y", "z"]].max().max(),
                f"{start:.1f}s",
                color=color,
                verticalalignment="bottom",
            )
            ax3.text(
                end,
                gravity_data[["x", "y", "z"]].max().max(),
                f"{end:.1f}s",
                color=color,
                verticalalignment="bottom",
            )

    # Add legend for steps
    for i, (step, color) in enumerate(
        zip(["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"], colors)
    ):
        ax1.axvspan(0, 0, alpha=0.3, color=color, label=step)

    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def create_interactive_step_plot(
    data: List[Tuple[pd.DataFrame, str]], 
    steps: Dict[str, Tuple[float, float]]
):
    """
    Create an interactive plot with sliders to adjust step times.
    Steps 1-4 have only start time sliders with fixed 10-second duration.
    Steps 5-6 have both start and end time sliders.

    Parameters:
    - data: List of tuples containing preprocessed DataFrames and their file names
    - steps: Dictionary with 'step1' to 'step6' keys, each containing a tuple with start and end times of detected periods
    """
    
    # Store original steps for potential reset
    original_steps = steps.copy()
    
    # Ensure all steps exist with default values if not detected
    all_steps = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6']
    default_start = 5
    for step in all_steps:
        if step not in steps or steps[step] is None:
            if step == 'step1':
                steps[step] = (default_start, default_start + 10)  # 10-second duration for step 1
                default_start += 15  # Space out default positions
            elif step == 'step2':
                steps[step] = (default_start, default_start + 10)  # 10-second duration for step 2
                default_start += 25  # 25-second gap between step 2 and step 3
            elif step == 'step3':
                steps[step] = (default_start, default_start + 10)  # 10-second duration for step 3
                default_start += 15  # Space out default positions
            elif step == 'step4':
                steps[step] = (default_start, default_start + 10)  # 10-second duration for step 4
                default_start += 25  # 25-second gap between step 4 and step 5
            elif step == 'step5':
                steps[step] = (default_start, default_start + 5)   # 5-second duration for step 5
                default_start += 5  # 5-second gap between step 5 and step 6
            elif step == 'step6':
                steps[step] = (default_start, default_start + 5)   # 5-second duration for step 6
                default_start += 15  # Space out default positions
    
    # Extract data for different sensors
    acc_data = next(df for df, name in data if "Accelerometer" in name)
    gyro_data = next(df for df, name in data if "Gyroscope" in name)
    gravity_data = next(df for df, name in data if "Gravity" in name)

    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14))
    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')
    plt.subplots_adjust(bottom=0.35, top=0.95, hspace=0.1)
    
    # Plot sensor data
    l1, = ax1.plot(acc_data["seconds_elapsed"], acc_data["magnitude"], label="Magnitude")
    ax1.set_ylabel("Accelerometer Magnitude")
    ax1.set_title("Accelerometer Data")
    ax1.grid(True, linestyle="--", alpha=0.7)

    l2, = ax2.plot(gyro_data["seconds_elapsed"], gyro_data["magnitude"], label="Magnitude")
    ax2.set_ylabel("Gyroscope Magnitude")
    ax2.set_title("Gyroscope Data")
    ax2.grid(True, linestyle="--", alpha=0.7)

    l3x, = ax3.plot(gravity_data["seconds_elapsed"], gravity_data["x"], label="X")
    l3y, = ax3.plot(gravity_data["seconds_elapsed"], gravity_data["y"], label="Y")
    l3z, = ax3.plot(gravity_data["seconds_elapsed"], gravity_data["z"], label="Z")
    ax3.set_xlabel("Seconds Elapsed")
    ax3.set_ylabel("Gravity")
    ax3.set_title("Gravity Data")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Colors for step highlighting
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#FF5733", "#C70039"]

    # Store step highlighting rectangles
    step_rects = {
        'acc': [],
        'gyro': [],
        'gravity': []
    }

    # Initial step rectangles
    for i, (step, period) in enumerate(steps.items()):
        if period:
            start, end = period
            color = colors[i]
            
            # Add rectangles for each sensor plot
            step_rects['acc'].append(ax1.axvspan(start, end, alpha=0.3, color=color, label=f"Step {i+1}"))
            step_rects['gyro'].append(ax2.axvspan(start, end, alpha=0.3, color=color))
            step_rects['gravity'].append(ax3.axvspan(start, end, alpha=0.3, color=color))

    ax1.legend()

    # Determine data time range
    max_time = max(
        acc_data["seconds_elapsed"].max(), 
        gyro_data["seconds_elapsed"].max(), 
        gravity_data["seconds_elapsed"].max()
    )

    # Create sliders
    sliders = []
    fixed_duration_steps = ['step1', 'step2', 'step3', 'step4']
    variable_duration_steps = ['step5', 'step6']

    # Create sliders for fixed duration steps (only start time)
    for i, step in enumerate(fixed_duration_steps):
        if not steps[step]:
            continue

        start, _ = steps[step]
        
        # Create axis for start slider
        ax_start = plt.axes([0.1 + (i % 2) * 0.45, 0.24 - (i // 2) * 0.04, 0.35, 0.03])

        # Create start time slider
        start_slider = widgets.Slider(
            ax_start, 
            f'{step} Start', 
            0, 
            max_time - 10,  # Ensure there's room for 10-second duration
            valinit=start,
            valstep=0.1
        )
        
        sliders.append({'start': start_slider, 'end': None})

    # Create sliders for variable duration steps (both start and end times)
    for i, step in enumerate(variable_duration_steps):
        if not steps[step]:
            continue

        start, end = steps[step]
        
        # Create axes for start and end sliders
        ax_start = plt.axes([0.1 + (i % 2) * 0.45, 0.16 - (i // 2) * 0.08, 0.35, 0.03])
        ax_end = plt.axes([0.1 + (i % 2) * 0.45, 0.12 - (i // 2) * 0.08, 0.35, 0.03])
        
        # Create start and end time sliders
        start_slider = widgets.Slider(
            ax_start, 
            f'{step} Start', 
            0, 
            max_time, 
            valinit=start,
            valstep=0.1
        )
        
        end_slider = widgets.Slider(
            ax_end, 
            f'{step} End', 
            0, 
            max_time, 
            valinit=end,
            valstep=0.1
        )
        
        sliders.append({'start': start_slider, 'end': end_slider})

    # Add buttons
    done_ax = plt.axes([0.8, 0.04, 0.1, 0.04])
    skip_ax = plt.axes([0.65, 0.04, 0.1, 0.04])
    detect_again_ax = plt.axes([0.5, 0.04, 0.1, 0.04])
    relax_detection_ax = plt.axes([0.35, 0.04, 0.1, 0.04])
    cancel_step_ax = plt.axes([0.20, 0.04, 0.1, 0.04])
    
    done_button = widgets.Button(done_ax, 'Done')
    skip_button = widgets.Button(skip_ax, 'Skip')
    detect_again_button = widgets.Button(detect_again_ax, 'Detect Again')
    relax_detection_button = widgets.Button(relax_detection_ax, 'Relax')
    cancel_step_button = widgets.Button(cancel_step_ax, 'Cancel Step')
    
    # Help variables
    new_start_time = [None]    
    steps_to_relax = [None]
    skip = False
    steps_to_cancel = [None]
    

    def update(val):
        # Update step rectangles based on slider values
        for i, (step, period) in enumerate(steps.items()):
            if not period or i >= len(sliders):
                continue

            slider_pair = sliders[i]
            new_start = slider_pair['start'].val
            
            # For steps 1-4, automatically set end time to start + 10
            if step in fixed_duration_steps:
                new_end = new_start + 10
            else:
                new_end = slider_pair['end'].val
                # Ensure end time is after start time
                if new_end <= new_start:
                    new_end = new_start + 0.1
                    slider_pair['end'].set_val(new_end)

            # Shift subsequent steps if needed
            min_gap = 0.1  # Minimum gap between steps
            for j in range(i + 1, len(sliders)):
                next_slider = sliders[j]
                if new_end + min_gap > next_slider['start'].val:
                    # Calculate how much we need to shift
                    shift_amount = new_end + min_gap - next_slider['start'].val
                    
                    # Update the next step's start time
                    next_start = next_slider['start'].val + shift_amount
                    next_slider['start'].set_val(next_start)
                    
                    # Update the next step's end time
                    if step in fixed_duration_steps:
                        next_end = next_start + 10
                    else:
                        next_end = next_slider['end'].val + shift_amount
                    next_slider['end'].set_val(next_end)
                    
                    # Update the new_end for the next iteration
                    new_end = next_end

            # Remove old rectangles
            for rect_list in step_rects.values():
                if i < len(rect_list):  # Check if rectangle exists
                    rect_list[i].remove()

            # Add new rectangles
            step_rects['acc'][i] = ax1.axvspan(new_start, new_end, alpha=0.3, color=colors[i], label=f"Step {i+1}")
            step_rects['gyro'][i] = ax2.axvspan(new_start, new_end, alpha=0.3, color=colors[i])
            step_rects['gravity'][i] = ax3.axvspan(new_start, new_end, alpha=0.3, color=colors[i])

            # Update steps dictionary
            steps[step] = (new_start, new_end)

        # Update legend
        ax1.legend()
        fig.canvas.draw_idle()

    def on_done(event):
        plt.close()

    def on_skip(event):
        plt.close()
        steps.clear()
        skip = True
        
    def on_detect_again(event):
        # Create a simple dialog for input
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Show an input dialog
        new_start = simpledialog.askfloat("Input", 
                                         "Enter new starting time (seconds):",
                                         minvalue=0,
                                         maxvalue=max_time)
        
        if new_start is not None:
            new_start_time[0] = new_start
            plt.close()
            steps.clear()
        else:
            new_start_time[0] = None
            exit("Bad input")
            
    def on_relax(event):
        # Create simple input dialog
        user_input = simpledialog.askstring(
            "Select Steps", 
            "Enter step numbers to relax (1-6)\nSeparate numbers with spaces or commas:",
            initialvalue="1 2"
        )
        
        if user_input:
            # Convert input string to list of numbers
            try:
                # Handle both comma and space separated inputs
                numbers = [int(x.strip()) for x in user_input.replace(',', ' ').split()]
                
                # Validate numbers are between 1 and 6
                valid_numbers = [n for n in numbers if 1 <= n <= 6]
                
                if valid_numbers:
                    steps_to_relax[0] = valid_numbers
                    plt.close()
                    steps.clear()
                else:
                    steps_to_relax[0] = None
                    exit("No valid numbers entered")
                
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Please enter valid numbers between 1 and 6"
                )
                
    def on_cancel_step(event):
        user_input = simpledialog.askstring(
            "Cancel Steps", 
            "Enter step numbers to cancel (1-6)\nSeparate numbers with spaces or commas:",
            initialvalue="3 5"
        )
        
        if user_input:
            # Convert input string to list of numbers
            try:
                # Handle both comma and space separated inputs
                numbers = [int(x.strip()) for x in user_input.replace(',', ' ').split()]
                
                # Validate numbers are between 1 and 6
                valid_numbers = [n for n in numbers if 1 <= n <= 6]
                
                if valid_numbers:
                    steps_to_cancel[0] = valid_numbers
                    plt.close()
                else:
                    steps_to_cancel[0] = None
                    exit("No valid numbers entered")
                
            except ValueError:
                messagebox.showerror(
                    "Invalid Input",
                    "Please enter valid numbers between 1 and 6"
                )
        

    # Attach update function to all sliders
    for slider_pair in sliders:
        slider_pair['start'].on_changed(update)
        if slider_pair['end'] is not None:
            slider_pair['end'].on_changed(update)

    done_button.on_clicked(on_done)
    skip_button.on_clicked(on_skip)
    detect_again_button.on_clicked(on_detect_again)
    relax_detection_button.on_clicked(on_relax)
    cancel_step_button.on_clicked(on_cancel_step)

    plt.show()
    
    # If steps dictionary is empty, it means skip was pressed
    if skip:
        return "skip"
    elif new_start_time[0] is not None:
        return ("detect_again", new_start_time[0])
    elif steps_to_relax[0] is not None:
        return ("relax", steps_to_relax[0])
    elif steps_to_cancel[0] is not None:
        return {step: duration for step, duration in steps.items() if step not in [f'step{n}' for n in steps_to_cancel[0]]}
    else:
        return steps

def correct_step_times(
    detected_steps: Dict[str, Optional[Tuple[float, float]]],
    corrections: List[Tuple[str, Optional[float], Optional[float]]] = [],
) -> Dict[str, Optional[Tuple[float, float]]]:
    """
    Apply corrections to the start and end times of detected steps.

    Parameters:
    - detected_steps: Dictionary output from detect_steps function
    - corrections: List of tuples, each containing:
        (step_name, new_start_time, new_end_time)
        Use None for new_start_time or new_end_time to keep the original value

    Returns:
    - Dictionary with corrected step times
    """
    corrected_steps = detected_steps.copy()

    for correction in corrections:
        step_name, new_start, new_end = correction

        if step_name not in corrected_steps:
            print(f"Warning: Step {step_name} not found in detected steps.")
            continue

        if corrected_steps[step_name] is None:
            if new_start is not None and new_end is not None:
                corrected_steps[step_name] = (new_start, new_end)
                print(f"Created new step {step_name}: ({new_start}, {new_end})")
            else:
                print(f"Warning: Cannot create {step_name} with partial information.")
            continue

        original_start, original_end = corrected_steps[step_name]

        if new_start is not None and new_end is not None:
            if new_start < new_end:
                corrected_steps[step_name] = (new_start, new_end)
                print(f"Fully corrected {step_name}: ({new_start}, {new_end})")
            else:
                print(f"Warning: Invalid correction for {step_name}. Start time must be less than end time.")
        elif new_start is not None:
            if new_start < original_end:
                corrected_steps[step_name] = (new_start, original_end)
                print(f"Corrected start time for {step_name}: ({new_start}, {original_end})")
            else:
                print(
                    f"Warning: Invalid start time correction for {step_name}. New start time must be less than original end time."
                )
        elif new_end is not None:
            if new_end > original_start:
                corrected_steps[step_name] = (original_start, new_end)
                print(f"Corrected end time for {step_name}: ({original_start}, {new_end})")
            else:
                print(
                    f"Warning: Invalid end time correction for {step_name}. New end time must be greater than original start time."
                )

    return corrected_steps

def segment_data(
    data: List[Tuple[pd.DataFrame, str]],
    steps: Dict[str, Tuple[float, float]],
    sensor_names: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Segment data according to the steps and sensor types provided, organizing streams by step.
    All measurements (x, y, z, magnitude, azimuth, inclination, inter-sample time) are aligned
    in a single DataFrame for each sensor type.

    Parameters:
        data (list): List of tuples containing DataFrames and their corresponding sensor names.
        steps (dict): Detected steps with start and end times.
        sensor_names (list): List of sensor types to segment.

    Returns:
        dict: A nested dictionary where the first level contains steps and the second level contains
              sensor measurements in aligned DataFrames.
    """
    # Initialize dictionary to store segmented data
    segments = {}

    # Iterate over each step and time period
    for step, period in steps.items():
        # Initialize dictionary for current step
        segments[step] = {}
        
        # Only segment if the step has a valid period
        if period:
            start, end = period
            
            # Extract data for each sensor in the specified time period
            accel_data = None
            gravity_data = None
            total_accel_data = None
            gyro_data = None
            
            for df, name in data:
                # Process only specified sensors
                if name in sensor_names:
                    # Segment the data within the time period
                    segmented_data = df[(df["seconds_elapsed"] >= start) & (df["seconds_elapsed"] < end)].copy()
                    
                    # Calculate inter-sample time
                    segmented_data["inter_sample_time"] = segmented_data["seconds_elapsed"].diff()
                    
                    # Identify sensor type and store the corresponding data
                    if "Accelerometer.csv" in name:
                        accel_data = segmented_data
                    elif "Gravity.csv" in name:
                        gravity_data = segmented_data
                    elif "Gyroscope.csv" in name:
                        gyro_data = segmented_data
                    elif "TotalAcceleration.csv" in name:
                        total_accel_data = segmented_data
            
            # Store acceleration data if available
            if accel_data is not None:
                # Calculate azimuth and inclination
                accel_data['azimuth'] = np.arctan2(accel_data["y"], accel_data["x"]) * (180 / np.pi)
                accel_data['inclination'] = np.arccos(accel_data["z"] / accel_data["magnitude"]) * (180 / np.pi)

                # Store all measurements in a single DataFrame
                segments[step]['accelerometer'] = accel_data[[
                    "seconds_elapsed",
                    "x", "y", "z",
                    "magnitude",
                    "azimuth",
                    "inclination",
                    "inter_sample_time"
                ]]

            # Store gravity data if available
            if gravity_data is not None:
                # Calculate azimuth and inclination
                gravity_data['azimuth'] = np.arctan2(gravity_data["y"], gravity_data["x"]) * (180 / np.pi)
                gravity_data['inclination'] = np.arccos(gravity_data["z"] / gravity_data["magnitude"]) * (180 / np.pi)

                # Store all measurements in a single DataFrame
                segments[step]['gravity'] = gravity_data[[
                    "seconds_elapsed",
                    "x", "y", "z",
                    "magnitude",
                    "azimuth",
                    "inclination",
                    "inter_sample_time"
                ]]
                
            # Store total accel data if available
            if total_accel_data is not None:
                # Calculate azimuth and inclination
                total_accel_data['azimuth'] = np.arctan2(total_accel_data["y"], total_accel_data["x"]) * (180 / np.pi)
                total_accel_data['inclination'] = np.arccos(total_accel_data["z"] / total_accel_data["magnitude"]) * (180 / np.pi)

                # Store all measurements in a single DataFrame
                segments[step]['total_acceleration'] = total_accel_data[[
                    "seconds_elapsed",
                    "x", "y", "z",
                    "magnitude",
                    "azimuth",
                    "inclination",
                    "inter_sample_time"
                ]]

            # Store gyroscope data if available (without azimuth/inclination)
            if gyro_data is not None:
                segments[step]['gyroscope'] = gyro_data[[
                    "seconds_elapsed",
                    "x", "y", "z",
                    "magnitude",
                    "inter_sample_time"
                ]]

    return segments


def create_and_save_segments(
    segments: Dict[str, pd.DataFrame], base_dir: str, output_dir: str, metadata: Dict[str, Any]
):
    """
    Create directories and save each segment of data into its corresponding directory,
    naming files using metadata from a JSON file.

    Parameters:
        segments (dict): Dictionary containing dataframes to be saved.
        base_dir (str): Base directory path to store the segments.
        output_dir (str): Directory path to store the segmented data.
        metadata_path (str): Path to the JSON metadata file containing pseudonym and recording.
    """

    pseudonym = metadata["pseudonym"]

    # Get the name of the recording from the base directory
    recording_name = os.path.basename(base_dir)

    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each segment and save to appropriate directory
    for segment_key, data in segments.items():
        # Extract step name from segment key
        step_name = segment_key.split("-")[0]

        # Create directory for each step if it doesn't exist
        step_dir = os.path.join(output_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)

        # Generate the filename
        filename = f"{pseudonym}-{recording_name}-{segment_key}.csv"
        file_path = os.path.join(step_dir, filename)

        # Save the DataFrame to a CSV file
        data.to_csv(file_path, index=False)
        print(f"Saved {segment_key} data to {file_path}")


def save_segmented_data(
    segments: Dict[str, Dict[str, Dict[str, pd.DataFrame]]], 
    base_dir: str, 
    output_dir: str, 
    metadata: Dict[str, Any]
) -> Dict[str, str]:
    """
    Save all sensor streams organized by settings (on hand, on desk, etc.).

    Parameters:
        segments (dict): Nested dictionary containing dataframes of segmented data.
                        Structure: {step: {sensor_type: {measurement: DataFrame}}}
        base_dir (str): Base directory path of the recording.
        output_dir (str): Base directory path to store the segmented data.
        metadata (dict): Metadata containing pseudonym and other information.

    Returns:
        dict: Status dictionary indicating whether each step was processed successfully.
    """
    pseudonym = metadata["pseudonym"]
    recording_name = os.path.basename(base_dir)
    
    # Map step names to setting directories
    step_settings = {
        "step1": "on_hand",
        "step2": "on_desk",
        "step3": "on_hand_audio",
        "step4": "on_desk_audio",
        "step5": "walking_1",
        "step6": "walking_2"
    }
    
    step_status = {}  # Dictionary to keep track of the status of each step

    # Create base directories for each setting
    for setting in step_settings.values():
        setting_dir = os.path.join(output_dir, setting)
        os.makedirs(setting_dir, exist_ok=True)
    
    # Process each step
    for step, setting in step_settings.items():
        if step not in segments:
            step_status[step] = "No Data"
            continue
            
        try:
            # Get the directory for this setting and recording
            setting_dir = os.path.join(output_dir, setting, f"{setting} - {recording_name}")
            os.makedirs(setting_dir, exist_ok=True)
            
            # Process each sensor type (accelerometer, gravity, gyroscope)
            for sensor_type, stream in segments[step].items():
                # Create file path for each sensor type within the setting directory
                file_path = os.path.join(setting_dir, f"{sensor_type}.csv")
                
                # Combine all measurements for the sensor type into a single DataFrame
                sensor_data = pd.DataFrame()
                for stream_name, df in stream.items():
                    if sensor_data.empty:
                        sensor_data = df.copy()
                    else:
                        sensor_data = pd.concat([sensor_data, df.drop(columns=["seconds_elapsed"])], axis=1)
                
                # Save the combined DataFrame for this sensor type
                sensor_data.to_csv(file_path, index=False)
            
            step_status[step] = "Successful"
            
        except Exception as e:
            step_status[step] = f"Failed: {e}"
            
    return step_status


def load_or_create_tracking_file(tracking_file_path: str) -> pd.DataFrame:
    """
    Load existing tracking file or create a new one if it doesn't exist.
    """

    # Check if the tracking file exists
    if os.path.exists(tracking_file_path):
        # Load and return the existing file
        return pd.read_csv(tracking_file_path)
    else:
        # Create a new DataFrame with predefined columns if the file doesn't exist
        columns = [
            "recording_name",
            "device_name",
            "device_id",
            "processed",
            "all good",
            "error",
            "last_processed_date",
            "notes",
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(tracking_file_path, index=False)  # Save the new tracking fil
        return df


def update_tracking_file(
    tracking_df: pd.DataFrame,
    recording_path: str,
    metadata: Dict[str, Any],
    processed: bool,
    all_good: bool = True,
    error: bool = False,
    notes: str = "",
) -> pd.DataFrame:
    """
    Update the tracking file with information about the processed recording.
    
    Parameters:
        tracking_df (pd.DataFrame): The tracking DataFrame to update.
        recording_path (str): Path to the recording directory.
        metadata (dict): Metadata containing additional information.
        processed (bool): Whether the recording was processed successfully.
        all_good (bool): Flag indicating if everything went well (default: True).
        error (str): Description of any error that occurred (default: "").
        notes (str): Any additional notes about the processing (default: "").

    Returns:
        pd.DataFrame: Updated tracking DataFrame.
    """
    # Extract the recording name from the path
    recording_name = os.path.basename(recording_path)
    
    # Parse device_name and device_id from the recording name
    device_name, device_id = recording_name.split('_', 2)[:2]
    
    # Create a new row with updated tracking information
    new_row = pd.DataFrame(
        {
            "recording_name": [recording_name],
            "device_name": [device_name],
            "device_id": [device_id],
            "processed": [processed],
            "all good": [all_good],
            "error": [error],
            "last_processed_date": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
            "notes": [notes],
        }
    )

    # Check if the recording already exists in the tracking file
    existing_row = tracking_df[tracking_df["recording_name"] == recording_name]
    if len(existing_row) > 0:
        # If it exists, update the existing row
        index = existing_row.index[0]
        tracking_df.loc[index] = new_row.iloc[0]
    else:
        # If it doesn't exist, concatenate the new row to the DataFrame
        tracking_df = pd.concat([tracking_df, new_row], ignore_index=True)

    return tracking_df


def get_user_corrections() -> List[Tuple[str, Optional[float], Optional[float]]]:
    """
    Interactively ask the user for step corrections.

    Returns:
        List of tuples containing step name and optional start and end times.
    """
    corrections = []
    valid_steps = {
        "step1",
        "step2",
        "step3",
        "step4",
        "step5",
        "step6",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    }

    print("Enter corrections for each step. Press Enter without input to skip a step.")

    while True:
        step_name = input("Enter step name (or 'q' to finish): ").strip().lower()
        if step_name == "q":
            break
        elif step_name == "skip":
            return "skip"

        if step_name not in valid_steps:
            print("Invalid step name. Please enter a valid step name.")
            continue

        # Convert numeric input to 'step' format
        if step_name.isdigit():
            step_name = f"step{step_name}"

        start_time = input(f"Enter start time for {step_name}: ").strip()
        if not start_time:
            print("Start time is required. Please enter a valid start time.")
            continue

        try:
            start_time = float(start_time)
        except ValueError:
            print("Invalid start time. Please enter a valid number.")
            continue

        if step_name in {"step1", "step2", "step3", "step4"}:
            end_time = start_time + 10
            print(f"End time for {step_name} automatically set to {end_time}")
        else:
            end_time = input(f"Enter end time for {step_name}: ").strip()
            if not end_time:
                end_time = None
            else:
                try:
                    end_time = float(end_time)
                except ValueError:
                    print("Invalid end time. Please enter a valid number.")
                    continue

        corrections.append((step_name, start_time, end_time))

    return corrections

def skip_recording(recording_path):
    """
    Move a recording to a separate directory and update the tracking file to mark it as skipped.
    
    Parameters:
    -----------
    recording_path : str
        Path to the directory containing the recordings
    """
    
    # Pfad zu dem Ordner, in den "skipped" Recordings verschoben werden sollen
    SKIP_FOLDER = "../Data/raw data/skipped recordings"
    
    # Prüfen, ob der Zielordner existiert, ansonsten erstellen
    if not os.path.exists(SKIP_FOLDER):
        os.makedirs(SKIP_FOLDER)
            
    # Move the recording to the skipped directory
    new_path = os.path.join(SKIP_FOLDER, os.path.basename(recording_path))
    shutil.move(recording_path, new_path)
    print(f"Recording has been moved to {SKIP_FOLDER}.")    
    
    print(f"Skipped recording: {os.path.basename(recording_path)}")

def close_all_plots():
    """
    Close all open matplotlib plots and clear the current figure.
    This function ensures that all plot windows are closed and memory is freed.
    """
    # Close all figure windows
    plt.close("all")

    # Clear the current figure
    plt.clf()

    # Clear the current axes
    plt.cla()

    # Reset the plot parameters to default
    plt.rcParams.update(plt.rcParamsDefault)

    # Close all figure windows
    plt.close("all")

    # If you're using IPython or Jupyter Notebook, you might also want to clear the output
    try:
        display.clear_output()
    except ImportError:
        print("Warning: display.clear_output() is not available. Please close the plot windows manually.")
        pass


def process_recording(
    recording_path: str, output_path: str, tracking_df: pd.DataFrame
) -> Tuple[pd.DataFrame, str]:
    """
    Process a single recording directory and update tracking information.
    """

    # Check if all required files are present, skip if files are missing
    if not check_files(recording_path):
        return tracking_df, f"Skipping {recording_path} due to missing files."

    # Load metadata for the recording
    metadata = load_metadata(recording_path)
    print(
        f"Pseudonym: {metadata['pseudonym']}, Device: {metadata['device_name']}, Recording: {metadata['number_of_recordings']}"
    )

    try:
        sensors = ["Accelerometer.csv", "Gyroscope.csv", "Gravity.csv", "TotalAcceleration.csv"]

        # Load the raw sensor data
        data = load_data(sensors, recording_path)
        
        # Check data quality
        is_data_good, max_gap = quality_control_sensors(data)
        
        if not is_data_good:
            if max_gap > 0.1:
                print(f"Data does not meet minimum sampling rate requirements.")
            else:
                print("Recording is too short")
            
            # Plot the sensor data for visual inspection
            plot_sensor_data(data)
            
            # Move the recording to the skipped directory        
            skip_recording(recording_path)
            
            # Update tracking file to mark the recording as skipped
            tracking_df = update_tracking_file(tracking_df, recording_path, metadata, processed=False, all_good=False, error=False, notes="skipped due to bad data quality")            
            
            # Close all plots
            close_all_plots()
            
            return tracking_df, "Recording skipped due to bad data quality."
        
        # Detect steps in the data
        search_start_time = 0
        relax_factor = 1.0 
        result = None
                
        while True:
            
            if isinstance(result, tuple) and result[0] == "relax":
                steps = None
                # Get the steps to relax
                steps_to_relax = result[1]
                # Increase relax_factor for specified steps
                steps = detect_steps(data, search_start_time=search_start_time,
                                steps_to_relax=steps_to_relax, relax_factor=2.5)
            else:
                steps = None
                # Normal detection
                steps = detect_steps(data, search_start_time=search_start_time)            

            # Ask user for corrections
            result = create_interactive_step_plot(data, steps)
            
            if result == "skip":
                # Move the recording to the skipped directory        
                skip_recording(recording_path)
                
                # Update tracking file to mark the recording as skipped
                tracking_df = update_tracking_file(tracking_df, recording_path, metadata, processed=False, all_good=False, error=False, notes="skipped due to user input")            
                
                # Close all plots
                close_all_plots()

                return tracking_df, "Recording skipped and moved successfully."
            elif isinstance(result, tuple) and result[0] == "detect_again":
                # User wants to detect again from a new starting point
                search_start_time = result[1]
                print(f"Detecting steps again from time: {search_start_time:.2f} seconds")
                continue
            elif isinstance(result, tuple) and result[0] == "relax":
                # User wants to detect specific steps again with relaxed conditions
                print(f"Detecting steps {result[1]} with relaxed conditions")
                continue
    
            else:
                # User is satisfied with the current detection/corrections
                corrected_steps = result
                break
        
        print("-" * 35)
        print("Step      Start    End")
        print("-" * 35)
        for step, period in corrected_steps.items():
            if period:
                print(f"{step:9} {period[0]:7.2f} {period[1]:7.2f}")
            else:
                print(f"{step:9} {'-':7} {'-':7}")
        print("-" * 35)

        # Confirm before proceeding
        confirmation = input("Please confirm before proceeding (yes/no): ").strip().lower()
        if confirmation not in ["yes", "y"]:
            print("Operation cancelled by the user.")
            exit()
        
        # Segment the data into streams for each step
        streams = segment_data(data, corrected_steps, sensors)

        # Save the data streams separated by settings
        step_status = save_segmented_data(streams, recording_path, output_path, metadata)
        
        # Update the tracking file
        tracking_df = update_tracking_file(tracking_df, recording_path, metadata, processed=True)

        # Close all plots
        close_all_plots()

        return tracking_df, "Processing completed successfully."

    except Exception as e:
        # Handle any errors
        error_message = f"Error processing {recording_path}: {str(e)}"
        print(f"Error details: {type(e).__name__}")
        print(traceback.format_exc())
        exit(1)


def process_all_recordings(base_directory: str, output_directory: str, control_file_path: str):
    """
    Process all recording directories within the base directory and track progress.
    """

    # Load or create the tracking file to track processing status
    control_df = load_or_create_tracking_file(control_file_path)

    # Iterate through all directories in the base directory
    for dir_name in os.listdir(base_directory):
        recording_path = os.path.join(base_directory, dir_name)

        # If found
        if os.path.isdir(recording_path):
            # Extract the recording name from the path
            recording_name = os.path.basename(recording_path)

            print(f"\nProcessing recording: {recording_name}")

            # Check if the recording has already been processed
            existing_record = control_df[control_df["recording_name"] == recording_name]
            if len(existing_record) > 0 and existing_record["processed"].iloc[0]:
                print(f"Skipping {recording_name} as it has already been processed.")
                continue

            # Process the recording and update the control DataFrame
            control_df, message = process_recording(recording_path, output_directory, control_df)
            print(message)
            print("-" * 40)

            # Save the updated tracking file after each recording
            control_df.to_csv(control_file_path, index=False)

    print("\nAll recordings processed. Check the tracking file for details.")


base_directory = "../Data/raw data/all data/"
output_directory = "../Data/raw data/separated by setting/"
control_file_path = "../Data/raw data/processing_progress.csv"
print("Comprobar el pfad")

process_all_recordings(base_directory, output_directory, control_file_path)
