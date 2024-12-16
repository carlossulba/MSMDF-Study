# RecordingSegmenter.py

from dataclasses import dataclass
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import json
import logging
import copy
import re
import scipy.signal
import shutil
from typing import List, Tuple, Dict


class StepThresholds:
    def __init__(self):
        # Define base thresholds for each step
        a = 1

    def get_adjusted_thresholds(self, step_name: str, relax_factor: float = 1.0) -> dict:
        """Get thresholds for a step, adjusted by relax factor if needed."""
        base = self.thresholds[step_name]
        adjusted = {}
        
        for measure, (lower, upper) in base.items():
            # For lower bounds, divide by relax_factor to make condition more lenient
            # For upper bounds, multiply by relax_factor to make condition more lenient
            adjusted[measure] = (lower / relax_factor, upper * relax_factor)
            
        return adjusted


@dataclass
class Recording:
    """ Class for storing information about a recording. """
    recording_path: str
    recording_name: str = "Unknown"	
    device_name: str = "Unknown"
    device_id: str = "Unknown"
    pseudonym: str = "Unknown"
    csv_files: List[str] = None
    data = List[Tuple[str, pd.DataFrame]]
    
    def __post_init__(self):
        if not os.path.exists(self.recording_path):
            raise ValueError(f"Recording directory '{self.recording_name}' does not exist.")
        
        pattern = r'^(.*?)_(.*?)-(20\d\d_\d\d_\d\d_\d\d_\d\d_\d\d)(.*)$'
        
        # get basename out of path
        match = re.match(pattern, os.path.basename(self.recording_path))
        if not match:
            # If it doesn't match, handle it gracefully
            raise ValueError(f"Recording name '{self.recording_path}' does not match expected pattern.")

        # Parse recording name
        device_name, device_id, datetime_str, remainder = match.groups()
        
        short_recording_name = remainder.strip("_- ")
        
        if short_recording_name:
            short_recording_name = f"{datetime_str} - {short_recording_name}"
        else:
            short_recording_name = datetime_str
          
        # Set the attributes  
        self.device_name = device_name
        self.device_id = device_id
        self.recording_name = short_recording_name
        
        
        self.csv_files = glob.glob(os.path.join(self.recording_path, "*.csv"))
        
        metadata_df = pd.read_csv(os.path.join(self.recording_path, "Metadata.csv"))
        with open(os.path.join(self.recording_path, "StudyMetadata.json"), "r") as f:
            study_metadata = json.load(f)
        self.pseudonym = study_metadata[0].get("value", study_metadata[0].get("title"))


class RecordingSegmenter:
    """
    Main class for segmenting a recording into the different steps.
    (on hand, on desk, on hand audio, on desk audio, walking 1, walking 2)
    """
    
    def __init__(self, base_directory: str, output_directory: str, control_file_path: str, skip_directory: str , log_level: str = 'INFO'):
        """
        Initialize the RecordingSegmenter object.
        
        Args:
        - base_directory: The directory containing the raw recording files
        - output_directory: The directory where the segmented files will be saved
        - control_file_path: The path to the control file
        """
        self.base_directory = base_directory
        self.output_directory = output_directory
        self.control_file_path = control_file_path
        self.control_file = None
        self.skip_directory = skip_directory
        
        self.max_acceptable_gap = 1 / 10  # min 10 Hz sampling rate
        self.max_duration = 360.0  # 6 minutes
        self.min_duration = 60.0  # 1 minute
        
        self._setup_logger(log_level)
        self.__post_init__()
    
    def _setup_logger(self, log_level: str):
        """Setup logging for the classifier trainer."""
        basic_format = '%(levelname)s - %(message)s'
        detailed_format = '%(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
        
        # Use detailed format for levels other than INFO
        log_format = basic_format if log_level.upper() == 'INFO' else detailed_format
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format=log_format
        )
        self.logger = logging.getLogger(__name__)

    def __post_init__(self):
        """ Validate initialization parameters and create necessary files. """
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory '{self.base_directory}' does not exist.")
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            self.logger.info(f"Created output directory: {self.output_directory}")
        else:
            self.logger.info(f"Using existing output directory: {self.output_directory}")
        
        if not os.path.exists(self.control_file_path):
            # create control file
            self._create_control_file()
            self.control_file = pd.read_csv(self.control_file_path)
            self.logger.info(f"Created control file: {self.control_file_path}")
        else:
            self.control_file = pd.read_csv(self.control_file_path)
            self.logger.info(f"Using existing control file: {self.control_file_path}")
    
    def _create_control_file(self):
        """ Create a control file to track the processing status of recordings. """
        # Define the columns for the control file
        columns = [
            "pseudonym",
            "device_id",
            "device_name",
            "recording_name",
            "processed",
            "notes",
            "last_update"
        ]
        
        # Create and Save the control file
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.control_file_path, index=False)
    
    def _update_control_file(self, recording: Recording, processed: bool, notes: str = ""):
        """ Update the control file with the processing status of a recording. """
        # Create the new row
        new_row = pd.DataFrame({
            "pseudonym": [recording.pseudonym],
            "device_id": [recording.device_id],
            "device_name": [recording.device_name],
            "recording_name": [recording.recording_name],
            "processed": [processed],
            "notes": [notes],
            "last_update": [pd.Timestamp.now().strftime("%Y-%m-%d")]
        })
        
        # Add or update the row in the control file
        existing_row = self.control_file[self.control_file["recording_name"] == recording.recording_name]
        if len(existing_row) > 0:
            # If it exists, update the existing row
            index = existing_row.index[0]
            self.control_file.loc[index] = new_row.iloc[0]
            self.logger.info(f"Updated control file for recording: {recording.recording_name}")
        else:
            # If it doesn't exist, concatenate the new row to the DataFrame
            self.control_file = pd.concat([self.control_file, new_row], ignore_index=True)
            self.logger.info(f"Added new entry to control file for recording: {recording.recording_name}")
            
        self.control_file.to_csv(self.control_file_path, index=False)

    def process_recordings(self):
        """ Process all recordings in the base directory and update the control file. """
        # Iterate through all directories in the base directory (each dir should be a recording)
        for dir_name in os.listdir(self.base_directory):
            recording = Recording(os.path.join(self.base_directory, dir_name))
            
            # Check if the recording has already been processed
            existing_record = self.control_file[self.control_file["recording_name"] == recording.recording_name]
            if len(existing_record) > 0 and existing_record["processed"].iloc[0]:
                self.logger.info(f"Skipping recording: {recording.recording_name} as it has already been processed.")
                continue
            
            self.logger.info(f"Processing recording: {recording.recording_name}")
            
            # Process the recording and update the control file
            self.process_recording(recording)
        
        self.logger.info("All recordings processed. Check the control file for details.")
 
    def process_recording(self, recording: Recording):
        """ Process a single recording and update the control file. """
        # control the quality of the recording
        acceptable_quality, message = self._quality_control(recording)
        
        if acceptable_quality:
            # detect steps
            steps, message = self._detect_steps(recording)
            
            if steps is None:
                self._move_recording(recording, self.skip_directory)
                self._update_control_file(recording, processed=False, notes=f"no steps detected: {message}")
                return
            
            # segment the data
            segments = self._segment_data(recording, steps)
            
            # add extra columns
            segments = self._add_extra_columns(segments)
            
            # save the segmented data
            status = self._save_segmented_data(segments, self.base_directory, self.output_directory, recording)
            
            # update the control file
            self._update_control_file(recording, processed=True, notes="successful")
        else:
            self._move_recording(recording, self.skip_directory)
            self._update_control_file(recording, processed=False, notes=f"bad data quality: {message}")
    
    def _quality_control(self, recording: Recording):
        """ Check if all the required files are present and the data quality is acceptable. """
        # Define the required files for every recording.
        required_files = [
            "Accelerometer.csv",
            "Gravity.csv",
            # "TotalAcceleration.csv",
            "Gyroscope.csv",
        ]
        
        # Check if all required files are present
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(recording.recording_path, file))]
        if missing_files:
            self.logger.warning(f"Recording {recording.recording_name} is missing some required files. Skipping.")
            return False, "missing files"
        
        # Load the data
        recording.data = self._load_data(required_files, recording.recording_path)
        
        for sensor_name, df in recording.data:
            # Check for large time gaps in the data
            time_values = df["seconds_elapsed"]
            time_diffs = time_values.diff()
            if any(time_diffs > self.max_acceptable_gap):
                self.logger.warning(f"Recording {recording.recording_name} contains time gaps greater than 100 ms. Skipping.")
                return False, "too large time gaps"
                
            # Check for allowed duration of the recording
            recording_duration = time_values.max() - time_values.min()
            if recording_duration > self.max_duration:
                self.logger.warning(f"Recording {recording.recording_name} is longer than {self.max_duration} seconds. Skipping.")
                return False, "recording too long"
            
            if recording_duration < self.min_duration:
                self.logger.warning(f"Recording {recording.recording_name} is shorter than {self.min_duration} seconds. Skipping.")
                return False, "recording too short"
            
            # Handle missing values (e.g., NaN) using linear interpolation
            if df.isnull().values.any():
                self.logger.warning(f"Interpolating missing values for {sensor_name} in recording {recording.recording_name}.")
                df.interpolate(method="linear", limit_direction="forward", axis=0, inplace=True)
                df.bfill(inplace=True)
                df.ffill(inplace=True)
        
        return True, ""
    
    def _load_data(self, files: List[str], recording_path: str) -> List[Tuple[str, pd.DataFrame]]:
        """
        Load CSV files and add magnitude column.

        Parameters:
        - files: List of CSV files to load
        - recording_path: The directory where the CSV files are located

        Returns:
        - List of tuples containing preprocessed DataFrames and their file names
        """

        data = []

        # Iterate through the list of files
        for file in files:
            df = pd.read_csv(os.path.join(recording_path, file))

            # If columns "x", "y", and "z" exist, calculate the magnitude.
            if all(col in df.columns for col in ["x", "y", "z"]):
                df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
                
            # sort the data by time
            df.sort_values("seconds_elapsed", inplace=True)

            data.append((file, df))

        return data

    def _detect_steps(self, recording: Recording):
        """ Detect the segments corresponding to each step in a specific recording. """
        # Resample the data to a common time scale
        acc_data = next(df for name, df in recording.data if "Accelerometer" in name)
        gravity_data = next(df for name, df in recording.data if "Gravity" in name)
        # total_acc_data = next(df for name, df in recording.data if "TotalAcceleration" in name)
        gyro_data = next(df for name, df in recording.data if "Gyroscope" in name)
        
        common_time = np.union1d(
            np.union1d(acc_data["seconds_elapsed"], gravity_data["seconds_elapsed"]),
            gyro_data["seconds_elapsed"],
            # np.union1d(total_acc_data["seconds_elapsed"], gyro_data["seconds_elapsed"]),
        )
        
        acc_data = np.interp(common_time, acc_data["seconds_elapsed"], acc_data["magnitude"])
        gravity_data_x = np.interp(common_time, gravity_data["seconds_elapsed"], gravity_data["x"])
        gravity_data_y = np.interp(common_time, gravity_data["seconds_elapsed"], gravity_data["y"])
        gravity_data_z = np.interp(common_time, gravity_data["seconds_elapsed"], gravity_data["z"])
        # total_acc_data = np.interp(common_time, total_acc_data["seconds_elapsed"], total_acc_data["magnitude"])
        gyro_data = np.interp(common_time, gyro_data["seconds_elapsed"], gyro_data["magnitude"])
        
        data = {
            "acc": acc_data,
            "gravity_x": gravity_data_x,
            "gravity_y": gravity_data_y,
            "gravity_z": gravity_data_z,
            # "total_acc": total_acc_data,
            "gyro": gyro_data
        }
        
        # detect the segments for each step
        on_desk_segment = self._detect_segment(
            data, "on_desk", common_time,
            start_time=0, end_time=common_time[-1], 
            min_duration=8, max_duration=10)
        if on_desk_segment == (None, None):
            on_desk_segment = (5, 15)
        
        on_hand_segment = self._detect_segment(
            data, "on_hand", common_time,
            start_time=0, end_time=on_desk_segment[0], 
            min_duration=8, max_duration=10)
        if on_hand_segment == (None, None):
            on_hand_segment = (on_desk_segment[0] - 15, on_desk_segment[0] - 5)
        
        on_desk_audio_segment = self._detect_segment(
            data, "on_desk_audio", common_time,
            start_time=on_desk_segment[1], end_time=common_time[-1], 
            min_duration=8, max_duration=10)
        if on_desk_audio_segment == (None, None):
            on_desk_audio_segment = (on_desk_segment[1] + 5, on_desk_segment[1] + 15)
        
        on_hand_audio_segment = self._detect_segment(
            data, "on_hand_audio", common_time,
            start_time=on_desk_segment[1], end_time=on_desk_audio_segment[0], 
            min_duration=8, max_duration=10)
        if on_hand_audio_segment == (None, None):
            mid_time = common_time[len(common_time) // 2]
            on_hand_audio_segment = (mid_time - 15, mid_time - 5)
        
        walking_1_segment = self._detect_segment(
            data, "walking_1", common_time,
            start_time=on_desk_audio_segment[1], end_time=common_time[-1], 
            min_duration=6, max_duration=13)
        if walking_1_segment == (None, None):
            walking_1_segment = (on_desk_audio_segment[1] + 10, on_desk_audio_segment[1] + 20)
        
        walking_2_segment = self._detect_segment(
            data, "walking_2", common_time,
            start_time=walking_1_segment[1], end_time=common_time[-1], 
            min_duration=6, max_duration=13)
        if walking_2_segment == (None, None):
            walking_2_segment = (walking_1_segment[1] + 5, walking_1_segment[1] + 15)
        
        segments = {
            "on_hand": on_hand_segment,
            "on_desk": on_desk_segment,
            "on_hand_audio": on_hand_audio_segment,
            "on_desk_audio": on_desk_audio_segment,
            "walking_1": walking_1_segment,
            "walking_2": walking_2_segment
        }
        
        max_time = common_time[-1]
        for step_name, (start, end) in segments.items():
            if end > max_time - 1:
                segments[step_name] = (max_time - 11, max_time - 1)
        
        # validate the detected segments
        segments, message = self._create_interactive_step_plot(common_time, data, segments)
        
        return segments, message
    
    def _detect_segment(self, data, step: str, common_time: np.ndarray,
                        start_time: float = 0.0, end_time: float = 300.0, 
                        min_duration: float = 4.0, max_duration: float = 10.0) -> Tuple[float, float]:
        """ 
        Detect a segment in the data based on the given thresholds, start and end times, and duration constraints. 
        Returns (segment_start, segment_end) or (None, None) if no suitable segment is found.
        """
        acc_mag = data["acc"]
        gyro_mag = data["gyro"]
        gravity_x = data["gravity_x"]
        gravity_y = data["gravity_y"]
        gravity_z = data["gravity_z"]
                
        # define thresholds based on step
        if step == "on_hand":    
            thresholds_mask = {
                'acc_mag': (0, 1.0),
                'gyro_mag': (0, 0.3),
                'gravity_x': (-2, 2),
                'gravity_y': (1, 10),
                'gravity_z': (0, 10)}
        elif step == "on_desk":
            thresholds_mask = {
                'acc_mag': (0, 0.8),
                'gyro_mag': (0, 0.1),
                'gravity_x': (-1, 1),
                'gravity_y': (-1, 1),
                'gravity_z': (8, 10)}
        elif step == "on_hand_audio":
            thresholds_mask = {
                'acc_mag': (0, 2),
                'gyro_mag': (0, 4.2),
                'gravity_x': (-2, 2),
                'gravity_y': (1, 10),
                'gravity_z': (0, 10)}
        elif step == "on_desk_audio":
            thresholds_mask = {
                'acc_mag': (0, 1.3),
                'gyro_mag': (0, 8),
                'gravity_x': (-1, 1),
                'gravity_y': (-1, 1),
                'gravity_z': (8, 10)}
        elif step == "walking_1":
            thresholds_mask = {
                'acc_mag': (0.08, 20),
                'gyro_mag': (0.08, 10),
                'gravity_x': (-10, 10),
                'gravity_y': (-10, 7),
                'gravity_z': (-6, 10)}
        elif step == "walking_2":
            thresholds_mask = {
                'acc_mag': (0.08, 20),
                'gyro_mag': (0.08, 10),
                'gravity_x': (-10, 10),
                'gravity_y': (-10, 7),
                'gravity_z': (-6, 10)}
                
        # Build mask
        mask = (
            (acc_mag >=     thresholds_mask['acc_mag'][0]) &    (acc_mag < thresholds_mask['acc_mag'][1]) &
            (gyro_mag >=    thresholds_mask['gyro_mag'][0]) &   (gyro_mag < thresholds_mask['gyro_mag'][1]) &
            (gravity_x >    thresholds_mask['gravity_x'][0]) &  (gravity_x < thresholds_mask['gravity_x'][1]) &
            (gravity_y >    thresholds_mask['gravity_y'][0]) &  (gravity_y < thresholds_mask['gravity_y'][1]) &
            (gravity_z >    thresholds_mask['gravity_z'][0]) &  (gravity_z < thresholds_mask['gravity_z'][1])
        )
        
        # Restrict to the specified time window
        if start_time is None or end_time is None:
            return None, None
        time_mask = (common_time >= start_time) & (common_time <= end_time)
        mask = mask & time_mask
        
        # Find continuous segments of True values
        # We can find transitions from False->True and True->False to identify segments
        indices = np.where(mask)[0]
        if len(indices) == 0:
            # No valid points
            return None, None
        
        # Group continuous indices
        segments = []
        segment_start_idx = indices[0]
        prev_idx = indices[0]
        
        for idx in indices[1:]:
            if idx != prev_idx + 1:
                # End of a continuous segment
                segments.append((segment_start_idx, prev_idx))
                segment_start_idx = idx
            prev_idx = idx
        segments.append((segment_start_idx, prev_idx))
        
        # Now filter out segments by duration constraints
        valid_segments = []
        for seg_start_idx, seg_end_idx in segments:
            seg_start_time = common_time[seg_start_idx]
            seg_end_time = common_time[seg_end_idx]
            duration = seg_end_time - seg_start_time
            
            if duration < min_duration:
                # Too short, discard entirely
                continue
            elif duration <= max_duration:
                # Fits well
                valid_segments.append((seg_start_time, seg_end_time))
            else:
                # Too long, split into max_duration chunks
                candidate_subsegs = []
                current_start = seg_start_time

                while current_start < seg_end_time:
                    current_end = current_start + max_duration
                    if current_end > seg_end_time:
                        # This is the last chunk
                        remainder = seg_end_time - current_start
                        if remainder >= min_duration:
                            candidate_subsegs.append((current_start, seg_end_time))
                        # If remainder < min_duration, discard the remainder
                        break
                    else:
                        candidate_subsegs.append((current_start, current_end))
                        current_start += 1

                # Now we need to pick the representative segment based on which step
                # [on_hand, on_hand_audio, on_desk, on_desk_audio] : minimal variability
                # [walking_1, walking_2] : maximal variability
                if candidate_subsegs:
                    chosen_subseg = None
                    chosen_variability = None
                    
                    for (sub_start, sub_end) in candidate_subsegs:
                        sub_indices = np.where((common_time >= sub_start) & (common_time <= sub_end))[0]
                        if len(sub_indices) > 1:
                            sub_acc_data = data["acc"][sub_indices]
                            sub_std = np.std(sub_acc_data)
                            
                            peaks, _ = scipy.signal.find_peaks(sub_acc_data, height=0.2, prominence=0.15)
                            num_peaks = len(peaks)
                            if step in ["walking_1", "walking_2"] and num_peaks < 5:
                                continue 
                            
                            if chosen_variability is None:
                                chosen_variability = sub_std
                                chosen_subseg = (sub_start, sub_end)
                            else:
                                if step in ["on_hand", "on_hand_audio", "on_desk", "on_desk_audio"]:
                                    if sub_std < chosen_variability:
                                        chosen_variability = sub_std
                                        chosen_subseg = (sub_start, sub_end)
                                elif step in ["walking_1", "walking_2"]:
                                    if sub_std > chosen_variability:
                                        chosen_variability = sub_std
                                        chosen_subseg = (sub_start, sub_end)
                    
                    if chosen_subseg:
                        valid_segments.append(chosen_subseg)
        
        # If no valid segment found, return None
        if not valid_segments:
            return None, None
        
        # Choose the "best fit" segment depending on the step:
        # - on_desk, walking_1: left-most (earliest start time)
        # - on_hand, on_desk_audio, on_hand_audio, walking_2: right-most (latest start time)
        if step in ["on_desk", "walking_1"]:
            chosen_segment = valid_segments[0]  # segments are in chronological order
        else:
            chosen_segment = valid_segments[-1]  # take the last one for right-most
        
        return chosen_segment

    def _create_interactive_step_plot(self, common_time: np.ndarray, data: List[Tuple[pd.DataFrame, str]], steps: Dict[str, Tuple[float, float]]):
        """
        Create an interactive plot with sliders to adjust step times.
        Steps 1-4 have only start time sliders with fixed 10-second duration.
        Steps 5-6 have both start and end time sliders.
        
        The user can:
        - Adjust steps via sliders
        - Click 'Done' to confirm final steps
        - Click 'Skip' to leave it for later without finalizing
        - Click 'Cancel' to indicate data is not usable
        """
        # Store original steps for potential reset
        original_steps = copy.deepcopy(steps)
        
        # Ensure all steps exist with default values if not detected
        all_steps = ['on_hand', 'on_desk', 'on_hand_audio', 'on_desk_audio', 'walking_1', 'walking_2']
        defaults = {}

        # Create figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14))
        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed')
        plt.subplots_adjust(bottom=0.35, top=0.95, hspace=0.1)
        
        # Plot sensor data
        ax1.plot(common_time, data["acc"])
        ax1.set_ylabel("Accelerometer Magnitude")
        ax1.set_title("Accelerometer Data")
        ax1.grid(True, linestyle="--", alpha=0.7)

        ax2.plot(common_time, data["gyro"])
        ax2.set_ylabel("Gyroscope Magnitude")
        ax2.set_title("Gyroscope Data")
        ax2.grid(True, linestyle="--", alpha=0.7)

        ax3.plot(common_time, data["gravity_x"], label="X")
        ax3.plot(common_time, data["gravity_y"], label="Y")
        ax3.plot(common_time, data["gravity_z"], label="Z")
        ax3.set_xlabel("Seconds Elapsed")
        ax3.set_ylabel("Gravity")
        ax3.set_title("Gravity Data")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.7)

        # Colors for step highlighting
        colors = ["#C4F589", "#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845"]

        # Highlight steps
        step_rects = {
            'acc': [],
            'gyro': [],
            'gravity': []
        }

        for i, (step_name, period) in enumerate(steps.items()):
            start, end = period
            color = colors[i]
            step_rects['acc'].append(ax1.axvspan(start, end, alpha=0.3, color=color, label=f"{step_name}"))
            step_rects['gyro'].append(ax2.axvspan(start, end, alpha=0.3, color=color))
            step_rects['gravity'].append(ax3.axvspan(start, end, alpha=0.3, color=color))

        ax1.legend()

        # Determine max_time from common_time
        max_time = common_time[-1] if len(common_time) > 0 else 360

        # Create sliders
        sliders = []
        fixed_duration_steps = ['on_hand', 'on_desk', 'on_hand_audio', 'on_desk_audio']
        variable_duration_steps = ['walking_1', 'walking_2']

        # Fixed duration steps: only start slider
        for i, step_name in enumerate(fixed_duration_steps):
            start, end = steps[step_name]
            
            ax_start = plt.axes([0.1 + (i % 2) * 0.45, 0.24 - (i // 2) * 0.04, 0.35, 0.03])
            
            start_slider = widgets.Slider(ax_start, f'{step_name} Start', 0, max_time - 10, valinit=start, valstep=0.1)
            
            sliders.append({'step': step_name, 'start': start_slider, 'end': None})

        # Variable duration steps: start and end sliders
        for i, step_name in enumerate(variable_duration_steps):
            start, end = steps[step_name]
            
            ax_start = plt.axes([0.1 + (i % 2) * 0.45, 0.16 - (i // 2) * 0.08, 0.35, 0.03])
            ax_end = plt.axes([0.1 + (i % 2) * 0.45, 0.12 - (i // 2) * 0.08, 0.35, 0.03])
            
            start_slider = widgets.Slider(ax_start, f'{step_name} Start', 0, max_time, valinit=start, valstep=0.1)
            end_slider = widgets.Slider(ax_end, f'{step_name} End', 0, max_time, valinit=end, valstep=0.1)
            
            sliders.append({'step': step_name, 'start': start_slider, 'end': end_slider})

        # Add buttons
        done_ax = plt.axes([0.8, 0.04, 0.1, 0.04])
        skip_ax = plt.axes([0.65, 0.04, 0.1, 0.04])
        cancel_step_ax = plt.axes([0.50, 0.04, 0.1, 0.04])
        
        done_button = widgets.Button(done_ax, 'Done')
        skip_button = widgets.Button(skip_ax, 'Skip')
        cancel_step_button = widgets.Button(cancel_step_ax, 'Cancel')
        
        user_action = [None]  # 'done', 'skip', 'cancel'
        
        def update(val):
            # Update step rectangles based on slider values
            for i, slider_info in enumerate(sliders):
                step_name = slider_info['step']
                start_val = slider_info['start'].val

                if step_name in fixed_duration_steps:
                    # Steps 1-4 fixed duration
                    end_val = start_val + 10
                else:
                    # Steps 5-6 variable duration
                    end_val = slider_info['end'].val

                    # Ensure end time is always after start time
                    if end_val <= start_val:
                        end_val = start_val + 0.1
                        slider_info['end'].set_val(end_val)

                # Remove old rectangles
                for sensor_rects in step_rects.values():
                    if i < len(sensor_rects):
                        sensor_rects[i].remove()

                # Add updated rectangles
                c = colors[i]
                step_rects['acc'][i] = ax1.axvspan(start_val, end_val, alpha=0.3, color=c)
                step_rects['gyro'][i] = ax2.axvspan(start_val, end_val, alpha=0.3, color=c)
                step_rects['gravity'][i] = ax3.axvspan(start_val, end_val, alpha=0.3, color=c)

                # Update steps dictionary
                steps[step_name] = (start_val, end_val)

            fig.canvas.draw_idle()

        # Connect sliders to update function
        for s in sliders:
            s['start'].on_changed(update)
            if s['end']:
                s['end'].on_changed(update)

        def done_clicked(event):
            user_action[0] = 'done'
            plt.close(fig)

        def skip_clicked(event):
            user_action[0] = 'skip'
            plt.close(fig)

        def cancel_clicked(event):
            user_action[0] = 'cancel'
            plt.close(fig)
               
        done_button.on_clicked(done_clicked)
        skip_button.on_clicked(skip_clicked)
        cancel_step_button.on_clicked(cancel_clicked)

        plt.show(block=True)     

        # Based on user action, return steps or None
        if user_action[0] == 'done':
            return steps, ""
        elif user_action[0] == 'skip':
            return None, 'skip'
        elif user_action[0] == 'skip':
            return None, 'cancel'

    def _segment_data(self, recording: Recording, steps: Dict[str, Tuple[float, float]]):
        """ Segment the recording's data according to the detected steps and save each in a different directory. """
        segments = {}

        for step_name, period in steps.items():
            # Initialize a dictionary for this step
            segments[step_name] = {}
            
            if period is None:
                continue
            
            start, end = period
            
            for sensor_name, df in recording.data:
                # Slice the data for the current step time window
                mask = (df["seconds_elapsed"] >= start) & (df["seconds_elapsed"] < end)
                seg_df = df[mask].copy()
                
                # If there's any data in this segment, store it
                if not seg_df.empty:
                    segments[step_name][sensor_name] = seg_df
        
        return segments
    
    def _add_extra_columns(self, segments: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Add additional computed columns to the segmented data:
        - inter_sample_time: difference between consecutive samples
        - azimuth, inclination: for accelerometer, gravity, and total acceleration data.
        - For gyroscope, only inter_sample_time is added.
        """
        for step, sensors in segments.items():
            for name, df in sensors.items():
                # Add inter-sample time
                df["inter_sample_time"] = df["seconds_elapsed"].diff()
                df["inter_sample_time"].fillna(0, inplace=True) 

                # Add magnitude column if needed
                if "magnitude" not in df.columns:
                    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
                
                # Add azimuth and inclination columns for accelerometer, gravity, and total acceleration
                # if "Accelerometer.csv" in name or "Gravity.csv" in name or "TotalAcceleration.csv" in name:
                if "Accelerometer.csv" in name or "Gravity.csv" in name:
                    df['azimuth'] = np.arctan2(df["y"], df["x"]) * (180 / np.pi)
                    
                    with np.errstate(invalid='ignore'):
                        df['inclination'] = np.arccos(df["z"] / df["magnitude"]) * (180 / np.pi)
                    
                    # Use interpolation to fill NaN values
                    df['inclination'] = df['inclination'].interpolate(method='linear', limit_direction='forward', axis=0)
                    df['inclination'] = df['inclination'].bfill().ffill()
                
                # Update the DataFrame in the dictionary
                sensors[name] = df

        return segments

    def _save_segmented_data(self, segments: Dict[str, Dict[str, pd.DataFrame]], base_dir: str, output_dir: str, recording: Recording) -> Dict[str, str]:
        """
        Save all sensor streams organized by settings (on_hand, on_desk, etc.).
        """
        step_status = {}

        # Create base directories for each setting
        for setting in segments.keys():
            setting_dir = os.path.join(output_dir, setting)
            os.makedirs(setting_dir, exist_ok=True)

        # Save each setting in correspinding setting directory
        for setting, sensor_data in segments.items():
            if not sensor_data:
                step_status[setting] = "No Data"
                continue

            try:
                # Create directory for this (setting, recording)
                recording_dir = f"{setting} - {recording.pseudonym} - {recording.device_id} - {recording.device_name} - {recording.recording_name}"
                recording_dir_path= os.path.join(output_dir, setting, recording_dir)
                recording_dir_path = os.path.normpath(recording_dir_path)
                os.makedirs(recording_dir_path, exist_ok=True)

                # Save each sensor DataFrame to a separate CSV
                for name, df in sensor_data.items():
                    file_path = os.path.join(recording_dir_path, name)
                    df.to_csv(file_path, index=False)

                step_status[setting] = "Successful"
            except Exception as e:
                step_status[setting] = f"Failed: {e}"

        return step_status

    def _move_recording(self, recording: Recording, target_dir: str = ""):
        """Move the entire recording directory to the specified directory or the skipped directory by default."""
        if target_dir == "":
            target_dir = self.skip_directory
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        source = recording.recording_path
        destination = os.path.join(target_dir, os.path.basename(recording.recording_path))
        
        try:
            shutil.move(source, destination)
            self.logger.info(f"Moved {source} to {destination}")
        except Exception as e:
            self.logger.error(f"Error moving {source} to {destination}: {e}")

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


# local main
if __name__ == "__main__":
    segmenter = RecordingSegmenter(    
        base_directory = "../Data/raw data/all data/",
        output_directory = "../Data/raw data/separated by setting/",
        control_file_path = "../Data/raw data/RecordingSegmenter_control_file.csv",
        skip_directory = "../Data/raw data/skipped recordings/"
    )

    segmenter.process_recordings()
