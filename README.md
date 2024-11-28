# Motion Sensor-Based Mobile Device Fingerprinting

This repository contains the code implementation for Motion Sensor-based Mobile Device Fingerprinting (MSMDF), including demonstration of the fingerprinting attack and possible countermeasures. The project is designed to showcase how motion sensors can be leveraged to uniquely identify devices and explore techniques to mitigate this potential privacy risk.

### Features
- **Attack Demonstration**: Demonstrates fingerprinting techniques using motion sensor data.
- **Countermeasures**: *Not implemented yet*.
- **Evaluation Notebook**: See [MSMDF Evaluation Notebook](https://github.com/carlossulba/MSMDF-Study/blob/main/Code/Classes%20and%20Notebooks/MSMDF%20Evaluation%20Notebook.ipynb) MSMDF Evaluation Notebook for experiments and results.

## Format of the Fingerprint
The extracted fingerprints are stored in a dictionary structure, where the keys represent specific experimental conditions, and the values are nested dictionaries for each device. Here is an overview of the structure:

- **Key**: Experimental condition (e.g., on-hand, on-desk, walking)
- **Value**: A dictionary containing:
  - Device IDs: Nested structure per device, holding:
    - fingerprints: List of fingerprints (1d feature array) extracted for the device.

**Example**:
```python
{
  'on desk': {
    'Device-01': [
      Fingerprint_01,
      Fingerprint_02,
      Fingerprint_03
    ],
    'Device-02': [
      Fingerprint_01,
      Fingerprint_02,
      Fingerprint_03
    ],
    ...
  },
  'on hand': {
    'Device-01': [
      Fingerprint_01,
      Fingerprint_02,
      Fingerprint_03
    ],
    ...
  }
}
```
.
# Overview of Attack
## Fingerprint Extraction
| **Parameter**         | **Description**                                             | **Examples / Values**                               | **Comment**                                              |
|------------------------|-------------------------------------------------------------|----------------------------------------------------|----------------------------------------------------------|
| `fingerprint_length`   | Duration of samples included in each fingerprint.           | E.g., 1, 2, 3, …, 9 seconds                        |                                                          |
| `sampling_rate`        | Rate at which data points are collected from all sensors.   | E.g., 10, 50, 100 Hz                               | From 10 to 200 Hz                                        |
| `enabled_settings`     | Experimental contexts included in the extraction.           | on_hand, on_desk, on_hand_audio, on_desk_audio, walking | Recordings of walking vary greatly on length             |
| `enabled_sensors`      | Motion sensors used for data collection.                    | Accelerometer, Gravity, Total acceleration, Gyroscope |                                                          |
| `enabled_streams`      | Data streams (columns) used from the sensors.               | X, Y, Z, Magnitude, Azimuth, Inclination, Inter sample time | Azimuth and Inclination only for accelerometer-related data |
| `enabled_features`     | Extracted features (scalars) from all streams.              | Mean, Variance, Standard Deviation, Spectral Energy, RMS, Skewness, etc. | 35 in total                                              |

## Classifier Training
| **Parameter**         | **Description**                                             | **Examples / Values**                               | **Comment**                                              |
|------------------------|-------------------------------------------------------------|----------------------------------------------------|----------------------------------------------------------|
| `num_devices`          | Number of devices used in the dataset.                 | E.g., 10, 50, 100                                  |                                                          |
| `training_set_ratio`   | Fraction of the data used for training compared to testing. | E.g., 0.6 (60% training, 40% testing)              |                                                          |
| `known_unknown_ratio`  | Ratio of known to unknown devices in an open-world scenario.| E.g., 0.9 (90% known, 10% unknown)                 | Helps evaluate performance in realistic scenarios.       |
| `cv_folds`             | Number of folds for cross-validation during training.       | E.g., 5, 10                                        | Higher folds give robust evaluations but increase time and resources. |
| `random_state`         | Random seed used to ensure reproducibility of results.      | E.g., 42, 12345                                    | Use a fixed seed for consistent experiments.             |
| `classifiers`          | Set of classifiers used for training and evaluation.        | E.g., SVM, Random Forest, Bagging Classifier       | 14 in total       |

.
# License
This project is open-source and available under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.en.html). By using this repository, you agree to the following conditions:
- Use responsibly and ethically.
- Cite this repository in your work or research.
- Ensure that any derivative works or modifications are open-sourced under the same license.

### How to Cite
Carlos (2024). Motion Sensor-Based Mobile Device Fingerprinting. GitHub Repository.
https://github.com/carlossulba/MSMDF-Study