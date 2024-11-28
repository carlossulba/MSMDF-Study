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

## License
This project is open-source and available under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.en.html). By using this repository, you agree to the following conditions:
- Use responsibly and ethically.
- Cite this repository in your work or research.
- Ensure that any derivative works or modifications are open-sourced under the same license.

### How to Cite
Carlos (2024). Motion Sensor-Based Mobile Device Fingerprinting. GitHub Repository.
https://github.com/carlossulba/MSMDF-Study