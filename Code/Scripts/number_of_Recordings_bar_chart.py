import os
from collections import defaultdict
import matplotlib.pyplot as plt

def generate_recording_chart(directory_path):
    # Dictionary to store the count of recordings per device
    device_recording_count = defaultdict(int)

    # Iterate through each recording directory
    if os.path.exists(directory_path):
        try:
            for root, dirs, files in os.walk(directory_path):
                for dir_name in dirs:
                    # Extract device name and device id from the directory name
                    if "_" in dir_name and "2024" in dir_name:
                        device_info = dir_name.split("2024", 1)[0].strip("-")
                        if "_" in device_info:
                            device_name, device_id = device_info.split("_", 1)
                            if device_name and device_id:
                                device_key = f"{device_name}_{device_id}"
                                device_recording_count[device_key] += 1
                    else:
                        print(f"Invalid directory name: {dir_name}")
                        break
                break  # Only process the top-level directories
        except Exception as e:
            print(f"Error reading directories in {directory_path}: {str(e)}")
            return
    
    # Count how many devices recorded each number of times
    recording_distribution = defaultdict(int)
    devices_in_each_category = defaultdict(list)
    for device_key, count in device_recording_count.items():
        recording_distribution[count] += 1
        devices_in_each_category[count].append(device_key)

        
    # Sort the categories and devices in each category
    sorted_devices_in_each_category = dict(sorted(devices_in_each_category.items()))

    # Print out the list of devices in each category
    for count, devices in sorted_devices_in_each_category.items():
        print(f"Devices with {count} recordings:")
        for device in devices:
            print(f"  - {device}")
        
    # Prepare data for the bar chart
    x_values = sorted(recording_distribution.keys())
    y_values = [recording_distribution[i] for i in x_values]
    
    # Determine colors for the bars
    colors = ['pink' if x < 8 else 'palegreen' for x in x_values]
    
    # Calculate total number of devices with 8 or more recordings
    total_devices_green = sum(y for x, y in zip(x_values, y_values) if x >= 8)

    
    # Generate the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, color=colors, width=0.6)
    plt.xlabel('Number of Recordings')
    plt.ylabel('Number of Devices')
    plt.title('Distribution of Recordings per Device')
    plt.xticks(x_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add text annotation for total devices with >= 8 recordings
    plt.annotate(f'Total Devices with >= 8 Recordings: {total_devices_green}', 
                 xy=(0.95, 0.95), xycoords='axes fraction', fontsize=10,
                 horizontalalignment='right', verticalalignment='top', color='green')
    
    plt.show()
    input("end")
    

# Example usage
generate_recording_chart("../Data/raw data/all data")
print("Done!")