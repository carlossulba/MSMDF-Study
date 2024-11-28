import os
import re
import json

import os
import json
import csv

def get_pseudonym_from_metadata(directory_path):
    metadata_path = os.path.join(directory_path, "StudyMetadata.json")
    csv_path = os.path.join(directory_path, "Metadata.csv")
    
    # Check if the metadata file exists
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    device_name = row.get("device name", "").strip()
                    device_id = row.get("device id", "").strip()
                    if device_name and device_id:
                        return f"{device_name}_{device_id}"
        except Exception as e:
            print(f"Error reading {csv_path}: {str(e)}")
    
    # Fallback, attempt to get pseudonym from JSON metadata
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                for item in metadata:
                    if item.get("title") == "Pseudonym" and item.get("value"):
                        return item.get("value").strip()
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {metadata_path}")
        except Exception as e:
            print(f"Error reading {metadata_path}: {str(e)}")
    
    # If all else fails, return a fallback message (although this is less likely now)
    return "No pseudonym found and no device information available"

def rename_directories(path):
    pattern = r'^(.*?)[-_]?(\d{4}-\d{2}-\d{2}[-_]\d{2}-\d{2}-\d{2})-(.+)$'
    
    for dirname in os.listdir(path):
        full_path = os.path.join(path, dirname)
        if os.path.isdir(full_path):
            match = re.match(pattern, dirname)
            if match:
                prefix, date, id = match.groups()
                date = date.replace('-', '_')
                
                # Get pseudonym from StudyMetadata.json
                pseudonym = get_pseudonym_from_metadata(full_path)
                
                if pseudonym:
                    if prefix:
                        new_name = f"{pseudonym}-{date}-{prefix}"
                    else:
                        new_name = f"{pseudonym}-{date}"
                    
                    new_path = os.path.join(path, new_name)
                    try:
                        os.rename(full_path, new_path)
                        print(f"Renamed: {dirname} -> {new_name}")
                    except Exception as e:
                        print(f"Error renaming {dirname}: {str(e)}")
                else:
                    print(f"Skipped: {dirname} (No pseudonym found in StudyMetadata.json)")
            else:
                print(f"Skipped: {dirname} (doesn't match expected format)")

# Path to the main directory containing all the subdirectories
directory_path = ".././././"

# Check if the directory exists
if os.path.exists(directory_path):
    rename_directories(directory_path)
else:
    print("Error: The directory does not exist.")