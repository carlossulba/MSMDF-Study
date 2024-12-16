import os

def delete_files(root_dir, filename):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == filename:
                file_path = os.path.join(dirpath, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def delete_files_from_python(root_dir, filename):
    delete_files(root_dir, filename)

if __name__ == "__main__":
    root_dir = "../Data/raw data/separated by setting"
    filename = "TotalAcceleration.csv"
    
    delete_files_from_python(root_dir, filename)