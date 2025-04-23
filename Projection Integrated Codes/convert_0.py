import os
import pandas as pd

# Set the root directory where your folders are located
root_dir = r'C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\output'  # <-- Change this to your directory

# Traverse the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'blender_objects_truck.csv':
            filepath = os.path.join(dirpath, filename)
            try:
                # Read CSV
                df = pd.read_csv(filepath)

                # Set 'phi' column to 0
                if 'phi' in df.columns:
                    df['phi'] = 0

                    # Overwrite the CSV file
                    df.to_csv(filepath, index=False)
                    print(f"Updated: {filepath}")
                else:
                    print(f"'phi' column not found in: {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
