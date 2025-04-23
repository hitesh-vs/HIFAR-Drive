import os
import csv

def modify_csv_add_phi(csv_file):
    temp_file = csv_file + ".temp"
    with open(csv_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['x', 'y', 'z', 'phi'])
        
        for row in reader:
            if len(row) >= 3:
                row.append('0')  # Add phi = 180
                writer.writerow(row)

    os.replace(temp_file, csv_file)
    print(f"✅ Modified: {csv_file}")

def process_all_truck_csvs(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "blender_objects_truck.csv":
                full_path = os.path.join(root, file)
                modify_csv_add_phi(full_path)

if __name__ == "__main__":
    base_dir = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\output"  # or wherever your files are
    process_all_truck_csvs(base_dir)
