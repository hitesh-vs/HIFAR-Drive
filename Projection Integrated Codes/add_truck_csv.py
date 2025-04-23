import os
import shutil

def copy_truck_csvs(source_base, target_base):
    for folder in os.listdir(source_base):
        src_objects_path = os.path.join(source_base, folder, "objects")
        target_objects_path = os.path.join(target_base, folder, "objects")
        
        src_file = os.path.join(src_objects_path, "blender_objects_truck.csv")
        target_file = os.path.join(target_objects_path, "blender_objects_truck.csv")
        
        if not os.path.exists(src_file):
            print(f"⚠️ Source file not found for: {folder}")
            continue

        os.makedirs(target_objects_path, exist_ok=True)
        
        try:
            shutil.copy2(src_file, target_file)
            print(f"✅ Copied to: {target_file}")
        except Exception as e:
            print(f"❌ Failed to copy for {folder}: {e}")

if __name__ == "__main__":
    source_dir = r"C:\Users\vshit\Downloads\output_truck\output_truck"
    target_dir = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\output"
    copy_truck_csvs(source_dir, target_dir)
