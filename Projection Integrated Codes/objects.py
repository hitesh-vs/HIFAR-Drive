# import cv2
# import numpy as np
# import os
# #from ultralytics import YOLO
# import numpy as np
# import matplotlib.pyplot as plt
# import re
# from detect_objects import predict_and_detect
# from objects_world import parse_object_file, project_to_3d
# import shutil

# def clear_directory(directory):
#     """Remove all contents from a directory and recreate it."""
#     if os.path.exists(directory):
#         shutil.rmtree(directory)  # Delete all files and subfolders
#     os.makedirs(directory, exist_ok=True)  # Recreate an empty directory


# def get_object_data(scene_directory):
#     #model = YOLO("yolov9c.pt")
#     # Intrinsic matrix
#     K = np.array([[1594.7, 0, 655.2961], 
#                 [0, 1607.7, 414.3627], 
#                 [0, 0, 1]])

#     dist_coeffs = [-0.352238385718678, -0.333547811831122]
#     objects_directory = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\scene_directory_objects"
#     clear_directory(objects_directory) 

#     # scene_directory = r"C:\Users\farha\OneDrive\Desktop\ObjectDetection\Images2render\scene9"  # The folder containing the images
#      # The folder to save the text files for detected objects
#     output_directory = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\output" # The folder to save the CSV files for Blender

#     # Create the output directory if it doesn't exist
#     os.makedirs(objects_directory, exist_ok=True)
#     os.makedirs(output_directory, exist_ok=True)

#     for filename in os.listdir(scene_directory):
#         file_path = os.path.join(scene_directory, filename)
    
#         # Check if it's a file and an image (common formats)
#         if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
#             print(f"Found image: {filename}")

#         image = cv2.imread(file_path)
    
#         if image is None:
#             print("Error: Could not read the image.")
#             return

#         result_img, detected_results, bottom_left_info, bottom_mid_info = predict_and_detect(model, image, conf=0.5)
#         # Create a unique directory for this image inside `output_directory`
#         image_name = os.path.splitext(filename)[0]  # Extract filename without extension
#         image_output_dir = os.path.join(output_directory, image_name)
#         objects_output_dir = os.path.join(image_output_dir, "objects")

#         os.makedirs(objects_output_dir, exist_ok=True)  # Create `objects` folder

#         # Save detected object information as a text file
#         txt_filename = image_name + ".txt"
#         output_file_path = os.path.join(objects_directory, txt_filename)

#         with open(output_file_path, "w") as f:
#             for detection in bottom_mid_info:
#                 f.write(f"{detection['label']}, {detection['coordinates'][0]}, {detection['coordinates'][1]}\n")


#         cv2.imshow("Image with Points", result_img)
#         # cv2.imwrite("scene4_8.jpg", result_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#         # print("Bottom midpoints saved")

#         input_file = output_file_path
#         objects_by_label = parse_object_file(input_file)

#         for label, points in objects_by_label.items():
#             # Project points to 3D
#             points_3d = project_to_3d(points, K, dist_coeffs, label)

#             # Save to a CSV file inside the corresponding `objects` folder
#             output_file = os.path.join(objects_output_dir, f'blender_objects_{label}.csv')
#             np.savetxt(output_file, points_3d, delimiter=',', header='x,y,z', comments='')

#             print(f"Saved {len(points)} points for label '{label}' to {output_file}")

# def main():
#     get_object_data()




    

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from objects_world import parse_object_file, project_to_3d
import shutil

def clear_directory(directory):
    """Remove all contents from a directory and recreate it."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Delete all files and subfolders
    os.makedirs(directory, exist_ok=True)  # Recreate an empty directory

def read_bottom_mid_info_from_file(file_path):
    """Read bottom mid info from a text file."""
    bottom_mid_info = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    label = parts[0].strip()
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    bottom_mid_info.append({
                        'label': label,
                        'coordinates': (x, y)
                    })
    except Exception as e:
        print(f"Error reading bottom mid info from {file_path}: {e}")
    
    return bottom_mid_info

def get_object_data(scene_directory, bottom_mid_info_directory):
    # Intrinsic matrix
    K = np.array([[1594.7, 0, 655.2961], 
                [0, 1607.7, 414.3627], 
                [0, 0, 1]])

    dist_coeffs = [-0.352238385718678, -0.333547811831122]
    objects_directory = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\scene_directory_objects"
    clear_directory(objects_directory) 

    output_directory = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\output" # The folder to save the CSV files for Blender

    # Create the output directory if it doesn't exist
    os.makedirs(objects_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(scene_directory):
        file_path = os.path.join(scene_directory, filename)
    
        # Check if it's a file and an image (common formats)
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            print(f"Found image: {filename}")
            
            image = cv2.imread(file_path)
            
            if image is None:
                print(f"Error: Could not read the image {filename}.")
                continue
                
            # Get the corresponding bottom_mid_info file
            image_name = os.path.splitext(filename)[0]  # Extract filename without extension
            bottom_mid_file = os.path.join(bottom_mid_info_directory, f"{image_name}_midpoints.txt")
            
            if not os.path.exists(bottom_mid_file):
                print(f"Warning: No bottom_mid_info file found for {filename}. Skipping.")
                continue
                
            # Read bottom_mid_info from the file
            bottom_mid_info = read_bottom_mid_info_from_file(bottom_mid_file)
            
            # Create output directories
            image_output_dir = os.path.join(output_directory, image_name)
            objects_output_dir = os.path.join(image_output_dir, "objects")
            os.makedirs(objects_output_dir, exist_ok=True)
            
            # # Draw detected points on the image (optional)
            # result_img = image.copy()
            # for detection in bottom_mid_info:
            #     x, y = int(detection['coordinates'][0]), int(detection['coordinates'][1])
            #     cv2.circle(result_img, (x, y), 5, (0, 255, 0), -1)
            #     cv2.putText(result_img, detection['label'], (x, y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the bottom_mid_info to the objects_directory
            txt_filename = image_name + ".txt"
            output_file_path = os.path.join(objects_directory, txt_filename)
            
            with open(output_file_path, "w") as f:
                for detection in bottom_mid_info:
                    f.write(f"{detection['label']}, {detection['coordinates'][0]}, {detection['coordinates'][1]}\n")
            
            # # Display the image with detected points (optional)
            # cv2.imshow("Image with Points", result_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Process and project to 3D
            objects_by_label = parse_object_file(output_file_path)
            
            for label, points in objects_by_label.items():
                # Project points to 3D
                points_3d = project_to_3d(points, K, dist_coeffs, label)
                
                # Save to a CSV file inside the corresponding `objects` folder
                output_file = os.path.join(objects_output_dir, f'blender_objects_{label}.csv')
                np.savetxt(output_file, points_3d, delimiter=',', header='x,y,z', comments='')
                
                print(f"Saved {len(points)} points for label '{label}' to {output_file}")

def main():
    scene_directory = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\input\curvedlanesdata\scene1_frontcam"  # Directory with your images
    bottom_mid_info_directory = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\input\curvedlanesdata\scene1_2dpts"  # Directory with your bottom_mid_info txt files
    
    get_object_data(scene_directory, bottom_mid_info_directory)

if __name__ == "__main__":
    main()