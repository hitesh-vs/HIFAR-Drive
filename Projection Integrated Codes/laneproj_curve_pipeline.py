


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.spatial import distance
import shutil

def ensure_directory_exists(file_path):
    """
    Create directory for the given file path if it doesn't exist.
    
    Parameters:
    - file_path: path to the file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def load_coordinates(file_path):
    """
    Load coordinates from a text file with comma-separated x,y values.
    
    Parameters:
    - file_path: path to the text file
    
    Returns:
    - numpy array of coordinates
    """
    try:
        with open(file_path, 'r') as file:
            # Read lines, split by comma, convert to float
            coordinates = [list(map(float, line.strip().split(','))) 
                           for line in file if line.strip()]
        return np.array(coordinates)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except ValueError:
        print("Error: Invalid coordinate format. Use x,y format.")
        return None

def skeletonize_and_sample_curve(points, img_size=(1024, 1024), num_points=50):
    """
    Skeletonize a binary mask and sample points along the skeleton.
    
    Parameters:
    - points: numpy array of coordinates defining the curve
    - img_size: size of the image (height, width)
    - num_points: number of points to sample along the skeleton
    
    Returns:
    - numpy array of sampled points along the curve's skeleton
    """
    # Create a binary mask from points
    mask = np.zeros(img_size, dtype=np.uint8)
    
    # Draw the curve as a thick line on the mask
    points_int = points.astype(np.int32)
    cv2.polylines(mask, [points_int], False, 255, thickness=5)
    
    # Apply skeletonization
    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.ximgproc.thinning(mask)
    
    # Find non-zero points (the skeleton)
    y_coords, x_coords = np.where(skeleton > 0)
    skeleton_points = np.column_stack((x_coords, y_coords))
    
    if len(skeleton_points) == 0:
        print("Warning: Skeletonization resulted in no points. Using original curve.")
        return sample_along_curve_evenly(points, num_points)
    
    # Sort points to create a continuous curve
    # Start with the leftmost point
    sorted_indices = np.argsort(skeleton_points[:, 0])
    start_point = skeleton_points[sorted_indices[0]]
    
    ordered_points = [start_point]
    remaining_points = list(skeleton_points[sorted_indices[1:]])
    
    # Iteratively find the nearest point
    while remaining_points and len(ordered_points) < len(skeleton_points):
        last_point = ordered_points[-1]
        
        # Find the closest point to the last added point
        distances = [distance.euclidean(last_point, p) for p in remaining_points]
        nearest_idx = np.argmin(distances)
        
        # Add the nearest point and remove it from remaining points
        ordered_points.append(remaining_points[nearest_idx])
        remaining_points.pop(nearest_idx)
    
    ordered_points = np.array(ordered_points)
    
    # Sample evenly along the ordered curve
    return sample_along_curve_evenly(ordered_points, num_points)

def sample_along_curve_evenly(points, num_points):
    """
    Sample evenly spaced points along a curve represented by discrete points.
    
    Parameters:
    - points: numpy array of curve points
    - num_points: desired number of sampled points
    
    Returns:
    - evenly sampled points along the curve
    """
    # Calculate cumulative distances along the curve
    dists = np.zeros(len(points))
    for i in range(1, len(points)):
        dists[i] = dists[i-1] + distance.euclidean(points[i], points[i-1])
    
    # Calculate total curve length
    total_length = dists[-1]
    
    # Generate evenly spaced distances
    sample_dists = np.linspace(0, total_length, num_points)
    
    # Interpolate to find the points at these distances
    sampled_points = []
    for d in sample_dists:
        # Find the segment containing this distance
        idx = np.searchsorted(dists, d)
        if idx == 0:
            sampled_points.append(points[0])
        elif idx >= len(points):
            sampled_points.append(points[-1])
        else:
            # Linearly interpolate between the two nearest points
            prev_idx = idx - 1
            segment_length = dists[idx] - dists[prev_idx]
            if segment_length > 0:
                t = (d - dists[prev_idx]) / segment_length
                interp_point = points[prev_idx] + t * (points[idx] - points[prev_idx])
                sampled_points.append(interp_point)
            else:
                sampled_points.append(points[idx])
    
    return np.array(sampled_points)

def sample_line_points_with_depth(input_file, depth_map_path, num_points=50, 
                                  output_coords_file='sampled_lane_points.txt',
                                  img_size=(1024, 1024),
                                  visualize=True):
    """
    Sample points from a lane coordinate file using skeletonization and extract depth values.
    
    Parameters:
    - input_file: path to input coordinate file
    - depth_map_path: path to the depth map numpy file
    - num_points: number of points to sample
    - output_coords_file: file to save sampled coordinates
    - output_depth_file: file to save depth values
    - img_size: size of the image (height, width)
    - visualize: whether to visualize the results
    """
    # Load coordinates from file
    points = load_coordinates(input_file)
    
    if points is None:
        return
    
    # Ensure output directories exist
    ensure_directory_exists(output_coords_file)
    # ensure_directory_exists(output_depth_file) 
    
    # Sample points along the curve using skeletonization
    sampled_points = skeletonize_and_sample_curve(points, img_size, num_points)
    
    # Save sampled points to file
    np.savetxt(output_coords_file, sampled_points, delimiter=',', fmt='%.4f')
    print(f"Points saved to {output_coords_file}")
    
    # Load depth map
    depth_map = np.load(depth_map_path)
    # print(f"Depth map shape: {depth_map.shape}") 
    
    # Extract depth values for sampled points
    # depth_values = [] 
    for x, y in sampled_points:
        # Convert to integer coordinates
        x_int, y_int = int(x), int(y)
        
        # Ensure coordinates are within depth map bounds
        x_int = max(0, min(x_int, depth_map.shape[1] - 1))
        y_int = max(0, min(y_int, depth_map.shape[0] - 1))
        
        # Extract depth value (NumPy uses [row, column] ordering)
        # depth_values.append(depth_map[y_int, x_int]) 
    
    # Save depth values
    # np.save(output_depth_file, np.array(depth_values)) 
    # print(f"Depth values saved to {output_depth_file}") 
    
    if visualize:
        # Create a visualization of the mask, skeleton, and sampled points
        mask = np.zeros(img_size, dtype=np.uint8)
        cv2.polylines(mask, [points.astype(np.int32)], False, 128, thickness=5)
        
        # Apply skeletonization for visualization
        skeleton = cv2.ximgproc.thinning(mask.copy())
        
        # Create a colored visualization
        vis_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        vis_img[mask > 0] = [0, 0, 128]  # Original mask in blue
        vis_img[skeleton > 0] = [0, 128, 0]  # Skeleton in green
        
        # Draw sampled points on top
        for i, (x, y) in enumerate(sampled_points):
            cv2.circle(vis_img, (int(x), int(y)), 3, [255, 0, 0], -1)  # Sampled points in red
        
        # plt.figure(figsize=(15, 8))
        
        # # Show curve extraction
        # plt.subplot(2, 2, 1)
        # plt.imshow(vis_img)
        # plt.title('Curve Extraction')
        # plt.axis('equal')
        
        # # Show depth visualization
        # plt.subplot(2, 2, 2)
        # scatter = plt.scatter(sampled_points[:, 0], sampled_points[:, 1], 
        #                     c=depth_values, cmap='viridis', s=50)
        # plt.colorbar(scatter, label='Depth')
        # plt.title('Sampled Lane Points with Depth')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.axis('equal')
        
        # # Show depth profile
        # plt.subplot(2, 2, 3)
        # plt.plot(depth_values, marker='o')
        # plt.title('Depth Values Along Curve')
        # plt.xlabel('Point Index')
        # plt.ylabel('Depth')
        
        # # 3D Visualization
        # ax = plt.subplot(2, 2, 4, projection='3d')
        # ax.scatter(sampled_points[:, 0], sampled_points[:, 1], depth_values, 
        #           c=depth_values, cmap='viridis', s=50)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Depth')
        # ax.set_title('3D Visualization of Lane Points')
        
        # plt.tight_layout()
        # plt.show()
    
    return sampled_points



def sampled_points(current_folder, depth_placeholder):
    # current_folder = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\current_scene_masks"
    # depth_placeholder = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\scene4_depth\0037_pred.npy"

    # Step 1: Remove existing "_sampled" folders
    subfolders = [f.path for f in os.scandir(current_folder) if f.is_dir()]
    for subfolder in subfolders:
        if subfolder.endswith("_sampled"):  # Delete only sampled folders
            shutil.rmtree(subfolder)
            print(f"Deleted old folder: {subfolder}")

    # Step 2: Process subfolders and create new "_sampled" folders
    subfolders = [f.path for f in os.scandir(current_folder) if f.is_dir()]
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        sampled_folder = os.path.join(current_folder, f"{subfolder_name}_sampled")
        os.makedirs(sampled_folder, exist_ok=True)  # Create fresh "_sampled" folder

        txt_files = [file for file in os.listdir(subfolder) if file.endswith('.txt')]

        for i, txt_file in enumerate(txt_files):
            input_file = os.path.join(subfolder, txt_file)
            output_file = os.path.join(sampled_folder, f"sampled_lane_points_{i}.txt")

            # Call the function to process the file
            sample_line_points_with_depth(
                input_file=input_file,
                depth_map_path=depth_placeholder,
                num_points=50,
                output_coords_file=output_file,  
                img_size=(1280, 960),  
                visualize=True  
            )

    print("Processing completed successfully.")


def main():
    sampled_points()
# Example usage
if __name__ == '__main__':
    main()
        