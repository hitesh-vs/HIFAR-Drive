import numpy as np
import matplotlib.pyplot as plt
import re

# Intrinsic matrix
K = np.array([[1594.7, 0, 655.2961], 
              [0, 1607.7, 414.3627], 
              [0, 0, 1]])

dist_coeffs = [-0.352238385718678, -0.333547811831122]

def project_to_3d(image_points, K, dist_coeffs, label):
    """
    Project 2D image plane coordinates to 3D world coordinates 
    using ground plane assumption with special handling for stopsigns
    
    Parameters:
    - image_points: nx2 array of (x,y) pixel coordinates
    - K: 3x3 intrinsic matrix
    - depth_values: corresponding depth values for each point
    - dist_coeffs: radial distortion coefficients
    - label: object label to determine projection method
    
    Returns:
    - 3D points in camera coordinate system
    """
    # Validate input lengths
    # assert len(image_points) == len(depth_values), "Number of points and depth values must match"
    
    # # Normalize depth values
    # depth_values = (depth_values - depth_values.min()) / ((depth_values.max() - depth_values.min()) + 1e-6)
    # depth_values = depth_values * 100
    
    # Prepare points for matrix multiplication
    # Add homogeneous coordinate
    homogeneous_points = np.column_stack([image_points, np.ones(len(image_points))])
    
    # Set the Y-value based on object type
    # For stop signs, use a fixed height (typical stop sign height is around 7 feet / 2.1 meters)
    STOPSIGN_HEIGHT = 0.55 # Using a more realistic height in meters
    TRAFFIC_HEIGHT = 1.5
    
    # Compute 3D points using matrix multiplication
    points_3d = []
    depth_values = np.ones(len(image_points))  # Dummy depth values for now
    
    for point, z in zip(homogeneous_points, depth_values):
        # Compute K^-1 * point
        ray_direction = np.linalg.inv(K) @ point
        
        # Different projection method based on object type
        if label == "stopsign" or label == "street_sign":
            # For stop signs, we project to a plane at STOPSIGN_HEIGHT
            # Calculate lambda that puts the point at height STOPSIGN_HEIGHT
            #STOPSIGN_HEIGHT_NEW = STOPSIGN_HEIGHT*point[1] / 369 # Using a more realistic height in meters
            lambda_factor = STOPSIGN_HEIGHT / ray_direction[1]
            print('stop sign detected')
            # Compute 3D point coordinates
            X = lambda_factor * ray_direction[0]
            Y = STOPSIGN_HEIGHT
            Z = lambda_factor * ray_direction[2]

        elif label == "traffic_light_green" or label == "traffic_light_red":
            # For stop signs, we project to a plane at STOPSIGN_HEIGHT
            # Calculate lambda that puts the point at height STOPSIGN_HEIGHT
            lambda_factor = TRAFFIC_HEIGHT / ray_direction[1]
            print('traffic light detected')
            # Compute 3D point coordinates
            X = lambda_factor * ray_direction[0]
            Y = TRAFFIC_HEIGHT
            Z = lambda_factor * ray_direction[2]

        else:
            # Standard ground plane projection (Y=0)
            lambda_factor = -1.5 / ray_direction[1]
            print('other object detected')
            # Compute 3D point coordinates
            X = lambda_factor * ray_direction[0]
            Y = 0  # Ground plane
            Z = lambda_factor * ray_direction[2]
        
        points_3d.append([X, Y, Z])

    # Coordinate frame rotation matrix
    rotation_matrix = np.array([
        [1, 0, 0],   # x stays the same
        [0, 0, -1],  # y becomes -z
        [0, 1, 0]    # z becomes y
    ])
    
    # Apply rotation to each point
    rotated_points_3d = [rotation_matrix @ point for point in points_3d]

    rotation_matrix_flip = np.array([
        [-1, 0, 0],  # X becomes -X
        [0, 1, 0],   # Y stays the same
        [0, 0, -1]   # Z becomes -Z
    ])

    rotated_points_3d = [rotation_matrix_flip @ point for point in rotated_points_3d]
    
    return np.array(rotated_points_3d)

# def parse_object_file(file_path):
#     """
#     Parse a file containing object detections in the format 'label, x, y'
    
#     Returns:
#     - Dictionary with labels as keys and lists of points as values
#     - Dictionary with labels as keys and dummy depth values as values
#     """
#     with open(file_path, 'r') as f:
#         content = f.read()
    
#     # Find all instances of 'label, x, y'
#     pattern = r'(\w+),\s*(\d+),\s*(\d+)'
#     matches = re.findall(pattern, content)
    
#     # Group by labels
#     objects_by_label = {}
    
#     for match in matches:
#         label, x, y = match
#         if label not in objects_by_label:
#             objects_by_label[label] = []
        
#         objects_by_label[label].append([int(x), int(y)])
    
#     # Convert to numpy arrays and create dummy depth values
#     for label in objects_by_label:
#         objects_by_label[label] = np.array(objects_by_label[label])
    
#     # Create dummy depth values (you may want to replace this with real depth data)
#     # depth_values_by_label = {}
#     # for label, points in objects_by_label.items():
#     #     # Create linear depth values from 1 to 2 for each label's points
#     #     depth_values_by_label[label] = np.linspace(1, 2, len(points))
    
#     return objects_by_label

def parse_object_file(file_path):
    """
    Parse a file containing object detections in the format 'label, x, y'
    
    Returns:
    - Dictionary with labels as keys and lists of points as values
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Updated pattern to match decimal numbers
    pattern = r'(\w+),\s*([\d\.]+),\s*([\d\.]+)'
    matches = re.findall(pattern, content)
    
    # Group by labels
    objects_by_label = {}
    
    for match in matches:
        label, x, y = match
        if label not in objects_by_label:
            objects_by_label[label] = []
        
        # Convert to float instead of int
        objects_by_label[label].append([float(x), float(y)])
    
    # Convert to numpy arrays
    for label in objects_by_label:
        objects_by_label[label] = np.array(objects_by_label[label])
    
    return objects_by_label

def main():
    # Path to your input file
    input_file = 'objectloc_830.txt'  # Change to your file path
    
    # Parse the object file
    objects_by_label = parse_object_file(input_file)
    
    # Process each label and save as separate CSV
    for label, points in objects_by_label.items():
        # depth_values = depth_values_by_label[label] 
        
        # Project to 3D with label information
        points_3d = project_to_3d(points, K, dist_coeffs, label)
        
        # Save points to a CSV file for Blender, using the label as part of the filename
        output_file = f'blender_s8i30_csv/objects/blender_objects_{label}.csv'
        np.savetxt(output_file, points_3d, delimiter=',', 
                  header='x,y,z', comments='')
        
        print(f"Saved {len(points)} points for label '{label}' to {output_file}")
    
    # Visualize all points in one 3D plot (optional)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Different colors for different labels
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # Plot each label with a different color
    for i, (label, points) in enumerate(objects_by_label.items()):
        # depth_values = depth_values_by_label[label] 
        points_3d = project_to_3d(points, K, dist_coeffs, label)
        
        color = colors[i % len(colors)]  # Cycle through colors
        
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=color, marker='o', s=10, label=f"{label}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Projection of Objects')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('3d_projection_objects.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()