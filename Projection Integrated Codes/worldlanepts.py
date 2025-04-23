import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
# Intrinsic matrix
K = np.array([[1594.7, 0, 655.2961], 
              [0, 1607.7, 414.3627], 
              [0, 0, 1]])

dist_coeffs = [-0.352238385718678, -0.333547811831122]

def project_to_3d(image_points, K, dist_coeffs):
    """
    Project 2D image plane coordinates to 3D world coordinates 
    using ground plane assumption
    
    Parameters:
    - image_points: nx2 array of (x,y) pixel coordinates
    - K: 3x3 intrinsic matrix
    - depth_values: corresponding depth values for each point
    - dist_coeffs: radial distortion coefficients
    
    Returns:
    - 3D points in camera coordinate system
    """
    # Validate input lengths
    # assert len(image_points) == len(depth_values), "Number of points and depth values must match" 
    
    # # Normalize depth values
    # depth_values = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())
    # depth_values =  depth_values * 100
    
    # Prepare points for matrix multiplication
    # Add homogeneous coordinate
    homogeneous_points = np.column_stack([image_points, np.ones(len(image_points))])
    
    # Ground plane fixed Y value
    GROUND_PLANE_Y = 0
    
    # Compute 3D points using matrix multiplication
    points_3d = []
    for point in homogeneous_points:
        # Compute K^-1 * point
        ray_direction = np.linalg.inv(K) @ point

        
        # Normalize the ray direction
        ray_magnitude = np.linalg.norm(ray_direction)
        
        # Compute lambda (normalization factor)
        # Solve: -1.5 = λ * ray_direction[1]
        lambda_factor = -1.5 / ray_direction[1]
        
        # Compute 3D point coordinates
        X = lambda_factor * ray_direction[0]
        Y = GROUND_PLANE_Y
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
    

def blender_points(current_folder, output_directory):
    # current_folder = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\current_scene_masks"
    # output_directory = r"C:\Users\farha\OneDrive\Desktop\Lanes_integrated\output"
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through subfolders
    subfolders = [f.path for f in os.scandir(current_folder) if f.is_dir()]
    for subfolder in subfolders:
        if subfolder.endswith("_sampled"):  
            imagenumber = subfolder.replace("_sampled", "")  # Extract the image number
            image_output_folder = os.path.join(output_directory, os.path.basename(imagenumber))
            os.makedirs(image_output_folder, exist_ok=True)  # Create corresponding output subfolder
            
            # Process and save points
            i = 0
            while os.path.exists(f'{subfolder}/sampled_lane_points_{i}.txt'):  # Ensure file exists
                points = np.loadtxt(f'{subfolder}/sampled_lane_points_{i}.txt', delimiter=',')
                points_3d = project_to_3d(points, K, dist_coeffs)
                np.savetxt(f'{image_output_folder}/blender_points_mask{i}.csv', points_3d, delimiter=',',
                           header='x,y,z', comments='')
                i += 1

                
                
        # Plot 3D points
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Scatter plot of 3D points
    # scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
    #             c='red', marker='o', s=5)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Projection of Image Points')

    # plt.tight_layout()
    
    # # Save the figure
    # #plt.savefig('3d_projection_mask_13_mask4.png', dpi=300, bbox_inches='tight')
    
    # # Optional: also show the plot
    # plt.show()

def main():

    blender_points()

if __name__ == '__main__':
    main()
# import numpy as np

# def accurate_project_to_3d(image_points, K, depth_values, dist_coeffs=None):
#     """
#     Accurately project 2D image plane coordinates to 3D world coordinates
    
#     Parameters:
#     - image_points: nx2 array of (x,y) pixel coordinates
#     - K: 3x3 intrinsic matrix
#     - depth_values: corresponding depth values for each point
#     - dist_coeffs: optional distortion coefficients
    
#     Returns:
#     - 3D points in camera coordinate system
#     """
#     # Validate inputs
#     image_points = np.asarray(image_points)
#     depth_values = np.asarray(depth_values)
#     assert image_points.shape[0] == depth_values.shape[0], "Point count mismatch"
    
#     # Undistort points if distortion coefficients are provided
#     if dist_coeffs is not None:
#         # Placeholder for undistortion - you'd typically use cv2.undistortPoints()
#         # This is a simplified approximation
#         image_points_undistorted = image_points * (1 + dist_coeffs[0] * np.sum(image_points**2, axis=1)[:, np.newaxis])
#     else:
#         image_points_undistorted = image_points
    
#     # Prepare homogeneous coordinates
#     image_points_homogeneous = np.column_stack([
#         image_points_undistorted, 
#         np.ones(len(image_points_undistorted))
#     ])
    
#     # Compute 3D points in camera coordinate system
#     points_3d = []
#     for point, depth in zip(image_points_homogeneous, depth_values):
#         # Inverse projection
#         # K^-1 * pixel_point gives a ray in camera coordinate system
#         ray = np.linalg.inv(K) @ point
        
#         # Scale ray by actual depth
#         point_3d = ray * depth
        
#         points_3d.append(point_3d)
    
#     # Convert to numpy array
#     points_3d = np.array(points_3d)
    
#     return points_3d

# def main():
#     # Intrinsic matrix (example)
#     K = np.array([[1594.7, 0, 655.2961], 
#                   [0, 1607.7, 414.3627], 
#                   [0, 0, 1]])
    
#     # Example usage
#     image_points = np.random.rand(100, 2) * 1000  # Random image points
#     depth_values = np.random.uniform(1, 10, 100)  # Random depth values
    
#     # Project to 3D
#     points_3d = accurate_project_to_3d(image_points, K, depth_values)
    
#     # Visualization (optional)
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
    
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
    
#     ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
#                c='red', marker='o', s=5)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Accurate 3D Projection')
    
#     plt.show()

# if __name__ == '__main__':
#     main()