import bpy
import csv
import numpy as np
import os

def srgb_to_linear(c):
    """Convert an sRGB color component to linear RGB"""
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4

def create_plane_mesh(size=100, color_hex="353535", roughness=0.85):
    import bpy

    # Create a plane with the specified size
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground_Plane"
    
    # Convert hex color to RGB and then to linear
    r_srgb = int(color_hex[0:2], 16) / 255
    g_srgb = int(color_hex[2:4], 16) / 255
    b_srgb = int(color_hex[4:6], 16) / 255
    
    r = srgb_to_linear(r_srgb)
    g = srgb_to_linear(g_srgb)
    b = srgb_to_linear(b_srgb)
    
    # Create material for the plane
    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    
    # Clear default nodes
    mat.node_tree.nodes.clear()
    
    # Create nodes
    output_node = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Set color and roughness
    principled_node.inputs['Base Color'].default_value = (r, g, b, 1.0)
    principled_node.inputs['Roughness'].default_value = roughness
    
    # Connect nodes
    mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material to plane
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)
    
    return plane

def import_point_cloud_as_spheres(filepath, sphere_radius=0.02):
    # Read points from CSV
    points = []
    with open(filepath, 'r') as csvfile:
        # Skip header if exists
        try:
            next(csvfile)
        except:
            csvfile.seek(0)  # Reset file pointer if no header
        
        csvreader = csv.reader(csvfile)
        points = [list(map(float, row)) for row in csvreader]
    
    # Create a parent empty object to hold all spheres
    parent = bpy.data.objects.new(f"PointCloudParent_{os.path.basename(filepath)}", None)
    bpy.context.collection.objects.link(parent)
    
    # Create a sphere mesh that will be instanced
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, segments=8, ring_count=6)
    sphere_template = bpy.context.active_object
    sphere_template.name = "SphereTemplate"
    sphere_template.hide_set(True)  # Hide the template from the viewport
    
    # Create material for the spheres
    mat = create_sphere_material()
    
    # Apply material to the template
    if sphere_template.data.materials:
        sphere_template.data.materials[0] = mat
    else:
        sphere_template.data.materials.append(mat)
    
        # Sampling: pick a subset of points
    sample_ratio = 0.07  # Only use 20% points
    sampled_indices = np.random.choice(len(points), int(len(points) * sample_ratio), replace=False)
    sampled_points = [points[i] for i in sampled_indices]

    # Instance spheres at sampled point locations
    for i, point in enumerate(sampled_points):
        obj = sphere_template.copy()
        obj.data = sphere_template.data.copy()
        obj.name = f"Point_{i}"
        
        obj.location = point
        bpy.context.collection.objects.link(obj)
        obj.parent = parent
    
    return parent, len(points)

def create_sphere_material(color=(1, 0, 0, 1)):
    # Create a new material for spheres
    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    
    node_tree = mat.node_tree
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    
    # Create glossy shader for a more appealing look
    glossy_node = node_tree.nodes.new(type='ShaderNodeBsdfGlossy')
    glossy_node.inputs['Color'].default_value = color
    glossy_node.inputs['Roughness'].default_value = 0.1
    
    node_tree.links.new(glossy_node.outputs['BSDF'], output_node.inputs['Surface'])
    return mat

def main():
    # Clear existing objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    # Directory containing CSV files
    directory = r'C:\Users\vshit\Downloads\20230814_Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach\20230814_Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach\scene6_output\image_14'
    
    # Color palette for different lanes
    colors = [
        (1, 1, 1, 1),    # Red
        (1, 1, 1, 1),    # Green
        (1, 1, 1, 1),    # Blue
        (1, 1, 1, 1),    # Yellow
        (1, 1, 1, 1),    # Magenta
        (1, 1, 1, 1)     # Cyan
    ]
    
    # List to store processed files
    processed_files = []
    total_points = 0
    
    # Iterate through CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            
            # Skip files already processed
            if filepath in processed_files:
                continue
            
            # Determine color based on iteration
            color_index = len(processed_files) % len(colors)
            sphere_color = colors[color_index]
            
            # Create material with this color
            sphere_mat = create_sphere_material(sphere_color)
            
            # Adjust sphere radius based on point cloud density
            # Smaller radius for dense clouds, larger for sparse
            sphere_radius = 0.02
            
            # Import point cloud as spheres
            point_cloud_parent, num_points = import_point_cloud_as_spheres(filepath, sphere_radius)
            total_points += num_points
            
            # Iterate through all children and assign the material
            for child in point_cloud_parent.children:
                if child.data.materials:
                    child.data.materials[0] = sphere_mat
                else:
                    child.data.materials.append(sphere_mat)
            
            # Add to processed files
            processed_files.append(filepath)
            
            print(f"Processed {filepath} with {num_points} points using color {sphere_color}")
    
    print(f"Total points visualized: {total_points}")
    
    # Create the ground plane with specified parameters
    create_plane_mesh(size=300, color_hex="353535", roughness=0.85)
    
    # Set render engine to Cycles for better rendering
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Render settings
    bpy.context.scene.cycles.samples = 128
    
    # Add lighting
    #bpy.ops.object.light_add(type='SUN', radius=1, location=(5, 5, 10))
    #sun = bpy.context.active_object
    #sun.data.energy = 2.0

# Run the script
main()