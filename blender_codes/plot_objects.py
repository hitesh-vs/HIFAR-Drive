import bpy
import os
import csv
import math
import re
import glob

def clear_existing_cars():
    """
    Clear existing car objects from the scene
    """
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select all objects with 'Car' in their name
    for obj in bpy.data.objects:
        if 'Car' in obj.name:
            obj.select_set(True)
    
    # Delete selected objects
    if any(obj.select_get() for obj in bpy.data.objects):
        bpy.ops.object.delete()

def append_object_from_blend(blend_file_path, object_name):
    """
    Append a specific object from another blend file.
    
    Parameters:
    blend_file_path (str): Full path to the source .blend file
    object_name (str): Name of the object to append
    
    Returns:
    bpy.types.Object or None: The appended object or None if it fails
    """
    # Check if the file exists
    if not os.path.exists(blend_file_path):
        print(f"Error: File '{blend_file_path}' not found.")
        return None
    
    # The internal path to objects in a blend file is always "/Object/"
    inner_path = "Object"
    
    # Append the object
    try:
        bpy.ops.wm.append(
            filepath=os.path.join(blend_file_path, inner_path, object_name),
            directory=os.path.join(blend_file_path, inner_path),
            filename=object_name
        )
        print(f"Successfully appended object '{object_name}'")
        
        # Return the newly appended object
        if object_name in bpy.data.objects:
            return bpy.data.objects[object_name]
        else:
            # Try to find the object by searching for recently added objects
            for obj in bpy.data.objects:
                if object_name in obj.name:
                    print(f"Found object with similar name: {obj.name}")
                    return obj
            
            print(f"Warning: Object '{object_name}' was appended but cannot be found")
            return None
    except Exception as e:
        print(f"Error: Failed to append object '{object_name}'. {str(e)}")
        return None

def apply_texture_to_material(obj, image_path):
    """
    Apply a texture to materials in an object that start with "Material"
    
    Parameters:
    obj (bpy.types.Object): The object to modify materials for
    image_path (str): Path to the image texture file
    
    Returns:
    bool: True if texture was applied successfully, False otherwise
    """
    if not os.path.exists(image_path):
        print(f"Error: Image texture '{image_path}' not found.")
        return False
    
    # Load the image
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return False
    
    # Find materials that start with "Material"
    material_modified = False
    
    for slot in obj.material_slots:
        if slot.material and slot.material.name.startswith("Material"):
            mat = slot.material
            
            # Create a copy of the material for this specific instance
            new_mat = mat.copy()
            slot.material = new_mat
            mat = new_mat
            
            # Make sure material uses nodes
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear existing nodes
            nodes.clear()
            
            # Create texture coordinate node
            tex_coord = nodes.new('ShaderNodeTexCoord')
            tex_coord.location = (-800, 0)
            
            # Create mapping node for better texture control
            mapping = nodes.new('ShaderNodeMapping')
            mapping.location = (-600, 0)
            
            # Create image texture node
            tex_image = nodes.new('ShaderNodeTexImage')
            tex_image.image = img
            tex_image.location = (-400, 0)
            
            # Create BSDF shader
            principled_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            principled_bsdf.location = (0, 0)
            
            # Create output node
            output = nodes.new('ShaderNodeOutputMaterial')
            output.location = (200, 0)
            
            # Link nodes
            links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
            links.new(tex_image.outputs['Color'], principled_bsdf.inputs['Base Color'])
            links.new(principled_bsdf.outputs['BSDF'], output.inputs['Surface'])
            
            material_modified = True
            print(f"Applied texture {os.path.basename(image_path)} to material: {mat.name}")
    
    if not material_modified:
        print(f"No materials starting with 'Material' found in {obj.name}")
    
    return material_modified

def get_scene_id_from_csv_path(csv_path):
    """
    Extract the scene ID (like '0004') from the CSV file path
    Specifically looks for a 4-digit folder in the path structure
    
    Parameters:
    csv_path (str): Path to CSV file
    
    Returns:
    str: Scene ID or None if not found
    """
    # Split the path into components
    path_parts = csv_path.split(os.sep)
    
    # Look for a 4-digit folder name in the path
    for part in path_parts:
        if re.match(r'^\d{4}$', part):
            return part
    
    # Alternative approach: check each directory level for a 4-digit pattern
    current_path = os.path.dirname(csv_path)
    max_levels = 5  # Maximum number of parent directories to check
    
    for _ in range(max_levels):
        folder_name = os.path.basename(current_path)
        if re.match(r'^\d{4}$', folder_name):
            return folder_name
        
        # Move up one directory level
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # Reached the root directory
            break
        current_path = parent_path
    
    # If all else fails, try to find any 4-digit sequence in the path
    match = re.search(r'[/\\](\d{4})[/\\]', csv_path)
    if match:
        return match.group(1)
    
    return None

def get_image_textures_for_scene(texture_folder, scene_id):
    """
    Get a list of image textures for a specific scene
    
    Parameters:
    texture_folder (str): Path to the folder containing texture images
    scene_id (str): Scene ID to filter by (e.g., '0004')
    
    Returns:
    list: List of paths to image textures for the scene
    """
    # Find all images matching the pattern scene_id_index_*.jpg
    pattern = os.path.join(texture_folder, f"{scene_id}_*.*")
    images = glob.glob(pattern)
    
    # Sort images by their index number
    def get_index(path):
        filename = os.path.basename(path)
        match = re.search(r'_(\d+)_', filename)
        if match:
            return int(match.group(1))
        return 0
    
    sorted_images = sorted(images, key=get_index)

    # Print found images for debugging
    if sorted_images:
        print(f"Found {len(sorted_images)} textures for scene {scene_id}:")
        for img in sorted_images:
            print(f"  - {os.path.basename(img)}")
    else:
        print(f"No textures found matching pattern {scene_id}_*.*")
        
    return sorted_images

def place_sign_with_dynamic_texture(csv_file_path, blend_file_path, object_name, texture_folder, scale_factor=1.0):
    """
    Place sign objects at specified coordinates from a CSV file and apply dynamic textures
    
    Parameters:
    csv_file_path (str): Path to the CSV file with coordinates
    blend_file_path (str): Path to the blend file containing the sign model
    object_name (str): Name of the sign object in the blend file
    texture_folder (str): Path to the folder containing texture images
    scale_factor (float): Scale factor to apply to the sign model
    """
    # Get the scene ID from the CSV file path
    scene_id = get_scene_id_from_csv_path(csv_file_path)
    if not scene_id:
        print(f"Could not extract scene ID from CSV path: {csv_file_path}")
        return
    
    print(f"Detected scene ID: {scene_id}")
    
    # Get all texture images for this scene
    texture_images = get_image_textures_for_scene(texture_folder, scene_id)
    if not texture_images:
        print(f"No texture images found for scene {scene_id} in folder {texture_folder}")
        return
    
    print(f"Found {len(texture_images)} texture images for scene {scene_id}")
    
    # Get prototype sign to duplicate
    prototype_sign = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_sign:
        print("Failed to import sign prototype. Exiting.")
        return
    
    # Set scale of prototype sign
    prototype_sign.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype sign (we'll use it as a template for duplicating)
    prototype_sign.hide_viewport = True
    prototype_sign.hide_render = True
    
    # Read coordinates from CSV
    sign_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Get the texture for this sign
                if sign_count < len(texture_images):
                    texture_path = texture_images[sign_count]
                else:
                    # If we have more signs than textures, reuse the last texture
                    texture_path = texture_images[-1] if texture_images else None
                    print(f"Warning: More signs than textures. Reusing last texture for sign {sign_count}")
                
                # Skip if no texture is available
                if not texture_path:
                    print(f"No texture available for sign {sign_count}, skipping")
                    continue
                
                # Parse coordinates and rotation
                try:
                    if len(row) >= 4:  # If we have x, y, z, phi
                        x, y, z, phi = map(float, row)
                    elif len(row) == 3:  # If we have just x, y, z
                        x, y, z = map(float, row)
                        phi = 0.0  # Default rotation
                    else:
                        print(f"Skipping invalid row: {row}")
                        continue
                    
                    # Duplicate the sign
                    sign_copy = prototype_sign.copy()
                    sign_copy.data = prototype_sign.data.copy()
                    sign_copy.name = f"StreetSign_{scene_id}_{sign_count}"
                    bpy.context.collection.objects.link(sign_copy)
                    
                    # Show the sign copy
                    sign_copy.hide_viewport = False
                    sign_copy.hide_render = False
                    
                    # Position the sign at the XYZ coordinates from CSV
                    sign_copy.location = (x, y, z)
                    
                    # Set the Z rotation (phi) in radians if available
                    if len(row) >= 4:
                        phi_radians = math.radians(phi)
                        sign_copy.rotation_euler = (
                            sign_copy.rotation_euler.x,
                            sign_copy.rotation_euler.y,
                            phi_radians
                        )
                    
                    # Apply the specific texture for this sign
                    apply_texture_to_material(sign_copy, texture_path)
                    
                    print(f"Placed StreetSign_{scene_id}_{sign_count} at position ({x}, {y}, {z}) with texture {os.path.basename(texture_path)}")
                    sign_count += 1
                    
                except Exception as e:
                    print(f"Error placing sign: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {sign_count} street signs in the scene")

def place_cars_from_csv(csv_file_path, blend_file_path, object_name, scale_factor=0.01):
    """
    Place car objects at specified coordinates from a CSV file
    
    Parameters:
    csv_file_path (str): Path to the CSV file with coordinates and rotation
    blend_file_path (str): Path to the blend file containing the car model
    object_name (str): Name of the car object in the blend file
    scale_factor (float): Scale factor to apply to the car model
    """
    # Get prototype car to duplicate
    prototype_car = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_car:
        print("Failed to import car prototype. Exiting.")
        return
    
    # Set scale of prototype car
    prototype_car.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype car (we'll use it as a template for duplicating)
    prototype_car.hide_viewport = True
    prototype_car.hide_render = True
    
    # Read coordinates from CSV
    car_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    x, y, z, phi = map(float, row)
                    
                    # Duplicate the car
                    car_copy = prototype_car.copy()
                    car_copy.data = prototype_car.data.copy()
                    car_copy.name = f"Car_{car_count}"
                    bpy.context.collection.objects.link(car_copy)
                    
                    # Show the car copy
                    car_copy.hide_viewport = False
                    car_copy.hide_render = False
                    
                    # Position the car at the XYZ coordinates from CSV
                    car_copy.location = (x, y, prototype_car.location.z)
                    
                    # Set the Z rotation (phi) in radians
                    phi_radians = math.radians(phi)
                    
                    # Blender uses Euler rotations, with Z being the vertical axis for rotation
                    car_copy.rotation_euler = (
                        car_copy.rotation_euler.x,
                        car_copy.rotation_euler.y,
                        phi_radians
                    )
                    
                    print(f"Placed Car_{car_count} at position ({x}, {y}, {z}) with rotation {phi}")
                    car_count += 1
                    
                except Exception as e:
                    print(f"Error placing car: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {car_count} cars in the scene")

def place_ptruck_from_csv(csv_file_path, blend_file_path, object_name, scale_factor=0.01):
    """
    Place pickup truck objects at specified coordinates from a CSV file
    """
    # Get prototype car to duplicate
    prototype_car = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_car:
        print("Failed to import pickup truck prototype. Exiting.")
        return
    
    # Set scale of prototype car
    prototype_car.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype car (we'll use it as a template for duplicating)
    prototype_car.hide_viewport = True
    prototype_car.hide_render = True
    
    # Read coordinates from CSV
    car_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    x, y, z, phi = map(float, row)
                    
                    # Duplicate the car
                    car_copy = prototype_car.copy()
                    car_copy.data = prototype_car.data.copy()
                    car_copy.name = f"PickupTruck_{car_count}"
                    bpy.context.collection.objects.link(car_copy)
                    
                    # Show the car copy
                    car_copy.hide_viewport = False
                    car_copy.hide_render = False
                    
                    # Position the car at the XYZ coordinates from CSV
                    car_copy.location = (x, y, 0.9)
                    
                    # Set the Z rotation (phi) in radians
                    phi_radians = math.radians(phi)
                    
                    # Blender uses Euler rotations, with Z being the vertical axis for rotation
                    car_copy.rotation_euler = (
                        car_copy.rotation_euler.x,
                        car_copy.rotation_euler.y,
                        phi_radians
                    )
                    
                    print(f"Placed PickupTruck_{car_count} at position ({x}, {y}, {z}) with rotation {phi}")
                    car_count += 1
                    
                except Exception as e:
                    print(f"Error placing pickup truck: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {car_count} pickup trucks in the scene")

def place_person_from_csv(csv_file_path, blend_file_path, object_name, scale_factor=0.01):
    """
    Place person objects at specified coordinates from a CSV file
    """
    # Get prototype person to duplicate
    prototype_person = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_person:
        print("Failed to import person prototype. Exiting.")
        return
    
    # Set scale of prototype person
    prototype_person.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype person (we'll use it as a template for duplicating)
    prototype_person.hide_viewport = True
    prototype_person.hide_render = True
    
    # Read coordinates from CSV
    person_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    x, y, z = map(float, row)
                    
                    # Duplicate the person
                    person_copy = prototype_person.copy()
                    person_copy.data = prototype_person.data.copy()
                    person_copy.name = f"Person_{person_count}"
                    bpy.context.collection.objects.link(person_copy)
                    
                    # Show the person copy
                    person_copy.hide_viewport = False
                    person_copy.hide_render = False
                    
                    # Position the person at the XYZ coordinates from CSV
                    person_copy.location = (x, y, prototype_person.location.z)
                    
                    # Set the Z rotation (phi) in radians
                    #phi_radians = math.radians(phi)
                    
                    #person_copy.rotation_euler = (
                        #person_copy.rotation_euler.x,
                        #person_copy.rotation_euler.y,
                        #phi_radians
                    #)
                    
                    print(f"Placed Person_{person_count} at position ({x}, {y}, {z}) with rotation {phi}")
                    person_count += 1
                    
                except Exception as e:
                    print(f"Error placing person: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {person_count} people in the scene")

def place_truck_from_csv(csv_file_path, blend_file_path, object_name, scale_factor=0.01):
    """
    Place truck objects at specified coordinates from a CSV file
    """
    # Get prototype truck to duplicate
    prototype_truck = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_truck:
        print("Failed to import truck prototype. Exiting.")
        return
    
    # Set scale of prototype truck
    prototype_truck.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype truck (we'll use it as a template for duplicating)
    prototype_truck.hide_viewport = True
    prototype_truck.hide_render = True
    
    # Read coordinates from CSV
    truck_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    x, y, z, phi = map(float, row)
                    
                    # Duplicate the truck
                    truck_copy = prototype_truck.copy()
                    truck_copy.data = prototype_truck.data.copy()
                    truck_copy.name = f"Truck_{truck_count}"
                    bpy.context.collection.objects.link(truck_copy)
                    
                    # Show the truck copy
                    truck_copy.hide_viewport = False
                    truck_copy.hide_render = False
                    
                    # Position the truck at the XYZ coordinates from CSV
                    truck_copy.location = (x, y, prototype_truck.location.z)
                    
                    # Set the Z rotation (phi) in radians
                    phi_radians = math.radians(phi)
                    
                    truck_copy.rotation_euler = (
                        truck_copy.rotation_euler.x,
                        truck_copy.rotation_euler.y,
                        phi_radians
                    )
                    
                    print(f"Placed Truck_{truck_count} at position ({x}, {y}, {z}) with rotation {phi}")
                    truck_count += 1
                    
                except Exception as e:
                    print(f"Error placing truck: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {truck_count} trucks in the scene")
    
def place_dustbin_from_csv(csv_file_path, blend_file_path, object_name, scale_factor=0.01):
    """
    Place truck objects at specified coordinates from a CSV file
    """
    # Get prototype truck to duplicate
    prototype_truck = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_truck:
        print("Failed to import truck prototype. Exiting.")
        return
    
    # Set scale of prototype truck
    prototype_truck.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype truck (we'll use it as a template for duplicating)
    prototype_truck.hide_viewport = True
    prototype_truck.hide_render = True
    
    # Read coordinates from CSV
    truck_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    x, y, z = map(float, row)
                    
                    # Duplicate the truck
                    truck_copy = prototype_truck.copy()
                    truck_copy.data = prototype_truck.data.copy()
                    truck_copy.name = f"Dustbin_{truck_count}"
                    bpy.context.collection.objects.link(truck_copy)
                    
                    # Show the truck copy
                    truck_copy.hide_viewport = False
                    truck_copy.hide_render = False
                    
                    # Position the truck at the XYZ coordinates from CSV
                    truck_copy.location = (x, y, prototype_truck.location.z)
                    
                    # Set the Z rotation (phi) in radians
                    #phi_radians = math.radians(phi)
                    
                    #truck_copy.rotation_euler = (
                       # truck_copy.rotation_euler.x,
                        #truck_copy.rotation_euler.y,
                        #phi_radians
                   # )
                    
                    print(f"Placed Dustbin_{truck_count} at position ({x}, {y}, {z})")
                    truck_count += 1
                    
                except Exception as e:
                    print(f"Error placing truck: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {truck_count} dustbins in the scene")
    
def place_pole_from_csv(csv_file_path, blend_file_path, object_name, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """
    Place pole objects at specified coordinates from a CSV file with custom scaling on each axis
    """
    # Get prototype pole to duplicate
    prototype_pole = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_pole:
        print("Failed to import pole prototype. Exiting.")
        return
    
    # Set scale of prototype pole - now with different scaling per axis
    prototype_pole.scale = (scale_x, scale_y, scale_z)
    
    # Hide the prototype pole (we'll use it as a template for duplicating)
    prototype_pole.hide_viewport = True
    prototype_pole.hide_render = True
    
    # Read coordinates from CSV
    pole_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates
                try:
                    x, y, z = map(float, row)
                    
                    # Duplicate the pole
                    pole_copy = prototype_pole.copy()
                    pole_copy.data = prototype_pole.data.copy()
                    pole_copy.name = f"Pole_{pole_count}"
                    bpy.context.collection.objects.link(pole_copy)
                    
                    # Show the pole copy
                    pole_copy.hide_viewport = False
                    pole_copy.hide_render = False
                    
                    # Position the pole at the XYZ coordinates from CSV
                    pole_copy.location = (x, y, 0.4)
                    
                    print(f"Placed Pole_{pole_count} at position ({x}, {y}, {z})")
                    pole_count += 1
                    
                except Exception as e:
                    print(f"Error placing pole: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {pole_count} poles in the scene")
    
def place_cone_from_csv(csv_file_path, blend_file_path, object_name, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """
    Place cone objects at specified coordinates from a CSV file with custom scaling on each axis
    """
    # Get prototype pole to duplicate
    prototype_cone = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_cone:
        print("Failed to import pole prototype. Exiting.")
        return
    
    # Set scale of prototype pole - now with different scaling per axis
    prototype_cone.scale = (scale_x, scale_y, scale_z)
    
    # Hide the prototype pole (we'll use it as a template for duplicating)
    prototype_cone.hide_viewport = True
    prototype_cone.hide_render = True
    
    # Read coordinates from CSV
    cone_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates
                try:
                    x, y, z = map(float, row)
                    
                    # Duplicate the pole
                    cone_copy = prototype_cone.copy()
                    cone_copy.data = prototype_cone.data.copy()
                    cone_copy.name = f"Cone_{cone_count}"
                    bpy.context.collection.objects.link(cone_copy)
                    
                    # Show the pole copy
                    cone_copy.hide_viewport = False
                    cone_copy.hide_render = False
                    
                    # Position the pole at the XYZ coordinates from CSV
                    cone_copy.location = (x, y, 0)
                    
                    print(f"Placed Cone_{cone_count} at position ({x}, {y}, {z})")
                    cone_count += 1
                    
                except Exception as e:
                    print(f"Error placing cone: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {cone_count} cones in the scene")

def place_stopsign_from_csv(csv_file_path, blend_file_path, object_name, scale_factor=0.01):
    """
    Place stop sign objects at specified coordinates from a CSV file
    """
    # Get prototype stop sign to duplicate
    prototype_sign = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_sign:
        print("Failed to import stop sign prototype. Exiting.")
        return
    
    # Set scale of prototype stop sign
    prototype_sign.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype stop sign (we'll use it as a template for duplicating)
    prototype_sign.hide_viewport = True
    prototype_sign.hide_render = True
    
    # Read coordinates from CSV
    sign_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    if len(row) >= 4:
                        x, y, z, phi = map(float, row)
                    else:
                        x, y, z = map(float, row)
                        phi = 0.0  # Default rotation
                    
                    # Duplicate the stop sign
                    sign_copy = prototype_sign.copy()
                    sign_copy.data = prototype_sign.data.copy()
                    sign_copy.name = f"StopSign_{sign_count}"
                    bpy.context.collection.objects.link(sign_copy)
                    
                    # Show the stop sign copy
                    sign_copy.hide_viewport = False
                    sign_copy.hide_render = False
                    
                    # Position the stop sign at the XYZ coordinates from CSV
                    sign_copy.location = (x, y, z if z else prototype_sign.location.z)
                    
                    # Set rotation if available
                    if len(row) >= 4:
                        phi_radians = math.radians(phi)
                        sign_copy.rotation_euler = (
                            sign_copy.rotation_euler.x,
                            sign_copy.rotation_euler.y,
                            phi_radians
                        )
                    
                    print(f"Placed StopSign_{sign_count} at position ({x}, {y}, {z})")
                    sign_count += 1
                    
                except Exception as e:
                    print(f"Error placing stop sign: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {sign_count} stop signs in the scene")

def place_traffic_signal_from_csv(csv_file_path, blend_file_path, object_name, signal_type, scale_factor=0.01):
    """
    Place traffic signal objects at specified coordinates from a CSV file
    """
    # Get prototype traffic signal to duplicate
    prototype_signal = append_object_from_blend(blend_file_path, object_name)
    
    if not prototype_signal:
        print(f"Failed to import {signal_type} traffic signal prototype. Exiting.")
        return
    
    # Set scale of prototype traffic signal
    prototype_signal.scale = (scale_factor, scale_factor, scale_factor)
    
    # Hide the prototype traffic signal (we'll use it as a template for duplicating)
    prototype_signal.hide_viewport = True
    prototype_signal.hide_render = True
    
    # Read coordinates from CSV
    signal_count = 0
    try:
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            for row in csvreader:
                # Parse coordinates and rotation
                try:
                    x, y, z = map(float, row)
                    
                    # Duplicate the traffic signal
                    signal_copy = prototype_signal.copy()
                    signal_copy.data = prototype_signal.data.copy()
                    signal_copy.name = f"TrafficSignal{signal_type}_{signal_count}"
                    bpy.context.collection.objects.link(signal_copy)
                    
                    # Show the traffic signal copy
                    signal_copy.hide_viewport = False
                    signal_copy.hide_render = False
                    
                    # Position the traffic signal at the XYZ coordinates from CSV
                    signal_copy.location = (x, y, z)
                    
                    # Set the Z rotation (phi) in radians
                    #phi_radians = math.radians(phi)
                    #signal_copy.rotation_euler = (
                     #   signal_copy.rotation_euler.x,
                      #  signal_copy.rotation_euler.y,
                       # phi_radians
                    #)
                    
                    print(f"Placed TrafficSignal{signal_type}_{signal_count} at position ({x}, {y}, {z})")
                    signal_count += 1
                    
                except Exception as e:
                    print(f"Error placing traffic signal: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    
    print(f"Placed {signal_count} {signal_type} traffic signals in the scene")

def create_plane_mesh(size=100, color_hex="353535", roughness=0.85):
    """
    Create a ground plane with specified parameters
    
    Parameters:
    size (float): Size of the plane
    color_hex (str): Hex color code for the plane material (without #)
    roughness (float): Roughness value for the material
    """
    # Create a plane with the specified size
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground_Plane"
    
    # Convert hex color to RGB
    r = int(color_hex[0:2], 16) / 255
    g = int(color_hex[2:4], 16) / 255
    b = int(color_hex[4:6], 16) / 255
    
    # Create material for the plane
    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    
    # Clear default nodes
    mat.node_tree.nodes.clear()
    
    # Create a Principled BSDF node
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
    
    print(f"Created ground plane with size={size}, color=#{color_hex}, roughness={roughness}")
    return plane

# Asset configuration - Dictionary of asset paths and object names
ASSET_CONFIG = {
    'car': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\Vehicles\SedanAndHatchbackArrow.blend",
        'object_name': "Car",
        'scale_factor': 0.012
    },
    'sedan': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\Vehicles\SedanAndHatchbackArrow.blend",
        'object_name': "Car",
        'scale_factor': 0.012
    },
    #'person': {
        #'blend_file': r"C:\Users\vshit\Desktop\pedestrian_asset.blend",
        #'object_name': "person",
        #'scale_factor': 0.7
    #},
    'stop_sign': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\SpeedLimitSign.blend",
        'object_name': "street_sign",
        'scale_factor': 1.0
    },
    'traffic_signal_green': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\TrafficSignalGreen.blend",
        'object_name': "Traffic_signal_green",
        'scale_factor': 0.3
    },
    'traffic_signal_red': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\TrafficSignalRed.blend",
        'object_name': "Traffic_signal1",
        'scale_factor': 0.3
    },
    'traffic_light': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\TrafficSignalGreen.blend",
        'object_name': "Traffic_signal_green",
        'scale_factor': 0.3
    },
    'truck': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\Vehicles\Truck.blend",
        'object_name': "Truck",
        'scale_factor': 0.0007
    },
    'pickup_truck': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\Vehicles\PickupTruckArrow.blend",
        'object_name': "PickupTruck",
        'scale_factor': 0.6
    },
    'street_sign': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\SpeedLimitSign.blend",
        'object_name': "street_sign",
        'scale_factor': 1.0
    },
    'suv': {
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\Vehicles\SUVArrow.blend",
        'object_name': "Jeep_3_",
        'scale_factor': 3.5
    },
    'trash_can':{
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\Dustbin.blend",
        'object_name': "Dustbin",
        'scale_factor': 1.25
    },
    'person':{
        'blend_file': r"C:\Users\vshit\Desktop\scene5person.blend",
        'object_name': "person_0",
        'scale_factor': 1
    },
    'pole':{
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\TrafficAssets.blend",
        'object_name': "Cylinder.001",
        'scale_x': 0.1,
        'scale_y': 0.1,
        'scale_z': 0.4  # Example: making the pole twice as tall
    },
    'cone':{
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\TrafficConeAndCylinder.blend",
        'object_name': "cone",
        'scale_x': 1,
        'scale_y': 1,
        'scale_z': 1  # Example: making the pole twice as tall
    },
     'barrel':{
        'blend_file': r"C:\Users\vshit\Downloads\P3Data\P3Data\Assets\TrafficConeAndCylinder.blend",
        'object_name': "cone",
        'scale_x': 1,
        'scale_y': 1,
        'scale_z': 1  # Example: making the pole twice as tall
    }
    
}

def discover_and_place_objects(base_folder, texture_folder=None):
    """
    Discover CSV files in a folder and place objects accordingly
    
    Parameters:
    base_folder (str): Path to the folder containing CSV files
    texture_folder (str, optional): Path to the folder containing texture images
    """
    print(f"Scanning for CSV files in: {base_folder}")
    
    # Check if the folder exists
    if not os.path.exists(base_folder):
        print(f"Error: Folder {base_folder} does not exist")
        return
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(base_folder, "*.csv"))
    
    # Filter for blender_objects CSVs
    object_csvs = [f for f in csv_files if "blender_objects" in os.path.basename(f).lower()]
    
    if not object_csvs:
        print(f"No 'blender_objects' CSV files found in {base_folder}")
        return
    
    print(f"Found {len(object_csvs)} object CSV files")
    
    # Process each CSV file
    for csv_file in object_csvs:
        filename = os.path.basename(csv_file)
        
        # Extract object type from filename (assumes naming like "blender_objects_TYPE.csv")
        match = re.search(r'blender_objects_([a-zA-Z_]+)\.csv', filename)
        if not match:
            print(f"Skipping {filename} - doesn't match expected naming pattern")
            continue
        
        object_type = match.group(1).lower()
        print(f"Detected object type: {object_type}")
        
        # Check if this object type is supported in ASSET_CONFIG
        if object_type not in ASSET_CONFIG:
            print(f"Warning: No asset configuration found for '{object_type}', skipping")
            continue
        
        # Get asset config for this object type
        asset = ASSET_CONFIG[object_type]
        
        # Place objects based on type
        if object_type in ['street_sign','stop_sign'] and texture_folder:
            print(f"Placing street signs with textures from {csv_file}")
            place_sign_with_dynamic_texture(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                texture_folder,
                asset['scale_factor']
            )
        elif object_type in ['car', 'sedan', 'suv']:
            print(f"Placing cars from {csv_file}")
            place_cars_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset['scale_factor']
            )
        elif object_type == 'person':
            print(f"Placing people from {csv_file}")
            place_person_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset['scale_factor']
            )
        elif object_type == 'trash_can':
            print(f"Placing people from {csv_file}")
            place_dustbin_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset['scale_factor']
            )
        elif object_type == 'stopsign':
            print(f"Placing stop signs from {csv_file}")
            place_stopsign_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset['scale_factor']
            )
        elif object_type == 'truck':
            print(f"Placing trucks from {csv_file}")
            place_truck_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset['scale_factor']
            )
        elif object_type == 'pickup_truck':
            print(f"Placing pickup trucks from {csv_file}")
            place_ptruck_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset['scale_factor']
            )
        elif object_type in ['traffic_light_green','traffic_light']:
            print(f"Placing green traffic lights from {csv_file}")
            place_traffic_signal_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                "Green",
                asset['scale_factor']
            )
        elif object_type == 'traffic_light_red':
            print(f"Placing red traffic lights from {csv_file}")
            place_traffic_signal_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                "Red",
                asset['scale_factor']
            )
        elif object_type == 'pole':
            print(f"Placing pole from {csv_file}")
            place_pole_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset.get('scale_x', 0.1),
                asset.get('scale_y', 0.1),
                asset.get('scale_z', 0.4)
            )
        elif object_type in ['cone','barrel']:
            print(f"Placing cone from {csv_file}")
            place_cone_from_csv(
                csv_file,
                asset['blend_file'],
                asset['object_name'],
                asset.get('scale_x', 1.4),
                asset.get('scale_y', 1.4),
                asset.get('scale_z', 1.4)
            )
        
        else:
            print(f"Warning: Object type '{object_type}' is in ASSET_CONFIG but no placement function is defined")

def main():
    """
    Main function to automate scene creation with assets from a specified folder
    """
    # Clear existing objects
    clear_existing_cars()
    
    # Create ground plane
    #create_plane_mesh(size=100, color_hex="353535", roughness=0.85)
    
    # Base folder containing object CSV files
    # Example: r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\output\0012\objects"
    base_folder = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\output6\0014\objects"
    
    # Path to texture folder for street signs
    texture_folder = r"C:\Users\vshit\Downloads\Lanes_integrated\Lanes_integrated\input\curvedlanesdata\scene5_out_signs\bboxes\crops"
    
    # Discover and place objects
    discover_and_place_objects(base_folder, texture_folder)
    
    print("Scene population complete")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()