import bpy

# Deselect everything
bpy.ops.object.select_all(action='DESELECT')

# Collect objects to delete
objects_to_delete = []

for obj in bpy.data.objects:
    # Keep cameras only
    if obj.type != 'CAMERA':
        objects_to_delete.append(obj)

# Unhide objects (in case they are hidden in viewport or render)
for obj in objects_to_delete:
    obj.hide_set(False)
    obj.hide_render = False

# Select and delete them
for obj in objects_to_delete:
    obj.select_set(True)

bpy.ops.object.delete()

# Optional: Clean up unused data blocks (meshes, materials, etc.)
bpy.ops.outliner.orphans_purge(do_recursive=True)
