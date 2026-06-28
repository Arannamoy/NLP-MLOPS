import bpy
import bmesh
import random
import csv
import os


output_dir = os.path.abspath("synthetic_piles_data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_path = os.path.join(output_dir, "ground_truth_volumes.csv")


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


volume_data = []


num_piles = 100 

for i in range(num_piles):
    clear_scene()


    # radius = random.uniform(1.5, 4.0)
    # height = random.uniform(0.8, 2.5)


    # # location=(0,0,height/2)
    # bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=radius, radius2=0, depth=height, location=(0, 0, height/2))
    # obj = bpy.context.active_object
    # obj.name = f"WoodPile_{i}"


    # bpy.ops.object.modifier_add(type='SUBSURF')
    # obj.modifiers["Subdivision"].levels = 3
    # bpy.ops.object.modifier_apply(modifier="Subdivision")

  
    # tex = bpy.data.textures.new(f"PileNoise_{i}", type='CLOUDS')
    # tex.noise_scale = random.uniform(0.5, 1.5)

    # bpy.ops.object.modifier_add(type='DISPLACE')
    # mod = obj.modifiers["Displace"]
    # mod.texture = tex
    # mod.strength = random.uniform(0.05, 0.2) 
    # bpy.ops.object.modifier_apply(modifier="Displace")



    length = random.uniform(8.0, 15.0)
    width = random.uniform(3.0, 6.0)
    height = random.uniform(1.5, 3.5)


    bpy.ops.mesh.primitive_grid_add(size=1, x_subdivisions=150, y_subdivisions=150)
    obj = bpy.context.active_object
    obj.scale = (width, length, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.name = f"WoodPile_{i}"


    for v in obj.data.vertices:
        x, y, z = v.co

        nx = x / (width / 2)
        ny = y / (length / 2)
        

        dist = (nx**2) + (ny**2) * 0.4 
        
        if dist < 1.0:

            v.co.z = height * (1.0 - dist)
        else:
            v.co.z = 0


    tex = bpy.data.textures.new(f"PileNoise_{i}", type='MUSGRAVE') 
    tex.noise_scale = random.uniform(1.0, 2.5)

    bpy.ops.object.modifier_add(type='DISPLACE')
    mod = obj.modifiers["Displace"]
    mod.texture = tex
    mod.strength = random.uniform(0.1, 0.3)
    mod.direction = 'Z' 
    bpy.ops.object.modifier_apply(modifier="Displace")



    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    volume = bm.calc_volume()
    bm.free()


    obj_file_name = f"pile_{i}.obj"
    obj_file_path = os.path.join(output_dir, obj_file_name)


    bpy.ops.wm.obj_export(filepath=obj_file_path, export_selected_objects=True)


    volume_data.append([obj_file_name, volume])
    print(f"Generated {obj_file_name} - Volume: {volume:.4f} m^3")


with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["FileName", "Volume_m3"])
    writer.writerows(volume_data)

print(f"\n {num_piles} files generated successfully in: {output_dir}")




# blender --background --python generate_piles.py