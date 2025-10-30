import numpy as np
import matplotlib.pyplot as plt
import mapbox_earcut as earcut
import random
import os
import json

def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2

def sample_point_in_triangle(x0, y0, x1, y1, x2, y2):
    r1, r2 = random.random(), random.random()
    u = 1 - np.sqrt(r1)
    v = r2 * np.sqrt(r1)
    w = 1 - u - v
    x = u * x0 + v * x1 + w * x2
    y = u * y0 + v * y1 + w * y2
    return x, y

vertices =[527656.8552018718, 5225046.534108513,
                   531833.481796785, 5223363.526086701,
                   532578.9832550277, 5225627.245186118,
                   536367.231058161, 5225649.312807018,
                   538033.8041706986, 5221770.103385767,
                   533001.1069081093, 5217850.31086144,
                   531856.1745954718, 5219140.625539481,
                   522745.1767349267, 5221506.545418932,
                   523739.7069170366, 5225585.184831546,
                   523352.4720197042, 5227713.70650898,
                   525747.4452677837, 5228507.798102497,
                   527656.8552018718, 5225046.534108513,]

nb_Turbines=10
nb_instances=100
min_spacing=80

Areas=[]
total_area=0
vertices_array = np.array(vertices, dtype=np.float32).reshape(-1, 2)
num_vertices=len(vertices_array)
rings = np.array([num_vertices], dtype=np.uint32)

# Triangulate
triangles = earcut.triangulate_float32(vertices_array, rings)
print(triangles)
for i in range(0,len(triangles),3):
    t0,t1,t2= triangles[i],triangles[i+1],triangles[i+2]
    x0, y0 = vertices_array[t0]
    x1, y1 = vertices_array[t1]
    x2, y2 = vertices_array[t2]

    area_current_tri=triangle_area(x0,y0,x1,y1,x2,y2)

    Areas.append(area_current_tri)
    total_area+=area_current_tri

cumulative_area_distribution=np.cumsum(np.array(Areas)/total_area)
all_layouts = []

output_folder = "turbine_layouts"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

for layout_idx in range(nb_instances):
    turbine_positions = []
    attempts = 0
    while len(turbine_positions) < nb_Turbines:
        attempts += 1
        if attempts > 1000:  # safety check to avoid infinite loops
            print(f"Warning: Could not place all turbines for layout {layout_idx}")
            break

        chosen_triangle_index = -1

        # 3.1 Pick a triangle (area-weighted)
        r = random.random()
        for idx, c in enumerate(cumulative_area_distribution):
            if r <= c:
                chosen_triangle_index = idx
                break

        # 3.2 Get triangle vertices
        t0, t1, t2 = triangles[3*chosen_triangle_index : 3*chosen_triangle_index + 3]
        x0, y0 = vertices_array[t0]
        x1, y1 = vertices_array[t1]
        x2, y2 = vertices_array[t2]

        # 3.3 Sample point inside triangle
        x, y = sample_point_in_triangle(x0, y0, x1, y1, x2, y2)

        # 3.4 Check spacing constraint
        valid = True
        for px, py in turbine_positions:
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance < min_spacing:
                valid = False
                break

        if valid:
            turbine_positions.append((x, y))

    all_layouts.append(turbine_positions)
    
    if turbine_positions:
        filename = f"layout_random_{layout_idx:03d}.txt"
        filepath = os.path.join(output_folder, filename)
        
        # This list comprehension converts the list of pairs into a single flat list.
        # Example: [(x1, y1), (x2, y2)] becomes [x1, y1, x2, y2]
        flat_positions = [coord for pos in turbine_positions for coord in pos]
        
        # 'with open' safely handles the file.
        with open(filepath, 'w') as f:
            # json.dump writes the new flat_positions list to the file in the format [num, num, ...].
            json.dump(flat_positions, f, indent=None)
print(f"Generated {len(all_layouts)} layouts with {nb_Turbines} turbines each.")
