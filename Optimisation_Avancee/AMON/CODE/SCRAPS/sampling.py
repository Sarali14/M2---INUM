import os
import random
import json
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import unary_union
import shapefile

# --- AMON modules ---
import sys
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")
sys.path.insert(0, str(ROOT))
import windfarm_eval
import constraints as cst
import data as d
import windfarm_setting as wf

# --- Constants ---
NB_TURBINES = 10
TOTAL_LAYOUTS = 500
BOUNDARY_SHP = ROOT / "data/boundary_zone.shp"
PARAM_FILE = ROOT / "instances/1/param3.txt"
OUTPUT_DIR = Path("CODE/turbine_layouts_mixed")
MIN_SPACING = 80.0  # nominal spacing in meters

# -----------------------------------------------------------------
# Load buildable zone and site info
# -----------------------------------------------------------------
nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(str(PARAM_FILE))
fmGROSS, WS, WD, max_index, max_ws = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)

buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

if isinstance(buildable_zone, np.ndarray):
    # Sometimes returns an array of polygons -> convert safely
    try:
        polys = []
        for p in buildable_zone:
            if isinstance(p, (Polygon, MultiPolygon)):
                polys.append(p)
            elif isinstance(p, (list, np.ndarray)) and len(p) >= 3:
                polys.append(Polygon(p))
        buildable_zone = unary_union(polys)
        print(f"[INFO] Converted numpy array -> MultiPolygon ({len(polys)} parts)")
    except Exception as e:
        raise TypeError(f"Unexpected buildable_zone type {type(buildable_zone)}") from e
else:
    # Already a Polygon or MultiPolygon
    print(f"[INFO] buildable_zone is already a {type(buildable_zone).__name__}")
# -----------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------
def random_point_in_polygon(polygon, max_tries=1000):
    """Uniformly sample a point inside a polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            return x, y
    raise RuntimeError("Failed to sample point inside polygon.")

def generate_layout_in_zone(polygon, n_turbines, min_spacing):
    """Generate a layout with spacing constraint inside a polygon."""
    points = []
    for _ in range(n_turbines):
        for _ in range(1000):
            x, y = random_point_in_polygon(polygon)
            if all(np.hypot(x - px, y - py) >= min_spacing for px, py in points):
                points.append((x, y))
                break
        else:
            return None  # failed
    return points

def jitter_layout(layout, polygon, spacing_factor=1.0, placing_factor=0.0):
    """Add small noise to create near/infeasible versions."""
    jittered = []
    for x, y in layout:
        dx = random.uniform(-spacing_factor, spacing_factor)
        dy = random.uniform(-spacing_factor, spacing_factor)
        new_x, new_y = x + dx, y + dy

        # Occasionally push a turbine slightly outside polygon
        if random.random() < placing_factor:
            angle = random.uniform(0, 2*np.pi)
            dist = random.uniform(0, spacing_factor * 3)
            new_x += dist * np.cos(angle)
            new_y += dist * np.sin(angle)

        jittered.append((new_x, new_y))
    return jittered

# -----------------------------------------------------------------
# Generate datasets
# -----------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)
counts = {"feasible": 0, "near": 0, "infeasible": 0}

for i in range(TOTAL_LAYOUTS):
    # Decide layout type
    if i < 0.5 * TOTAL_LAYOUTS:
        mode = "feasible"
    elif i < 0.8 * TOTAL_LAYOUTS:
        mode = "near"
    else:
        mode = "infeasible"

    # Generate a feasible layout as baseline
    base_layout = None
    for _ in range(100):
        layout = generate_layout_in_zone(buildable_zone, NB_TURBINES, MIN_SPACING)
        if layout:
            base_layout = layout
            break
    if base_layout is None:
        print(f"Failed to make feasible layout at index {i}")
        continue

    # Apply mode-specific modifications
    if mode == "feasible":
        final_layout = base_layout
    elif mode == "near":
        final_layout = jitter_layout(base_layout, buildable_zone, spacing_factor=30.0, placing_factor=0.1)
    else:  # infeasible
        final_layout = jitter_layout(base_layout, buildable_zone, spacing_factor=80.0, placing_factor=0.3)

    # Flatten and save
    flat = [coord for p in final_layout for coord in p]
    filename = OUTPUT_DIR / f"layout_mixed_{i:03d}.txt"
    with open(filename, "w") as f:
        json.dump(flat, f)

    counts[mode] += 1
    if i % 50 == 0:
        print(f"Generated {i}/{TOTAL_LAYOUTS} layouts...")

print("\nGeneration complete.")
print("Feasible:", counts["feasible"], "Near-feasible:", counts["near"], "Infeasible:", counts["infeasible"])
print(f"Files saved in: {OUTPUT_DIR.resolve()}")
