#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Windfarm layout blackbox and constraints — 300 essais, sortie fichiers TXT + console

- 300 essais de génération de 10 éoliennes
- Affichage console: uniquement les coordonnées en liste aplatie [x1,y1,x2,y2,...]
- Fichiers de sortie: turbine_layouts/layout_constrainted_000.txt ... layout_constrainted_2
99.txt
"""

import sys
import os
import random
from pathlib import Path

# --- ajustement du chemin pour trouver les modules du projet ---
ROOT = Path(__file__).resolve().parents[1]  # le dossier AMON
sys.path.insert(0, str(ROOT))

# --- imports du projet ---
import windfarm_eval                 # la vraie blackbox
import constraints as cst
import data as d
import windfarm_setting as wf

# --- imports externes ---
import shapefile  # pip install pyshp
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity


# ============================================================
# Helpers
# ============================================================

def memoize(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper


@memoize
def settings(param_file_path):
    # This line has been corrected to ensure no unintended line breaks
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_path)
    fmGROSS, WS, WD, max_index, max_ws = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)
    return fmGROSS, WS_BB, WD_BB, D, buildable_zone


def eval_bb_local(param_file_path, x):
    # This will now use the provided param_file_path, ensuring consistency
    return windfarm_eval.windfarm_eval(param_file_path, x)


# ============================================================
# Geometry / shapefile
# ============================================================

BOUNDARY_SHP = r"data/boundary_zone.shp"
SCALE_FACTOR = 0.2

def shapefile_to_multipolygon(shp_path, scale_factor=1.0):
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Introuvable: {shp_path}")
    sf = shapefile.Reader(shp_path)

    polys = []
    for s in sf.shapes():
        parts = list(s.parts) + [len(s.points)]
        rings = [s.points[parts[i]:parts[i+1]] for i in range(len(parts)-1)]
        if not rings:
            continue
        poly = Polygon(rings[0], holes=rings[1:] if len(rings) > 1 else None)
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)

    if not polys:
        raise ValueError("Aucun polygone valide trouvé dans le shapefile.")
    
    boundary = unary_union(polys)
    if scale_factor != 1.0:
        boundary = affinity.scale(boundary, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return boundary

# IMPORTANT: The boundary used for random point generation should also be consistent.
# Here we'll rely on the settings function to get D, etc.
# For now, let's keep this as it is, assuming this boundary is consistent with your param3.txt
# If not, you might need to load it dynamically using `settings` too.
boundary = shapefile_to_multipolygon(BOUNDARY_SHP, scale_factor=SCALE_FACTOR)


def random_point_in_polygon(polygon, max_tries=10000):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            return x, y
    raise RuntimeError("Impossible de tirer un point dans la boundary.")


def generate_points_in_boundary(polygon, n_points):
    coords = []
    for _ in range(n_points):
        x, y = random_point_in_polygon(polygon)
        coords.extend([x, y])
    return coords


# ============================================================
# Attempt one layout
# ============================================================

# Define the actual parameter file path
ACTUAL_PARAM_FILE_PATH = str(ROOT / "instances/1/param3.txt")

def attempt_one_layout(NB=10, max_rejects=1000, eps=1e-9, seed=None):
    if seed is not None:
        random.seed(seed)
    X0 = []
    n = 0
    Sec = 0

    while n < NB and Sec < max_rejects:
        xtest = generate_points_in_boundary(boundary, n_points=1)
        X1 = X0 + xtest

        # Pass the actual parameter file path
        _, s_d, sum_dist = eval_bb_local(param_file_path=ACTUAL_PARAM_FILE_PATH, x=X1)

        # Check for ValueError fallback values as well
        # The windfarm_eval script returns s_d=1e6, sum_dist=1e6 on ValueError
        if abs(s_d) >= 1e5 or abs(sum_dist) >= 1e5: # Check for large values indicating an error
            print(f"DEBUG: Error values detected for a potential layout during generation. s_d={s_d}, sum_dist={sum_dist}")
            Sec += 1 # Treat as a reject
        elif abs(s_d) <= eps and abs(sum_dist) <= eps:
            Sec = 0
            X0 = X1
            n += 1
        else:
            Sec += 1

    return X0 if n == NB else None


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # REMOVE THE OVERRIDE BLOCK - IT'S NO LONGER NEEDED AND CAUSES INCONSISTENCY
    # d.read_param_file = _read_param_file_override

    # ============================
    # Préparer le dossier turbine_layouts
    # ============================
    OUT_DIR = Path("CODE") / "turbine_layouts_test"
    # Ensure the directory is clean before generating new files
    if OUT_DIR.exists():
        for f in OUT_DIR.glob("layout_new_*.txt"):
            os.remove(f)
        print(f"Cleaned existing layout_new_*.txt files in {OUT_DIR}")
    OUT_DIR.mkdir(exist_ok=True)


    # ============================
    # Boucle externe : N essais
    # ============================
    N_TRIES = 200
    NB_TURBINES = 10
    MAX_REJECTS = 1000
    EPS = 1e-9

    for i in range(N_TRIES):
        seed = i
        coords_flat = attempt_one_layout(NB=NB_TURBINES, max_rejects=MAX_REJECTS, eps=EPS, seed=seed)
        if coords_flat is None:
            # If a layout couldn't be generated successfully within max_rejects
            # you might want to log this or handle it differently.
            # For now, we'll save an empty list to avoid errors, but it means
            # this instance will have 0 turbines, which might be an issue.
            # Consider if you want to skip saving these failed generations.
            coords_flat = []
            print(f"WARNING: Failed to generate a valid {NB_TURBINES}-turbine layout for seed {seed}")


        # Sauvegarde dans fichier individuel
        filename = OUT_DIR / f"layout_test_{i:03d}.txt"
        with open(filename, "w", encoding="utf-8") as ftxt:
            ftxt.write(str(coords_flat))

        print(f"Saved {filename.name} (length: {len(coords_flat)//2} turbines)")
