import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import os
import shapefile
from pathlib import Path
from scipy.stats import qmc

# --- Polygon vertices ---
DATA_DIR = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/data")
SHP_FILE = DATA_DIR / "poly2.shp"   # <-- just change this file name

N_instances = 5000

# ---- NEW: choose min and max number of turbines per layout ----
N_min = 10      # <-- set this as you want
N_max = 40     # <-- set this as you want (must be >= N_min)

margin = 100.0  # meters
out_dir = "samples_LH_square_training_modified"
spacing = 20.0  # (currently unused, kept for future constraints)

# --- Read shapefile geometry using pyshp ---
sf = shapefile.Reader(str(SHP_FILE))
print(f"{SHP_FILE.name} contains {len(sf.shapes())} shapes")

# Use the largest shape (if multiple polygons)
shapes = sf.shapes()
largest_shape = max(shapes, key=lambda s: Polygon(s.points).area)
points = largest_shape.points
xs = 1.2 * np.array([p[0] for p in points])
ys = 1.2 * np.array([p[1] for p in points])

# --- Compute bounding box and margin ---
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

xmin = min_x - margin
xmax = max_x + margin
ymin = min_y - margin
ymax = max_y + margin

print(f"Bounding box :\n  x ∈ [{xmin:.1f}, {xmax:.1f}]\n  y ∈ [{ymin:.1f}, {ymax:.1f}]")

os.makedirs(out_dir, exist_ok=True)

# --- Generate layouts ---
for idx in range(N_instances):
    # random number of turbines in [N_min, N_max]
    n_turbines = np.random.randint(N_min, N_max + 1)

    sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    lhs_unit = sampler.random(n=n_turbines)

    lhs_scaled = np.zeros_like(lhs_unit)
    lhs_scaled[:, 0] = xmin + (xmax - xmin) * lhs_unit[:, 0]
    lhs_scaled[:, 1] = ymin + (ymax - ymin) * lhs_unit[:, 1]

    # flatten [x1,y1,x2,y2,...] as before
    layout_flat = lhs_scaled.flatten().tolist()
    file_path = os.path.join(out_dir, f"Sample_LH_{idx:04d}.txt")

    with open(file_path, "w") as f:
        f.write(repr(layout_flat))

    print(f"Saved {file_path} with {n_turbines} turbines")


"""
# --- Optional: example plot using a layout with N_max turbines ---
sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
lhs_unit = sampler.random(n=N_max)

lhs_scaled = np.empty_like(lhs_unit)
lhs_scaled[:, 0] = xmin + (xmax - xmin) * lhs_unit[:, 0]
lhs_scaled[:, 1] = ymin + (ymax - ymin) * lhs_unit[:, 1]

# --- Create polygon and bounding box geometries ---
poly = Polygon(zip(xs, ys))
bbox = box(xmin, ymin, xmax, ymax)

# --- Plot everything ---
plt.figure(figsize=(8, 7))
# Original polygon
x_poly, y_poly = poly.exterior.xy
plt.plot(x_poly, y_poly, color="blue", linewidth=2, label="Polygon")
# Bounding box
x_box, y_box = bbox.exterior.xy
plt.plot(x_box, y_box, "k--", label="Bounding box")
# LHS points (example with N_max turbines)
plt.scatter(lhs_scaled[:, 0], lhs_scaled[:, 1], s=10, alpha=0.6, label=f"LHS samples (N={N_max})")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Latin Hypercube samples around polygon")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
