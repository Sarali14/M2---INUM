import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box,shape
import os
import shapefile
from pathlib import Path
from scipy.stats import qmc

# --- Polygon vertices ---
DATA_DIR = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/data")
SHP_FILE = DATA_DIR / "poly2.shp"   # <-- just change this file name
N_instances = 1
n_samples = 30
margin = 100.0  # meters
out_dir = "samples_LH_square_test_30_NM"
spacing=20.0

# --- Read shapefile geometry using pyshp ---
sf = shapefile.Reader(str(SHP_FILE))
print(f"{SHP_FILE.name} contains {len(sf.shapes())} shapes")

# Use the largest shape (if multiple polygons)
shapes = sf.shapes()
largest_shape = max(shapes, key=lambda s: Polygon(s.points).area)
points = largest_shape.points
xs = 1.2*np.array([p[0] for p in points])
ys = 1.2*np.array([p[1] for p in points])

# --- Compute bounding box and margin ---
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

xmin = min_x - margin
xmax = max_x + margin
ymin = min_y - margin
ymax = max_y + margin

print(f"Bounding box :\n  x ∈ [{xmin:.1f}, {xmax:.1f}]\n  y ∈ [{ymin:.1f}, {ymax:.1f}]")

# --- Latin Hypercube Sampling in 2D ---
"""def lhs(n_samples, dim):
    cut = np.linspace(0, 1, n_samples + 1)
    u = np.random.rand(n_samples, dim)
    a, b = cut[:-1], cut[1:]
    points = a[:, None] + (b - a)[:, None] * u
    for j in range(dim):
        np.random.shuffle(points[:, j])
    return points"""


os.makedirs(out_dir, exist_ok=True)

for idx in range(N_instances):
    sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    lhs_unit = sampler.random(n=n_samples)  
    lhs_scaled = np.zeros_like(lhs_unit)
    lhs_scaled[:, 0] = xmin + (xmax - xmin) * lhs_unit[:, 0]
    lhs_scaled[:, 1] = ymin + (ymax - ymin) * lhs_unit[:, 1]
    # Save as a Python list of [x, y] pairs, compatible with ast.literal_eval

    layout_flat = lhs_scaled.flatten().tolist()
    file_path = os.path.join(out_dir, f"Sample_LH_{idx:04d}.txt")
    
    with open(file_path, "w") as f:
        f.write(repr(layout_flat))

    print(f"Saved {file_path}")

sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
lhs_unit = sampler.random(n=n_samples)

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
# LHS points
plt.scatter(lhs_scaled[:, 0], lhs_scaled[:, 1], c="red", s=10, alpha=0.6, label="LHS samples (scipy)")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Latin Hypercube samples around polygon")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

