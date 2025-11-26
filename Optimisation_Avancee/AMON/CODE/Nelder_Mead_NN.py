import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from shapely.geometry import Polygon, box,shape
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import shapefile

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "CODE/Neural_Network_surrogate"))
from penalized_surrogate import penalized_surrogate 
from surrogate_eap import predict_eap

DATA_DIR = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/data")
SHP_FILE = DATA_DIR / "poly2.shp"

Instance = str(ROOT / "instances/2/param.txt")
X0_filz = str(ROOT / "CODE/samples_LH_square_test_30_NM/Sample_LH_0000.txt")
results_file = ROOT / "instances/2/nelder_mead_log.txt"

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
xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()

def initialize_simplex(X, delta):
    if isinstance(X, str):  # if user passed a filename instead of array
        with open(X, "r", encoding="utf-8") as f:
            s = f.read()
        s = s.replace("[", "").replace("]", "").replace(",", " ")
        X = [float(t) for t in s.split()]
    X = np.array(X, dtype=float)
    n = len(X)
    simplex = [X.copy()]
    base=X.copy()
    for i in range(n):
        X_new = base.copy()
        X_new[i] += delta
        simplex.append(X_new)
    return simplex

l=10

simplex = initialize_simplex(X0_filz,delta=200)

def ordering(penalized_function, simplex, instance_path, lambd):
    func_vals = [penalized_function(x.tolist(), lambd=lambd) for x in simplex]
    sorted_indices = np.argsort(func_vals)
    sorted_simplex = [simplex[i] for i in sorted_indices]
    sorted_func = [func_vals[i] for i in sorted_indices]
    return sorted_func, sorted_simplex


def save_plot(X, iteration, xmin, xmax, ymin, ymax, frame_dir="frames_nm"):
    """
    Plot the positions of the 20 turbines inside the rectangular terrain.
    """
    # Create folder if needed
    os.makedirs(frame_dir, exist_ok=True)

    # Reshape X into Nx2
    X = np.array(X).reshape(-1, 2)
    xs = X[:, 0]
    ys = X[:, 1]

    plt.figure(figsize=(7, 6))

    # Draw terrain rectangle
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    plt.plot(rect_x, rect_y, "k-", linewidth=2, label="Terrain")

    # Plot turbines
    plt.scatter(xs, ys, c="red", s=30, label="Turbines")

    plt.title(f"Nelder-Mead — Iteration {iteration}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis("equal")
    plt.xlim(xmin - 50, xmax + 50)
    plt.ylim(ymin - 50, ymax + 50)
    plt.grid(True)

    # Save figure
    filename = os.path.join(frame_dir, f"iter_{iteration:04d}.png")
    plt.savefig(filename, dpi=120)
    plt.close()

def Nelder_Mead(simplex, instance_path, penalized_function, lambd,
                alpha=1.5, gamma=3.0, rho=0.5, sigma=0.5,
                tol=1e-6, max_iter=1000):
    iter_count = 0
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Iteration\tBest_penalized_EAP\tMove\tStdDev\n")

    while iter_count < max_iter:
        # --- Step 1: Evaluate and order simplex ---
        func_vals = [penalized_function(x.tolist(), lambd=lambd) for x in simplex]
        sorted_indices = np.argsort(func_vals)  # best first
        simplex = [simplex[i] for i in sorted_indices]
        func_vals = [func_vals[i] for i in sorted_indices]

        # --- Step 2: Convergence check ---
        if np.std(func_vals)/ (abs(np.mean(func_vals)) + 1e-12) < tol:
            break

        # --- Step 3: Centroid excluding worst ---
        centroid = np.mean(simplex[:-1], axis=0)

        # --- Step 4: Reflection ---
        X_r = centroid + alpha * (centroid - simplex[-1])
        f_Xr = penalized_function(X_r.tolist(), lambd=lambd)

        if f_Xr < func_vals[0]:  # Expansion
            X_e = centroid + gamma * (X_r - centroid)
            f_Xe = penalized_function(X_e.tolist(),  lambd=lambd)
            simplex[-1] = X_e if f_Xe < f_Xr else X_r
            move_type= "Expansion"

        elif f_Xr < func_vals[-2]:  # Accept reflection
            simplex[-1] = X_r
            move_type= " Reflection"

        else:  # Contraction
            if f_Xr < func_vals[-1]:  # outside contraction
                X_cont = centroid + rho * (X_r - centroid)
                move_type="Outside Contraction"
            else:  # inside contraction
                X_cont = centroid + rho * (simplex[-1] - centroid)
                move_type="Inside Contraction"
            f_Xcont = penalized_function(X_cont.tolist(), lambd=lambd)
            if f_Xcont < func_vals[-1]:
                simplex[-1] = X_cont
            else:
                # --- Shrink step: executed if reflection, expansion, and contraction fail ---
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])

                    move_type = "Shrink"

                    # Recompute function values for the updated simplex
                    func_vals = [penalized_function(x.tolist(), lambd=lambd) for x in simplex]
        func_vals = [penalized_function(x.tolist(), lambd=lambd) for x in simplex]
        sorted_indices = np.argsort(func_vals)
        simplex = [simplex[i] for i in sorted_indices]
        func_vals = [func_vals[i] for i in sorted_indices]
        best_so_far = func_vals[0]
        print(f"Iteration {iter_count}: Best objective = {-best_so_far:.12f}, Move = {move_type}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"{iter_count}\t{-best_so_far:.6f}\t{move_type}\t{np.std(func_vals):.6e}\n")
        save_plot(simplex[0], iter_count, xmin, xmax, ymin, ymax)
        iter_count += 1

    # Return best vector
    func_vals = [penalized_function(x.tolist(), lambd=lambd) for x in simplex]
    best_index = np.argmin(func_vals)
    return simplex[best_index], func_vals[best_index]

"""print("Initial simplex function values:")
for x in simplex:
    print(penalized_function(x, Instance, l))"""
# Run Nelder-Mead optimization
best_X, best_val = Nelder_Mead(simplex, Instance, penalized_surrogate, lambd=l)

# === Build the final video ===
frame_dir = "frames_nm"
frames = []

for k in sorted(os.listdir(frame_dir)):
    if k.endswith(".png"):
        frames.append(imageio.imread(os.path.join(frame_dir, k)))

imageio.mimsave("nm_evolution_30.gif", frames, fps=4)

print("Video saved as nm_evolution_30.gif")

# Print results
print("=== Nelder-Mead Optimization Finished ===")
print("Best solution vector (X):")
print(best_X)
#print("\nPenalized objective value:")
#print(best_val)
print("\nMaximized penalized EAP:")
print(-best_val)

with open(results_file, "a", encoding="utf-8") as f:
    f.write("\n=== Optimization Finished ===\n")
    f.write(f"Best penalized EAP: {-best_val:.6f}\n")
    f.write("Best solution vector (X):\n")
    np.savetxt(f, np.array(best_X), fmt="%.6f")
