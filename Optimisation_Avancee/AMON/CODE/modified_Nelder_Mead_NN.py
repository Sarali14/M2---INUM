import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import shapefile  # <-- needed for shapefile.Reader
import time

# -------------------------------------------------------------------
# Paths and imports
# -------------------------------------------------------------------
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CODE"))
sys.path.insert(0, str(ROOT / "CODE/Neural_Network_surrogate"))
from penalized_surrogate import penalized_surrogate 
from surrogate_eap import predict_eap

Instance = str(ROOT / "instances/2/param.txt")
X0_filz = str(ROOT / "CODE/samples_LH_square_9_scipy/Sample_LH_0003.txt")
SHP_FILE = ROOT / "data/poly2.shp"
results_file = ROOT / "instances/2/nelder_mead_log_NN.txt"
out_path = ROOT / "CODE/optimized_nelder_mead_NN.txt"

# -------------------------------------------------------------------
# Read polygon from shapefile
# -------------------------------------------------------------------
sf = shapefile.Reader(str(SHP_FILE))
print(f"{SHP_FILE.name} contains {len(sf.shapes())} shapes")

# Use the largest shape (if multiple polygons)
shapes = sf.shapes()
largest_shape = max(shapes, key=lambda s: Polygon(s.points).area)
points = largest_shape.points
xs = 1.2 * np.array([p[0] for p in points])
ys = 1.2 * np.array([p[1] for p in points])

# --- Compute bounding box (if you want it later) ---
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

# -------------------------------------------------------------------
# Read base layout: n turbines (fixed)
# -------------------------------------------------------------------
with open(X0_filz, "r", encoding="utf-8") as f:
    s = f.read()
s = s.replace("[", "").replace("]", "").replace(",", " ")
BASE_LAYOUT = np.array([float(t) for t in s.split()], dtype=float)
assert len(BASE_LAYOUT) % 2 == 0, "BASE_LAYOUT length must be 2n."

n_turbines = len(BASE_LAYOUT) // 2
print(f"Base layout has {n_turbines} turbines (fixed).")
print("We will add turbine number n+1 =", n_turbines + 1)

# -------------------------------------------------------------------
# Simplex initialization
# -------------------------------------------------------------------
def initialize_simplex(X, delta):
    X = np.array(X, dtype=float)
    n = len(X)
    simplex = [X.copy()]
    base = X.copy()
    for i in range(n):
        X_new = base.copy()
        X_new[i] += delta
        simplex.append(X_new)
    return simplex

l = 10.0  # penalty parameter

# -------------------------------------------------------------------
# Initial guess for turbine n+1
# -------------------------------------------------------------------
x_new_init = -550.0     # your manual guess
y_new_init = 250.0    # your manual guess

X_new_init = np.array([x_new_init, y_new_init], dtype=float)

# Simplex is now 2D: only [x_{n+1}, y_{n+1}]
simplex = initialize_simplex(X_new_init, delta=150.0)

# -------------------------------------------------------------------
# penalized surrogate for the new turbine
# -------------------------------------------------------------------
def penalized_for_new_turbine(x_new, lambd):
    """
    x_new: array-like of shape (2,) → [x_{n+1}, y_{n+1}]
    returns penalized objective for the FULL layout [BASE_LAYOUT, x_new]
    """
    x_new = np.array(x_new, dtype=float)
    full_layout = np.concatenate([BASE_LAYOUT, x_new])  # length 2*(n+1)
    return penalized_surrogate(full_layout.tolist(), lambd=lambd)

# -------------------------------------------------------------------
# Nelder–Mead algorithm (returns best point, value, path, iters, reached_max)
# -------------------------------------------------------------------
def Nelder_Mead(simplex, penalized_function, lambd,
                alpha=1.5, gamma=3.0, rho=0.5, sigma=0.5,
                tol=1e-6, max_iter=100):
    iter_count = 0
    path = []  # will store best (n+1)-th turbine positions per iteration

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Iteration\tBest_penalized_EAP\tMove\tStdDev\n")

    while iter_count < max_iter:
        # --- Step 1: Evaluate and order simplex ---
        func_vals = [penalized_function(x, lambd=lambd) for x in simplex]
        sorted_indices = np.argsort(func_vals)  # best first
        simplex = [simplex[i] for i in sorted_indices]
        func_vals = [func_vals[i] for i in sorted_indices]

        # --- Step 2: Convergence check ---
        if np.std(func_vals) / (abs(np.mean(func_vals)) + 1e-12) < tol:
            break

        # --- Step 3: Centroid excluding worst ---
        centroid = np.mean(simplex[:-1], axis=0)

        # --- Step 4: Reflection ---
        X_r = centroid + alpha * (centroid - simplex[-1])
        f_Xr = penalized_function(X_r, lambd=lambd)

        if f_Xr < func_vals[0]:  # Expansion
            X_e = centroid + gamma * (X_r - centroid)
            f_Xe = penalized_function(X_e, lambd=lambd)
            simplex[-1] = X_e if f_Xe < f_Xr else X_r
            move_type = "Expansion"

        elif f_Xr < func_vals[-2]:  # Accept reflection
            simplex[-1] = X_r
            move_type = "Reflection"

        else:  # Contraction
            if f_Xr < func_vals[-1]:  # outside contraction
                X_cont = centroid + rho * (X_r - centroid)
                move_type = "Outside Contraction"
            else:  # inside contraction
                X_cont = centroid + rho * (simplex[-1] - centroid)
                move_type = "Inside Contraction"
            f_Xcont = penalized_function(X_cont, lambd=lambd)
            if f_Xcont < func_vals[-1]:
                simplex[-1] = X_cont
            else:
                # --- Shrink step ---
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    move_type = "Shrink"
                func_vals = [penalized_function(x, lambd=lambd) for x in simplex]

        # Recompute and re-sort
        func_vals = [penalized_function(x, lambd=lambd) for x in simplex]
        sorted_indices = np.argsort(func_vals)
        simplex = [simplex[i] for i in sorted_indices]
        func_vals = [func_vals[i] for i in sorted_indices]
        best_so_far = func_vals[0]

        # Store current best position of turbine n+1
        current_best_X = simplex[0].copy()
        path.append(current_best_X)

        print(f"Iteration {iter_count}: Best objective = {-best_so_far:.12f}, Move = {move_type}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(f"{iter_count}\t{-best_so_far:.6f}\t{move_type}\t{np.std(func_vals):.6e}\n")

        iter_count += 1

    # Return best vector + path + iteration info
    func_vals = [penalized_function(x, lambd=lambd) for x in simplex]
    best_index = np.argmin(func_vals)
    reached_max_iter = (iter_count >= max_iter)
    return simplex[best_index], func_vals[best_index], path, iter_count, reached_max_iter

# -------------------------------------------------------------------
# Run Nelder–Mead for turbine (n+1)
# -------------------------------------------------------------------
alpha = 1.5
gamma = 3.0
rho = 0.5
sigma = 0.5
tol = 1e-6
max_iter = 100

start_time = time.time()
best_X_new, best_val, path, n_iter, reached_max = Nelder_Mead(
    simplex,
    penalized_for_new_turbine,
    lambd=l,
    alpha=alpha,
    gamma=gamma,
    rho=rho,
    sigma=sigma,
    tol=tol,
    max_iter=max_iter
)
end_time = time.time()
runtime = end_time - start_time
print(f"\nNelder Mead runtime: {runtime:.4f} seconds")

# Build full best layout: [base layout, best_X_new]
full_best_layout = np.concatenate([BASE_LAYOUT, best_X_new])

full_init_layout = np.concatenate([BASE_LAYOUT, X_new_init])

eap_start = predict_eap(full_init_layout.tolist())
eap_opt   = predict_eap(full_best_layout.tolist())

print("=== Nelder-Mead Optimization Finished ===")
print(f"Best position for turbine n+1 (turbine {n_turbines + 1}): [x, y]")
print(best_X_new)
print("\nMaximized penalized EAP:")
print(-best_val)

# -------------------------------------------------------------------
# Save final full layout as [x, y, x, y, ...] in a .txt file
# -------------------------------------------------------------------
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(full_best_layout.tolist(), f, indent=2)

print(f"\nSaved optimized layout to {out_path}")

with open(results_file, "a", encoding="utf-8") as f:
    f.write("\n=== Optimization Finished ===\n")
    f.write(f"Best penalized EAP: {-best_val:.6f}\n")
    f.write(f"Best position for turbine n+1 (turbine {n_turbines + 1}):\n")
    np.savetxt(f, np.array(best_X_new), fmt="%.6f")
    f.write("Full layout vector (including turbine n+1):\n")
    np.savetxt(f, full_best_layout, fmt="%.6f")

# -------------------------------------------------------------------
# Pretty summary "table" like the others
# -------------------------------------------------------------------
start_pos = X_new_init.tolist()
end_pos = best_X_new.tolist()

print("\n" + "=" * 90)
print(f"{'NELDER–MEAD (NN) SETTINGS':<40} {'TURBINE (n+1) & RUNTIME':<40}")
print("-" * 90)
print(f"{'Alpha (reflection)':<28} {alpha:<10.3f} {'Start [x, y]':<22} {start_pos}")
print(f"{'Gamma (expansion)':<28} {gamma:<10.3f} {'Final [x, y]':<22} {end_pos}")
print(f"{'Rho (contraction)':<28} {rho:<10.3f} {'Runtime (s)':<22} {runtime:.4f}")
print(f"{'Sigma (shrink)':<28} {sigma:<10.3f} {'Iterations used':<22} {n_iter}")
print(f"{'Tolerance':<28} {tol:<10.2e} {'Reached max_iter?':<22} {reached_max}")
print(f"{'Max iterations':<28} {max_iter:<10d} {'Penalty λ':<22} {l:.3f}")
print(f"{'EAP at start (NN)':<28} {eap_start:<10.3f}")
print(f"{'EAP at optimum (NN)':<28} {eap_opt:<10.3f}")
print("=" * 90 + "\n")

# -------------------------------------------------------------------
# Prepare data for plotting
# -------------------------------------------------------------------
coords_opt = full_best_layout.reshape(-1, 2)   # shape (n+1, 2)
path_arr = np.array(path)                      # shape (n_steps, 2)

# Index of the (n+1)-th turbine in coords_opt
turbine_new_index = coords_opt.shape[0] - 1    # last one

poly = Polygon(zip(xs, ys))

plt.figure(figsize=(8, 7))

# Original polygon
x_poly, y_poly = poly.exterior.xy
plt.plot(x_poly, y_poly, color="blue", linewidth=2, label="Polygon")

# 1) Plot all turbines in black (final positions)
for i in range(coords_opt.shape[0]):
    if i != turbine_new_index:
        plt.scatter(coords_opt[i, 0], coords_opt[i, 1], c="black", s=40)

# 2) Plot the trajectory of the (n+1)-th turbine in red
plt.plot(path_arr[:, 0], path_arr[:, 1], '-o',
         c="red", linewidth=1, markersize=4,
         label=f"Path of turbine {turbine_new_index + 1}")

# start point
plt.scatter(path_arr[0, 0], path_arr[0, 1],
            facecolors="none", edgecolors="blue", s=100,
            label="Start (n+1)")

# final point
plt.scatter(path_arr[-1, 0], path_arr[-1, 1],
            c="green", s=60, label="Final (n+1)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Nelder–Mead path of turbine n+1 (NN surrogate)")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
