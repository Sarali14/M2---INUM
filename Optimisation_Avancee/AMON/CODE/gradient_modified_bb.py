import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, shape
import shapefile
import json
import time

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

import windfarm_eval

SHP_FILE = ROOT / "data/poly2.shp"
Instance = str(ROOT / "instances/2/param.txt")
X0_filz = str(ROOT / "CODE/samples_LH_square_9_scipy/Sample_LH_0000.txt")

sf = shapefile.Reader(str(SHP_FILE))
print(f"{SHP_FILE.name} contains {len(sf.shapes())} shapes")

# Use the largest shape (if multiple polygons)
shapes = sf.shapes()
largest_shape = max(shapes, key=lambda s: Polygon(s.points).area)
points = largest_shape.points
xs = 1.2 * np.array([p[0] for p in points])
ys = 1.2 * np.array([p[1] for p in points])

# --- Compute bounding box and margin (if needed later) ---
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()


def gradient_EAP(instance_path, X, h, l, free_turbines=None):
    """
    Compute the gradient of penalized EAP w.r.t turbine positions using central finite differences.

    Parameters:
    - instance_path: path to the windfarm instance file
    - X: list or array of turbine positions [x0,y0,x1,y1,...]
    - h: finite difference step size
    - l: penalty coefficient
    - free_turbines: list of turbine indices (0-based) that are allowed to move.
                     If None -> all turbines are free.

    Returns:
    - grad: numpy array of derivatives w.r.t each x and y coordinate
    """
    # If X is a file path, read it
    if isinstance(X, str):
        with open(X, "r", encoding="utf-8") as f:
            s = f.read()
        s = s.replace("[", "").replace("]", "").replace(",", " ")
        X = [float(t) for t in s.split()]

    X = np.array(X, dtype=float)
    n_turbines = len(X) // 2
    grad = np.zeros_like(X, dtype=float)

    if free_turbines is None:
        free_turbines = range(n_turbines)

    for i in free_turbines:
        Xp = X.copy()
        Xm = X.copy()
        Yp = X.copy()
        Ym = X.copy()

        # perturb x_i
        Xp[2 * i] += h
        Xm[2 * i] -= h
        # perturb y_i
        Yp[2 * i + 1] += h
        Ym[2 * i + 1] -= h

        # Evaluate EAP at perturbed positions
        EAP_Xp, spacing_Xp, placing_Xp = windfarm_eval.windfarm_eval(instance_path, Xp.tolist())
        EAP_Xm, spacing_Xm, placing_Xm = windfarm_eval.windfarm_eval(instance_path, Xm.tolist())
        EAP_Yp, spacing_Yp, placing_Yp = windfarm_eval.windfarm_eval(instance_path, Yp.tolist())
        EAP_Ym, spacing_Ym, placing_Ym = windfarm_eval.windfarm_eval(instance_path, Ym.tolist())

        # Central difference on the penalized objective: EAP - l*(spacing+placing)
        grad[2 * i] = (
            (EAP_Xp - l * (spacing_Xp + placing_Xp)) -
            (EAP_Xm - l * (spacing_Xm + placing_Xm))
        ) / (2 * h)

        grad[2 * i + 1] = (
            (EAP_Yp - l * (spacing_Yp + placing_Yp)) -
            (EAP_Ym - l * (spacing_Ym + placing_Ym))
        ) / (2 * h)

    return grad


def gradient_descent(instance_path, X_init, h, alpha, tol, max_iter, l,
                     free_turbines=None, track_index=None):
    """
    Gradient ascent on the penalized EAP (EAP - l*(spacing+placing)).

    Returns:
    - X_opt (list): optimized positions
    - path (list): list of (x, y) positions of tracked turbine
    - it (int): number of iterations performed
    - alpha (float): final step size
    - reached_max_iter (bool): True if loop stopped due to max_iter
    """
    X = np.array(X_init, dtype=float)
    it = 0

    path = []
    if track_index is not None:
        x_track = X[2 * track_index]
        y_track = X[2 * track_index + 1]
        path.append((x_track, y_track))

    grad = gradient_EAP(instance_path, X, h, l, free_turbines=free_turbines)
    grad_norm = np.linalg.norm(grad)
    print(f"Initial gradient norm: {grad_norm}")

    while grad_norm > tol and it < max_iter:
        grad = np.array(gradient_EAP(instance_path, X, h, l, free_turbines=free_turbines),
                        dtype=float)
        grad_norm = np.linalg.norm(grad)

        try:
            EAP_val, spacing, placing = windfarm_eval.windfarm_eval(instance_path, X.tolist())
            EAP_val_penalise = EAP_val - l * (spacing + placing)
        except ValueError:
            print(f"Iteration {it + 1}: Invalid turbine positions, reducing step size")
            alpha *= 0.5
            continue

        print(f"Iter {it + 1}: EAP_pen={EAP_val_penalise:.6f}, alpha={alpha}, Grad norm={grad_norm:.6f}")

        # tentative update (gradient ASCENT on penalized EAP)
        X_new = X + alpha * grad

        try:
            EAP_new, spacing_new, placing_new = windfarm_eval.windfarm_eval(
                instance_path, X_new.tolist()
            )
            EAP_new_penalise = EAP_new - l * (spacing_new + placing_new)
        except ValueError:
            alpha *= 0.5
            continue

        # if penalized EAP did not improve, reduce step size
        if EAP_new_penalise < EAP_val_penalise:
            alpha *= 0.5
        else:
            X = X_new  # accept step

            if track_index is not None:
                x_track = X[2 * track_index]
                y_track = X[2 * track_index + 1]
                path.append((x_track, y_track))

        it += 1

    reached_max_iter = (it >= max_iter)

    return X.tolist(), path, it, alpha, reached_max_iter


# --- Load initial 9-turbine positions ---
with open(X0_filz, "r") as f:
    s = f.read().replace("[", "").replace("]", "").replace(",", " ")
X_9 = [float(t) for t in s.split()]      # length = 18  (9 turbines)

# --- Add a 10th turbine (initial guess) ---
x10_init = 1050.0
y10_init = 150.0
X_init_10 = X_9 + [x10_init, y10_init]

# Index of the 10th turbine in [x0,y0,x1,y1,...]
n_turbines = len(X_init_10) // 2
tenth_index = n_turbines - 1           # 0-based index (so 9 if there are 10 turbines)

free_turbines = [tenth_index]          # only move the 10th turbine

# Parameters
h = 12
l = 10
alpha_init = 1000
tol = 1e-3
max_iter = 500

# --- Optional: check gradient only for the 10th turbine ---
grad = gradient_EAP(Instance, X_init_10, h, l, free_turbines=free_turbines)
print("Gradient (only 10th turbine non-zero):", grad)
print("Norm of gradient:", np.linalg.norm(grad))

# --- Run gradient descent: optimize only the 10th turbine ---
start_time = time.time()

X_opt, path, n_iter, alpha_final, reached_max_iter = gradient_descent(
    Instance,
    X_init_10,
    h=h,
    alpha=alpha_init,
    tol=tol,
    max_iter=max_iter,
    l=l,
    free_turbines=free_turbines,
    track_index=tenth_index
)

end_time = time.time()
runtime = end_time - start_time
print(f"\nGradient descent runtime: {runtime:.4f} seconds")
print("Optimized positions (9 fixed + optimized 10th):", X_opt)

out_path = ROOT / "CODE/optimized_position_bb.txt"
with open(out_path, "w") as f:
    json.dump(X_opt, f, indent=2)

print(f"\nSaved evaluations to {out_path}")

# ---- Neat summary "table" ----
start_pos_10 = [x10_init, y10_init]
final_pos_10 = X_opt[2 * tenth_index: 2 * tenth_index + 2]

print("\n" + "=" * 90)
print(f"{'ALGORITHM SETTINGS':<40} {'10th TURBINE & RUNTIME':<40}")
print("-" * 90)
print(f"{'Finite diff step h':<28} {h:<10.3f} {'Start position [x, y]':<22} {start_pos_10}")
print(f"{'Initial step size alpha':<28} {alpha_init:<10.3f} {'Final position [x, y]':<22} {final_pos_10}")
print(f"{'Final step size alpha':<28} {alpha_final:<10.3f}")
print(f"{'Tolerance':<28} {tol:<10.2e} {'Runtime (s)':<22} {runtime:.4f}")
print(f"{'Max iterations':<28} {max_iter:<10d} {'Iterations used':<22} {n_iter}")
print(f"{'Penalty coefficient λ':<28} {l:<10.3f}")
print(f"{'Reached max_iter?':<28} {str(reached_max_iter):<10}")
print("=" * 90 + "\n")

# --- Prepare coordinates for plotting ---
coords_opt = np.array(X_opt).reshape(-1, 2)   # shape (10, 2)
path_arr = np.array(path)                     # shape (n_steps, 2)

poly = Polygon(zip(xs, ys))

plt.figure(figsize=(8, 7))

# Original polygon
x_poly, y_poly = poly.exterior.xy
plt.plot(x_poly, y_poly, color="blue", linewidth=2, label="Polygon")

# 1) Plot all turbines in black (final positions)
for i in range(coords_opt.shape[0]):
    if i != tenth_index:
        plt.scatter(coords_opt[i, 0], coords_opt[i, 1], c="black", s=40)

# 2) Plot the trajectory of the 10th turbine in red
plt.plot(path_arr[:, 0], path_arr[:, 1], '-o', c="red", linewidth=1, markersize=4,
         label="Path of 10th turbine")

# start point
plt.scatter(path_arr[0, 0], path_arr[0, 1],
            facecolors="none", edgecolors="red", s=100, label="Start (10th)")

# final point
plt.scatter(path_arr[-1, 0], path_arr[-1, 1],
            c="red", s=60, label="Final (10th)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient descent path of 10th turbine")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
