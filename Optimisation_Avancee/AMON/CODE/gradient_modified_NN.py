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
sys.path.insert(0, str(ROOT / "CODE/Neural_Network_surrogate"))

from penalized_surrogate import penalized_surrogate
from surrogate_eap import predict_eap

SHP_FILE = ROOT / "data/poly2.shp"
Instance = str(ROOT / "instances/2/param.txt")
X0_filz = str(ROOT / "CODE/samples_LH_square_9_scipy/Sample_LH_0000.txt")

sf = shapefile.Reader(str(SHP_FILE))
print(f"{SHP_FILE.name} contains {len(sf.shapes())} shapes")

# Use the largest shape (if multiple polygons)
shapes = sf.shapes()
largest_shape = max(shapes, key=lambda s: Polygon(s.points).area)
points = largest_shape.points
xs =1.2* np.array([p[0] for p in points])
ys =1.2* np.array([p[1] for p in points])

# bounding box
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()


def gradient_EAP(X, h, l, free_turbines=None):
    """Central difference gradient of penalized NN surrogate."""
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

        Xp[2*i] += h
        Xm[2*i] -= h
        Yp[2*i+1] += h
        Ym[2*i+1] -= h

        pen_Xp = penalized_surrogate(Xp.tolist(), lambd=l)
        pen_Xm = penalized_surrogate(Xm.tolist(), lambd=l)
        pen_Yp = penalized_surrogate(Yp.tolist(), lambd=l)
        pen_Ym = penalized_surrogate(Ym.tolist(), lambd=l)

        grad[2*i] = (pen_Xp - pen_Xm) / (2*h)
        grad[2*i+1] = (pen_Yp - pen_Ym) / (2*h)

    return grad


def gradient_descent(X_init, h, alpha, tol, max_iter, l,
                     free_turbines=None, track_index=None):

    X = np.array(X_init, dtype=float)
    it = 0

    path = []
    if track_index is not None:
        path.append((X[2*track_index], X[2*track_index+1]))

    grad = gradient_EAP(X.tolist(), h, l, free_turbines)
    grad_norm = np.linalg.norm(grad)
    print(f"Initial gradient norm: {grad_norm}")

    while grad_norm > tol and it < max_iter:

        grad = gradient_EAP(X.tolist(), h, l, free_turbines)
        grad_norm = np.linalg.norm(grad)

        try:
            pen_val = penalized_surrogate(X.tolist(), lambd=l)
        except ValueError:
            print(f"Iteration {it+1}: invalid pos, shrinking alpha")
            alpha *= 0.5
            it +=1
            continue

        print(f"Iter {it+1}: pen_surrogate={pen_val:.6f}, alpha={alpha}, grad_norm={grad_norm:.6f}")

        X_new = X - alpha * grad
        """if free_turbines is not None:
            for i in free_turbines:
                X_new[2*i]   = np.clip(X_new[2*i],   min_x, max_x)
                X_new[2*i+1] = np.clip(X_new[2*i+1], min_y, max_y)"""

        try:
            pen_new = penalized_surrogate(X_new.tolist(), lambd=l)
        except ValueError:
            alpha *= 0.5
            it+=1
            continue

        if pen_new < pen_val:
            # accept
            X = X_new
            pen_val = pen_new
            if track_index is not None:
                path.append((X[2*track_index], X[2*track_index+1]))
        else:
            # reject step, just reduce step size
            alpha *= 0.5

        it += 1

    reached_max = (it >= max_iter)
    return X.tolist(), path, it, alpha, reached_max


# Load initial positions
with open(X0_filz, "r") as f:
    s = f.read().replace("[", "").replace("]", "").replace(",", " ")
X_9 = [float(t) for t in s.split()]

# Add 10th turbine
x10_init = 1050.0
y10_init = 150.0
X_init_10 = X_9 + [x10_init, y10_init]

n_turbines = len(X_init_10) // 2
tenth_index = n_turbines - 1
free_turbines = [tenth_index]

# parameters
h = 12
l = 10
alpha_init = 1000
tol = 1e-3
max_iter = 100

# gradient test
grad = gradient_EAP(X_init_10, h, l, free_turbines)
print("Gradient (only 10th turbine non-zero):", grad)
print("Norm:", np.linalg.norm(grad))

# run gradient descent
start = time.time()

X_opt, path, n_iter, alpha_final, reached_max = gradient_descent(
    X_init_10, h=h, alpha=alpha_init, tol=tol, max_iter=max_iter,
    l=l, free_turbines=free_turbines, track_index=tenth_index)

runtime = time.time() - start

print(f"\nGradient descent runtime: {runtime:.4f} seconds")
print("Optimized positions:", X_opt)

eap_start = predict_eap(X_init_10)
eap_opt   = predict_eap(X_opt)
# save
with open(ROOT/"CODE/optimized_position_NN.txt", "w") as f:
    json.dump(X_opt, f, indent=2)

print("\nSaved evaluations.\n")

# =======================
# 👍 SUMMARY TABLE (added)
# =======================

start_pos = [x10_init, y10_init]
final_pos = X_opt[2*tenth_index:2*tenth_index+2]

print("\n" + "="*90)
print(f"{'ALGORITHM SETTINGS':<40} {'10th TURBINE & RUNTIME':<40}")
print("-"*90)
print(f"{'Finite diff step h':<28} {h:<10.3f} {'Start [x, y]':<22} {start_pos}")
print(f"{'Initial alpha':<28} {alpha_init:<10.3f} {'Final [x, y]':<22} {final_pos}")
print(f"{'Final alpha':<28} {alpha_final:<10.3f} {'Runtime (s)':<22} {runtime:.4f}")
print(f"{'Tolerance':<28} {tol:<10.2e} {'Iterations used':<22} {n_iter}")
print(f"{'Max iterations':<28} {max_iter:<10d} {'Reached max_iter?':<22} {reached_max}")
print(f"{'Penalty λ':<28} {l:<10.3f}")
print(f"{'EAP at start (NN)':<28} {eap_start:<10.3f}")
print(f"{'EAP at optimum (NN)':<28} {eap_opt:<10.3f}")
print("="*90 + "\n")

# plotting
coords_opt = np.array(X_opt).reshape(-1, 2)
path_arr = np.array(path)
poly = Polygon(zip(xs, ys))

plt.figure(figsize=(8,7))
x_poly, y_poly = poly.exterior.xy
plt.plot(x_poly, y_poly, color="blue", linewidth=2)

for i in range(coords_opt.shape[0]):
    if i != tenth_index:
        plt.scatter(coords_opt[i, 0], coords_opt[i, 1], c="black", s=40)

plt.plot(path_arr[:,0], path_arr[:,1], '-o', c="red", markersize=4)
plt.scatter(path_arr[0,0], path_arr[0,1], facecolors="none", edgecolors="red", s=100)
plt.scatter(path_arr[-1,0], path_arr[-1,1], c="red", s=60)

plt.axis("equal")
plt.grid()
plt.title("Gradient descent path of 10th turbine (NN surrogate)")
plt.show()
