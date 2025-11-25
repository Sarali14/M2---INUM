from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")
CODE_DIR=ROOT / "CODE"

file_ter=CODE_DIR / "evaluations_ter.txt"
file_bis=CODE_DIR / "evaluations_bis.txt"

with open(file_ter, "r") as f:
    eval_ter=json.load(f)

with open(file_bis, "r") as f:
    eval_bis=json.load(f)

if len(eval_ter) != len(eval_bis):
    raise ValueError(f"Length mismatch: ter={len(eval_ter)}, bis={len(eval_bis)}")

f_ter =1* np.array(eval_ter, dtype=float)  # shape (N,)
f_bis =1* np.array(eval_bis, dtype=float)  # shape (N,)

N = len(f_ter)
print(f"Loaded {N} points.")

# --- Weighted-sum method over the discrete set ---
# We assume MINIMIZATION of both objectives f1, f2
weights = np.linspace(0.0, 1.0, 21)  # 21 weights: 0.0, 0.05, ..., 1.0

best_indices = []
best_points = []   # (f1, f2, w, index)

for w in weights:
    # Combined objective: F_w = w*f1 + (1-w)*f2
    combined = w * f_ter + (1.0 - w) * f_bis
    idx = np.argmin(combined)   # best layout index for this weight
    best_indices.append(idx)
    best_points.append((f_ter[idx], f_bis[idx], w, idx))
    print(f"w = {w:.2f} → best index = {idx}, f_ter = {f_ter[idx]:.4f}, f_bis = {f_bis[idx]:.4f}")

# --- Remove duplicates (same layout chosen for multiple weights) ---
unique_by_index = {}
for f1_i, f2_i, w, idx in best_points:
    # keep the first occurrence for each layout index
    if idx not in unique_by_index:
        unique_by_index[idx] = (f1_i, f2_i, w)

pareto_approx_indices = sorted(unique_by_index.keys())
pareto_approx = np.array([unique_by_index[i][:2] for i in pareto_approx_indices])  # (M, 2)

print(f"\nApproximate Pareto points found (unique layouts): {len(pareto_approx_indices)}")
print(pareto_approx)
plt.figure(figsize=(8, 6))

# All points
plt.scatter(f_ter,f_bis, s=15, alpha=0.3, color='black',label="f_ter and f_bis  evaluated layouts")

# Weighted-sum approximate front (unique points)
if len(pareto_approx) > 0:
    plt.scatter(pareto_approx[:, 0], pareto_approx[:, 1],
                s=60, alpha=0.9,color='red' ,label="Weighted-sum solutions",
                marker="x")

plt.xlabel("Objective 1 (evaluations_ter)")
plt.ylabel("Objective 2 (evaluations_bis)")
plt.title("Weighted-sum Pareto Approximation (from surrogate evaluations)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
