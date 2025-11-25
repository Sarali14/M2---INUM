from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import ast


# ---------------------------------------------------
# Paths
# ---------------------------------------------------
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/CODE/Neural_Network_surrogate")
sys.path.insert(0, str(ROOT))

from penalized_surrogate_ter import penalized_surrogate_ter
from penalized_surrogate_bis import penalized_surrogate_bis


# ---------------------------------------------------
# Load initial population : 10-turbine layouts (20 numbers)
# ---------------------------------------------------
PARENTS_DIR = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/CODE/samples_test_FP")

parent_files = sorted(PARENTS_DIR.glob("*.txt"))
initial_X = []

for fpath in parent_files:
    with open(fpath, "r") as f:
        content = f.read().strip()
        layout = np.array(ast.literal_eval(content), dtype=float)

    if layout.size != 20:
        raise ValueError(f"Layout in {fpath} does not have 20 numbers (10 turbines).")

    initial_X.append(layout)

initial_X = np.array(initial_X)
pop_size = initial_X.shape[0]

print(f"Loaded {pop_size} full layouts from {PARENTS_DIR}")


# ---------------------------------------------------
# Multi-objective problem (20 variables)
# ---------------------------------------------------
class WindFarmProblem(Problem):

    def __init__(self):
        super().__init__(n_var=20,      # full layout: 10 turbines x,y
                         n_obj=2,
                         n_constr=0,
                         xl=-1000,
                         xu=1000)

    def _evaluate(self, X, out, *args, **kwargs):

        F1 = []
        F2 = []

        for layout in X:

            f1 = penalized_surrogate_ter(layout)
            f2 = penalized_surrogate_bis(layout)

            F1.append(f1)
            F2.append(f2)

        out["F"] = np.column_stack([F1, F2])


problem = WindFarmProblem()

algorithm = NSGA2(
    pop_size=pop_size,
    sampling=initial_X
)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 80),
    seed=1,
    verbose=True,
    save_history=True
)


# ---------------------------------------------------
# Print statistics
# ---------------------------------------------------
print("\nPareto front (objectives):")
print(res.F)
print("Number of Pareto-optimal solutions:", len(res.F))

F = res.F
print("F1 range:", F[:, 0].min(), "→", F[:, 0].max())
print("F2 range:", F[:, 1].min(), "→", F[:, 1].max())

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

dom_pairs = 0
for i in range(len(F)):
    for j in range(len(F)):
        if i != j and dominates(F[i], F[j]):
            dom_pairs += 1

print("Number of dominating pairs in final population:", dom_pairs)
print("Correlation between f1 and f2:", np.corrcoef(F[:, 0], F[:, 1])[0, 1])


# ---------------------------------------------------
# Save all Pareto layouts
# ---------------------------------------------------
save_dir = Path("pareto_full_10_turbines")
save_dir.mkdir(exist_ok=True)

for i, layout in enumerate(res.X):
    fname = save_dir / f"pareto_layout_{i:03d}.txt"
    with open(fname, "w") as f:
        f.write("[" + ", ".join(f"{v:.6f}" for v in layout) + "]")

print(f"Saved {len(res.X)} Pareto layouts into {save_dir.resolve()}")


# ---------------------------------------------------
# Plot front evolution
# ---------------------------------------------------
all_fronts = res.history
colors = plt.cm.viridis(np.linspace(0, 1, len(all_fronts)))

plt.figure(figsize=(8, 6))

for i, entry in enumerate(all_fronts):
    pop = entry.pop
    F = pop.get("F")
    plt.scatter(F[:, 0], F[:, 1], s=15, alpha=0.8, color=colors[i])

plt.xlabel("Objective 1 (ter)")
plt.ylabel("Objective 2 (bis)")
plt.title("NSGA-II evolution – 10 turbine optimization")
plt.grid(True)
plt.tight_layout()
plt.show()
