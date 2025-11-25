from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import ast  # to read [x, y] from txt

# ---------------------------------------------------
# Paths / imports
# ---------------------------------------------------
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")
sys.path.insert(0, str(ROOT))

Instances_ter = str(ROOT / "instances/2/param_ter.txt")
Instances_bis = str(ROOT / "instances/2/param_bis.txt")

#from penalized_surrogate_ter import penalized_surrogate_ter
#from penalized_surrogate_bis import penalized_surrogate_bis
import windfarm_eval as windfarm
# 🔁 CHANGE THIS: base layout with 9 turbines [x0,y0,...,x8,y8]
BASE_LAYOUT_FILE = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/CODE/samples_LH_square_9_scipy/Sample_LH_0000.txt")

# 🔁 CHANGE THIS: directory that contains [x, y] txt files for the 10th turbine
PARENTS_DIR = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/CODE/new_turbine_FP")


# ---------------------------------------------------
# Load base layout (fixed 9 turbines)
# ---------------------------------------------------
with open(BASE_LAYOUT_FILE, "r") as f:
    base_content = f.read().strip()
    # Example content: "[250.7681601299281, 300.123456, ...]"
    BASE_LAYOUT = np.array(ast.literal_eval(base_content), dtype=float)

BASE_LAYOUT = BASE_LAYOUT.flatten()

# Sanity check: should be 9 turbines -> 18 values
n_turbines_fixed = BASE_LAYOUT.size // 2
assert n_turbines_fixed == 9, f"Expected 9 turbines in base layout, got {n_turbines_fixed}"

# Index where the 10th turbine will be appended
# Full layout will be [base_9 (18 values), x9, y9] -> length 20
def build_full_layout(x_last):
    """x_last: array-like [x10, y10] -> full layout of 10 turbines"""
    return np.concatenate([BASE_LAYOUT, np.array(x_last)])


# ---------------------------------------------------
# Build initial population from directory of [x, y] files
# ---------------------------------------------------
parent_files = sorted(PARENTS_DIR.glob("*.txt"))
initial_X = []

for fpath in parent_files:
    with open(fpath, "r") as f:
        content = f.read().strip()
        # content like "[123.45, 67.89]"
        coords = np.array(ast.literal_eval(content), dtype=float)
    if coords.size != 2:
        raise ValueError(f"File {fpath} does not contain exactly 2 values.")
    initial_X.append(coords)

initial_X = np.array(initial_X)
pop_size = initial_X.shape[0]

print(f"Loaded {pop_size} parent turbines from {PARENTS_DIR}")


# ---------------------------------------------------
# Define problem: only 2 decision variables (x10, y10)
# ---------------------------------------------------
class WindFarmProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,           # x10, y10
                         n_obj=2,
                         n_constr=0,
                         xl=-1000,         # bounds for x10, y10 (adapt if needed)
                         xu=1000)

    def _evaluate(self, X, out, *args, **kwargs):
        F1 = []
        F2 = []

        for x_last in X:      # x_last = [x10, y10]
            layout = build_full_layout(x_last)   # full 20D vector

            eap_ter,spacing,placing = windfarm.windfarm_eval(Instances_ter,layout.tolist())
            eap_bis,spacing,placing = windfarm.windfarm_eval(Instances_bis,layout.tolist())

            f1= -eap_ter+ 10*(spacing+placing)
            f2= -eap_bis+ 10*(spacing+placing)

            F1.append(f1)
            F2.append(f2)

        out["F"] = np.column_stack([F1, F2])


problem = WindFarmProblem()

# NSGA2 with custom initial population = your parent turbines
algorithm = NSGA2(
    pop_size=pop_size,
    sampling=initial_X        # <-- parents as initial population
)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True,
               save_history=True)

print("Pareto front (objectives):")
print(res.F)
print("Number of Pareto-optimal solutions:", len(res.F))

F = res.F
print("F1 range:", F[:, 0].min(), "→", F[:, 0].max())
print("F2 range:", F[:, 1].min(), "→", F[:, 1].max())

def dominates(a, b):
    """Return True if a dominates b (for minimization)."""
    return np.all(a <= b) and np.any(a < b)

F = res.F
n = len(F)
dom_pairs = 0

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if dominates(F[i], F[j]):
            dom_pairs += 1

print("Number of dominating pairs in final population:", dom_pairs)


corr = np.corrcoef(F[:, 0], F[:, 1])[0, 1]
print("Correlation between f1 and f2:", corr)
# ---------------------------------------------------
# Save FULL layouts [x0,y0,...,x8,y8,x9,y9] from Pareto front
# ---------------------------------------------------
save_dir = Path("2_criteria_opti")
save_dir.mkdir(exist_ok=True)

pareto_decisions = res.X   # [x10, y10] per solution

for i, x_last in enumerate(pareto_decisions):
    layout = build_full_layout(x_last)  # [x0,y0,...,x8,y8,x9,y9]

    filename = save_dir / f"pareto_layout_{i:03d}.txt"
    list_string = "[" + ", ".join(f"{v:.6f}" for v in layout) + "]"
    with open(filename, "w") as f:
        f.write(list_string)

print(f"Saved {len(pareto_decisions)} FULL Pareto layouts to {save_dir.resolve()}")


# ---------------------------------------------------
# Plot all fronts across generations
# ---------------------------------------------------
all_fronts = res.history
colors = plt.cm.viridis(np.linspace(0, 1, len(all_fronts)))

plt.figure(figsize=(8, 6))

for i, entry in enumerate(all_fronts):
    pop = entry.pop
    F = pop.get("F")
    plt.scatter(F[:, 0], F[:, 1], s=20, color=colors[i], label=f"Gen {i}")

plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.title("Pareto Fronts Across Generations (NSGA-II)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

plt.subplots_adjust(right=0.75)   # give space to legend
plt.tight_layout()
plt.show()
