from pathlib import Path
import sys, ast, json

# ============================
# PATHS
# ============================
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")
sys.path.insert(0, str(ROOT))
sys.path.insert(0,str(ROOT/"CODE/Neural_Network_surrogate"))

from penalized_surrogate_ter import penalized_surrogate_ter
from penalized_surrogate_bis import penalized_surrogate_bis

# Base layout (9 turbines)
base_file = ROOT / "CODE/samples_LH_square_9_scipy/Sample_LH_0000.txt"

# Directory containing the 10th turbine samples
folder_FP = ROOT / "CODE/new_turbine_FP"
all_files = sorted(folder_FP.glob("*.txt"))

print(f"Found {len(all_files)} new turbines in: {folder_FP}")

# ============================
# LOAD 9-TURBINE BASE LAYOUT
# ============================
with open(base_file, "r") as f:
    s = f.read().replace("[", "").replace("]", "").replace(",", " ")
base9 = [float(v) for v in s.split()]        # length = 18

print("Loaded base layout with 9 turbines.")

# ============================
# STORAGE FOR RESULTS
# ============================
f_ter_list = []   # first objective
f_bis_list = []   # second objective (if needed)

# ============================
# FUNCTION TO READ X10,Y10
# ============================
def read_xy(file):
    with open(file, "r") as f:
        content = f.read().strip()
        data = ast.literal_eval(content)
    return data  # should be [x10, y10]


# ============================
# PROCESS EACH NEW TURBINE
# ============================
for i, file in enumerate(all_files):

    x10, y10 = read_xy(file)   # read the 10th turbine

    layout10 = base9 + [x10, y10]   # full layout

    # First objective
    f_ter = penalized_surrogate_ter(layout10,lambd=10)

    # Second objective — customize if needed
    # For now: same surrogate with λ=5 example
    f_bis = penalized_surrogate_bis(layout10, lambd=10)

    f_ter_list.append(f_ter)
    f_bis_list.append(f_bis)

    print(f"[{i+1}/{len(all_files)}] {file.name}  →  f1={f_ter:.4f}, f2={f_bis:.4f}")


# ============================
# SAVE RESULTS
# ============================
out_ter = ROOT / "CODE/evaluations_ter.txt"
out_bis = ROOT / "CODE/evaluations_bis.txt"

with open(out_ter, "w") as f:
    json.dump(f_ter_list, f, indent=2)

with open(out_bis, "w") as f:
    json.dump(f_bis_list, f, indent=2)

print("\nSaved:")
print(out_ter)
print(out_bis)
