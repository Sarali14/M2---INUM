from pathlib import Path
import sys, ast
import numpy as np

# --- Paths setup ---
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CODE"))

import windfarm_eval as windfarm
import data as d
import windfarm_setting as wf
from constraints import buildable_zone
from penalized_surrogate import penalized_surrogate

# ---- 1. Load instance ----
instance_path = str(ROOT / "instances/2/param_ter.txt")

# ---- 2. Load your Latin Hypercube layout ----
# Folder containing 1 file with 30 turbines (60 coords)
lh_folder = ROOT / "CODE/samples_LH_1"
lh_files = list(lh_folder.glob("*.txt"))
assert len(lh_files) == 1, f"Expected 1 LH file, found {len(lh_files)}"

layout_file = lh_files[0]
print(f"Using layout file: {layout_file.name}")

# Read coordinates
with open(layout_file, "r") as f:
    content = f.read().strip()
    layout = ast.literal_eval(content)

# Flatten if needed (some generators store [[x,y],...])
if all(isinstance(t, list) for t in layout):
    X_test = [coord for pair in layout for coord in pair]
else:
    X_test = layout

assert len(X_test) == 2 * nb_wt, f"Layout has {len(X_test)//2} turbines, expected {nb_wt}"

# ---- 3. True evaluation ----
EAP_true, spacing_true, placing_true = windfarm.windfarm_eval(instance_path, X_test)
pen_true = -EAP_true + lambd * (spacing_true + placing_true)

# ---- 4. Surrogate evaluation ----
pen_sur = penalized_surrogate(X_test)

# ---- 5. Compare ----
print("\n=== Penalized Objective Comparison ===")
print(f"EAP_true           = {EAP_true:.6f}")
print(f"Spacing_true       = {spacing_true:.6f}")
print(f"Placing_true       = {placing_true:.6f}")
print(f"Penalized_true     = {pen_true:.6f}")
print()
print(f"Penalized_surrogate = {pen_sur:.6f}")
diff = abs(pen_sur - pen_true)
print(f"Difference          = {diff:.6f}")
if pen_true != 0:
    print(f"Relative difference = {diff/abs(pen_true)*100:.3f}%")
