from pathlib import Path
import sys,ast,random
import numpy as np
import json
from penalized_surrogate import penalized_surrogate
#from pred_NN_test1 import read_layout

# --- Paths ---
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")
sys.path.insert(0, str(ROOT))

# --- Initialize surrogate ---
instance_path = str(ROOT / "instances/2/param_ter.txt")

# --- Folder with layouts ---
coordinates_folder = ROOT / "CODE/samples_LH_square_FP"
all_instances = sorted(coordinates_folder.glob("*.txt"))

def read_layout(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        layout = ast.literal_eval(content)  # convert string to list 
    if all(isinstance(t, list) for t in layout):
        layout_flat = [coord for turbine in layout for coord in turbine]  # flatten [[x,y],,...] -> [x1,y1,...,x10,y10]
    else:
        layout_flat = layout
    return layout_flat

evaluations = []

for layout_file in all_instances:
    layout = read_layout(layout_file)

    # Flatten [[x,y], [x,y], ...] → [x1, y1, x2, y2, ...]
    #layout_flat = [coord for pair in layout for coord in pair]

    try:
        value = penalized_surrogate(layout)
        print(f"Evaluated {layout_file.name}: {value:.6f}")
    except Exception as e:
        print(f"Error evaluating {layout_file.name}: {e}")
        value = None

    evaluations.append(value)

# --- Save all results ---
out_path = ROOT / "CODE/evaluations.txt"
with open(out_path, "w") as f:
    json.dump(evaluations, f, indent=2)

print(f"\n Saved evaluations to {out_path}")
