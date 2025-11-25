from pathlib import Path
from modified_surrogate_eap import predict_eap
import sys
import numpy as np

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

from constraints import placing_constraint, spacing_constraint_min, buildable_zone
import windfarm_setting as wf
import data as d

instance_path = str(ROOT / "instances/2/param.txt")

(nb_wt,
 diameter,
 hub_height,
 scale_factor,
 power_curve,
 boundary_file,
 exclusion_zone_file,
 wind_speed,
 wind_direction) = d.read_param_file(instance_path)

# Convert relative to absolute paths
boundary_file_path = str(ROOT / boundary_file)
if exclusion_zone_file != "na":
    exclusion_zone_file_path = str(ROOT / exclusion_zone_file)
else:
    exclusion_zone_file_path = "na"

# ---- 3. Create terrain and buildable zone ----
lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(
    boundary_file_path,
    exclusion_zone_file_path,
    scale_factor
)
ok_zone = buildable_zone(boundary_shapely, exclusion_zones_shapely)

# ---- 4. Parameters ----
D = diameter
lambd = 10


def _ensure_flat_coords(X):
    """
    Ensure X is a flat list [x1, y1, ..., xN, yN].

    Accepts:
      - flat list/tuple [x1, y1, ..., xN, yN]
      - list of pairs [[x1, y1], [x2, y2], ...]
      - 1D numpy array
    """
    if isinstance(X, np.ndarray):
        X = X.flatten().tolist()
    else:
        X = list(X)

    if len(X) == 0:
        return X

    # If it's a list of pairs, flatten it
    if all(isinstance(t, (list, tuple)) for t in X):
        X = [coord for pt in X for coord in pt]

    return X


def _split_coords(X_flat):
    """
    X_flat: flat list [x1, y1, x2, y2, ..., xN, yN]
    returns x_coords, y_coords as lists
    """
    x_coords = X_flat[0::2]
    y_coords = X_flat[1::2]
    return x_coords, y_coords


def penalized_surrogate(X, ok_zone=ok_zone, D=D, lambd=lambd):
    """
    Penalized objective using:
      - NN surrogate for EAP (variable number of turbines, padded internally)
      - exact spacing and placing constraints from constraints.py

    Parameters
    ----------
    X : layout
        - [x1, y1, ..., xN, yN] or
        - [[x1, y1], ..., [xN, yN]] or
        - 1D numpy array of same
        N must be <= N_max used in NN training.
    ok_zone : Shapely MultiPolygon
        Buildable zone (same type you use in placing_constraint)
    D : float
        Minimal distance between two turbines (same as in spacing_constraint_min)
    lambd : float
        Penalty coefficient

    Returns
    -------
    float: -EAP_surrogate + lambd * (spacing + placing)
    """

    # 0) Make sure coordinates are flat for constraints
    X_flat = _ensure_flat_coords(X)

    # 1) surrogate EAP (predict_eap will do its own flatten + padding)
    eap = predict_eap(X_flat)

    # 2) constraints using your exact functions (on REAL coordinates, not padded)
    x_coords, y_coords = _split_coords(X_flat)

    spacing = spacing_constraint_min(x_coords, y_coords, D)
    placing = placing_constraint(x_coords, y_coords, ok_zone)

    # 3) penalized objective
    return -eap + lambd * (spacing + placing)
