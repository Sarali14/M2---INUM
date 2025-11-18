from pathlib import Path
from surrogate_eap import predict_eap
import sys
ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

from constraints import placing_constraint, spacing_constraint_min,buildable_zone
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


def _split_coords(X):
    """
    X: flat list [x1, y1, x2, y2, ..., xN, yN]
    returns x_coords, y_coords as lists
    """
    x_coords = X[0::2]
    y_coords = X[1::2]
    return x_coords, y_coords


def penalized_surrogate(X, ok_zone=ok_zone, D=D, lambd=lambd):
    """
    Penalized objective using:
      - NN surrogate for EAP
      - exact spacing and placing constraints from constraints.py

    Parameters
    ----------
    X : list or 1D array
        Flattened layout [x1, y1, ..., xN, yN]
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

    # 1) surrogate EAP
    eap = predict_eap(X)

    # 2) constraints using your exact functions
    x_coords, y_coords = _split_coords(X)

    spacing = spacing_constraint_min(x_coords, y_coords, D)
    placing = placing_constraint(x_coords, y_coords, ok_zone)

    # 3) penalized objective
    return -eap + lambd * (spacing + placing)
