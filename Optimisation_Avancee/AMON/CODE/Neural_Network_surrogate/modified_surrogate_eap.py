import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

# Full path to the checkpoint (matches modified_pred_NN.py)
CKPT_PATH = HERE / "modified_eap_surrogate.pt"
import constraints as cst

# ---- Network architecture (same as in modified_pred_NN.py) ----
class MyNet(nn.Module):
    def __init__(self, n):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(n, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

# ---- Load checkpoint once at import time ----
_checkpoint = torch.load(CKPT_PATH, map_location="cpu")

_n_features = _checkpoint["n_features"]     # = 3 * N_max
_N_max      = _checkpoint["N_max"]          # same N_MAX as in training

_model = MyNet(_n_features)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.eval()

_X_mean = _checkpoint["X_mean"]
_X_std  = _checkpoint["X_std"]
_Y_mean = _checkpoint["Y_mean"]
_Y_std  = _checkpoint["Y_std"]


def _maybe_flatten(layout):
    """
    Accept either [x1, y1, ..., xN, yN]
    or [[x1, y1], [x2, y2], ...]
    and return a flat list [x1, y1, ..., xN, yN].
    """
    if isinstance(layout, np.ndarray):
        layout = layout.tolist()

    if len(layout) == 0:
        return []

    # list of pairs -> flatten
    if all(isinstance(t, (list, tuple)) for t in layout):
        return [coord for pt in layout for coord in pt]
    else:
        return list(layout)


def layout_to_padded_features(layout_flat, N_max=_N_max):
    """
    Convert a flattened layout [x1, y1, ..., xN, yN] to a padded feature
    vector of size 3 * N_max:

        (x1, y1, a1, x2, y2, a2, ..., xN_max, yN_max, aN_max)

    where a_i = 1 for real turbines (i <= N) and a_i = 0 for padding.
    """
    coords = np.array(layout_flat, dtype=np.float32)
    assert len(coords) % 2 == 0, "Layout length should be 2*N (x,y pairs)."
    num_turbines = len(coords) // 2

    if num_turbines > N_max:
        raise ValueError(
            f"Layout has {num_turbines} turbines, but N_max={N_max}. "
            "Increase N_MAX in training if needed."
        )

    feats = np.zeros((N_max, 3), dtype=np.float32)

    for i in range(num_turbines):
        x = coords[2 * i]
        y = coords[2 * i + 1]
        feats[i, 0] = x
        feats[i, 1] = y
        feats[i, 2] = 1.0  # active turbine

    return feats.flatten().tolist()


def predict_eap(layout_flat):
    """
    Surrogate EAP prediction.

    layout_flat:
        - [x1, y1, ..., xN, yN]   or
        - [[x1, y1], [x2, y2], ...]
    returns: float (predicted EAP)
    """
    # 1) Make sure we have a flat [x1, y1, ..., xN, yN]
    layout_flat = _maybe_flatten(layout_flat)

    # 2) Build padded + masked feature vector (length = 3 * N_max)
    feat_vec = layout_to_padded_features(layout_flat, N_max=_N_max)

    # 3) To tensor (batch size 1)
    x = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0)  # shape (1, n_features)

    # 4) Normalize inputs (same as training)
    x_norm = (x - _X_mean) / _X_std

    # 5) Predict & denormalize
    with torch.no_grad():
        y_norm = _model(x_norm)
        y = y_norm * _Y_std + _Y_mean

    return float(y.item())
