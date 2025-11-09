import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

import constraints as cst

# ---- Network architecture (same as in training) ----
class MyNet(nn.Module):
    def __init__(self, n):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(n, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- Load checkpoint once at import time ----
_checkpoint = torch.load("eap_surrogate.pt", map_location="cpu")

_n_features = _checkpoint["n_features"]
_model = MyNet(_n_features)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.eval()

_X_mean = _checkpoint["X_mean"]
_X_std  = _checkpoint["X_std"]
_Y_mean = _checkpoint["Y_mean"]
_Y_std  = _checkpoint["Y_std"]


def predict_eap(layout_flat):
    """
    Surrogate EAP prediction.

    layout_flat: list or 1D array [x1, y1, ..., xN, yN]
    returns: float (predicted EAP)
    """
    x = torch.tensor(layout_flat, dtype=torch.float32).unsqueeze(0)  # shape (1, n_features)

    # normalize inputs
    x_norm = (x - _X_mean) / _X_std

    with torch.no_grad():
        y_norm = _model(x_norm)
        y = y_norm * _Y_std + _Y_mean

    return float(y.item())

