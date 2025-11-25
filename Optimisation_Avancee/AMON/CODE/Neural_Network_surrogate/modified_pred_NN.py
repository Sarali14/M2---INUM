import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys, ast, random
import matplotlib.pyplot as plt
import time

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))
start_time = time.time()
import windfarm_eval as windfarm

Instances = str(ROOT / "instances/2/param.txt")

# WARNING: if you run this script from CODE/Neural_Network_surrogate,
# Path.cwd() is that folder, so adjust if needed.
coordinates_folder = ROOT / "CODE/samples_LH_square_training_modified"

all_instances = list(coordinates_folder.glob("*.txt"))
random.shuffle(all_instances)


print("Total layouts found:", len(all_instances))

# ---- LIMIT FOR DEBUG ----
MAX_SAMPLES = len(all_instances)   # choose 100, 200, 300... whatever you want
all_instances = all_instances[:MAX_SAMPLES]

print("Using only", len(all_instances), "layouts for this run")
# -------------------------
# CONFIG: max number of turbines
# -------------------------
# Choose N_MAX as the maximum number of turbines you want the surrogate to handle.
# Right now your layouts are for 10 turbines, but you can put 20 for future use
# (you'll then need training data with variable N to fully exploit it).
N_MAX = 40

def read_layout(file_path):
    """
    Read the layout from file.
    Returns a *flattened* list: [x1, y1, x2, y2, ..., xN, yN]
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
        layout = ast.literal_eval(content)  # convert string to list
    if all(isinstance(t, list) for t in layout):
        # [[x,y], [x,y], ...] -> [x1, y1, ..., xN, yN]
        layout_flat = [coord for turbine in layout for coord in turbine]
    else:
        layout_flat = layout
    return layout_flat

def layout_to_padded_features(layout_flat, N_max=N_MAX):
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
            "Increase N_MAX in the script."
        )

    # feats[i] = [x_i, y_i, a_i]
    feats = np.zeros((N_max, 3), dtype=np.float32)

    # Fill first num_turbines slots with real turbines
    for i in range(num_turbines):
        x = coords[2 * i]
        y = coords[2 * i + 1]
        feats[i, 0] = x
        feats[i, 1] = y
        feats[i, 2] = 1.0  # active

    # Remaining rows (i >= num_turbines) stay zeros: (0, 0, 0) => padding

    # Flatten to 1D vector of length 3 * N_max
    return feats.flatten().tolist()

X = []
Y_eap = []

for layout_file in all_instances:  # loop over your .txt files
    # read the layout from file (flattened coords)
    layout_flat = read_layout(layout_file)  # [x1, y1, ..., xN, yN]

    # Padded + masked features for the NN surrogate
    input_vec = layout_to_padded_features(layout_flat, N_max=N_MAX)
    X.append(input_vec)

    # IMPORTANT: black-box evaluation still uses the REAL layout (no padding)
    eap, spacing, placing = windfarm.windfarm_eval(Instances, layout_flat)
    Y_eap.append(eap)

X = torch.tensor(X, dtype=torch.float32)
n_samples, n_features = X.shape

train_instances = int(n_samples * 0.8)
test_instances =n_samples - train_instances

# Standardization (now applied to [x, y, a] padded features)
X_mean = X.mean(dim=0, keepdim=True)
X_std = X.std(dim=0, keepdim=True)
X_std[X_std == 0] = 1.0

X_norm = (X - X_mean) / X_std

Y_eap = torch.tensor(Y_eap, dtype=torch.float32).unsqueeze(1)

Y_mean = Y_eap.mean()
Y_std = Y_eap.std()
Y_norm = (Y_eap - Y_mean) / Y_std

print("Y_mean =", Y_mean.item(), "Y_std =", Y_std.item())
print("EAP range:", Y_eap.min().item(), "to", Y_eap.max().item())

class myNet(nn.Module):
    def __init__(self, n):
        super(myNet, self).__init__()
        self.fc1 = nn.Linear(n, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

model = myNet(n_features)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

epochs = 130

X_train = X_norm[:train_instances]
X_test = X_norm[train_instances:]

Y_train = Y_norm[:train_instances]
Y_test = Y_norm[train_instances:]

train_error = []
epochs_list = []
test_error = []

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    Y_pred_train = model(X_train)
    Y_pred_train_MSE = loss_function(Y_pred_train, Y_train)
    Y_pred_train_MSE.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        Y_pred_test = model(X_test)
        Y_pred_test_MSE = loss_function(Y_pred_test, Y_test)

    train_error.append(Y_pred_train_MSE.item())
    test_error.append(Y_pred_test_MSE.item())
    epochs_list.append(epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: train={Y_pred_train_MSE.item():.4f}, test={Y_pred_test_MSE.item():.4f}")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "n_features": n_features,   # now = 3 * N_MAX
    "N_max": N_MAX,             # <--- save N_max for the prediction script
    "X_mean": X_mean,
    "X_std": X_std,
    "Y_mean": Y_mean,
    "Y_std": Y_std,
}

torch.save(checkpoint, "CODE//Neural_Network_surrogate/modified_eap_surrogate.pt")
print("Saved surrogate to modified_eap_surrogate.pt")
end_time = time.time()
print(f"Total script runtime: {end_time - start_time:.2f} seconds")
