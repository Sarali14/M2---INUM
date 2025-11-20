import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import sys,ast,random
import matplotlib.pyplot as plt

ROOT = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

import windfarm_eval as windfarm

Instances = str(ROOT / "instances/2/param_ter.txt")
coordinates_folder=Path.cwd() / "CODE/samples_LH_square_training"

all_instances = list(coordinates_folder.glob("*.txt"))
random.shuffle(all_instances)

"""print("Looking for layouts in:", coordinates_folder)
print("Found", len(all_instances), "files")
if len(all_instances) == 0:
    raise RuntimeError("No layout files found – check folder path and names.")
"""
def read_layout(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        layout = ast.literal_eval(content)  # convert string to list
    if all(isinstance(t, list) for t in layout):
        layout_flat = [coord for turbine in layout for coord in turbine]  # flatten [[x,y],...] -> [x1,y1,...,x10,y10]
    else:
        layout_flat = layout  
    return layout_flat

X=[]
Y_eap=[]

for layout_file in all_instances:  # loop over your 500 .txt files
    # read the layout from file
    layout = read_layout(layout_file)  # list of [x1, y1, ..., x10, y10]
    # flatten if necessary
    #layout_flat = [coord for coord in layout]  # already flattened if read correctly
    X.append(layout)
    eap,spacing,placing= windfarm.windfarm_eval(Instances,layout)
    Y_eap.append(eap)

X=torch.tensor(X,dtype=torch.float32)

n_samples,n_features=X.shape

train_instances=int(n_samples*0.8)
test_instances=n_samples-train_instances

X_mean=X.mean(dim=0, keepdim=True)
X_std=X.std(dim=0,keepdim=True)

X_std[X_std==0]=1.0

X_norm=(X-X_mean)/X_std

Y_eap=torch.tensor(Y_eap,dtype=torch.float32).unsqueeze(1)

Y_mean=Y_eap.mean()
Y_std=Y_eap.std()
Y_norm=(Y_eap - Y_mean)/Y_std

print("Y_mean =", Y_mean.item(), "Y_std =", Y_std.item())
print("EAP range:", Y_eap.min().item(), "to", Y_eap.max().item())

class myNet(nn.Module):
    def __init__(self,n):
        super(myNet,self).__init__()
        self.fc1=nn.Linear(n,16)
        self.fc2=nn.Linear(16,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)

        return x

model=myNet(n_features)

optimizer=optim.Adam(model.parameters(),lr=0.0001)
loss_function=nn.MSELoss()

epochs=700

X_train=X_norm[:train_instances]
X_test=X_norm[train_instances:]

Y_train=Y_norm[:train_instances]
Y_test=Y_norm[train_instances:]

train_error=[]
epochs_list=[]
test_error=[]

for epoch in range(1,epochs+1):
    model.train()
    optimizer.zero_grad()
    Y_pred_train=model(X_train)
    Y_pred_train_MSE=loss_function(Y_pred_train,Y_train)
    Y_pred_train_MSE.backward()
    optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        Y_pred_test = model(X_test)
        Y_pred_test_MSE = loss_function(Y_pred_test, Y_test)

"""train_error.append(Y_pred_train_MSE.item())
    test_error.append(Y_pred_test_MSE.item())
    epochs_list.append(epoch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: train={Y_pred_train_MSE.item():.4f}, test={Y_pred_test_MSE.item():.4f}")"""
checkpoint = {
    "model_state_dict": model.state_dict(),
    "n_features": n_features,   # this is X.shape[1]
    "X_mean": X_mean,
    "X_std": X_std,
    "Y_mean": Y_mean,
    "Y_std": Y_std,
}

torch.save(checkpoint, "CODE//Neural_Network_surrogate/eap_surrogate_ter.pt")
print("Saved surrogate to eap_surrogate_ter.pt")


"""
# ---- Plot ----
plt.plot(epochs_list, train_error, label="Training Set", color='magenta')
plt.plot(epochs_list, test_error, label="Validation Set", color='gray')
best_idx = int(np.argmin(test_error))  
best_epoch = epochs_list[best_idx]
plt.axvline(best_epoch, linestyle='--', color='black')
plt.text(best_epoch + 0.5, min(test_error), "Optimal\ncomplexity", color='black')
plt.xlabel("Training Epoch")
plt.ylabel("Error (MSE)")
plt.legend()
plt.show()

Y_pred_test = Y_pred_test * Y_std + Y_mean
Y_test_exact = Y_test * Y_std + Y_mean

Y_true = Y_test_exact.squeeze().numpy()
Y_pred = Y_pred_test.squeeze().numpy()

plt.figure()
plt.scatter(Y_true, Y_pred, alpha=0.7)
min_val = min(Y_true.min(), Y_pred.min())
max_val = max(Y_true.max(), Y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # ideal line y = x
plt.xlabel("Exact EAP")
plt.ylabel("Predicted EAP")
plt.title("Test set: exact vs predicted EAP")
plt.grid(True)
plt.show()
"""  
