import numpy as np
import torch
from pathlib import Path
import random
import sys
import ast
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path("/home/sarah-ali/M2---INUM/Master_2/Optimisation_Avancee/AMON")   # AMON folder
sys.path.insert(0, str(ROOT))

import windfarm_eval

Instances = str(ROOT / "instances/1/param3.txt")
coordinates_folder=Path.cwd() / "CODE/turbine_layouts"

all_instances = list(coordinates_folder.glob("*.txt"))
random.shuffle(all_instances)

def read_layout(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        layout = ast.literal_eval(content)  # convert string to list
    if all(isinstance(t, list) for t in layout):
        layout_flat = [coord for turbine in layout for coord in turbine]  # flatten [[x,y],...] -> [x1,y1,...,x10,y10]
    else:
        layout_flat = layout  # already flat
    
    # Check length
    if len(layout_flat) != 20:
        raise ValueError(f"Layout {file_path} does not have 20 coordinates: {layout_flat}")
    return layout_flat

def penalized_function(X_file,lambd):
    EAP,spacing,placing= windfarm_eval.windfarm_eval(Instances,X_file)
    return EAP - lambd*(spacing+placing)

X = []  # inputs
Y = []  # penalized outputs

lambd = 0.0001
#print("Looking in folder:", coordinates_folder)
#print("Number of layout files found:", len(all_instances))
#print(all_instances[:5]) 
for layout_file in all_instances:  # loop over your 500 .txt files
    # read the layout from file
    layout = read_layout(layout_file)  # list of [x1, y1, ..., x10, y10]
    
    # flatten if necessary
    layout_flat = [coord for coord in layout]  # already flattened if read correctly
    X.append(layout_flat)
    
    # compute penalized cost using your black-box
    cost = penalized_function(layout_file, lambd)  # pass file, not the whole list
    Y.append(cost)
    
X_tensor=torch.tensor(X, dtype=torch.float32)

X_mean = X_tensor.mean(dim=0, keepdim=True)
X_std = X_tensor.std(dim=0, keepdim=True)

# Avoid dividing by zero
X_std[X_std == 0] = 1.0

# Normalize
X_norm = (X_tensor - X_mean) / X_std

Y_tensor=torch.tensor(Y,dtype=torch.float32).unsqueeze(1)

Y_mean = Y_tensor.mean()
Y_std = Y_tensor.std()
Y_norm = (Y_tensor - Y_mean) / Y_std
class myNet(nn.Module):
  def __init__(self): 
    super(myNet, self).__init__()
    self.fc1=nn.Linear(20,64) 
    self.fc2=nn.Linear(64,64)
    self.fc3=nn.Linear(64,64)
    self.fc4=nn.Linear(64,1) 

  def forward(self, x):
    x=F.relu(self.fc1(x)) #here we see the application as qsked in question ( fully_connected -> relu -> fully_connected ... relu -> fully connected)
    x=F.relu(self.fc2(x))
    x=F.relu(self.fc3(x))
    x=self.fc4(x)

    return x

myNet = myNet() #initiate the class

optimizer=optim.Adam(myNet.parameters(),lr=0.0001) #define the adam optimizer as asked
loss_function= nn.MSELoss() #chose the MSE loss function as we are wrking with continuous values
best_val_loss = float('inf')  # start with "infinite" validation loss
epochs_no_improve = 0
patience = 30  # number of epochs to wait before stopping
best_model_wts = myNet.state_dict() 
#train_error=[] #defined this liste to save the training errors and plot them later
#epochs_list=[]
print("Y mean:", Y_mean.item(), "Y std:", Y_std.item())

epochs=2000

X_train=X_norm[:400]
X_test=X_norm[400:]

Y_train=Y_norm[:400]
Y_test=Y_norm[400:]

for epoch in range(1,epochs+1): #"loop over epochs for the training"
    myNet.train()
    optimizer.zero_grad()
    train_Y=myNet(X_train)
    train_Y_mse=loss_function(train_Y,Y_train)
    train_Y_mse.backward()
    optimizer.step()

    myNet.eval()  # set model to evaluation mode
    with torch.no_grad():
        val_pred = myNet(X_test)
        val_loss = loss_function(val_pred, Y_test)

    if epoch % 100 == 0:
      print(f"Epoch [{epoch}/{epochs}],Train Loss: {train_Y_mse.item():.6f}, Val Loss: {val_loss.item():.6f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        epochs_no_improve = 0
        best_model_wts = myNet.state_dict()  # save best weights
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        myNet.load_state_dict(best_model_wts)  # restore best weights
        break

myNet.eval()

# Suppose you have a new layout file
new_layout_file = "CODE/layout_new_000.txt"
new_layout = read_layout(new_layout_file)  # list of 20 coordinates
new_tensor = torch.tensor([new_layout], dtype=torch.float32)  # batch size 1

new_tensor_norm = (new_tensor - X_mean) / X_std

with torch.no_grad():
    predicted_norm = myNet(new_tensor_norm)
    predicted_value = predicted_norm * Y_std + Y_mean
print("Predicted penalized value:", predicted_value.item())
print("Exact penalized value :" ,penalized_function(new_layout_file,lambd))
