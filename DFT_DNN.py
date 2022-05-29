import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from DNN import DNN, EarlyStopping, train_withEarlyStopping
import matplotlib.pyplot as plt


print()
DirPath = "/data3/DFT/DFT_Data/"
train_list_nuclei = list(input("Enter the type of nuclei you want to train, like Ca48 : ").split())
print()
print(f"List of nuclei used for training : {train_list_nuclei}")
print()

# test_list_nuclei = list(input("Enter the type of nuclei you want to test after training, like Ca48 : ").split())
# print()
# print(f"List of nuclei used for testing : {test_list_nuclei}")
# print()

den_list = []
for nucleus in train_list_nuclei:
    den = pd.read_csv(DirPath+nucleus+"_den.PKO2", sep="\s+", header=0)
    den_list.append(den.to_numpy())
np_den = np.concatenate(den_list)
# print(f"first : {np_den[0][7]}")
np_den = np.delete(np_den, 7, axis=1)
# print(f"after : {np_den[0][7]}")

pot_list = []
for nucleus in train_list_nuclei:
    pot_P = pd.read_csv(DirPath+nucleus+"_pot-P.PKO2", sep="\s+", header=0)
    # pot_N = pd.read_csv(DirPath+nucleus+"_pot-N.PKO2", sep="\s+", header=0)
    pot_list.append(pot_P.to_numpy())
    # pot_list.append(pot_N.to_numpy())
np_pot = np.concatenate(pot_list)
# print(f"first : {np_pot[0][5]}")
np_pot = np.delete(np_pot, (5,6), axis=1)

Input = np.delete(np_den, 0, axis=1)
Target = np_pot[:,2] # S_x
# print(f"after : {np_pot[0][5]}")

T_Input = torch.as_tensor(Input, dtype=torch.float32)
T_Target = torch.as_tensor(Target, dtype=torch.float32)

# plt.scatter(T_Input[:,0], T_Target)
# plt.show()

N_samples = T_Input.shape[0]
batch_size = 10

DSet = TensorDataset(T_Input, T_Target)
N_train = int(N_samples*0.8)
N_valid = int(N_samples - N_train)
DTrain, DValid = random_split(
    DSet, lengths=[N_train, N_valid])

DTrainLoader = DataLoader(DTrain, batch_size=batch_size, shuffle=True)
DValidLoader = DataLoader(DValid, batch_size=batch_size, shuffle=True)

# Define the training model of DFT potential
# Number of outputs can be more than 1
model = DNN(T_Input.shape[1], 1)

loss_fcn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
ConvergenceSteps = 30

outfmodel = f"./Models/DNN_Tanh10x10.pth"
# torch.save(model.state_dict(), outfmodel)
print(f"Model Output : {outfmodel}")

early_stopping = EarlyStopping(ConvergenceSteps, 0, outfmodel, True)

print("Training start!")
train_withEarlyStopping(model, loss_fcn, optimizer,
                        DTrainLoader, DValidLoader, early_stopping, True)

model.eval()
# with torch.no_grad:
pot_pred = model(T_Input)

# print(pot_pred)
# print(f"shape : {pot_pred.shape}")
    
plt.scatter(np_den[:,0], pot_pred.detach().numpy())
plt.show()
    



    