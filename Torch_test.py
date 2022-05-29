import uproot
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from DNN import DNN, EarlyStopping, train_withEarlyStopping, LabelSmoothing

Zmin = 45
Zmax = 62
Znum = Zmax - Zmin + 1

infname = "/data3/AMNT/RDF/r0156_ZPID_Preparation_HLE_forCL.root"

t2 = uproot.open(infname+":t2")
# library = "ak"(default) or "np" or "pd"
N_samples = t2.num_entries
N_samples_str = "1e7"
N_samples = int(float(N_samples_str))
IC = t2["IC"].array(entry_start=0, entry_stop=N_samples, library="np")
ICdx = t2["ICdx"].array(entry_start=0, entry_stop=N_samples, library="np")
IC0Cor = t2["IC0Cor"].array(entry_start=0, entry_stop=N_samples, library="np")
Mint = t2["Mint"].array(entry_start=0, entry_stop=N_samples, library="np")
# mICE = t2["mICE"].array(entry_start=0, entry_stop=N_samples, library="np")
# mICdE = t2["mICdE"].array(entry_start=0, entry_stop=N_samples, library="np")
TargetZ = t2["TargetZ"].array(
    entry_start=0, entry_stop=N_samples, library="np") - Zmin

Inputs = np.column_stack((IC, IC0Cor, ICdx, Mint))

MMS = MinMaxScaler()
MMS.fit(Inputs)
Inputs = MMS.transform(Inputs)

outfMMS = f'./Models/MinMaxScaler_forTorch_{N_samples_str}.pkl'
with open(outfMMS, 'wb') as outf:
    pickle.dump(MMS, outf)
print('MinMaxScaler Output : ', outfMMS)

T_Input = torch.as_tensor(Inputs, dtype=torch.float32)
T_Target = torch.as_tensor(TargetZ, dtype=torch.int32)
# Make soft labels artificially
T_Target = LabelSmoothing(T_Target, Znum)
NInputVs = T_Input.shape[1]

# print(T_Input)
# print(f"N_samples : {T_Input.shape[0]}")


# Create Dataset and DataLoader to handle the Input and Target easily
batch_size = 500
DSet = TensorDataset(T_Input, T_Target)
N_train = int(N_samples*0.8)
N_valid = int(N_samples - N_train)
DTrain, DValid = random_split(
    DSet, lengths=[N_train, N_valid])

DTrainLoader = DataLoader(DTrain, batch_size=batch_size, shuffle=True)
DValidLoader = DataLoader(DValid, batch_size=batch_size, shuffle=True)

model = DNN(NInputVs, Znum)
loss_fcn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
ConvergenceSteps = 10

outfmodel = f"./Models/DNN_Tanh20x20_ICICdxIC0CorMint_{N_samples_str}_SoftLabels.pth"
# torch.save(model.state_dict(), outfmodel)
print(f"Model Output : {outfmodel}")

early_stopping = EarlyStopping(ConvergenceSteps, 0, outfmodel, True)

print("Training start!")
train_withEarlyStopping(model, loss_fcn, optimizer,
                        DTrainLoader, DValidLoader, early_stopping, True)

# # For early stopping using validation set
# N_conv = 0
# epoch = 0
# vloss_old = 10000000.
# vloss_new = 0.
# while((N_conv < ConvergenceSteps) & (epoch < 1000)):
#     print(f"Epoch : {epoch:04d}", end='  ')
#     vloss_new = train(model, loss_fcn, optimizer,
#                       DTrainLoader, DValidLoader, 1)
#     if(vloss_new < vloss_old):
#         print(
#             f'Validation Loss Decreased({vloss_old:.7f}--->{vloss_new:.7f}) \t Saving The Model')
#         torch.save(model.state_dict(), outfmodel)
#         N_conv = 0
#         vloss_old = vloss_new
#     else:
#         N_conv += 1

#     epoch += 1

print()
print("Done!")
