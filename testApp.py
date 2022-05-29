import uproot
import numpy as np
import torch
from array import array
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import ROOT
from DNN import DNN
import time

ROOT.gInterpreter.Declare(
    '#include "/home/cyj6602/Analysis/RootMacros/RDF/RDF_Class.h"')

Dim_In = 20
Zmin = 45
Zmax = 62
Znum = Zmax - Zmin + 1

infname = "/data3/AMNT/ZPID/r0156_PreProcessing_forCL.root"
# "/data3/AMNT/RDF/r0156_ZPID_Preparation_HLE_forCL.root"

t2 = uproot.open(infname+":t2")
# library = "ak"(default) or "np" or "pd"
N_samples_str = "1e6"  # Number of samples used for training model

Nentries = t2.num_entries
Nentries = int(2e7)
IC = t2["IC"].array(entry_start=0, entry_stop=Nentries, library="np")
ICdx = t2["ICdx"].array(entry_start=0, entry_stop=Nentries, library="np")
IC0Cor = t2["IC0Cor"].array(entry_start=0, entry_stop=Nentries, library="np")
Mint = t2["Mint"].array(entry_start=0, entry_stop=Nentries, library="np")
# mICE = t2["mICE"].array(entry_start=0, entry_stop=Nentries, library="np")
# mICdE = t2["mICdE"].array(entry_start=0, entry_stop=Nentries, library="np")
TargetZ = t2["TargetZ"].array(
    entry_start=0, entry_stop=Nentries, library="np") - Zmin

Inputs = np.column_stack((IC, IC0Cor, ICdx, Mint))

infMMS = f'./Models/MinMaxScaler_forTorch_{N_samples_str}.pkl'
print("Using MinMaxScaler in the file : ", infMMS)
MMS = pickle.load(open(infMMS, 'rb'))
Inputs = MMS.transform(Inputs)

T_Input = torch.as_tensor(Inputs, dtype=torch.float32)
# T_Target = torch.as_tensor(TargetZ, dtype=torch.long)
NInputVs = T_Input.shape[1]


infmodel = f"./Models/DNN_Tanh20x20_ICICdxIC0CorMint_{N_samples_str}_SoftLabels.pth"
print("Using Model in the file : ", infmodel)
model = DNN(NInputVs, Znum)
model.load_state_dict(torch.load(infmodel))
# Set model to evaluate (opposed to .train())
model.eval()
with torch.no_grad():
    T_Output = model(T_Input)
    Predict_Prob = torch.softmax(T_Output, dim=-1, dtype=torch.float32)
Outputs = torch.argmax(Predict_Prob, dim=-1) + Zmin
# print(Predict_Prob)
# print(Outputs)

# nThreads = 16
# ROOT.ROOT.EnableImplicitMT(nThreads)

t_start = time.time()

fin = ROOT.TFile(infname, "read")
torg = fin.Get("t2")

outfname = f"/data3/AMNT/ZPID/r0156_ZPID_byDNN_Tanh20x20_ICIC0CorICdxMint_{N_samples_str}_SoftLabels.root"
fout = ROOT.TFile(outfname, "recreate")
tnew = torg.CloneTree(0)

# rdf = ROOT.RDataFrame(infname, "read")
# rdf.DefineSlot("Prob_pred", , {}).DefineSlot("Z_pred", , {})
# rdf.snapshot("t2", outfname)

Z_pred = array('i', [0])
Prob_pred = array('f', Znum*[0.0])
tnew.Branch("Z_pred", Z_pred, "Z_pred/I")
tnew.Branch("Prob_pred", Prob_pred, f"Prob_pred[{Znum}]/F")
# tnew.Branch("Z_target", Z_target, "Z_target/I")
# Nentries = 100000

# NonZeroEnt = 0
for i in range(0, Nentries):
    if i % 1000 == 0:
        print(f"\rProcessing : {1.*i/Nentries*100:.3f}%", end='')

    torg.GetEntry(i)

    # Z_pred[0] = Outputs[NonZeroEnt]
    # Prob_pred[0] = Predict_Prob[NonZeroEnt]
    for j in range(Znum):
        Prob_pred[j] = Predict_Prob[i][j]
    Z_pred[0] = Outputs[i]
    # Z_target[0] = TargetZ[i]
    # print('Output : ', Outputs[i])
    # print('TargetZ : ', TargetZ[i])
    # print('Z_pred : ', Z_pred)
    tnew.Fill()

tnew.Write()
fout.Close()
fin.Close()

print()
print(f'Output : {outfname}')
print(f'Execution Time (sec) : {time.time()-t_start}')


# print("T_Inputs : ", T_Inputs)
# print("NInputs : ", NInputs)
# print(model)

# y_pred = model.forward(T_Inputs[0])
# print("Output : ", y_pred)
# print("Sum of Softmax : ", y_pred.sum())
