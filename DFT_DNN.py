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


def GetInputAndTarget(train_list_nuclei, Rcut):

    # * Dir path of .PKO2 files
    DirPath = "/data3/DFT/DFT_Data/"
    
    # * convert density, potential in .PKO2 file to numpy array
    den_list = []
    pot_list = []
    for nucleus in train_list_nuclei:
        den = pd.read_csv(DirPath+nucleus+"_den.PKO2", sep="\s+", header=0)
        np_den_temp = den.to_numpy()
        radius = np_den_temp[:,0]
        den_list.append(np_den_temp[radius<Rcut])
        pot_P = pd.read_csv(DirPath+nucleus+"_pot-P.PKO2", sep="\s+", header=0)
        # pot_N = pd.read_csv(DirPath+nucleus+"_pot-N.PKO2", sep="\s+", header=0)
        pot_list.append(pot_P.to_numpy()[radius<Rcut])
        # pot_list.append(pot_N.to_numpy())
    
    np_den = np.concatenate(den_list)    
    np_pot = np.concatenate(pot_list)
    
    # * Delete T_H and T_x (all-zero) columns
    np_pot = np.delete(np_pot, (5,6), axis=1)

    return np_den, np_pot, den_list, pot_list

def PreProcessing(np_den, np_pot):
    
    # * Delete radius colmun and rhoc (all-zero) column
    Input = np.delete(np_den, (0,7), axis=1)
    # * Transform the input variables
    # * 1 - Log transform : rho->1/3*log(rho), drho->log(drho)
    # Input[:,0:6] = 1./3*np.log(np.absolute(Input[:,0:6]))
    # Input[:,6:10] = np.log(np.absolute(Input[:,6:10])/np.power(np.absolute(np_den[:,1:5]),4/3))
    # Mins = np.min(Input, axis=0)
    # Maxs = np.max(Input, axis=0)
    # print(Mins)
    # print(Maxs)
    # print(Maxs-Mins)
    # * 2 - rho's -> rho(n) + rho(p) and rho(n) - rho(p)
    # * Take only vector density for V_H calculation
    Input = Input[:,(2,3,8,9)]
    # temp = np.zeros(Input.shape)
    # temp[:,0] = Input[:,0] + Input[:,1]
    # temp[:,1] = (Input[:,0] - Input[:,1]) #/(Input[:,0] + Input[:,1])
    # Input = temp
    # print(Input.shape)

    # * Make Potentials to be positive, divide with transformed density (1/3*log(n))
    Target = np_pot[:,3] # 1 for S_H, 2 for S_x, 3 for V_H 4 for V_x
    
    MMS_Input = MinMaxScaler()
    Input = MMS_Input.fit_transform(Input)
    
    MMS_Target = MinMaxScaler()
    Target = MMS_Target.fit_transform(Target.reshape(-1,1))
    # print(Target.shape)
    
    # print(Input)
    # print(Target)
    
    return Input, Target, MMS_Input, MMS_Target

def PreProcessing_test(np_den, np_pot, MMS_Input, MMS_Target):
    
    # * Delete radius colmun and rhoc (all-zero) column
    Input = np.delete(np_den, (0,7), axis=1)
    
    # * Take only vector density for V_H calculation
    Input = Input[:,(2,3,8,9)]

    # * Make Potentials to be positive, divide with transformed density (1/3*log(n))
    Target = np_pot[:,3] # 1 for S_H, 2 for S_x, 3 for V_H 4 for V_x
    
    Input = MMS_Input.transform(Input)
    Target = MMS_Target.transform(Target.reshape(-1,1))
    
    return Input, Target


def main():
    
    # * Get the list of train nuclei and test nuclei
    print()
    train_list_nuclei = list(input("Enter the type of nuclei you want to train, like Ca48 : ").split())
    print()
    print(f"List of nuclei used for training : {train_list_nuclei}")
    print()
    test_list_nuclei = list(input("Enter the type of nuclei you want to test after training, like Ca48 : ").split())
    print()
    print(f"List of nuclei used for testing : {test_list_nuclei}")
    print()
    
    # * Set the radius [fm] cut (to cut the low density region)
    Rcut = 9.
    
    # * Get the inputs (density related terms) and targets (potential related terms)
    np_den, np_pot, np_den_list, np_pot_list = GetInputAndTarget(train_list_nuclei, Rcut)
    test_np_den, test_np_pot, test_np_den_list, test_np_pot_list = GetInputAndTarget(test_list_nuclei, Rcut)    
    
    # * Preprocessing (typically using MinMaxScaler) inputs and targets
    Input, Target, MMS_Input, MMS_Target = PreProcessing(np_den, np_pot)
    test_Input, test_Target = PreProcessing_test(test_np_den, test_np_pot, MMS_Input, MMS_Target)
    
    # * Convert into tensors which are used in PyTorch Training
    T_Input = torch.as_tensor(Input, dtype=torch.float32)
    T_Target = torch.as_tensor(Target, dtype=torch.float32)
    test_T_Input = torch.as_tensor(test_Input, dtype=torch.float32)
    test_T_Target = torch.as_tensor(test_Target, dtype=torch.float32)

    # * Define some parameters for dataset splitting and loading
    N_samples = T_Input.shape[0]
    batch_size = 5
    DSet = TensorDataset(T_Input, T_Target)
    N_train = int(N_samples*0.6)
    N_valid = int(N_samples - N_train)
    DTrain, DValid = random_split(
        DSet, lengths=[N_train, N_valid])
    DTrainLoader = DataLoader(DTrain, batch_size=batch_size, shuffle=True, drop_last=True)
    DValidLoader = DataLoader(DValid, batch_size=batch_size, shuffle=True, drop_last=True)
    # DTotLoader = DataLoader(DSet, batch_size=batch_size, shuffle=True, drop_last=True)

    # * Define the training model of DFT potential
    # * Number of outputs can be more than 1
    Num_HiddenLayers = 2
    Num_Nodes = 10
    model = DNN(Num_HiddenLayers, Num_Nodes, T_Input.shape[1], 1)
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ConvergenceSteps = 30
    
    # * Write the path of DNN weight file
    outfmodel = f"./Models/DNN_Tanh10x10.pth"
    print(f"Model Output : {outfmodel}")

    # * Train with early stopping and save the DNN weights iteratively
    early_stopping = EarlyStopping(ConvergenceSteps, 0, outfmodel, True)
    print("Training start!")
    train_withEarlyStopping(model, loss_fcn, optimizer,
                            DTrainLoader, DValidLoader, early_stopping, True)

    # * Train without early stopping
    # tloss_arr = []
    # N_epoch = 1000
    # for i in range(N_epoch):
    #     tloss = train(model, loss_fcn, optimizer, DTotLoader, True)
    #     tloss_arr.append(tloss)
    
    # * Calculate output for all nuclei (train + test)
    Total_Input = torch.cat((T_Input, test_T_Input), dim=0)
    model.eval()
    with torch.no_grad():
        pot_pred = model(Total_Input)       
    Total_Output = pot_pred.detach().numpy()
    
    # * Inverse transform to original potential energy scale
    Total_Output = MMS_Target.inverse_transform(Total_Output)
    T_Target = MMS_Target.inverse_transform(T_Target)
    test_T_Target = MMS_Target.inverse_transform(test_T_Target)
    
    # * Separate Output to train set and test set
    T_Output = Total_Output[:T_Target.shape[0],]
    test_T_Output = Total_Output[T_Target.shape[0]:,]

    # * Set color and line characteristics of target and output
    Target_Color_Opt = ['r^-','b^-','g^-','c^-', 'k^-']
    Output_Color_Opt = ['rs-','bs-','gs-','cs-', 'ks-']
    
    # * Draw potential:radius for train nuclei
    i = 0
    Ntemp = 0
    for nuclei in train_list_nuclei:
        nuclei_pot_Output = T_Output[Ntemp:Ntemp+np_den_list[i].shape[0]]
        nuclei_pot_Target = T_Target[Ntemp:Ntemp+np_den_list[i].shape[0]]
        plt.plot(np_den_list[i][:,0], nuclei_pot_Output, Output_Color_Opt[i], label = f'{nuclei}_Output', linewidth=1, mfc='none')
        plt.plot(np_den_list[i][:,0], nuclei_pot_Target, Target_Color_Opt[i], label = f'{nuclei}_Target', linewidth=1, mfc='none')
        Ntemp += np_den_list[i].shape[0]
        i += 1
    plt.xlabel("radius [fm]")
    plt.ylabel("Potential [MeV]")
    plt.legend()
    plt.show()
    
    # * Draw potential:radius for test nuclei
    i = 0
    Ntemp = 0
    for nuclei in test_list_nuclei:
        nuclei_pot_Output = test_T_Output[Ntemp:Ntemp+test_np_den_list[i].shape[0]]
        nuclei_pot_Target = test_T_Target[Ntemp:Ntemp+test_np_den_list[i].shape[0]]
        plt.plot(test_np_den_list[i][:,0], nuclei_pot_Output, Output_Color_Opt[i], label = f'{nuclei}_Output', linewidth=1, mfc='none')
        plt.plot(test_np_den_list[i][:,0], nuclei_pot_Target, Target_Color_Opt[i], label = f'{nuclei}_Target', linewidth=1, mfc='none')
        Ntemp += test_np_den_list[i].shape[0]
        i += 1
    plt.xlabel("radius [fm]")
    plt.ylabel("Potential [MeV]")
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()