import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


class DNN(nn.Module):
    def __init__(self, Num_HiddenLayers, Num_Nodes, Num_InputVs, Num_Outputs):
        super(DNN, self).__init__()
        # self.flatten = nn.Flatten()
        # Num_Nodes = 10
        if (Num_HiddenLayers<1):
            print("Number of hidden layers should be larger than 1")
            return
        Layers = []
        Layers.append(nn.Linear(Num_InputVs, Num_Nodes))
        Layers.append(nn.Tanh())
        for i in range(Num_HiddenLayers-1):
            Layers.append(nn.Linear(Num_Nodes, Num_Nodes))
            Layers.append(nn.Tanh())
        Layers.append(nn.Linear(Num_Nodes, Num_Outputs))
        
        self.HLayer = nn.Sequential(*Layers)
            
        # self.HLayer = nn.Sequential(
        #     nn.Linear(Num_InputVs, Num_Nodes),
        #     # nn.BatchNorm1d(Num_Nodes),
        #     nn.Tanh(),
        #     nn.Linear(Num_Nodes, Num_Nodes),
        #     # nn.BatchNorm1d(Num_Nodes),
        #     nn.Tanh(),
        #     # nn.Linear(Num_Nodes, Num_Nodes),
        #     # nn.BatchNorm1d(Num_Nodes),
        #     # nn.Tanh(),
        #     # nn.Linear(5, 5),
        #     # nn.BatchNorm1d(5),
        #     # nn.Tanh(),
        #     # nn.Linear(5, 5),
        #     # nn.BatchNorm1d(5),
        #     # nn.Tanh(),
        #     # nn.Linear(5, 5),
        #     # nn.BatchNorm1d(5),
        #     # nn.Tanh(),
        #     # nn.Linear(5, 5),
        #     # nn.BatchNorm1d(5),
        #     # nn.Tanh(),
        #     # nn.Dropout(p=0.2), # Unactivate some neurons for generalization
        #     nn.Linear(Num_Nodes, Num_Outputs)
        # )

    def forward(self, x):
        # x = self.flatten(x)
        y_pred = self.HLayer(x)
        return y_pred


class EarlyStopping:
    def __init__(self, ConvergenceSteps=10, Delta=0., outfmodel="Model.pth", verbose=True):
        """
        Args:
            ConvergenceSteps (int, optional): The number of patience after valid_loss improvement. Defaults to 10.
            MaxIter (int, optional): Maximum number of train. Defaults to 1000.
            delta (_type_, optional): Minimum magnitude of change in valid_loss for accepting improvement. Defaults to 0..
        """
        self.ConvergenceSteps = ConvergenceSteps
        self.Delta = Delta
        self.N_conv = 0
        self.vloss_min = None  # float("inf")
        # self.vloss_new = 0.
        self.flag_stop = False
        self.outfmodel = outfmodel
        self.verbose = verbose

    def __call__(self, valid_loss, model):
        if self.vloss_min == None:
            self.vloss_min = valid_loss
        elif valid_loss < self.vloss_min:
            if self.verbose:
                print(
                    f'Validation Loss Decreased({self.vloss_min:.7f}--->{valid_loss:.7f}) \t Saving The Model')
            torch.save(model.state_dict(), self.outfmodel)
            self.N_conv = 0
            self.vloss_min = valid_loss
        else:
            self.N_conv += 1
            print(
                f'Early stopping steps : {self.N_conv} out of {self.ConvergenceSteps}')
            if self.N_conv >= self.ConvergenceSteps:
                self.flag_stop = True


def train(model, loss_fcn, optimizer, DTrainLoader, verbose):
    # Train the DNN with training dataset
    model.train()
    train_loss = 0.0

    # for batch, (Input, Target) in enumerate(DTrainLoader):
    for Input_train, Target_train in DTrainLoader:
        # Compute prediction error
        y_train_pred = model(Input_train)
        loss_train_batch = loss_fcn(y_train_pred, Target_train)

        # BackPropagation
        optimizer.zero_grad()
        loss_train_batch.backward()
        optimizer.step()
        train_loss += loss_train_batch.item()

    # Check the validation dataset
    # model.eval()
    # with torch.no_grad():
    #     valid_loss = 0.0
    #     for Input_valid, Target_valid in DValidLoader:
    #         y_valid_pred = model(Input_valid)
    #         valid_loss_batch = loss_fcn(y_valid_pred, Target_valid)
    #         valid_loss += valid_loss_batch.item()

    if verbose:
        train_loss /= len(DTrainLoader)
        # train_loss *= batch_size
        # valid_loss /= len(DValidLoader)
        # valid_loss *= batch_size
        print(f"Training Loss : {train_loss:>7f}" ) # Validation Loss : {valid_loss:>7f}")

    return train_loss #, valid_loss


def train_withEarlyStopping(model, loss_fcn, optimizer, DTrainLoader, DValidLoader, early_stopping: EarlyStopping, verbose):

    while(early_stopping.flag_stop == False):

        # Train the DNN with training dataset
        model.train()
        train_loss = 0.0

        # for batch, (Input, Target) in enumerate(DTrainLoader):
        for Input_train, Target_train in DTrainLoader:
            # Compute prediction error
            y_train_pred = model(Input_train)
            loss_train_batch = loss_fcn(y_train_pred, Target_train)

            # BackPropagation
            optimizer.zero_grad()
            loss_train_batch.backward()
            optimizer.step()
            train_loss += loss_train_batch.item()

        # Check the validation dataset
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for Input_valid, Target_valid in DValidLoader:
                y_valid_pred = model(Input_valid)
                valid_loss_batch = loss_fcn(y_valid_pred, Target_valid)
                valid_loss += valid_loss_batch.item()

        if verbose:
            train_loss /= len(DTrainLoader)
            # train_loss *= batch_size
            valid_loss /= len(DValidLoader)
            # valid_loss *= batch_size
            print(
                f"Training Loss : {train_loss:>7f}   Validation Loss : {valid_loss:>7f}")

        early_stopping(valid_loss, model)
