import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


class DNN(nn.Module):
    def __init__(self, Num_InputVs, Num_Outputs):
        super(DNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.HLayer = nn.Sequential(
            nn.Linear(Num_InputVs, 10),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            # nn.Dropout(p=0.2), # Unactivate some neurons for generalization
            nn.Linear(10, Num_Outputs)
            # nn.Softmax(dim=-1) # nn.CrossEntropyLoss automatically apply Softmax
        )

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


def train(model, loss_fcn, optimizer, DTrainLoader, DValidLoader, verbose):
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

    return valid_loss


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

# def EarlyStopping(ConvergenceSteps: int, MaxIter: int, train_method):
#     N_conv = 0
#     epoch = 0
#     vloss_old = 10000000.
#     vloss_new = 0.
#     while((N_conv < ConvergenceSteps) & (epoch < 1000)):
#         print(f"Epoch : {epoch:04d}", end='  ')
#         vloss_new = train(model, loss_fcn, optimizer,
#                           DTrainLoader, DValidLoader, 1)
#         if(vloss_new < vloss_old):
#             print(
#                 f'Validation Loss Decreased({vloss_old:.7f}--->{vloss_new:.7f}) \t Saving The Model')
#             torch.save(model.state_dict(), outfmodel)
#             N_conv = 0
#             vloss_old = vloss_new
#         else:
#             N_conv += 1

#     epoch += 1


# Smoothing only near target label
def LabelSmoothing(Hard_label, NTarget):

    print("Label smoothing start!")
    # default_label = torch.zeros((NTarget))
    Soft_label_1 = torch.Tensor([0.8, 0.15, 0.05])
    Soft_label_2 = torch.Tensor([0.1, 0.8, 0.09, 0.01])
    Soft_label_3 = torch.Tensor([0.01, 0.09, 0.8, 0.09, 0.01])
    Soft_label_4 = torch.Tensor([0.01, 0.09, 0.8, 0.1])
    Soft_label_5 = torch.Tensor([0.05, 0.15, 0.8])

    out_list = []

    for i in range(len(Hard_label)):
        S_label = torch.zeros((NTarget))  # default_label
        if Hard_label[i] == 0:
            S_label[0:3] = Soft_label_1
        elif Hard_label[i] == 1:
            S_label[0:4] = Soft_label_2
        elif Hard_label[i] == NTarget-2:
            S_label[-4:] = Soft_label_4
        elif Hard_label[i] == NTarget-1:
            S_label[-3:] = Soft_label_5
        else:
            S_label[Hard_label[i]-2:Hard_label[i]+3] = Soft_label_3
        # print(S_label)
        out_list.append(S_label)

    Soft_label = torch.stack(out_list, dim=0)
    print("Label smoothing done!")

    return Soft_label
