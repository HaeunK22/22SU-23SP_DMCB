"""training with RNA and Methylation. Common : variant = 5 : 5."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from MLSurv import *
from loss import SurvLoss, latent_loss
from dataload import Data
from earlystopping import EarlyStopping
from torch import nn
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from statistics import mean
import os
parser = argparse.ArgumentParser()

parser.add_argument('--input1', '-i1', type=str, default='./../data/RNA_common_sorted.tsv')
# parser.add_argument('--input1', '-i1', type=str, default='./../data/RNA_common_sort_protein.tsv')
parser.add_argument('--input2', '-i2', type=str, default='./../data/methyl_common_sort.tsv')
parser.add_argument('--label', '-l', type=str, default='./../data/labels_common.tsv')
parser.add_argument('--feat_num', '-f', type=int, default=1000)
parser.add_argument('--coefficient', '-c', type=str, default='2,1,200')
parser.add_argument('--hiddendim', '-d', type=str, default='500,200')
parser.add_argument('--patience', '-p', type=int, default= '300')
parser.add_argument('--device', '-gpu', type=int, default=0)

config = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu')

input_dim = config.feat_num
coefficients = config.coefficient
hiddendim = config.hiddendim
patience = config.patience
alpha, beta, gamma = [float(c) for c in coefficients.split(',')]
dim1, dim2 = [int(c) for c in hiddendim.split(',')]

path = "./../figures/" + 'hid' + hiddendim + 'pat' + str(patience) + 'c5:s5'
if not os.path.exists(path):
    os.mkdir(path)
        
# stdscaler = StandardScaler()
stdscaler = MinMaxScaler()

learning_rate = 5e-5
n_epochs = 3000

mse_loss = nn.MSELoss()
surv_loss = SurvLoss(device)

# Data preparation
X1 = pd.read_table(config.input1, nrows = input_dim, header=None).transpose()
X2 = pd.read_table(config.input2, nrows = config.feat_num).transpose()
X1 = stdscaler.fit_transform(np.array(X1))
X2 = stdscaler.fit_transform(np.array(X2))
X1 = torch.tensor(np.array(X1), dtype = torch.float).to(device)
X2 = torch.tensor(np.array(X2), dtype = torch.float).to(device)

label = pd.read_table(config.label)
time, event = torch.tensor(label['time'], dtype=torch.float), torch.tensor(label['event'], dtype=torch.float)
time, event = time.unsqueeze(1).to(device), event.unsqueeze(1).to(device)
train_idx, test_idx, _, _ = train_test_split(np.arange(X1.shape[0]), event, test_size = 0.2, random_state = 2)

test_X1 = X1[test_idx, :]
test_X2 = X2[test_idx, :]
test_time = time[test_idx]
test_event = event[test_idx]

kf = KFold(n_splits= 5)
val_acc = []
for i, (train_index, val_index) in enumerate(kf.split(train_idx)):
    print(f'--------------Fold {i}---------------')
    train_X1 = X1[train_index, :]
    val_X1 = X1[val_index, :]
    train_X2 = X2[train_index, :]
    val_X2 = X2[val_index, :]
    train_time = time[train_index]
    val_time = time[val_index]
    train_event = time[train_index]
    val_event = time[val_index]
    
    training_data = Data(train_X1, train_X2, train_time, train_event, device)
    val_data = Data(val_X1, val_X2, val_time, val_event, device)
    
    train_dataloader = DataLoader(training_data, batch_size = train_index.size)
    val_dataloader = DataLoader(val_data, batch_size = val_index.size)
    
    model = MLSurv(input_dim, dim1, dim2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, train_loss, valid_loss, train_acc, avg_val_acc, valid_acc, va_loss, dis_loss, sv_loss = train_model(model, train_dataloader, val_dataloader, patience, n_epochs, optimizer)
    val_acc.append(valid_acc) 


print(f'--------------Train--------------------')
train_idx, val_idx, _, _ = train_test_split(train_idx, event[train_idx], test_size = 0.25, random_state = 4)
train_X1 = X1[train_idx, :]
val_X1 = X1[val_idx, :]
train_X2 = X2[train_idx, :]
val_X2 = X2[val_idx, :]
train_time = time[train_idx]
val_time = time[val_idx]
train_event = time[train_idx]
val_event = time[val_idx]
    
training_data = Data(train_X1, train_X2, train_time, train_event, device)
val_data = Data(val_X1, val_X2, val_time, val_event, device)
test_data = Data(test_X1, test_X2, test_time, test_event, device)
    
train_dataloader = DataLoader(training_data, batch_size = train_idx.size)
val_dataloader = DataLoader(val_data, batch_size = val_idx.size)
test_dataloader = DataLoader(test_data, batch_size = test_idx.size)

model = MLSurv(input_dim, dim1, dim2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model, train_loss, valid_loss, train_acc, avg_val_acc, valid_acc, va_loss, dis_loss, sv_loss = train_model(model, train_dataloader, val_dataloader, patience, n_epochs, optimizer)
test_acc = test_loop(test_dataloader, model)
print(f'# of total data : {X1.shape[0]}')
print(f'# of train data : {training_data.__len__()}')
print(f'# of val data : {val_data.__len__()}')
print(f'# of test data : {test_data.__len__()}')
plot_fold_acc(val_acc, test_acc, path)
visualize_loss(train_loss, valid_loss, path)
visualize_3train_loss(va_loss, dis_loss, sv_loss, path)
visualize_cindex(train_acc, avg_val_acc, test_acc, path)
torch.save(model, path + "/model2.pt")
