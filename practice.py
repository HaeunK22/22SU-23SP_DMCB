"""training with RNA and Methylation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from MLSurv import *
from loss import SurvLoss, latent_loss
from dataload import Data
from earlystopping_inher import EarlyStopping
from torch import nn
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
import os
parser = argparse.ArgumentParser()

parser.add_argument('--input1', '-i1', type=str, default='./../data/RNA_common_sorted.tsv')
# parser.add_argument('--input1', '-i1', type=str, default='./../data/RNA_common_sort_protein.tsv')
parser.add_argument('--input2', '-i2', type=str, default='./../data/methyl_common_sort.tsv')
parser.add_argument('--label', '-l', type=str, default='./../data/labels_common.tsv')
parser.add_argument('--feat_num', '-f', type=int, default=1000)


config = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

input_dim = config.feat_num

path = "./../figures/" + 'lat100,c5:s5,es300'
if not os.path.exists(path):
    os.mkdir(path)

# stdscaler = StandardScaler()
stdscaler = MinMaxScaler()

X1 = pd.read_table(config.input1, nrows = input_dim, header=None).transpose()
X2 = pd.read_table(config.input2, nrows = config.feat_num).transpose()
X1 = stdscaler.fit_transform(np.array(X1))
X2 = stdscaler.fit_transform(np.array(X2))
X1 = torch.tensor(np.array(X1), dtype = torch.float).to(device)
X2 = torch.tensor(np.array(X2), dtype = torch.float).to(device)

label = pd.read_table(config.label)
time, event = torch.tensor(label['time'], dtype=torch.float), torch.tensor(label['event'], dtype=torch.float)
time, event = time.unsqueeze(1).to(device), event.unsqueeze(1).to(device)
train_idx, test_idx, _, _ = train_test_split(np.arange(X1.shape[0]), event, test_size = 0.2, random_state = 25)
train_idx, val_idx, _, _ = train_test_split(train_idx, event[train_idx], test_size = 0.25, random_state = 1)

train_X1 = X1[train_idx, :]
val_X1 = X1[val_idx, :]
test_X1 = X1[test_idx, :]
train_X2 = X2[train_idx, :]
val_X2 = X2[val_idx, :]
test_X2 = X2[test_idx, :]

train_time = time[train_idx]
val_time = time[val_idx]
test_time = time[test_idx]
train_event = event[train_idx]
val_event = event[val_idx]
test_event = event[test_idx]

training_data = Data(train_X1, train_X2, train_time, train_event, device)
val_data = Data(val_X1, val_X2, val_time, val_event, device)
test_data = Data(test_X1, test_X2, test_time, test_event, device)

train_dataloader = DataLoader(training_data, batch_size = train_idx.size)
val_dataloader = DataLoader(val_data, batch_size = val_idx.size)
test_dataloader = DataLoader(test_data, batch_size = test_idx.size)


model = MLSurv(input_dim, 500, 100)
model = model.to(device)

learning_rate = 5e-5
n_epochs = 3000
patience = 300

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()
surv_loss = SurvLoss(device)
    

model, train_loss, valid_loss, train_acc, valid_acc, va_loss, dis_loss, sv_loss = train_model(model, patience, n_epochs, optimizer)
test_acc = test_loop(test_dataloader, model)
visualize_loss(train_loss, valid_loss, path)
visualize_3train_loss(va_loss, dis_loss, sv_loss, path)
visualize_cindex(train_acc, valid_acc, test_acc, path)
torch.save(model, path + "/protein_model.pt")
