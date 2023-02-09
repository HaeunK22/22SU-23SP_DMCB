"""training with single modality."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from MLSurv_single import *
from loss import SurvLoss, latent_loss
from dataload import Data
from earlystopping import EarlyStopping
from torch import nn
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lifelines.utils import concordance_index
parser = argparse.ArgumentParser()

# parser.add_argument('--input1', '-i1', type=str, default='./../data/RNA_common_sorted.tsv')
parser.add_argument('--input2', '-i2', type=str, default='./../data/methyl_common_sort.tsv')
parser.add_argument('--label', '-l', type=str, default='./../data/labels_common.tsv')
parser.add_argument('--feat_num', '-f', type=int, default=1000)

config = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

input_dim = config.feat_num

stdscaler = StandardScaler()

# X1 = pd.read_table(config.input1, nrows = input_dim, header=None).transpose()
X2 = pd.read_table(config.input2, nrows = config.feat_num).transpose()
# X1 = stdscaler.fit_transform(np.array(X1))
X1 = stdscaler.fit_transform(np.array(X2))
X1 = torch.tensor(X1, dtype = torch.float).to(device)

label = pd.read_table(config.label)
time, event = torch.tensor(label['time'], dtype=torch.float), torch.tensor(label['event'], dtype=torch.float)
time, event = time.unsqueeze(1).to(device), event.unsqueeze(1).to(device)
train_idx, test_idx, _, _ = train_test_split(np.arange(X1.shape[0]), event, test_size = 0.2, random_state = 25)
train_idx, val_idx, _, _ = train_test_split(train_idx, event[train_idx], test_size = 0.25, random_state = 1)
# kf = KFold()
# for i, (train_idx, test_idx) kf.split(train_idx)

train_X = X1[train_idx, :]
val_X = X1[val_idx, :]
test_X = X1[test_idx, :]

train_time = time[train_idx]
val_time = time[val_idx]
test_time = time[test_idx]
train_event = event[train_idx]
val_event = event[val_idx]
test_event = event[test_idx]

training_data = Data(train_X, train_time, train_event, device)
val_data = Data(val_X, val_time, val_event, device)
test_data = Data(test_X, test_time, test_event, device)

train_dataloader = DataLoader(training_data, batch_size = 256)
val_dataloader = DataLoader(val_data, batch_size = 256)
test_dataloader = DataLoader(test_data, batch_size = 256)

encoder1 = Encoder1(input_dim, 256)
decoder1 = Decoder1(64, 256, input_dim)
model = MLSurv_single(encoder1, decoder1)
model = model.to(device)

learning_rate = 1e-4
n_epochs = 1000
batch_size = 256
patience = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()
surv_loss = SurvLoss(device)

def train_model(model, patience, n_epochs, optimizer):
    
    # Track training loss
    train_losses = []
    # Track validation loss
    valid_losses = []
    # Track average training loss
    avg_train_losses = []
    # Track average validation loss
    avg_valid_losses = []
    # Track average c-index for training data
    avg_train_acc = []
    # Track average c-index for validation data
    avg_valid_acc = []
    
    # early_stopping object
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch, (X, time_batch, event_batch) in enumerate(train_dataloader, 1):
            # loss = 0
            # Compute predition and loss
            decoder_, mu1, sigma1, output, z1 = model.forward(X)
            loss = mse_loss(decoder_, X) + \
                latent_loss(model.z_mean, model.z_sigma) + \
                10*surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Check train performance
            train_losses.append(loss.item())
            
        probs_by_interval = output.permute(1, 0).detach().cpu().numpy()
        train_c_index = [concordance_index(event_times = time_batch.cpu().detach().numpy(),
                                     predicted_scores = interval_probs,
                                     event_observed = event_batch.cpu().detach().numpy())
                for interval_probs in probs_by_interval]
            
    
        # validate   
        model.eval()
        for X, time_batch, event_batch in val_dataloader:
            decoder_, mu1, sigma1, output, z1 = model.forward(X)
            loss = mse_loss(decoder_, X) + \
                latent_loss(model.z_mean, model.z_sigma) + \
                10*surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
            valid_losses.append(loss.item())
        probs_by_interval = output.permute(1, 0).detach().cpu().numpy()
        valid_c_index = [concordance_index(event_times = time_batch.cpu().detach().numpy(),
                                     predicted_scores = interval_probs,
                                     event_observed = event_batch.cpu().detach().numpy())
                for interval_probs in probs_by_interval]

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        train_c_index = np.mean(train_c_index)
        valid_c_index = np.mean(valid_c_index)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_train_acc.append(train_c_index)
        avg_valid_acc.append(valid_c_index)
        
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # make check point when validation loss decreases
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

   # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load('0208_checkpoint.pt'))
    
    return  model, avg_train_losses, avg_valid_losses, avg_train_acc, avg_valid_acc
    
# iterate over test dataset to check model performance
def test_loop(dataloader, model):
    print("---------------- Start test loop.")
    model.eval()
    with torch.no_grad():
        for X1, time_batch, event_batch in dataloader:
            decoder_, mu1, sigma1, output, z1 = model.forward(X1)
    
        probs_by_interval = output.permute(1, 0).detach().cpu().numpy()
        c_index = [concordance_index(event_times = time_batch.cpu().detach().numpy(),
                                    predicted_scores = interval_probs,
                                    event_observed = event_batch.cpu().detach().numpy())
                for interval_probs in probs_by_interval]
        recon_error = mse_loss(decoder_,X1).item()
        latent_error = latent_loss(model.z_mean, model.z_sigma).item()
        surv_error = surv_loss.forward(output, time_batch, event_batch, model.output_intervals.double().to(device)).item()
        tot_error = recon_error + latent_error + surv_error
        print(f"Test Accuracy : ")
        print(f"Recon: {recon_error:>7f},\
            Latent: {latent_error:>7f},\
            Surv: {surv_error:>7f},\
            Total: {tot_error:>7f},\
            Cindex : {np.mean(c_index):>7f}")
        
# Visualize loss     
def visualize_loss(t_loss, v_loss):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(t_loss)+1), t_loss, label='Training Loss')
    plt.plot(range(1,len(v_loss)+1), v_loss, label='Validation Loss')

    # validation loss의 최저값 지점을 찾기
    minposs = v_loss.index(min(v_loss))+1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.5, 1.6)
    plt.xlim(0, len(t_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('1000_DNAm_loss_plot.png', bbox_inches = 'tight')
    
# Visualize c-index     
def visualize_cindex(t_acc, v_acc):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(t_acc)+1), t_acc, label='Training C-index')
    plt.plot(range(1,len(v_acc)+1), v_acc, label='Validation C-index')

    plt.xlabel('epochs')
    plt.ylabel('c-index')
    plt.ylim(0.4, 1.0)
    plt.xlim(0, len(t_acc)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('1000_DNAm_c-index_plot.png', bbox_inches = 'tight')
    

# t_loss_array = []
# v_loss_array = []
# for i in range(50):
#     model, train_loss, valid_loss = train_model(model, patience, n_epochs, optimizer)
    
model, train_loss, valid_loss, train_acc, valid_acc = train_model(model, patience, n_epochs, optimizer) 
visualize_loss(train_loss, valid_loss)
visualize_cindex(train_acc, valid_acc)
test_loop(test_dataloader, model)
