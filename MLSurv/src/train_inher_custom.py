"""training with RNA and Methylation."""
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
from sklearn.model_selection import KFold, cross_val_score
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
import os
parser = argparse.ArgumentParser()

parser.add_argument('--input1', '-i1', type=str, default='./../data/RNA_common_sorted.tsv')
parser.add_argument('--input2', '-i2', type=str, default='./../data/methyl_common_sort.tsv')
parser.add_argument('--label', '-l', type=str, default='./../data/labels_common.tsv')
parser.add_argument('--feat_num', '-f', type=int, default=1000)
parser.add_argument('--coefficient', '-c', type=str, default='2,1,200')
parser.add_argument('--device', '-gpu', type=int, default=0)

config = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu')

input_dim = config.feat_num
coefficients = config.coefficient
alpha,beta,gamma = [float(c) for c in coefficients.split(',')]


path = "./../figures/" + coefficients + 'inher'
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

encoder1 = Encoder(input_dim, 256)
decoder1 = Decoder(64, 256, input_dim)
encoder2 = Encoder(input_dim, 256)
decoder2 = Decoder(64, 256, input_dim)
model = MLSurv(encoder1, decoder1, encoder2, decoder2)
model = model.to(device)

learning_rate = 5e-5
n_epochs = 3000
batch_size = 256
patience = 500

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
    
    ### Delete if not needed ###
    t_vae_losses = []
    t_disentangle_losses = []
    t_surv_losses = []
    v_vae_losses = []
    v_disentangle_losses = []
    v_surv_losses = []
    avg_t_vae_losses = []
    avg_t_disentangle_losses = []
    avg_t_surv_losses = []
    avg_v_vae_losses = []
    avg_v_disentangle_losses = []
    avg_v_surv_losses = []
    
    
    # early_stopping object
    early_stopping = EarlyStopping(patience = patience, verbose = True, path = path + '/checkpointinher.pt')
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch, (X1, X2, time_batch, event_batch) in enumerate(train_dataloader, 1):
            # loss = 0
            # Compute predition and loss
            decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, output, inputmlp = model.forward(X1, X2)
            # VAE Loss
            vae_loss = mse_loss(decoder1_, X1) + mse_loss(decoder2_, X2) + mse_loss(decoder3_, X1) + mse_loss(decoder4_, X2) + \
                latent_loss(mu1, sigma1) + latent_loss(mu2, sigma2)
            # Disentangle Loss
            disentangle_loss = mse_loss(comm1, comm2)
            # Survival Prediction Loss
            surv_error = surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
            # Total loss
            loss = alpha*vae_loss + \
                beta*disentangle_loss + \
                gamma*surv_error
                
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Check train performance
            train_losses.append(loss.item())
            ### Delete if not needed ###
            t_vae_losses.append(vae_loss.item())
            t_disentangle_losses.append(disentangle_loss.item())
            t_surv_losses.append(surv_error.item())
            
            
        surv = pd.DataFrame(output.detach().cpu().numpy()).T
        durations = torch.flatten(train_time).detach().cpu().numpy()
        events = torch.flatten(train_event).detach().cpu().numpy()
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        train_ctd = ev.concordance_td('antolini')
    
        # validate   
        model.eval()
        for X1, X2, time_batch, event_batch in val_dataloader:
            decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, output, inputmlp = model.forward(X1, X2)
            # VAE Loss
            vae_loss = mse_loss(decoder1_, X1) + mse_loss(decoder2_, X2) + mse_loss(decoder3_, X1) + mse_loss(decoder4_, X2) + \
                latent_loss(mu1, sigma1) + latent_loss(mu2, sigma2)
            # Disentangle Loss
            disentangle_loss = mse_loss(comm1, comm2)
            # Survival Prediction Loss
            surv_error = surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
            # Total Loss
            loss = alpha*vae_loss + \
                beta*disentangle_loss + \
                gamma*surv_error
            valid_losses.append(loss.item())
            ### Delete if not needed ###
            v_vae_losses.append(vae_loss.item())
            v_disentangle_losses.append(disentangle_loss.item())
            v_surv_losses.append(surv_error.item())
        surv = pd.DataFrame(output.detach().cpu().numpy()).T
        durations = torch.flatten(val_time).detach().cpu().numpy()
        events = torch.flatten(val_event).detach().cpu().numpy()
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        valid_ctd = ev.concordance_td('antolini')

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_train_acc.append(train_ctd)
        avg_valid_acc.append(valid_ctd)
        
        ### Delete if not needed ###
        t_vae_loss = np.average(t_vae_losses)
        t_disentangle_loss = np.average(t_disentangle_losses)
        t_surv_loss = np.average(t_surv_losses)
        v_vae_loss = np.average(v_vae_losses)
        v_disentangle_loss = np.average(v_disentangle_losses)
        v_surv_loss = np.average(v_surv_losses)
        avg_t_vae_losses.append(t_vae_loss)
        avg_t_disentangle_losses.append(t_disentangle_losses)
        avg_t_surv_losses.append(t_surv_losses)
        avg_v_vae_losses.append(v_vae_losses)
        avg_v_disentangle_losses.append(v_disentangle_losses)
        avg_v_surv_losses.append(v_surv_losses)

        
        epoch_len = len(str(n_epochs))
        # print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
        #              f'train_loss: {train_loss:.5f} ' +
        #              f'valid_loss: {valid_loss:.5f}')
        # print(print_msg)
        #if (epoch % 100 == 0):
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] \n' +
                f'TRAIN\n' + 
                f'vae : {t_vae_loss:.5f} | disentangle : {t_disentangle_loss:.5f} | surv : {t_surv_loss:.5f} | total loss : {train_loss:.5f}\n' +
                f'VALIDATION\n' +
                f'vae : {v_vae_loss:.5f} | disentangle : {v_disentangle_loss:.5f} | surv : {v_surv_loss:.5f} | total loss : {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        ### Delete if not needed ###
        t_vae_losses = []
        t_disentangle_losses = []
        t_surv_losses = []
        v_vae_losses = []
        v_disentangle_losses = []
        v_surv_losses = []

        # make check point when validation loss decreases
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

   # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load(path + '/checkpointinher.pt'))
    
    return  model, avg_train_losses, avg_valid_losses, avg_train_acc, avg_valid_acc, avg_t_vae_losses, avg_t_disentangle_losses, avg_t_surv_losses
    
# iterate over test dataset to check model performance
def test_loop(dataloader, model):
    print("---------------- Start test -------------------")
    model.eval()
    with torch.no_grad():
        for X1, X2, time_batch, event_batch in dataloader:
            decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, output, inputmlp = model.forward(X1, X2)
    
        surv = pd.DataFrame(output.detach().cpu().numpy()).T
        durations = torch.flatten(test_time).detach().cpu().numpy()
        events = torch.flatten(test_event).detach().cpu().numpy()
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        test_ctd = ev.concordance_td('antolini')
        # VAE Loss
        vae_loss = mse_loss(decoder1_, X1) + mse_loss(decoder2_, X2) + mse_loss(decoder3_, X1) + mse_loss(decoder4_, X2) + \
                latent_loss(mu1, sigma1) + latent_loss(mu2, sigma2)
        # Disentangle Loss
        disentangle_loss = mse_loss(comm1, comm2)
        # Survival Prediction Loss
        surv_error = surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
        # Total Loss
        loss = alpha*vae_loss + \
            beta*disentangle_loss + \
            gamma*surv_error
        print(f"Test Accuracy : ")
        print(f"VAE loss: {vae_loss:>7f},\
            Disentangle loss: {disentangle_loss:>7f},\
            Surv loss: {surv_error:>7f},\
            Total loss: {loss:>7f},\
            Ctd : {test_ctd}")
    
    return test_ctd
        
# Visualize loss
def visualize_loss(t_loss, v_loss, path):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(t_loss)+1), t_loss, label='Training Loss')
    plt.plot(range(1,len(v_loss)+1), v_loss, label='Validation Loss')

    # validation loss의 최저값 지점을 찾기
    minposs = v_loss.index(min(v_loss))+1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(-25, 25)
    plt.xlim(0, len(t_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(path + '/MLSurvinher_loss_plot.png', bbox_inches = 'tight')
    
# Visualize vae loss, disentangle loss, survival prediction loss
def visualize_3train_loss(va_loss, dis_loss, sv_loss, path):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(va_loss)+1), va_loss, label='VAE Loss')
    plt.plot(range(1,len(dis_loss)+1), dis_loss, label='Disentangle Loss')
    plt.plot(range(1,len(sv_loss)+1), sv_loss, label='Survival Loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(va_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(path + '/MLSurvinher_3trainloss_plot.png', bbox_inches = 'tight')
    
# Visualize c-td     
def visualize_cindex(t_acc, v_acc, path):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(t_acc)+1), t_acc, label='Training C-td')
    plt.plot(range(1,len(v_acc)+1), v_acc, label='Validation C-td')

    plt.xlabel('epochs')
    plt.ylabel('c-td')
    plt.ylim(0.4, 1.0)
    plt.xlim(0, len(t_acc)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.title('test c-td:'+ str(test_acc))
    plt.show()
    fig.savefig(path+'/MLSurvinher_c-td_plot.png', bbox_inches = 'tight')
    

model, train_loss, valid_loss, train_acc, valid_acc, va_loss, dis_loss, sv_loss = train_model(model, patience, n_epochs, optimizer) 
test_acc = test_loop(test_dataloader, model)
visualize_cindex(train_acc, valid_acc, path)
visualize_loss(train_loss, valid_loss, path)
visualize_3train_loss(va_loss, dis_loss, sv_loss, path)
