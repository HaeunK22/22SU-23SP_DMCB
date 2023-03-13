"""Data loader for single dataset"""
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class Data(Dataset):
    def __init__(self, X, time, observed, device):
        self.x_data = torch.tensor(X, device=device)
        self.time_data = torch.tensor(time, device=device)
        self.observed_data = torch.tensor(observed, device=device)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x_data[index]).clone().detach()
        time = torch.tensor(self.time_data[index]).clone().detach()
        observed = torch.tensor(self.observed_data[index]).clone().detach()
        return x, time, observed
