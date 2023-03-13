"""Data loader for two datasets"""
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class Data(Dataset):
    def __init__(self, X1, X2, time, observed, device):
        self.x1_data = torch.tensor(X1, device=device)
        self.x2_data = torch.tensor(X2, device=device)
        self.time_data = torch.tensor(time, device=device)
        self.observed_data = torch.tensor(observed, device=device)
    
    def __len__(self):
        return len(self.x1_data)
    
    def __getitem__(self, index):
        x1 = torch.tensor(self.x1_data[index]).clone().detach()
        x2 = torch.tensor(self.x1_data[index]).clone().detach()
        time = torch.tensor(self.time_data[index]).clone().detach()
        observed = torch.tensor(self.observed_data[index]).clone().detach()
        return x1, x2, time, observed
