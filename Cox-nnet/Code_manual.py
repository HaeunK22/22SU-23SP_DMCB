import torch
import numpy as np
from torch import nn

# Loss function
class PartialNLL(torch.nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    # Use matrix to avoid nested loops, for faster calculation
    def _make_R_rev(self, y):
        device = y.device
        R = np.zeros((y.shape[0], y.shape[0]))
        y = y.cpu().detach().numpy()
        for i in range(y.shape[0]):
            idx = np.where(y >= y[i])
            R[i, idx] = 1
        return torch.tensor(R, device=device)

    def forward(self, theta, time, observed):
        R_rev = self._make_R_rev(time).to(theta.device)
        exp_theta = torch.exp(theta)
        num_observed = torch.sum(observed)
        loss = -torch.sum((theta.reshape(-1) - torch.log(torch.sum((exp_theta * R_rev.t()), 0))) * observed) / num_observed
        if np.isnan(loss.data.tolist()):
            for a, b in zip(theta, exp_theta):
                print(a, b)
        return loss


class Coxnnet(nn.Module):
    def __init__(self, nfeat, n_hidden):
        super(Coxnnet, self).__init__()
        self.hidden_dim = n_hidden
        self.linear_tanh_stack = nn.Sequential(
            torch.nn.Linear(nfeat, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits
