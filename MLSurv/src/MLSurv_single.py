import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Encoder1(torch.nn.Module):
    def __init__(self, D_in, H):
        super(Encoder1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)

    def forward(self, x):

        return F.relu(self.linear1(x))


class Decoder1(torch.nn.Module):
    def __init__(self, latent, H, D_out):
        super(Decoder1, self).__init__()
        self.linear1 = torch.nn.Linear(latent, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class MLSurv_single(torch.nn.Module):
    latent_dim = 64

    def __init__(self, encoder, decoder):
        super(MLSurv_single, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(256, 64)
        self._enc_log_sigma = torch.nn.Linear(256, 64)
        self.output_intervals=torch.arange(0., 365 * 31, 365)
        

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(64, 30, bias=True),
            torch.nn.Sigmoid())

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(mu.device)

        self.z_mean = mu
        self.z_sigma = sigma
       
        
        return mu + sigma * Variable(std_z, requires_grad=False), mu, sigma  # Reparameterization trick

    def forward(self, x):
        h_enc1 = self.encoder(x)
        # print("h_enc1", h_enc1)
        z1, mu1, sigma1 = self._sample_latent(h_enc1)
        # print("z1", z1)
        # print("mu1", mu1)
        # print("sigma1", sigma1)
        decoder_ = self.decoder(z1)
        # print("decoder_", decoder_)
        out = self.risk_layer(z1)

        return decoder_,mu1,sigma1,out,z1
