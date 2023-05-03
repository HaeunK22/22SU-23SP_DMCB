import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)

    def forward(self, x):

        return F.relu(self.linear1(x))


class Decoder(torch.nn.Module):
    def __init__(self, latent, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class MLSurv_single(torch.nn.Module):

    def __init__(self, input_dim, dim1, dim2):
        super(MLSurv_single, self).__init__()
        self.encoder = Encoder(input_dim, dim1)
        self.decoder = Decoder(dim2, dim1, input_dim)
        
        self._enc_mu = torch.nn.Linear(dim1, dim2)
        self._enc_log_sigma = torch.nn.Linear(dim1, dim2)
        self.output_intervals=torch.arange(0., 365 * 31, 365)
        

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(int(dim2), 30, bias=True),
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
        h_enc1 = self.encoder(x)    # Encoder( Linear 3000 -> 500 / relu )

        z1, mu1, sigma1 = self._sample_latent(h_enc1)   # Gaussian Distribution: ( Linear 500 -> 200 )

        decoder_ = self.decoder(z1)

        out = self.risk_layer(z1)   # Risk Layer ( Linear 200 -> 30 / Sigmoid)

        return decoder_, mu1, sigma1, out, z1
