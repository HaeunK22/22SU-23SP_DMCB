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


# class Encoder2(torch.nn.Module):
#     def __init__(self, D_in, H):
#         super(Encoder2, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)

#     def forward(self, x):

#         return F.relu(self.linear1(x))


# class Decoder2(torch.nn.Module):
#     def __init__(self, latent, H, D_out):
#         super(Decoder2, self).__init__()
#         self.linear1 = torch.nn.Linear(latent, H)
#         self.linear2 = torch.nn.Linear(H, D_out)

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         return F.relu(self.linear2(x))


class MLSurv(torch.nn.Module):
    latent_dim = 64

    def __init__(self, encoder1, decoder1, encoder2, decoder2):
        super(MLSurv, self).__init__()
        self.encoder1 = encoder1
        self.decoder1 = decoder1
        self.encoder2 = encoder2
        self.decoder2 = decoder2
        self._enc_mu = torch.nn.Linear(256, 64)
        self._enc_log_sigma = torch.nn.Linear(256, 64)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(96, 30, bias=True),
            torch.nn.Sigmoid())

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False), mu, sigma  # Reparameterization trick

    def forward(self, x1, x2):
        h_enc1 = self.encoder1(x1)
        z1, mu1, sigma1 = self._sample_latent(h_enc1)

        comm1, spe1 = z1.split([32, 32], dim=1)
        decoder1_ = self.decoder1(z1)   # Decoder1 output

        h_enc2 = self.encoder2(x2)
        z2, mu2, sigma2 = self._sample_latent(h_enc2)
        
        comm2, spe2 = z2.split([32, 32], dim=1)
        decoder2_ = self.decoder2(z2)   # Decoder2 output

        connect1 = torch.cat([comm2, spe1], dim=1)
        decoder3_ = self.decoder1(connect1) # Decoder3 output
        connect2 = torch.cat([comm1, spe2], dim=1)
        decoder4_ = self.decoder2(connect2) # Decoder4 output

        inputmlp_com = (comm1 + comm2) / 2
        inputmlp = torch.cat((inputmlp_com, spe1, spe2), 1) 

        out = self.risk_layer(inputmlp)

        return decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, out, inputmlp
