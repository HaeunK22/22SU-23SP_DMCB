"""MLSurv for finding latent vector size. Common : variant = 4 : 6."""
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


class MLSurv(torch.nn.Module):

    def __init__(self, input_dim, dim1, dim2):
        super(MLSurv, self).__init__()
        self.encoder1 = Encoder(input_dim, dim1)
        self.decoder1 = Decoder(dim2, dim1, input_dim)
        self.encoder2 = Encoder(input_dim, dim1)
        self.decoder2 = Decoder(dim2, dim1, input_dim)
        self.hidden_layer1 = dim1
        self.hidden_layer2 = dim2
        self._enc_mu = torch.nn.Linear(dim1, dim2)
        self._enc_log_sigma = torch.nn.Linear(dim1, dim2)
        self.output_intervals=torch.arange(0., 365 * 31, 365)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(int(dim2/10*16), 30, bias=True),
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

    def forward(self, x1, x2):
        # VAE 1
        h_enc1 = self.encoder1(x1)  # Linear : input_dim -> hidden_dim1(default:500)
        z1, mu1, sigma1 = self._sample_latent(h_enc1)   # Latent vector, size hidden_dim2(default:200)

        comm1, spe1 = z1.split([int(self.hidden_layer2/10*4), int(self.hidden_layer2/10*6)], dim=1)
        decoder1_ = self.decoder1(z1)   # Decoder1 output : hidden_dim2(default:200) -> input_dim

        # VAE 2
        h_enc2 = self.encoder2(x2)
        z2, mu2, sigma2 = self._sample_latent(h_enc2)
        
        comm2, spe2 = z2.split([int(self.hidden_layer2/10*4), int(self.hidden_layer2/10*6)], dim=1)
        decoder2_ = self.decoder2(z2)

        # variational from latent 1 & inherent from latent 2
        connect1 = torch.cat([comm2, spe1], dim=1)
        decoder3_ = self.decoder1(connect1) # Decoder3 output
        
        # variational from latent 2 & inherent from latent 1
        connect2 = torch.cat([comm1, spe2], dim=1)
        decoder4_ = self.decoder2(connect2) # Decoder4 output

        # input of final risk layer
        inputmlp_com = (comm1 + comm2) / 2
        inputmlp = torch.cat((inputmlp_com, spe1, spe2), 1) 

        out = self.risk_layer(inputmlp)

        return decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, out, inputmlp
