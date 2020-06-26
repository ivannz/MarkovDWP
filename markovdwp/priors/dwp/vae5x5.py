import torch
import torch.nn.functional as F

from torch.nn import Module, Conv2d, ConvTranspose2d


class Encoder(Module):
    """The distribution of a latent code given a sample.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L251
    """
    def __init__(self, x_dim=1, z_dim=4, h_dim=64, softplus=True):
        super().__init__()
        self.x_dim, self.z_dim, self.h_dim = x_dim, z_dim, h_dim
        self.softplus = softplus

        self.features = torch.nn.Sequential(
            Conv2d(1 * x_dim, 1 * h_dim, 3, padding=1),  # don't forget padding
            torch.nn.ELU(),

            Conv2d(1 * h_dim, 1 * h_dim, 3, padding=1),  # don't forget padding
            torch.nn.ELU(),

            Conv2d(1 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(2 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(2 * h_dim, 2 * z_dim, 1)  # final layer has twice the z_dim for chunking
        )

    def forward(self, input):
        mu, log_var = torch.chunk(self.features(input), 2, dim=1)
        return mu, F.softplus(log_var)


class Decoder(Module):
    """The posterior of each sample, given it latent code.

    Details
    -------
    A very complex hierarchical model, but is still a biparameteric diagonal Gaussian.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L211
    """
    def __init__(self, z_dim=4, x_dim=1, h_dim=64, softplus=True):
        super().__init__()
        self.z_dim, self.x_dim, self.h_dim = z_dim, x_dim, h_dim
        self.softplus = softplus

        # NB they allow fixing output log_var
        # https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L244
        self.features = torch.nn.Sequential(
            # torch.nn.ELU(),  # weird (?) -- an artifact of module construction, they just reuse it below
            Conv2d(
                1 * z_dim, 2 * h_dim, 1),
            torch.nn.ELU(),

            ConvTranspose2d(
                2 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            ConvTranspose2d(
                2 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(
                2 * h_dim, 1 * h_dim, 1),  # ooh, no traspose and ker=1 here, mmm
            torch.nn.ELU(),

            Conv2d(
                1 * h_dim, 2 * x_dim, 1),  # ... and no Conv^\top here too
        )

    def forward(self, input):
        mu, log_var = torch.chunk(self.features(input), 2, dim=1)
        return mu, F.softplus(log_var)
