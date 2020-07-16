import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from torch.nn import Module, Conv2d, ConvTranspose2d


class Encoder(Module):
    """Latent code conditional distribution parameterization for 7x7 convolutions.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L135
    """
    def __init__(self, x_dim=1, z_dim=2, h_dim=32):
        super().__init__()
        self.x_dim, self.z_dim, self.h_dim = x_dim, z_dim, h_dim

        self.features = torch.nn.Sequential(
            Conv2d(1 * x_dim, 1 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(1 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(2 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(2 * h_dim, 2 * z_dim, 1)
        )

        self.kernel_size = 7, 7
        self.event_shape = z_dim, 1, 1

    def forward(self, input):
        assert input.shape[2:] == self.kernel_size

        loc, logscale = torch.chunk(self.features(input), 2, dim=1)
        return Independent(Normal(loc, F.softplus(logscale)), 3)


class Decoder(Module):
    """Sample conditional distribution parameterization for 7x7 convolutions.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L110
    """
    def __init__(self, z_dim=2, x_dim=1, h_dim=32):
        super().__init__()
        self.z_dim, self.x_dim, self.h_dim = z_dim, x_dim, h_dim

        self.features = torch.nn.Sequential(
            ConvTranspose2d(
                1 * z_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            ConvTranspose2d(
                2 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            ConvTranspose2d(
                2 * h_dim, 1 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(
                1 * h_dim, 2 * x_dim, 1)  # NO conv^\top here
        )

        self.kernel_size = 1, 1
        self.event_shape = x_dim, 7, 7

    def forward(self, input):
        assert input.shape[2:] == self.kernel_size

        loc, logscale = torch.chunk(self.features(input), 2, dim=1)
        return Independent(Normal(loc, F.softplus(logscale)), 3)
