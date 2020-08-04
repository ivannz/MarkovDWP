import torch

from torch.nn import Conv2d, ConvTranspose2d
from .base import BaseDeepIndependentGaussian


class Encoder(BaseDeepIndependentGaussian):
    """Latent code conditional distribution parameterization for 7x7 convolutions.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L135
    """
    def __init__(self, x_dim=1, z_dim=2, h_dim=32):
        super().__init__([x_dim, 7, 7], [z_dim, 1, 1])

        self.features = torch.nn.Sequential(
            Conv2d(1 * x_dim, 1 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(1 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(2 * h_dim, 2 * h_dim, 3),
            torch.nn.ELU(),

            Conv2d(2 * h_dim, 2 * z_dim, 1)
        )

        self.x_dim, self.z_dim, self.h_dim = x_dim, z_dim, h_dim


class Decoder(BaseDeepIndependentGaussian):
    """Sample conditional distribution parameterization for 7x7 convolutions.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L110
    """
    def __init__(self, z_dim=2, x_dim=1, h_dim=32):
        super().__init__([z_dim, 1, 1], [x_dim, 7, 7])

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

        self.z_dim, self.x_dim, self.h_dim = z_dim, x_dim, h_dim
