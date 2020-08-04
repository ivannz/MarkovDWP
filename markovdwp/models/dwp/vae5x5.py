import torch

from torch.nn import Conv2d, ConvTranspose2d
from .base import BaseDeepIndependentGaussian


class Encoder(BaseDeepIndependentGaussian):
    """The distribution of a latent code given a sample.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L251
    """
    def __init__(self, x_dim=1, z_dim=4, h_dim=64):
        super().__init__([x_dim, 5, 5], [z_dim, 1, 1])

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

        self.x_dim, self.z_dim, self.h_dim = x_dim, z_dim, h_dim


class Decoder(BaseDeepIndependentGaussian):
    """The posterior of each sample, given it latent code.

    Details
    -------
    A very complex hierarchical model, but is still a biparameteric diagonal Gaussian.

    Reference
    ---------
    See [Atanov et al. (2019; H.1 p.15)](http://arxiv.org/abs/1810.06943)

    https://github.com/bayesgroup/deep-weight-prior/blob/master/models/vae.py#L211
    """
    def __init__(self, z_dim=4, x_dim=1, h_dim=64):
        super().__init__([z_dim, 1, 1], [x_dim, 5, 5])

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

        self.z_dim, self.x_dim, self.h_dim = z_dim, x_dim, h_dim
