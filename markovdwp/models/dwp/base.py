import torch

from torch.nn.functional import softplus
from torch.distributions import Normal, Independent


class BaseDeepIndependentGaussian(torch.nn.Module):
    def __init__(self, input_shape, event_shape):
        super().__init__()
        self.input_shape = torch.Size(input_shape)
        self.event_shape = torch.Size(event_shape)

    def forward(self, input):
        n_dim_input, n_dim_event = len(self.input_shape), len(self.event_shape)
        assert input.shape[-n_dim_input:] == self.input_shape

        # flatten the composite batch dim, keeping feature dims intact
        output = self.features(input.flatten(0, -n_dim_input-1))

        # get location and scale with original batch dims (doubled features)
        output = output.reshape(*input.shape[:-n_dim_input], *output.shape[1:])
        loc, logscale = torch.chunk(output, 2, dim=-n_dim_event)
        # assert self.event_shape == loc.shape[-n_dim_event:]

        return Independent(Normal(loc, softplus(logscale)), n_dim_event)
